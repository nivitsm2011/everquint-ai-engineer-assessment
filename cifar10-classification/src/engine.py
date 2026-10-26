"""Training and evaluation loops, checkpointing and early stopping."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.amp import GradScaler, autocast
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, OneCycleLR
from torch.utils.data import DataLoader

from src.config import Config
from src.utils import AverageMeter

History = dict[str, list[float]]


@dataclass
class EpochResult:
    """Average loss and accuracy (0-1) over one pass, plus optional per-sample outputs."""

    loss: float
    accuracy: float
    predictions: Tensor | None = None
    targets: Tensor | None = None


@dataclass
class CheckpointState:
    """Training progress recovered from a checkpoint."""

    epoch: int
    best_val_acc: float
    history: History
    config: dict[str, Any] = field(default_factory=dict)


def build_optimizer(model: nn.Module, config: Config) -> Optimizer:
    """Create the optimiser named in ``config`` over all model parameters."""
    if config.optimizer == "sgd":
        return SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    return AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def build_scheduler(optimizer: Optimizer, config: Config, steps_per_epoch: int) -> LRScheduler:
    """Create the LR scheduler named in ``config``.

    ``OneCycleLR`` is stepped once per batch and ``CosineAnnealingLR`` once per
    epoch; :func:`train_one_epoch` handles both.
    """
    if config.scheduler == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
        )
    return CosineAnnealingLR(optimizer, T_max=config.epochs)


def make_grad_scaler(config: Config, device: torch.device) -> GradScaler | None:
    """Return a gradient scaler when mixed precision is requested and the device is CUDA."""
    if config.use_amp and device.type == "cuda":
        return GradScaler("cuda")
    return None


def _backward_and_step(
    loss: Tensor,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler | None,
    grad_clip: float | None,
) -> None:
    """Backpropagate ``loss`` and update weights, with optional AMP scaling and clipping."""
    if scaler is None:
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        return

    scaler.scale(loss).backward()
    if grad_clip:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    device: torch.device,
    scaler: GradScaler | None = None,
    grad_clip: float | None = None,
) -> EpochResult:
    """Train ``model`` for one epoch.

    Args:
        scaler: Enables mixed precision when provided; ``None`` keeps full precision.
        grad_clip: Maximum gradient norm; ``None`` or 0 disables clipping.

    Returns:
        Average training loss and accuracy over the epoch.
    """
    model.train()
    loss_meter, accuracy_meter = AverageMeter(), AverageMeter()
    step_per_batch = isinstance(scheduler, OneCycleLR)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, targets)
        _backward_and_step(loss, model, optimizer, scaler, grad_clip)
        if step_per_batch:
            scheduler.step()

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        accuracy_meter.update((logits.argmax(dim=1) == targets).float().mean().item(), batch_size)

    if not step_per_batch:
        scheduler.step()
    return EpochResult(loss_meter.average, accuracy_meter.average)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_predictions: bool = False,
) -> EpochResult:
    """Evaluate ``model`` on ``loader`` without updating any weights.

    Args:
        return_predictions: Also return every predicted class and true label (on CPU).
    """
    model.eval()
    loss_meter, accuracy_meter = AverageMeter(), AverageMeter()
    all_predictions: list[Tensor] = []
    all_targets: list[Tensor] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.argmax(dim=1)

        batch_size = targets.size(0)
        loss_meter.update(criterion(logits, targets).item(), batch_size)
        accuracy_meter.update((predictions == targets).float().mean().item(), batch_size)
        if return_predictions:
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    result = EpochResult(loss_meter.average, accuracy_meter.average)
    if return_predictions:
        result.predictions = torch.cat(all_predictions)
        result.targets = torch.cat(all_targets)
    return result


class EarlyStopping:
    """Signals when a monitored metric has not improved for ``patience`` epochs.

    Args:
        patience: Number of consecutive non-improving epochs tolerated.
        mode: ``"max"`` for metrics such as accuracy, ``"min"`` for losses.
        min_delta: Smallest change that counts as an improvement.
    """

    def __init__(self, patience: int, mode: str = "max", min_delta: float = 0.0) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.counter = 0

    def step(self, value: float) -> bool:
        """Record this epoch's metric; return ``True`` if training should stop."""
        if self._is_improvement(value):
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def state_dict(self) -> dict[str, Any]:
        """Return the tracker's state for checkpointing."""
        return {"best": self.best, "counter": self.counter}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state produced by :meth:`state_dict`."""
        self.best = state["best"]
        self.counter = state["counter"]


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    scaler: GradScaler | None,
    early_stopping: EarlyStopping,
    epoch: int,
    best_val_acc: float,
    config: Config,
    history: History,
) -> None:
    """Write everything needed to resume training or reload the model for evaluation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "early_stopping": early_stopping.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "config": asdict(config),
            "history": history,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: GradScaler | None = None,
    early_stopping: EarlyStopping | None = None,
    device: torch.device | str = "cpu",
) -> CheckpointState:
    """Load a checkpoint into ``model`` and, when given, the training objects.

    Pass only ``model`` to restore weights for evaluation; pass the optimiser,
    scheduler, scaler and early-stopping tracker as well to resume training.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint["scaler"] is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    if early_stopping is not None:
        early_stopping.load_state_dict(checkpoint["early_stopping"])
    return CheckpointState(
        epoch=checkpoint["epoch"],
        best_val_acc=checkpoint["best_val_acc"],
        history=checkpoint["history"],
        config=checkpoint["config"],
    )
