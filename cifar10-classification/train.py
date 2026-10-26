"""Training entry point.

Example:
    python train.py --model baseline_cnn
    python train.py --quick          # 2-epoch smoke test on a small subset
    python train.py --resume         # continue from outputs/checkpoints/<run>/last.pt
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from torch import nn

from src.config import Config, build_arg_parser, config_from_args
from src.data import get_dataloaders
from src.engine import (
    EarlyStopping,
    History,
    build_optimizer,
    build_scheduler,
    evaluate,
    load_checkpoint,
    make_grad_scaler,
    save_checkpoint,
    train_one_epoch,
)
from src.models import build_model, count_parameters
from src.utils import Timer, get_device, set_seed, setup_logger

QUICK_EPOCHS = 2
QUICK_SAMPLE_LIMIT = 2048
HISTORY_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time")


def parse_args(argv: Sequence[str] | None = None) -> tuple[Config, bool, bool]:
    """Parse the command line into ``(config, resume, quick)``."""
    parser = build_arg_parser("Train a CIFAR-10 classifier")
    parser.add_argument("--resume", action="store_true", help="Resume from the run's last.pt checkpoint.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Smoke test: {QUICK_EPOCHS} epochs on {QUICK_SAMPLE_LIMIT} images per split.",
    )
    parser.add_argument("--amp", action="store_true", help="Mixed-precision training (CUDA only).")
    args = parser.parse_args(argv)
    config = config_from_args(args)
    if args.amp:
        config = replace(config, use_amp=True)
    if args.quick:
        config = replace(config, epochs=QUICK_EPOCHS)
    return config, args.resume, args.quick


def save_history(path: Path, history: History, best_val_acc: float) -> None:
    """Write per-epoch metrics and learning rates as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump({"best_val_acc": best_val_acc, "history": history}, file, indent=2)


def run_training(config: Config, run_name: str, resume: bool, sample_limit: int | None) -> None:
    """Train ``config.model_name``, logging metrics and saving checkpoints and history."""
    set_seed(config.seed)
    device = get_device()
    logger = setup_logger(run_name, Path(config.log_dir) / f"{run_name}.log")

    train_loader, val_loader, _, _ = get_dataloaders(config, limit=sample_limit)
    model = build_model(config.model_name, dropout=config.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    scaler = make_grad_scaler(config, device)
    early_stopping = EarlyStopping(config.early_stopping_patience, mode="max")

    if config.use_amp and scaler is None:
        logger.warning("Mixed precision is only supported on CUDA; device is %s, training in full precision", device.type)
    logger.info(
        "Run %s | device %s | %s parameters | %d train / %d val images",
        run_name,
        device,
        f"{count_parameters(model):,}",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    checkpoint_dir = Path(config.checkpoint_dir) / run_name
    history: History = {key: [] for key in HISTORY_KEYS}
    start_epoch, best_val_acc = 0, 0.0
    if resume:
        state = load_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, scheduler, scaler, early_stopping, device
        )
        start_epoch, best_val_acc, history = state.epoch, state.best_val_acc, state.history
        logger.info("Resumed from epoch %d (best val acc %.2f%%)", start_epoch, best_val_acc * 100)

    grad_clip = config.grad_clip or None
    for epoch in range(start_epoch + 1, config.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        with Timer() as timer:
            train_result = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, scaler, grad_clip
            )
            val_result = evaluate(model, val_loader, criterion, device)

        for key, value in zip(
            HISTORY_KEYS,
            (train_result.loss, train_result.accuracy, val_result.loss, val_result.accuracy, lr, timer.elapsed),
        ):
            history[key].append(value)
        logger.info(
            "Epoch %02d/%d | train loss %.4f acc %.2f%% | val loss %.4f acc %.2f%% | lr %.5f | %.1fs",
            epoch,
            config.epochs,
            train_result.loss,
            train_result.accuracy * 100,
            val_result.loss,
            val_result.accuracy * 100,
            lr,
            timer.elapsed,
        )

        is_best = val_result.accuracy > best_val_acc
        best_val_acc = max(best_val_acc, val_result.accuracy)
        should_stop = early_stopping.step(val_result.accuracy)

        checkpoint = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            early_stopping=early_stopping,
            epoch=epoch,
            best_val_acc=best_val_acc,
            config=config,
            history=history,
        )
        save_checkpoint(checkpoint_dir / "last.pt", **checkpoint)
        if is_best:
            save_checkpoint(checkpoint_dir / "best.pt", **checkpoint)
            logger.info("New best validation accuracy: %.2f%%", best_val_acc * 100)
        save_history(Path(config.results_dir) / f"{run_name}_history.json", history, best_val_acc)

        if should_stop:
            logger.info("Early stopping: no improvement for %d epochs", config.early_stopping_patience)
            break

    logger.info("Finished. Best validation accuracy: %.2f%%", best_val_acc * 100)


def main(argv: Sequence[str] | None = None) -> None:
    """Command-line entry point."""
    config, resume, quick = parse_args(argv)
    run_name = f"{config.model_name}_quick" if quick else config.model_name
    run_training(config, run_name, resume, QUICK_SAMPLE_LIMIT if quick else None)


if __name__ == "__main__":
    main()
