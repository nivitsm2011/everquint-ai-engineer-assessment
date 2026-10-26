"""Tests for seeding, metrics helpers, checkpointing and configuration loading."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from src.config import Config, load_config, parse_config
from src.engine import (
    EarlyStopping,
    build_optimizer,
    build_scheduler,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)
from src.utils import AverageMeter, Timer, get_device, make_generator, set_seed


def test_set_seed_gives_identical_tensors_on_repeat() -> None:
    set_seed(7)
    first = (torch.rand(5), torch.randn(5), torch.randperm(10))
    set_seed(7)
    second = (torch.rand(5), torch.randn(5), torch.randperm(10))
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_different_seeds_give_different_tensors() -> None:
    set_seed(1)
    first = torch.rand(5)
    set_seed(2)
    assert not torch.equal(first, torch.rand(5))


def test_make_generator_is_reproducible() -> None:
    assert torch.equal(
        torch.randperm(20, generator=make_generator(3)), torch.randperm(20, generator=make_generator(3))
    )


def test_get_device_returns_a_known_device() -> None:
    assert get_device().type in {"cuda", "mps", "cpu"}


def test_average_meter_weights_by_sample_count() -> None:
    meter = AverageMeter()
    meter.update(2.0, n=2)
    meter.update(5.0, n=1)
    assert meter.average == pytest.approx(3.0)
    meter.reset()
    assert meter.average == 0.0


def test_timer_measures_elapsed_time() -> None:
    with Timer() as timer:
        sum(range(1000))
    assert timer.elapsed > 0


def test_early_stopping_stops_after_patience_without_improvement() -> None:
    stopper = EarlyStopping(patience=2, mode="max")
    assert [stopper.step(v) for v in (0.5, 0.6, 0.55, 0.58, 0.7)] == [False, False, False, True, False]


def test_early_stopping_min_mode() -> None:
    stopper = EarlyStopping(patience=1, mode="min")
    assert [stopper.step(v) for v in (1.0, 0.9, 0.95)] == [False, False, True]


def _tiny_training_objects(config: Config) -> tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
    optimizer = build_optimizer(model, config)
    return model, optimizer, build_scheduler(optimizer, config, steps_per_epoch=4)


def test_checkpoint_roundtrip_restores_everything(tmp_path: Path) -> None:
    config = Config(epochs=3, learning_rate=0.05)
    model, optimizer, scheduler = _tiny_training_objects(config)
    for _ in range(2):  # move optimiser and scheduler away from their initial state
        model(torch.randn(4, 3, 2, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
    stopper = EarlyStopping(patience=5)
    stopper.step(0.5)
    history = {"val_acc": [0.4, 0.5]}

    path = tmp_path / "ckpt" / "last.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, scheduler=scheduler, scaler=None, early_stopping=stopper,
        epoch=2, best_val_acc=0.5, config=config, history=history,
    )

    new_model, new_optimizer, new_scheduler = _tiny_training_objects(config)
    new_stopper = EarlyStopping(patience=5)
    state = load_checkpoint(path, new_model, new_optimizer, new_scheduler, None, new_stopper)

    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), new_model.state_dict().values()))
    assert new_optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])
    assert new_scheduler.state_dict()["_step_count"] == scheduler.state_dict()["_step_count"]
    assert new_stopper.state_dict() == stopper.state_dict()
    assert (state.epoch, state.best_val_acc, state.history) == (2, 0.5, history)
    assert state.config["learning_rate"] == 0.05


@pytest.mark.parametrize("scheduler_name", ["onecycle", "cosine"])
@pytest.mark.parametrize("grad_clip", [None, 1.0])
def test_train_one_epoch_updates_weights_and_steps_scheduler(scheduler_name: str, grad_clip: float | None) -> None:
    config = Config(epochs=2, scheduler=scheduler_name)
    model, optimizer, scheduler = _tiny_training_objects(config)
    before = [p.clone() for p in model.parameters()]
    loader = [(torch.randn(8, 3, 2, 2), torch.randint(0, 3, (8,))) for _ in range(4)]

    result = train_one_epoch(
        model, loader, nn.CrossEntropyLoss(), optimizer, scheduler, torch.device("cpu"), grad_clip=grad_clip
    )

    assert 0.0 <= result.accuracy <= 1.0 and result.loss > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    steps = 4 if scheduler_name == "onecycle" else 1  # per batch vs per epoch
    assert scheduler.last_epoch == steps


def test_train_one_epoch_with_grad_scaler_path() -> None:
    config = Config(epochs=2)
    model, optimizer, scheduler = _tiny_training_objects(config)
    loader = [(torch.randn(8, 3, 2, 2), torch.randint(0, 3, (8,))) for _ in range(4)]
    scaler = torch.amp.GradScaler("cpu")
    result = train_one_epoch(
        model, loader, nn.CrossEntropyLoss(), optimizer, scheduler, torch.device("cpu"), scaler, grad_clip=1.0
    )
    assert torch.isfinite(torch.tensor(result.loss))


def test_config_loads_yaml_and_applies_cli_overrides() -> None:
    config = parse_config(["--epochs", "3", "--lr", "1e-3", "--use_amp", "true", "--model", "resnet_cnn"])
    assert (config.epochs, config.learning_rate, config.use_amp, config.model_name) == (3, 0.001, True, "resnet_cnn")
    assert config.seed == load_config().seed


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        replace(Config(), model_name="bogus")
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(overrides={"not_a_field": 1})
