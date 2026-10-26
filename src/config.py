"""Experiment configuration: YAML loading with command-line overrides."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Sequence, get_type_hints

import yaml

DEFAULT_CONFIG_PATH = Path("configs/config.yaml")

MODEL_NAMES = ("baseline_cnn", "resnet_cnn")
OPTIMIZERS = ("sgd", "adamw")
SCHEDULERS = ("onecycle", "cosine")

# Short CLI spellings for frequently used fields.
FLAG_ALIASES = {
    "model_name": ("--model",),
    "learning_rate": ("--lr",),
    "batch_size": ("--batch-size",),
}


@dataclass(frozen=True)
class Config:
    """Typed, validated container for every tunable experiment setting."""

    seed: int = 42

    data_dir: str = "data"
    batch_size: int = 128
    num_workers: int | None = None
    val_split: int = 5000

    model_name: str = "baseline_cnn"
    dropout: float = 0.3

    epochs: int = 10
    learning_rate: float = 0.1
    weight_decay: float = 5e-4
    optimizer: str = "sgd"
    momentum: float = 0.9
    scheduler: str = "onecycle"
    grad_clip: float = 0.0
    use_amp: bool = False
    early_stopping_patience: int = 5

    checkpoint_dir: str = "outputs/checkpoints"
    plot_dir: str = "outputs/plots"
    log_dir: str = "outputs/logs"
    results_dir: str = "outputs/results"
    onnx_path: str = "outputs/model.onnx"

    def __post_init__(self) -> None:
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"model_name must be one of {MODEL_NAMES}, got {self.model_name!r}")
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {self.scheduler!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0 < self.val_split < 50_000:
            raise ValueError(f"val_split must be in (0, 50000), got {self.val_split}")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive")
        if self.num_workers is not None and self.num_workers < 0:
            raise ValueError("num_workers must be non-negative or null (auto)")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must be non-negative")


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load a YAML file into a :class:`Config`, then apply ``overrides``.

    Args:
        path: Location of the YAML configuration file.
        overrides: Optional key/value pairs that take precedence over the file.

    Raises:
        ValueError: If a key is not a known configuration field or a value is invalid.
    """
    with open(path, encoding="utf-8") as file:
        values: dict[str, Any] = yaml.safe_load(file) or {}
    values.update(overrides or {})

    field_types = get_type_hints(Config)
    unknown = sorted(set(values) - set(field_types))
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    # YAML 1.1 reads "1e-3" as a string, so coerce numeric fields explicitly.
    coerced = {
        key: field_types[key](value) if field_types[key] in (int, float) else value
        for key, value in values.items()
    }
    return replace(Config(), **coerced)


def build_arg_parser(description: str = "CIFAR-10 classification") -> argparse.ArgumentParser:
    """Create a parser with ``--config`` plus one optional flag per config field.

    Entry points can add their own arguments before parsing.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the YAML config file."
    )
    for field in fields(Config):
        flags = (f"--{field.name}", *FLAG_ALIASES.get(field.name, ()))
        parser.add_argument(
            *flags, dest=field.name, default=None, help=f"Override '{field.name}' from the config file."
        )
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Build a :class:`Config` from the YAML file named in ``args`` and any CLI overrides.

    Override values are parsed as YAML scalars, so ``--use_amp true`` yields a
    bool and ``--learning_rate 1e-3`` yields a float.
    """
    overrides = {
        field.name: yaml.safe_load(getattr(args, field.name))
        for field in fields(Config)
        if getattr(args, field.name) is not None
    }
    return load_config(args.config, overrides)


def parse_config(argv: Sequence[str] | None = None) -> Config:
    """Parse the command line into a :class:`Config`."""
    return config_from_args(build_arg_parser().parse_args(argv))
