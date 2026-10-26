"""Shared helpers: reproducibility, device selection, logging, metrics and timing."""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path
from types import TracebackType

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every random number generator used in the project.

    Covers ``random``, NumPy, PyTorch (CPU and all CUDA devices) and forces
    deterministic cuDNN kernels.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` that seeds NumPy and ``random`` inside DataLoader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded generator for reproducible DataLoader shuffling and splits."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_device() -> torch.device:
    """Pick the best available device in the order CUDA, Apple MPS, CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logger(name: str = "cifar10", log_file: str | Path | None = None) -> logging.Logger:
    """Create a logger that writes to stdout and, optionally, to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class AverageMeter:
    """Tracks the running average of a metric such as loss or accuracy."""

    def __init__(self) -> None:
        self.value = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self) -> None:
        """Clear all accumulated statistics."""
        self.value = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """Record ``value`` observed over ``n`` samples."""
        self.value = value
        self.sum += value * n
        self.count += n

    @property
    def average(self) -> float:
        """Mean of all recorded values, weighted by sample count."""
        return self.sum / self.count if self.count else 0.0


class Timer:
    """Context manager that measures wall-clock time in seconds.

    Example:
        >>> with Timer() as timer:
        ...     pass
        >>> timer.elapsed >= 0
        True
    """

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.elapsed = time.perf_counter() - self._start
