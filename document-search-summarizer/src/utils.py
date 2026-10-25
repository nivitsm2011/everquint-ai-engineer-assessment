"""Shared helpers: logging, timing decorator and a simple disk cache."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def timed(func: F) -> F:
    """Log how long ``func`` took, in milliseconds."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            ms = (time.perf_counter() - start) * 1000
            get_logger(func.__module__).info("%s took %.1f ms", func.__name__, ms)

    return wrapper  # type: ignore[return-value]


class DiskCache:
    """Tiny JSON-on-disk cache keyed by a hash of the input (e.g. LLM prompts)."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _file(self, key: str) -> Path:
        return self.directory / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".json")

    def get(self, key: str) -> Any | None:
        """Return the cached value for ``key`` or ``None``."""
        f = self._file(key)
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` (must be JSON-serialisable) under ``key``."""
        self._file(key).write_text(json.dumps(value), encoding="utf-8")
