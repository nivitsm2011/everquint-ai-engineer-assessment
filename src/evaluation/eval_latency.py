"""Efficiency: timing of indexing, search and summarization."""

from __future__ import annotations

from src.config import Config


def measure_latency(cfg: Config) -> dict[str, float]:
    """Return mean/p95 latency in milliseconds per pipeline stage."""
    raise NotImplementedError
