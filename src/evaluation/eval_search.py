"""Retrieval metrics: Recall@k, MRR, nDCG for bm25 vs dense vs hybrid."""

from __future__ import annotations

from src.config import Config


def evaluate_search(cfg: Config) -> dict[str, dict[str, float]]:
    """Return metrics per retrieval mode."""
    raise NotImplementedError
