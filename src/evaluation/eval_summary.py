"""Summary quality: ROUGE against references plus a human-evaluation sheet."""

from __future__ import annotations

from src.config import Config


def evaluate_summaries(cfg: Config) -> dict[str, float]:
    """Return ROUGE scores and write the human-eval CSV."""
    raise NotImplementedError
