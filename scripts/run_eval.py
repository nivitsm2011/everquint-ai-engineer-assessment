"""CLI: run the retrieval, summary and latency evaluations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.evaluation.eval_latency import measure_latency  # noqa: E402
from src.evaluation.eval_search import evaluate_search  # noqa: E402
from src.evaluation.eval_summary import evaluate_summaries  # noqa: E402
from src.evaluation.make_testset import make_testset  # noqa: E402


def main() -> None:
    """Entry point."""
    cfg = load_config()
    make_testset(cfg)
    evaluate_search(cfg)
    evaluate_summaries(cfg)
    measure_latency(cfg)


if __name__ == "__main__":
    main()
