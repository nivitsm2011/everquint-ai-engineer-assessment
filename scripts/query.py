"""CLI: search the corpus and summarize the results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.llm import get_llm  # noqa: E402
from src.search import search  # noqa: E402
from src.summarize import summarize  # noqa: E402


def main() -> None:
    """Entry point."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-n", type=int, default=cfg.get("search.top_n"))
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default=cfg.get("search.mode"))
    parser.add_argument("--length", choices=list(cfg.get("summary.lengths")), default=cfg.get("summary.default_length"))
    args = parser.parse_args()
    results = search(args.query, cfg, top_n=args.top_n, mode=args.mode)
    for i, r in enumerate(results, 1):
        sys.stdout.write(f"{i}. [{r.score:.3f}] {r.title}\n   {r.excerpt}\n")
    sys.stdout.write("\n" + summarize(args.query, results, cfg, get_llm(cfg), args.length) + "\n")


if __name__ == "__main__":
    main()
