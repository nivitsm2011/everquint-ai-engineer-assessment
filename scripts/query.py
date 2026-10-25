"""CLI: search the corpus. Query terms in excerpts are wrapped in **double asterisks**."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.search import METHODS, SearchEngine  # noqa: E402


def format_results(results: list, show_excerpts: int) -> str:  # type: ignore[type-arg]
    """Render results as plain text."""
    lines: list[str] = []
    for r in results:
        lines.append(f"{r.rank}. [{r.score:.4f}] {r.doc_id}  ({r.method})")
        lines.append(f"   {r.title}")
        for ex in r.excerpts[:show_excerpts]:
            lines.append(f"   - {ex.highlighted}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top_n", type=int, default=None, help="documents to retrieve (config default)")
    parser.add_argument("--method", choices=METHODS, default="hybrid")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--page_size", type=int, default=None)
    parser.add_argument("--excerpts", type=int, default=1, help="excerpts to print per document")
    args = parser.parse_args()
    results = SearchEngine().search(
        args.query, args.top_n, args.method, args.page, args.page_size
    )
    sys.stdout.write(format_results(results, args.excerpts) or "No results.\n")


if __name__ == "__main__":
    main()
