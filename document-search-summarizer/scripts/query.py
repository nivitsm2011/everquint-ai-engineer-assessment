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
    parser.add_argument(
        "--summarize", action="store_true", help="summarize the returned documents with the LLM"
    )
    parser.add_argument("--length", default="medium", help="short | medium | long | exact word count")
    args = parser.parse_args()
    results = SearchEngine().search(args.query, args.top_n, args.method, args.page, args.page_size)
    sys.stdout.write(format_results(results, args.excerpts) or "No results.\n")
    if args.summarize and results:
        from src.llm import LLMError
        from src.summarize import summarize

        try:
            s = summarize(results, args.length)
        except LLMError as exc:
            sys.exit(f"Summarization failed: {exc}")
        sys.stdout.write(
            f"--- Summary ({args.length}: target {s.target_words}, actual {s.word_count} words; "
            f"{s.provider}/{s.model}; {s.llm_calls} LLM calls{', cached' if s.cached else ''}; "
            f"{s.seconds}s) ---\n{s.text}\n"
        )


if __name__ == "__main__":
    main()
