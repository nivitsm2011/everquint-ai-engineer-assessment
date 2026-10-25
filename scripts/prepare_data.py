"""CLI: download, clean, chunk and save the corpus."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data_prep import prepare  # noqa: E402


def main() -> None:
    """Entry point."""
    prepare(load_config())


if __name__ == "__main__":
    main()
