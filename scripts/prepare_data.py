"""CLI: download (streamed), clean, chunk and save the corpus; prints corpus statistics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.data_prep import prepare  # noqa: E402


def main() -> None:
    """Entry point."""
    stats = prepare(load_config())
    sys.stdout.write(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
