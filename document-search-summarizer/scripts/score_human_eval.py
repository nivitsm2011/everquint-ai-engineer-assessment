"""CLI: read the filled-in human-evaluation sheet and report average scores (1-5)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.evaluation.eval_summary import score_human_eval  # noqa: E402


def main() -> None:
    """Entry point."""
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sheet", nargs="?", default=str(cfg.path("paths.human_eval_file")))
    args = parser.parse_args()
    result = score_human_eval(Path(args.sheet))
    cfg.path("paths.human_eval_scores_file").write_text(json.dumps(result, indent=2), encoding="utf-8")
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
