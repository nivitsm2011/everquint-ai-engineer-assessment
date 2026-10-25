"""CLI: domain-adapt the embedding model (Phase 7).

Stages (all resumable): pairs -> train -> index -> compare.  After it finishes, set
``finetune.enabled: true`` in config/config.yaml to make the app and search use the fine-tuned model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config  # noqa: E402
from src.finetune import build_finetuned_index, compare, make_pairs, train  # noqa: E402

STAGES = ("pairs", "train", "index", "compare")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = parser.parse_args()
    cfg = load_config()
    if cfg.get("finetune.enabled"):
        sys.exit("Set finetune.enabled: false while fine-tuning (the base model is the starting point).")
    if "pairs" in args.stages:
        make_pairs(cfg)
    if "train" in args.stages:
        train(cfg)
    if "index" in args.stages:
        build_finetuned_index(cfg)
    if "compare" in args.stages:
        sys.stdout.write(compare(cfg).to_string(index=False) + "\n")


if __name__ == "__main__":
    main()
