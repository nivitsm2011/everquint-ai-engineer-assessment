"""CLI: run evaluation stages 5a-5d end to end and write reports/metrics_summary.json.

Every stage is resumable (LLM outputs are cached on disk), so an interrupted run can simply be
started again.  Use --stages to run only some of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.evaluation.eval_latency import measure_latency  # noqa: E402
from src.evaluation.eval_search import evaluate_search  # noqa: E402
from src.evaluation.eval_summary import evaluate_summaries  # noqa: E402
from src.evaluation.make_testset import make_testset  # noqa: E402
from src.llm import get_llm  # noqa: E402

STAGES = ("testset", "search", "summary", "latency")


def _show(title: str, table: pd.DataFrame) -> None:
    sys.stdout.write(f"\n== {title} ==\n{table.to_string(index=False)}\n")


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    args = parser.parse_args()
    cfg = load_config()
    out_path = cfg.path("paths.metrics_summary_file")
    summary: dict[str, object] = json.loads(out_path.read_text("utf-8")) if out_path.exists() else {}

    if "testset" in args.stages:
        ts = make_testset(cfg)
        summary["testset"] = {"queries": len(ts), "seed": cfg.get("evaluation.seed")}
        _show("Test set (sample)", ts.head(10))
    if "search" in args.stages:
        m = evaluate_search(cfg)
        summary["search"] = json.loads(m.round(4).to_json(orient="records"))
        _show("Search accuracy", m.round(3))
    if "summary" in args.stages:
        res = evaluate_summaries(cfg)
        summary["summary"] = res
        _show("ROUGE F1", pd.read_csv(cfg.path("paths.summary_scores_file")))
        sys.stdout.write(
            f"\nHuman-evaluation sheet: {cfg.path('paths.human_eval_file')} "
            f"({res['human_eval_rows']} rows)\n"
        )
    if "latency" in args.stages:
        lat = measure_latency(cfg)
        summary["latency"] = json.loads(lat.to_json(orient="records"))
        _show("Latency", lat)

    llm = get_llm(cfg)
    summary["run"] = {
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "llm": f"{llm.provider}/{llm.model}",
        "embedding": cfg.get("embedding.model_name"),
        "test_docs": cfg.get("evaluation.test_docs"),
        "summary_eval_docs": cfg.get("evaluation.summary_eval_docs"),
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sys.stdout.write(f"\nWrote {out_path}\n")


if __name__ == "__main__":
    main()
