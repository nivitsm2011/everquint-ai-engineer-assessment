"""5d. Efficiency: average and p95 latency of index loading, each search method and summarization."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.config import Config
from src.indexing import load_embedder, load_indexes
from src.search import METHODS, SearchEngine
from src.utils import get_logger

log = get_logger(__name__)


def _row(component: str, seconds: list[float]) -> dict[str, object]:
    ms = np.array(seconds) * 1000
    return {
        "component": component,
        "n": len(ms),
        "mean_ms": round(float(ms.mean()), 1),
        "p95_ms": round(float(np.percentile(ms, 95)), 1),
    }


def measure_latency(cfg: Config) -> pd.DataFrame:
    """Time index load, query-time search per method (warm), and LLM summarization per length.

    Summarization times are taken from the real generation times recorded while evaluating
    (cached results are excluded), so no extra LLM calls are needed.
    """
    rows: list[dict[str, object]] = []
    loads = []
    for _ in range(int(cfg.get("evaluation.latency_repeats_index_load"))):
        t0 = time.perf_counter()
        load_indexes(cfg)
        loads.append(time.perf_counter() - t0)
    rows.append(_row("load indexes (BM25+TF-IDF+FAISS)", loads))
    t0 = time.perf_counter()
    embedder = load_embedder(cfg)
    rows.append(_row("load embedding model", [time.perf_counter() - t0]))

    engine = SearchEngine(cfg, embedder=embedder)
    queries = pd.read_csv(cfg.path("paths.testset_file"))["query"].tolist()
    engine.search(queries[0], top_n=10, method="hybrid")  # warm-up: loads chunk table
    for method in METHODS:
        times = []
        for q in queries:
            t0 = time.perf_counter()
            engine.search(q, top_n=10, method=method, page=1, page_size=10)
            times.append(time.perf_counter() - t0)
        rows.append(_row(f"search: {method} (top 10)", times))

    details_path = cfg.path("paths.summary_details_file")
    if details_path.exists():
        details = pd.read_csv(details_path)
        details = details[details["seconds"].notna() & (details["llm_calls"] > 0)]
        for length in cfg.get("evaluation.summary_lengths"):
            sub = details[(details["mode"] == "single") & (details["length"] == length)]
            if len(sub):
                rows.append(
                    _row(
                        f"summarize 1 doc: {length} (~{int(sub['words'].iloc[0])} words)",
                        sub["seconds"].tolist(),
                    )
                )
        multi = details[details["mode"] == "multi"]
        if len(multi):
            rows.append(
                _row(
                    f"summarize top-{int(cfg.get('evaluation.human_eval_top_docs'))} docs: "
                    f"{multi['length'].iloc[0]}",
                    multi["seconds"].tolist(),
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(cfg.path("paths.latency_file"), index=False)
    return table
