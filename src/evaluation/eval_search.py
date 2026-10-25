"""5b. Retrieval metrics for bm25, tfidf, dense and hybrid on the generated test queries."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import Config  # noqa: E402
from src.search import METHODS, SearchEngine  # noqa: E402
from src.utils import get_logger  # noqa: E402

log = get_logger(__name__)
CUTOFF = 10


def rank_of(target: str, ranked_ids: list[str]) -> int | None:
    """1-based rank of ``target`` in ``ranked_ids`` (None if absent)."""
    return ranked_ids.index(target) + 1 if target in ranked_ids else None


def metrics_from_ranks(ranks: list[int | None], ks: list[int]) -> dict[str, float]:
    """Metrics for single-relevant-document queries (so IDCG = 1 and Recall@k = Hit-Rate@k)."""
    n = len(ranks)
    found = [r for r in ranks if r is not None and r <= CUTOFF]
    out = {f"recall@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    out[f"mrr@{CUTOFF}"] = sum(1 / r for r in found) / n
    out[f"ndcg@{CUTOFF}"] = sum(1 / math.log2(r + 1) for r in found) / n
    out["precision@5"] = sum(1 for r in ranks if r is not None and r <= 5) / 5 / n
    return out


def evaluate_search(cfg: Config, engine: SearchEngine | None = None) -> pd.DataFrame:
    """Run every test query through every method; write metrics CSV, per-query ranks and a chart."""
    testset = pd.read_csv(cfg.path("paths.testset_file"))
    engine = engine or SearchEngine(cfg)
    ks = [int(k) for k in cfg.get("evaluation.k_values")]
    ranks_by_method: dict[str, list[int | None]] = {}
    for method in METHODS:
        ranks: list[int | None] = []
        for doc_id, query in zip(testset["doc_id"], testset["query"], strict=True):
            hits = engine.search(query, top_n=CUTOFF, method=method, page=1, page_size=CUTOFF)
            ranks.append(rank_of(doc_id, [h.doc_id for h in hits]))
        ranks_by_method[method] = ranks
        log.info("%s done", method)

    rows = [{"method": m, "queries": len(testset), **metrics_from_ranks(r, ks)}
            for m, r in ranks_by_method.items()]
    metrics = pd.DataFrame(rows)
    metrics.round(4).to_csv(cfg.path("paths.search_metrics_file"), index=False)
    ranks_df = testset.copy()
    for m, r in ranks_by_method.items():
        ranks_df[f"rank_{m}"] = r  # blank = not in top 10
    ranks_df.to_csv(cfg.path("paths.search_ranks_file"), index=False)
    plot_search_metrics(metrics, cfg)
    return metrics


def plot_search_metrics(metrics: pd.DataFrame, cfg: Config) -> None:
    """Grouped bar chart: one group per metric, one bar per retrieval method."""
    cols = [c for c in metrics.columns if c not in ("method", "queries", "precision@5")]
    colors = {"bm25": "#4C78A8", "tfidf": "#9D9D9D", "dense": "#F58518", "hybrid": "#54A24B"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.2
    for i, row in metrics.iterrows():
        xs = [j + (i - 1.5) * width for j in range(len(cols))]
        bars = ax.bar(xs, [row[c] for c in cols], width, label=row["method"], color=colors[row["method"]])
        ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    ax.set_xticks(range(len(cols)), [c.replace("recall", "Recall").replace("mrr", "MRR").replace("ndcg", "nDCG")
                                      for c in cols])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (higher is better)")
    ax.set_title(f"Search accuracy on {int(metrics['queries'].iloc[0])} generated queries")
    ax.legend(ncol=4, loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    out = cfg.path("paths.figures_dir") / "search_metrics.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info("wrote %s", out)
