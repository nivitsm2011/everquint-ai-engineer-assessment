"""Draw the architecture diagram used in the report (reports/figures/architecture.png)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports" / "figures" / "architecture.png"
OFFLINE, ONLINE, STORE, LLM = "#DCE9F7", "#E2F2E0", "#FFF1D6", "#F6DDE8"


def box(ax, x, y, w, h, text, color):  # type: ignore[no-untyped-def]
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#44546A", lw=1.2
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def arrow(ax, p, q):  # type: ignore[no-untyped-def]
    ax.annotate("", xy=q, xytext=p, arrowprops={"arrowstyle": "->", "color": "#44546A", "lw": 1.3})


def main() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.text(0.1, 5.95, "Offline: prepare once", fontsize=10, weight="bold")
    ax.text(0.1, 2.85, "Online: every query", fontsize=10, weight="bold")

    box(ax, 0.1, 4.7, 1.9, 0.9, "CNN/DailyMail\n(Hugging Face,\nstreamed)", OFFLINE)
    box(ax, 2.5, 4.7, 2.2, 0.9, "Clean + de-duplicate\n+ sentence-aware\n200-word chunks", OFFLINE)
    box(ax, 5.2, 4.7, 1.8, 0.9, "documents.parquet\nchunks.parquet", STORE)
    box(ax, 7.6, 5.0, 2.0, 0.6, "BM25 + TF-IDF", OFFLINE)
    box(ax, 7.6, 4.2, 2.0, 0.6, "MiniLM embeddings", OFFLINE)
    box(ax, 10.0, 4.2, 1.9, 0.6, "FAISS index\n(flat / IVF / HNSW)", STORE)
    box(ax, 10.0, 5.0, 1.9, 0.6, "BM25 / TF-IDF\npickles", STORE)
    for p, q in [
        ((2.0, 5.15), (2.5, 5.15)),
        ((4.7, 5.15), (5.2, 5.15)),
        ((7.0, 5.3), (7.6, 5.3)),
        ((7.0, 4.95), (7.6, 4.5)),
        ((9.6, 4.5), (10.0, 4.5)),
        ((9.6, 5.3), (10.0, 5.3)),
    ]:
        arrow(ax, p, q)

    box(ax, 0.1, 1.5, 1.6, 0.9, "User query\n(CLI / Streamlit,\nauto-suggest)", ONLINE)
    box(ax, 2.2, 1.9, 1.9, 0.9, "BM25 ranking\n(same preprocessing\nas corpus)", ONLINE)
    box(ax, 2.2, 0.7, 1.9, 0.9, "Dense ranking\n(embed query,\nFAISS search)", ONLINE)
    box(ax, 4.6, 1.3, 1.9, 1.0, "Hybrid: weighted\nReciprocal Rank\nFusion (k=60)", ONLINE)
    box(ax, 7.0, 1.3, 2.0, 1.0, "Group chunks by\ndocument, top-N,\nhighlighted excerpts", ONLINE)
    box(ax, 9.5, 1.3, 2.4, 1.0, "Paginated results\n(title, score,\nexcerpts, full text)", ONLINE)
    box(ax, 7.0, 0.05, 2.4, 0.9, "Map-reduce summary\nat chosen length\n(per-doc, then combine)", LLM)
    box(ax, 9.9, 0.05, 2.0, 0.9, "LLM: OpenAI or\nOllama (config)", LLM)
    box(ax, 4.6, 0.05, 2.0, 0.9, "Disk cache\n(doc ids, length,\nmodel)", STORE)
    for p, q in [
        ((1.7, 2.0), (2.2, 2.3)),
        ((1.7, 1.8), (2.2, 1.2)),
        ((4.1, 2.3), (4.6, 1.9)),
        ((4.1, 1.2), (4.6, 1.7)),
        ((6.5, 1.8), (7.0, 1.8)),
        ((9.0, 1.8), (9.5, 1.8)),
        ((8.2, 1.3), (8.2, 0.95)),
        ((9.4, 0.5), (9.9, 0.5)),
        ((7.0, 0.5), (6.6, 0.5)),
    ]:
        arrow(ax, p, q)
    arrow(ax, (6.0, 4.7), (6.0, 2.35))
    ax.text(6.1, 3.5, "indexes + chunks\nloaded once", fontsize=8, color="#44546A")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    sys.stdout.write(f"wrote {OUT}\n")


if __name__ == "__main__":
    main()
