"""CLI: build BM25, TF-IDF and FAISS indexes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.indexing import build_bm25, build_faiss, build_tfidf, embed_chunks  # noqa: E402


def main() -> None:
    """Entry point."""
    cfg = load_config()
    chunks = pd.read_parquet(cfg.path("paths.chunks_file"))
    build_bm25(chunks, cfg)
    build_tfidf(chunks, cfg)
    build_faiss(embed_chunks(chunks, cfg), cfg)


if __name__ == "__main__":
    main()
