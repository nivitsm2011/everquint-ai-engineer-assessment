"""CLI: build (or reuse) BM25, TF-IDF and FAISS indexes; print build time and size on disk."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import load_config  # noqa: E402
from src.indexing import (  # noqa: E402
    build_bm25,
    build_faiss,
    build_tfidf,
    embed_chunks,
    file_size_mb,
    load_embedder,
    load_indexes,
)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="ignore existing index files")
    args = parser.parse_args()
    cfg = load_config()
    chunks = pd.read_parquet(cfg.path("paths.chunks_file"))
    stats: dict[str, dict[str, float | str]] = {}

    def run(name: str, key: str, build) -> None:  # type: ignore[no-untyped-def]
        path = cfg.path(key)
        if path.exists() and not args.rebuild:
            stats[name] = {"build_seconds": 0.0, "status": "reused", "size_mb": file_size_mb(path)}
            return
        t0 = time.perf_counter()
        build()
        stats[name] = {
            "build_seconds": round(time.perf_counter() - t0, 1),
            "status": "built",
            "size_mb": file_size_mb(path),
        }

    run("bm25", "paths.bm25_index", lambda: build_bm25(chunks, cfg))
    run("tfidf", "paths.tfidf_index", lambda: build_tfidf(chunks, cfg))

    emb_path = cfg.path("paths.embeddings_file")
    if args.rebuild or not emb_path.exists():
        t0 = time.perf_counter()
        embeddings = embed_chunks(chunks, cfg)
        stats["embeddings"] = {
            "build_seconds": round(time.perf_counter() - t0, 1),
            "status": "built",
            "size_mb": file_size_mb(emb_path),
        }
    else:
        embeddings = np.load(emb_path)
        stats["embeddings"] = {"build_seconds": 0.0, "status": "reused", "size_mb": file_size_mb(emb_path)}
    run("faiss", "paths.faiss_index", lambda: build_faiss(embeddings, cfg))

    # Prove the persisted indexes load and answer a query.
    t0 = time.perf_counter()
    idx = load_indexes(cfg)
    load_s = round(time.perf_counter() - t0, 2)
    query = "prime minister announces new economic plan"
    ids = idx["chunk_ids"]
    bm = idx["bm25"].get_scores(query.split())
    qv = load_embedder(cfg).encode([query], normalize_embeddings=True).astype(np.float32)
    _, hits = idx["faiss"].search(qv, 3)
    titles = chunks.set_index("chunk_id")["title"]
    sys.stdout.write(f"\nSanity query: {query!r}\n")
    sys.stdout.write(f"  BM25 top1 : {titles[ids[int(np.argmax(bm))]][:90]}\n")
    for h in hits[0][:2]:
        sys.stdout.write(f"  Dense top : {titles[ids[int(h)]][:90]}\n")

    total_mb = round(sum(float(s["size_mb"]) for s in stats.values()), 1)
    summary = {
        "chunks": len(chunks),
        "faiss_type": cfg.get("index.faiss.type"),
        "embedding_model": cfg.get("embedding.model_name"),
        "load_all_indexes_seconds": load_s,
        "artifacts": stats,
        "total_size_mb": total_mb,
    }
    out = cfg.path("paths.index_stats_file")
    if not all(s["status"] == "reused" for s in stats.values()) or not out.exists():
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sys.stdout.write(f"\n{'artifact':<12}{'status':<8}{'build s':>9}{'size MB':>10}\n")
    for name, s in stats.items():
        sys.stdout.write(f"{name:<12}{s['status']:<8}{s['build_seconds']:>9}{s['size_mb']:>10}\n")
    sys.stdout.write(f"{'TOTAL':<20}{'':>9}{total_mb:>10}\nLoad all indexes: {load_s}s\n")


if __name__ == "__main__":
    main()
