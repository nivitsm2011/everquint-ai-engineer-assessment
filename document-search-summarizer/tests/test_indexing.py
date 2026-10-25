"""Index build/persist/load tests on a tiny synthetic corpus (no model download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.indexing as ix
from src.config import Config, load_config


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    base = load_config().data
    data = {**base, "paths": {**base["paths"]}}
    for key in ("bm25_index", "tfidf_index", "faiss_index", "embeddings_file"):
        data["paths"][key] = str(tmp_path / Path(base["paths"][key]).name)
    data["index"] = {**base["index"], "faiss": {**base["index"]["faiss"], "nlist": 4}}
    return Config(data=data)


@pytest.fixture()
def chunks() -> pd.DataFrame:
    words = ["apple", "banana", "cherry", "delta", "echo", "fox", "golf", "hotel"]
    rows = [
        {"chunk_id": f"c{i}", "bm25_text": " ".join([words[i % 8]] * 3 + [words[(i + 1) % 8]])}
        for i in range(64)
    ]
    return pd.DataFrame(rows)


def test_bm25_and_tfidf_roundtrip(cfg: Config, chunks: pd.DataFrame) -> None:
    ix.build_bm25(chunks, cfg)
    ix.build_tfidf(chunks, cfg)
    emb = np.random.default_rng(0).normal(size=(64, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ix.build_faiss(emb, cfg)
    idx = ix.load_indexes(cfg)
    scores = idx["bm25"].get_scores(["apple"])
    assert chunks.chunk_id[int(scores.argmax())] in {"c0", "c8", "c16", "c24", "c32", "c40", "c48", "c56"}
    assert idx["tfidf_matrix"].shape[0] == 64
    assert idx["faiss"].ntotal == 64


@pytest.mark.parametrize("kind", ["flat", "ivf", "hnsw"])
def test_faiss_variants_find_themselves(cfg: Config, kind: str) -> None:
    cfg.data["index"]["faiss"]["type"] = kind
    emb = np.random.default_rng(1).normal(size=(200, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    index = ix.build_faiss(emb, cfg)
    if kind == "ivf":
        index.nprobe = 4
    _, hits = index.search(emb[:5], 1)
    assert hits[:, 0].tolist() == [0, 1, 2, 3, 4]


def test_unknown_faiss_type(cfg: Config) -> None:
    cfg.data["index"]["faiss"]["type"] = "nope"
    with pytest.raises(ValueError):
        ix.build_faiss(np.zeros((4, 8), dtype=np.float32), cfg)
