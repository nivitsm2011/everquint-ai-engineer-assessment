"""Tests for the embedding-variant switch and fine-tuning data hygiene."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config, with_variant


def test_base_variant_is_untouched() -> None:
    cfg = load_config()
    assert with_variant(cfg, "base") is cfg
    assert cfg.get("embedding.model_name") == "sentence-transformers/all-MiniLM-L6-v2"


def test_finetuned_variant_swaps_model_and_dense_index_only() -> None:
    cfg = load_config()
    ft = with_variant(cfg, "finetuned")
    assert ft.get("embedding.model_name") == str(PROJECT_ROOT / "models" / "embedder_ft")
    assert ft.path("paths.faiss_index") == PROJECT_ROOT / "indexes" / "ft" / "dense.faiss"
    assert ft.path("paths.embeddings_file") == PROJECT_ROOT / "indexes" / "ft" / "embeddings.npy"
    assert ft.path("paths.bm25_index") == cfg.path("paths.bm25_index")  # lexical indexes are shared
    assert cfg.path("paths.faiss_index") == PROJECT_ROOT / "indexes" / "dense.faiss"  # original unchanged


def test_unknown_variant() -> None:
    with pytest.raises(ValueError):
        with_variant(load_config(), "other")


def test_training_pool_excludes_test_documents() -> None:
    """The pairs file (once generated) must not contain any test-set document."""
    cfg = load_config()
    pairs_path = cfg.path("finetune.pairs_file")
    if not pairs_path.exists():
        pytest.skip("pairs not generated yet")
    pairs = pd.read_csv(pairs_path)
    test_ids = set(pd.read_csv(cfg.path("paths.testset_file"))["doc_id"])
    assert not set(pairs["doc_id"]) & test_ids
    assert pairs["doc_id"].is_unique and (pairs["query"].str.len() > 0).all()
