"""Build and load BM25, TF-IDF and FAISS dense indexes (implemented in the indexing phase)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import Config


def build_bm25(chunks: pd.DataFrame, cfg: Config) -> Any:
    """Build and persist a BM25 index over chunk text."""
    raise NotImplementedError


def build_tfidf(chunks: pd.DataFrame, cfg: Config) -> Any:
    """Build and persist a TF-IDF vectoriser and matrix."""
    raise NotImplementedError


def embed_chunks(chunks: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Encode chunks with the configured embedding model."""
    raise NotImplementedError


def build_faiss(embeddings: np.ndarray, cfg: Config) -> Any:
    """Build and persist a FAISS inner-product index."""
    raise NotImplementedError


def load_indexes(cfg: Config) -> dict[str, Any]:
    """Load all persisted indexes from ``paths.index_dir``."""
    raise NotImplementedError
