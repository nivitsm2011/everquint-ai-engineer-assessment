"""First-start setup for hosted demos: build a small corpus and its indexes if they are missing."""

from __future__ import annotations

import pandas as pd

from src.config import Config
from src.data_prep import prepare
from src.indexing import build_bm25, build_faiss, build_tfidf, embed_chunks
from src.utils import get_logger

log = get_logger(__name__)
REQUIRED = (
    "paths.documents_file",
    "paths.chunks_file",
    "paths.bm25_index",
    "paths.tfidf_index",
    "paths.faiss_index",
)


def is_ready(cfg: Config) -> bool:
    """True when the corpus and every index file already exist."""
    return all(cfg.path(key).exists() for key in REQUIRED)


def ensure_ready(cfg: Config) -> bool:
    """Build corpus + indexes from scratch if anything is missing; return True if a build ran."""
    if is_ready(cfg):
        return False
    log.info("corpus or indexes missing: building a demo corpus (first start only)")
    prepare(cfg)
    chunks = pd.read_parquet(cfg.path("paths.chunks_file"))
    build_bm25(chunks, cfg)
    build_tfidf(chunks, cfg)
    build_faiss(embed_chunks(chunks, cfg), cfg)
    return True
