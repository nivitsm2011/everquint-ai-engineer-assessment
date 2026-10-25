"""Retrieval: BM25, dense, hybrid (reciprocal rank fusion), top-N and excerpts."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Config


@dataclass(frozen=True)
class SearchResult:
    """One ranked hit."""

    doc_id: str
    chunk_id: str
    title: str
    score: float
    excerpt: str


def bm25_search(query: str, cfg: Config, k: int) -> list[tuple[str, float]]:
    """Rank chunks by BM25; returns ``(chunk_id, score)``."""
    raise NotImplementedError


def dense_search(query: str, cfg: Config, k: int) -> list[tuple[str, float]]:
    """Rank chunks by embedding similarity; returns ``(chunk_id, score)``."""
    raise NotImplementedError


def reciprocal_rank_fusion(
    rankings: list[list[str]], weights: list[float], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse several ranked id lists with weighted RRF."""
    raise NotImplementedError


def search(query: str, cfg: Config, top_n: int | None = None, mode: str | None = None) -> list[SearchResult]:
    """Return the top-N results for ``query`` using bm25, dense or hybrid mode."""
    raise NotImplementedError
