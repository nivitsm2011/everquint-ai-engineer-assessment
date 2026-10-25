"""Document search: BM25, TF-IDF, dense and hybrid (RRF) retrieval with highlighted excerpts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.data_prep import bm25_normalize
from src.indexing import load_embedder, load_indexes
from src.utils import get_logger

log = get_logger(__name__)

Method = Literal["bm25", "tfidf", "dense", "hybrid"]
METHODS: tuple[str, ...] = ("bm25", "tfidf", "dense", "hybrid")
_NON_WORD = re.compile(r"\W+")
_PUNCT_SPLIT = re.compile(r"(\W*)(.*?)(\W*)$", re.DOTALL)


@dataclass(frozen=True)
class Excerpt:
    """A passage of a matching chunk. ``highlighted`` wraps query terms in ``**``."""

    chunk_id: str
    score: float
    text: str
    highlighted: str


@dataclass(frozen=True)
class Result:
    """One document-level search hit."""

    rank: int
    doc_id: str
    title: str
    score: float
    method: str
    excerpts: list[Excerpt] = field(default_factory=list)


def reciprocal_rank_fusion(
    rankings: list[list[int]], weights: list[float] | None = None, k: int = 60
) -> list[tuple[int, float]]:
    """Fuse ranked id lists: ``score(d) = sum_i w_i / (k + rank_i(d))`` (rank is 1-based)."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + w / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def query_terms(query: str) -> list[str]:
    """Preprocess a query exactly like the corpus (lowercase, tokenise, drop stopwords).

    If every word is a stopword, fall back to the plain lowercase tokens so the query is not empty.
    """
    tokens = bm25_normalize(query).split()
    return tokens or [t for t in _NON_WORD.sub(" ", query.lower()).split() if t]


def _top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest scores, best first (ignores non-positive scores)."""
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx[scores[idx] > 0]


def _term_forms(terms: list[str]) -> set[str]:
    """Query terms plus their trivially de-pluralised forms, for highlighting."""
    return set(terms) | {t.rstrip("s") for t in terms if len(t) > 3}


def _matches(word: str, forms: set[str]) -> bool:
    core = _NON_WORD.sub("", word.lower())
    return bool(core) and (core in forms or (len(core) > 3 and core.rstrip("s") in forms))


def make_excerpt(text: str, terms: list[str], max_words: int) -> tuple[str, str]:
    """Return ``(plain, highlighted)`` for the window with the most query-term hits.

    Matching is whole-word and case-insensitive; original casing and punctuation are kept.
    """
    words = text.split()
    forms = _term_forms(terms)
    hits = np.array([1 if _matches(w, forms) else 0 for w in words])
    if len(words) <= max_words:
        start = 0
    else:
        csum = np.concatenate([[0], np.cumsum(hits)])
        window = csum[max_words:] - csum[: len(words) - max_words + 1]
        start = int(np.argmax(window))  # earliest window with the most hits
    end = min(start + max_words, len(words))
    plain_words, marked = words[start:end], []
    for w, h in zip(plain_words, hits[start:end], strict=True):
        if h:
            pre, core, post = _PUNCT_SPLIT.match(w).groups()  # type: ignore[union-attr]
            marked.append(f"{pre}**{core}**{post}")
        else:
            marked.append(w)
    lead, tail = ("… " if start > 0 else ""), (" …" if end < len(words) else "")
    return lead + " ".join(plain_words) + tail, lead + " ".join(marked) + tail


class SearchEngine:
    """Loads the persisted indexes once and answers queries with any retrieval method."""

    def __init__(
        self,
        cfg: Config | None = None,
        indexes: dict[str, Any] | None = None,
        chunks: pd.DataFrame | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self._indexes = indexes
        self._chunks = chunks
        self._embedder = embedder

    # -- lazy resources -------------------------------------------------------------------------
    @property
    def indexes(self) -> dict[str, Any]:
        if self._indexes is None:
            self._indexes = load_indexes(self.cfg)
        return self._indexes

    @property
    def chunks(self) -> pd.DataFrame:
        if self._chunks is None:
            cols = ["chunk_id", "doc_id", "title", "text"]
            chunks = pd.read_parquet(self.cfg.path("paths.chunks_file"), columns=cols)
            if chunks["chunk_id"].tolist() != self.indexes["chunk_ids"]:
                raise RuntimeError("chunks.parquet and the indexes are out of sync; rebuild the indexes")
            self._chunks = chunks
        return self._chunks

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = load_embedder(self.cfg)
        return self._embedder

    # -- per-method chunk rankings ----------------------------------------------------------------
    def _bm25(self, terms: list[str], pool: int) -> tuple[np.ndarray, np.ndarray]:
        scores = np.asarray(self.indexes["bm25"].get_scores(terms))
        idx = _top_k(scores, pool)
        return idx, scores[idx]

    def _tfidf(self, terms: list[str], pool: int) -> tuple[np.ndarray, np.ndarray]:
        qvec = self.indexes["tfidf_vectorizer"].transform([" ".join(terms)])
        scores = (self.indexes["tfidf_matrix"] @ qvec.T).toarray().ravel()
        idx = _top_k(scores, pool)
        return idx, scores[idx]

    def _dense(self, query: str, pool: int) -> tuple[np.ndarray, np.ndarray]:
        qv = self.embedder.encode(
            [query], normalize_embeddings=bool(self.cfg.get("embedding.normalize"))
        ).astype(np.float32)
        scores, idx = self.indexes["faiss"].search(qv, pool)
        keep = idx[0] >= 0
        return idx[0][keep], scores[0][keep]

    def _rank_chunks(self, query: str, method: str, pool: int) -> list[tuple[int, float]]:
        """Return ``(chunk_row, score)`` best first for the chosen method."""
        terms = query_terms(query)
        if method == "bm25":
            idx, sc = self._bm25(terms, pool)
        elif method == "tfidf":
            idx, sc = self._tfidf(terms, pool)
        elif method == "dense":
            idx, sc = self._dense(query, pool)
        else:  # hybrid: weighted RRF of BM25 and dense
            b_idx, _ = self._bm25(terms, pool)
            d_idx, _ = self._dense(query, pool)
            w = self.cfg.get("search.weights")
            fused = reciprocal_rank_fusion(
                [b_idx.tolist(), d_idx.tolist()],
                [float(w["bm25"]), float(w["dense"])],
                int(self.cfg.get("search.rrf_k")),
            )
            return fused
        return list(zip(idx.tolist(), sc.tolist(), strict=True))

    # -- public API -------------------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_n: int | None = None,
        method: Method | None = None,
        page: int = 1,
        page_size: int | None = None,
    ) -> list[Result]:
        """Return page ``page`` of the top-``top_n`` documents for ``query``.

        Chunk hits are aggregated to documents using the best chunk's score; each result carries
        up to ``search.excerpts_per_doc`` highlighted excerpts. Blank queries return ``[]``.
        """
        method = method or self.cfg.get("search.mode")
        top_n = int(top_n or self.cfg.get("search.top_n"))
        page_size = int(page_size or self.cfg.get("search.page_size"))
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
        if top_n < 1 or page < 1 or page_size < 1:
            raise ValueError("top_n, page and page_size must be >= 1")
        if not query.strip():
            return []

        pool = max(int(self.cfg.get("search.candidate_pool")), top_n * 5)
        ranked = self._rank_chunks(query, method, pool)
        terms = query_terms(query)
        doc_ids = self.chunks["doc_id"].to_numpy()

        per_doc: dict[str, list[tuple[int, float]]] = {}
        for row, score in ranked:  # already best first, so the first chunk per doc is its max
            per_doc.setdefault(str(doc_ids[row]), []).append((row, score))
        top_docs = list(per_doc.items())[:top_n]

        start = (page - 1) * page_size
        max_words = int(self.cfg.get("search.excerpt_words"))
        n_excerpts = int(self.cfg.get("search.excerpts_per_doc"))
        results: list[Result] = []
        for offset, (doc_id, hits) in enumerate(top_docs[start : start + page_size]):
            excerpts = []
            for row, score in hits[:n_excerpts]:
                plain, marked = make_excerpt(str(self.chunks["text"].iat[row]), terms, max_words)
                excerpts.append(Excerpt(str(self.chunks["chunk_id"].iat[row]), float(score), plain, marked))
            results.append(
                Result(
                    rank=start + offset + 1,
                    doc_id=doc_id,
                    title=str(self.chunks["title"].iat[hits[0][0]]),
                    score=float(hits[0][1]),
                    method=method,
                    excerpts=excerpts,
                )
            )
        return results


_default_engine: SearchEngine | None = None


def search(
    query: str,
    top_n: int | None = None,
    method: Method | None = None,
    page: int = 1,
    page_size: int | None = None,
) -> list[Result]:
    """Module-level convenience wrapper around a lazily created :class:`SearchEngine`.

    Defaults come from ``config.yaml`` (``search.top_n``=5, ``search.mode``=hybrid,
    ``search.page_size``=5).
    """
    global _default_engine
    if _default_engine is None:
        _default_engine = SearchEngine()
    return _default_engine.search(query, top_n, method, page, page_size)


__all__ = ["Excerpt", "Result", "SearchEngine", "search", "reciprocal_rank_fusion"]
