"""Build, persist and load the BM25, TF-IDF and FAISS dense indexes."""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import Config
from src.utils import get_logger, timed

log = get_logger(__name__)


def _dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)  # noqa: S301 - files are produced locally by build_index


# --- lexical indexes ---------------------------------------------------------------------------
@timed
def build_bm25(chunks: pd.DataFrame, cfg: Config) -> BM25Okapi:
    """Build and persist a BM25Okapi index over the pre-tokenised ``bm25_text`` column."""
    corpus = [t.split() for t in chunks["bm25_text"]]
    model = BM25Okapi(corpus, k1=float(cfg.get("index.bm25.k1")), b=float(cfg.get("index.bm25.b")))
    _dump({"model": model, "chunk_ids": chunks["chunk_id"].tolist()}, cfg.path("paths.bm25_index"))
    return model


@timed
def build_tfidf(chunks: pd.DataFrame, cfg: Config) -> tuple[TfidfVectorizer, Any]:
    """Build and persist a TF-IDF vectoriser and sparse chunk matrix (the lexical baseline)."""
    ngram = tuple(cfg.get("index.tfidf.ngram_range"))
    vec = TfidfVectorizer(
        token_pattern=r"\S+",  # text is already tokenised/lowercased/stopword-free
        lowercase=False,
        ngram_range=ngram,
        min_df=int(cfg.get("index.tfidf.min_df")),
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vec.fit_transform(chunks["bm25_text"])
    _dump(
        {"vectorizer": vec, "matrix": matrix, "chunk_ids": chunks["chunk_id"].tolist()},
        cfg.path("paths.tfidf_index"),
    )
    return vec, matrix


# --- dense index -------------------------------------------------------------------------------
def get_device() -> str:
    """Use a GPU when one is present; everything also runs on CPU."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedder(cfg: Config) -> Any:
    """Load the sentence-transformers model named in the config."""
    from sentence_transformers import SentenceTransformer

    name, device = cfg.get("embedding.model_name"), get_device()
    try:  # prefer the local cache: avoids network round-trips once the model is downloaded
        model = SentenceTransformer(name, device=device, local_files_only=True)
    except OSError:
        model = SentenceTransformer(name, device=device)
    model.max_seq_length = int(cfg.get("embedding.max_seq_length"))
    return model


@timed
def embed_chunks(chunks: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Encode chunk text into L2-normalised float32 vectors.

    Work is done in blocks and each block is saved, so an interrupted run (e.g. laptop sleep)
    resumes instead of starting over. Parts are merged into ``paths.embeddings_file`` at the end.
    """
    emb_path = cfg.path("paths.embeddings_file")
    parts_dir = emb_path.parent / "embedding_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    block = int(cfg.get("embedding.checkpoint_every"))
    texts = chunks["text"].tolist()
    model = load_embedder(cfg)
    log.info("Embedding %d chunks on %s", len(texts), get_device())

    for n, start in enumerate(range(0, len(texts), block)):
        part = parts_dir / f"part_{n:05d}.npy"
        if part.exists():
            continue
        vecs = model.encode(
            texts[start : start + block],
            batch_size=int(cfg.get("embedding.batch_size")),
            normalize_embeddings=bool(cfg.get("embedding.normalize")),
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        np.save(part, vecs)
        log.info("embedded %d / %d", min(start + block, len(texts)), len(texts))

    embeddings = np.concatenate([np.load(p) for p in sorted(parts_dir.glob("part_*.npy"))])
    np.save(emb_path, embeddings)
    shutil.rmtree(parts_dir)
    return embeddings


@timed
def build_faiss(embeddings: np.ndarray, cfg: Config) -> faiss.Index:
    """Build and persist a FAISS inner-product index (cosine, as vectors are normalised).

    ``index.faiss.type``: ``flat`` (exact; default), ``ivf`` (IndexIVFFlat) or ``hnsw``
    (IndexHNSWFlat). The last two are the scalability path for much larger corpora.
    """
    dim = embeddings.shape[1]
    kind = str(cfg.get("index.faiss.type")).lower()
    if kind == "flat":
        index: faiss.Index = faiss.IndexFlatIP(dim)
    elif kind == "ivf":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(
            quantizer, dim, int(cfg.get("index.faiss.nlist")), faiss.METRIC_INNER_PRODUCT
        )
        index.train(embeddings)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(dim, int(cfg.get("index.faiss.hnsw_m")), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = int(cfg.get("index.faiss.ef_construction"))
    else:
        raise ValueError(f"Unknown index.faiss.type: {kind!r} (flat | ivf | hnsw)")
    index.add(embeddings)
    path = cfg.path("paths.faiss_index")
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    return index


def _tune_search_params(index: faiss.Index, cfg: Config) -> None:
    """Apply query-time recall/speed knobs for approximate indexes."""
    if isinstance(index, faiss.IndexIVF):
        index.nprobe = int(cfg.get("index.faiss.nprobe"))
    elif isinstance(index, faiss.IndexHNSW):
        index.hnsw.efSearch = int(cfg.get("index.faiss.ef_search"))


def load_indexes(cfg: Config) -> dict[str, Any]:
    """Load every persisted index from disk (no rebuilding)."""
    bm25 = _load(cfg.path("paths.bm25_index"))
    tfidf = _load(cfg.path("paths.tfidf_index"))
    dense = faiss.read_index(str(cfg.path("paths.faiss_index")))
    _tune_search_params(dense, cfg)
    if not (len(bm25["chunk_ids"]) == len(tfidf["chunk_ids"]) == dense.ntotal):
        raise RuntimeError("Indexes disagree on chunk count; rebuild with --rebuild")
    return {
        "bm25": bm25["model"],
        "tfidf_vectorizer": tfidf["vectorizer"],
        "tfidf_matrix": tfidf["matrix"],
        "faiss": dense,
        "chunk_ids": bm25["chunk_ids"],
    }


def file_size_mb(path: Path) -> float:
    """Size of a file in MB (0 if absent)."""
    return round(path.stat().st_size / 1e6, 1) if path.exists() else 0.0
