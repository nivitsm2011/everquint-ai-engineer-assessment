"""Search tests on a tiny synthetic corpus with a fake bag-of-words embedder (no model download)."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest

import src.indexing as ix
from src.config import Config, load_config
from src.data_prep import bm25_normalize
from src.search import SearchEngine, make_excerpt, query_terms, reciprocal_rank_fusion

TOPICS = {
    "d0": "The football team won the championship match after a thrilling penalty shootout.",
    "d1": "The central bank raised interest rates to fight inflation across the economy.",
    "d2": "Scientists discovered a new planet orbiting a distant star using the telescope.",
    "d3": "The football striker scored twice and the football club celebrated the title.",
}
VOCAB = sorted({w for t in TOPICS.values() for w in bm25_normalize(t).split()})


class FakeEmbedder:
    """Bag-of-words vectors over the toy vocabulary, L2-normalised."""

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        out = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in bm25_normalize(t).split():
                if w in VOCAB:
                    out[i, VOCAB.index(w)] += 1
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1, norm)


@pytest.fixture()
def engine(tmp_path: Path) -> SearchEngine:
    base = load_config().data
    data = {**base, "paths": {**base["paths"]}}
    for key in ("bm25_index", "tfidf_index", "faiss_index", "embeddings_file"):
        data["paths"][key] = str(tmp_path / Path(base["paths"][key]).name)
    data["index"] = {**base["index"], "tfidf": {**base["index"]["tfidf"], "min_df": 1}}  # toy corpus
    cfg = Config(data=data)
    rows = []
    for doc_id, text in TOPICS.items():  # two chunks per doc so aggregation is exercised
        rows.append({"chunk_id": f"{doc_id}_c00", "doc_id": doc_id, "title": text[:30], "text": text})
        rows.append(
            {"chunk_id": f"{doc_id}_c01", "doc_id": doc_id, "title": text[:30], "text": "Filler words only."}
        )
    chunks = pd.DataFrame(rows)
    chunks["bm25_text"] = chunks["text"].map(bm25_normalize)
    ix.build_bm25(chunks, cfg)
    ix.build_tfidf(chunks, cfg)
    emb = FakeEmbedder().encode(chunks["text"].tolist())
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(cfg.path("paths.faiss_index")))
    return SearchEngine(cfg, chunks=chunks[["chunk_id", "doc_id", "title", "text"]], embedder=FakeEmbedder())


@pytest.mark.parametrize("method", ["bm25", "tfidf", "dense", "hybrid"])
def test_every_method_finds_the_right_doc(engine: SearchEngine, method: str) -> None:
    res = engine.search("central bank interest rates", top_n=3, method=method)
    assert res[0].doc_id == "d1" and res[0].method == method and res[0].rank == 1
    assert len({r.doc_id for r in res}) == len(res)  # one hit per document


def test_document_score_is_best_chunk_and_excerpts_highlight(engine: SearchEngine) -> None:
    res = engine.search("football", top_n=4, method="bm25")
    assert {r.doc_id for r in res} == {"d0", "d3"}
    assert "**football**" in res[0].excerpts[0].highlighted.lower()
    assert res[0].score >= res[1].score


@pytest.mark.parametrize("method", ["bm25", "tfidf", "dense", "hybrid"])
def test_results_are_ranked_best_first(engine: SearchEngine, method: str) -> None:
    res = engine.search("football team championship", top_n=4, method=method)
    scores = [r.score for r in res]
    assert scores == sorted(scores, reverse=True) and len(scores) >= 2
    assert [r.rank for r in res] == list(range(1, len(res) + 1))


def test_hybrid_combines_both_rankers(engine: SearchEngine) -> None:
    """A document found by only one ranker still appears in the hybrid results."""
    hybrid = {r.doc_id for r in engine.search("striker scored", top_n=4, method="hybrid")}
    bm25 = {r.doc_id for r in engine.search("striker scored", top_n=4, method="bm25")}
    dense = {r.doc_id for r in engine.search("striker scored", top_n=4, method="dense")}
    assert hybrid == bm25 | dense


def test_pagination(engine: SearchEngine) -> None:
    full = engine.search("the football team", top_n=4, method="dense", page_size=4)
    p1 = engine.search("the football team", top_n=4, method="dense", page=1, page_size=2)
    p2 = engine.search("the football team", top_n=4, method="dense", page=2, page_size=2)
    assert [r.doc_id for r in p1 + p2] == [r.doc_id for r in full]
    assert [r.rank for r in p2] == [3, 4]
    assert engine.search("football", top_n=4, method="bm25", page=9) == []


def test_blank_query_and_bad_arguments(engine: SearchEngine) -> None:
    assert engine.search("   ") == []
    with pytest.raises(ValueError):
        engine.search("x", method="magic")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        engine.search("x", page=0)


def test_rrf_math() -> None:
    fused = dict(reciprocal_rank_fusion([[1, 2], [2, 3]], [1.0, 1.0], k=60))
    assert fused[2] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[2] > fused[1] > fused[3]
    weighted = dict(reciprocal_rank_fusion([[1], [2]], [1.0, 0.0], k=60))
    assert weighted[2] == 0.0


def test_query_preprocessing_matches_corpus() -> None:
    assert query_terms("The Queen of England") == bm25_normalize("The Queen of England").split()
    assert query_terms("the of and") == ["the", "of", "and"]  # all stopwords -> fall back


def test_excerpt_window_centres_on_hits() -> None:
    text = " ".join(["filler"] * 100 + ["Climate", "change,", "said", "scientists."] + ["filler"] * 100)
    plain, marked = make_excerpt(text, ["climate", "scientist"], 20)
    assert "**Climate**" in marked and "**scientists.**" not in marked and "**scientists**." in marked
    assert plain.startswith("… ") and plain.endswith(" …") and len(plain.split()) <= 22
