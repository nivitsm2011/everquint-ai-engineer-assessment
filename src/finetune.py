"""Phase 7: domain-adapt the embedding model on (LLM query, passage) pairs from TRAIN documents."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config, load_config, with_variant
from src.evaluation.eval_search import evaluate_search
from src.evaluation.make_testset import generate_query
from src.indexing import build_faiss, embed_chunks, get_device
from src.llm import LLMError, LLMUnavailableError, get_llm
from src.search import SearchEngine
from src.utils import DiskCache, get_logger

log = get_logger(__name__)


def make_pairs(cfg: Config) -> pd.DataFrame:
    """Create ``(query, passage)`` pairs from indexed documents that are NOT in the test set.

    The query is written by the LLM from the start of the article; the passage is the article's first
    chunk, i.e. the text the query was written from. Resumable: queries are cached on disk.
    """
    ft = cfg.get("finetune")
    docs = pd.read_parquet(cfg.path("paths.documents_file"), columns=["doc_id", "text"])
    test_ids = set(pd.read_csv(cfg.path("paths.testset_file"))["doc_id"])
    pool = docs[~docs["doc_id"].isin(test_ids)]
    rng = np.random.default_rng(int(ft["seed"]))
    sample = pool.iloc[np.sort(rng.choice(len(pool), size=int(ft["num_pairs"]), replace=False))]
    chunks = pd.read_parquet(cfg.path("paths.chunks_file"), columns=["chunk_id", "doc_id", "position", "text"])
    first = dict(zip(chunks.loc[chunks["position"] == 0, "doc_id"], chunks.loc[chunks["position"] == 0, "text"],
                     strict=True))
    llm, cache = get_llm(cfg), DiskCache(cfg.path("paths.cache_dir") / "train_queries")
    rows = []
    for i, (doc_id, text) in enumerate(zip(sample["doc_id"], sample["text"], strict=True), 1):
        try:
            q = generate_query(doc_id, text, cfg, llm, cache, doc_words=int(ft["query_doc_words"]))
        except LLMUnavailableError:
            raise
        except LLMError as exc:
            log.warning("skipping %s: %s", doc_id, exc)
            continue
        if q:
            rows.append({"doc_id": doc_id, "query": q, "passage": first[doc_id]})
        if i % 10 == 0:
            log.info("training queries: %d / %d", i, len(sample))
    pairs = pd.DataFrame(rows)
    assert not set(pairs["doc_id"]) & test_ids, "training pairs leaked test documents"
    path = cfg.path("finetune.pairs_file")
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(path, index=False)
    log.info("wrote %s (%d pairs, 0 test documents)", path, len(pairs))
    return pairs


def train(cfg: Config) -> None:
    """Fine-tune the base embedding model with MultipleNegativesRankingLoss and save it."""
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from sentence_transformers.datasets import NoDuplicatesDataLoader

    ft = cfg.get("finetune")
    pairs = pd.read_csv(cfg.path("finetune.pairs_file"))
    model = SentenceTransformer(cfg.get("embedding.model_name"), device=get_device())
    model.max_seq_length = int(cfg.get("embedding.max_seq_length"))
    examples = [InputExample(texts=[q, p]) for q, p in zip(pairs["query"], pairs["passage"], strict=True)]
    loader = NoDuplicatesDataLoader(examples, batch_size=int(ft["batch_size"]))
    loss = losses.MultipleNegativesRankingLoss(model)
    steps = len(loader) * int(ft["epochs"])
    log.info("training on %d pairs, %d steps (batch %d)", len(examples), steps, int(ft["batch_size"]))
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=int(ft["epochs"]),
        warmup_steps=max(1, int(steps * float(ft["warmup_ratio"]))),
        optimizer_params={"lr": float(ft["learning_rate"])},
        show_progress_bar=False,
    )
    out = cfg.path("finetune.model_dir")
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    log.info("saved fine-tuned model to %s", out)


def build_finetuned_index(cfg: Config) -> None:
    """Re-embed every chunk with the fine-tuned model and build its own FAISS index."""
    cfg_ft = with_variant(cfg, "finetuned")
    chunks = pd.read_parquet(cfg_ft.path("paths.chunks_file"))
    build_faiss(embed_chunks(chunks, cfg_ft), cfg_ft)


def compare(cfg: Config) -> pd.DataFrame:
    """Evaluate base vs fine-tuned on the test queries; save a long-format comparison table."""
    base = evaluate_search(cfg, SearchEngine(cfg), write=False).set_index("method")
    ft_cfg = with_variant(cfg, "finetuned")
    tuned = evaluate_search(ft_cfg, SearchEngine(ft_cfg), write=False).set_index("method")
    rows = []
    for method in ("dense", "hybrid", "bm25"):
        for metric in [c for c in base.columns if c != "queries"]:
            b, t = float(base.at[method, metric]), float(tuned.at[method, metric])
            rows.append({"method": method, "metric": metric, "base": round(b, 4),
                         "finetuned": round(t, 4), "delta": round(t - b, 4)})
    table = pd.DataFrame(rows)
    out = cfg.path("finetune.comparison_file")
    table.to_csv(out, index=False)
    return table


__all__ = ["make_pairs", "train", "build_finetuned_index", "compare", "load_config"]
