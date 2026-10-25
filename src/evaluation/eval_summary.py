"""5c. Summary quality: ROUGE vs the CNN/DailyMail highlights, lead-3 baseline, human-eval sheet."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer

from src.config import Config
from src.data_prep import split_sentences
from src.llm import LLMError, LLMUnavailableError
from src.search import SearchEngine
from src.summarize import Summarizer
from src.utils import get_logger

log = get_logger(__name__)
ROUGE_TYPES = ["rouge1", "rouge2", "rougeL"]
HUMAN_CRITERIA = ["coherence", "coverage", "faithfulness", "conciseness"]
DETAIL_COLS = ["mode", "doc_id", "length", "words", "summary", "word_count", "seconds", "llm_calls",
               "rouge1_f", "rouge2_f", "rougeL_f"]


def lead_n(text: str, n: int = 3) -> str:
    """The first ``n`` sentences of an article (the classic news-summarization baseline)."""
    return " ".join(split_sentences(text)[:n])


def rouge_f1(scorer: rouge_scorer.RougeScorer, reference: str, candidate: str) -> dict[str, float]:
    """ROUGE-1/2/L F1 (stemmed) of ``candidate`` against ``reference``."""
    s = scorer.score(reference, candidate)
    return {f"{t}_f": s[t].fmeasure for t in ROUGE_TYPES}


def _load_details(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=DETAIL_COLS)


def run_single_doc_summaries(cfg: Config, summarizer: Summarizer | None = None) -> pd.DataFrame:
    """Summarise the first ``summary_eval_docs`` test documents at each length; resumable.

    Every finished (doc, length) is appended to ``summary_details.csv`` immediately, so an
    interrupted run continues where it stopped and real generation times are preserved.
    """
    summarizer = summarizer or Summarizer(cfg)
    testset = pd.read_csv(cfg.path("paths.testset_file"))
    docs = pd.read_parquet(cfg.path("paths.documents_file"), columns=["doc_id", "reference_summary"])
    refs = dict(zip(docs["doc_id"], docs["reference_summary"], strict=True))
    ids = testset["doc_id"].tolist()[: int(cfg.get("evaluation.summary_eval_docs"))]
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)
    path = cfg.path("paths.summary_details_file")
    details = _load_details(path)
    done = set(zip(details.loc[details["mode"] == "single", "doc_id"],
                   details.loc[details["mode"] == "single", "length"], strict=True))

    for i, doc_id in enumerate(ids, 1):
        for length in cfg.get("evaluation.summary_lengths"):
            if (doc_id, length) in done:
                continue
            try:
                s = summarizer.summarize_document(doc_id, length)
            except LLMUnavailableError:
                raise
            except LLMError as exc:
                log.warning("skipping %s/%s: %s", doc_id, length, exc)
                continue
            row = {"mode": "single", "doc_id": doc_id, "length": length, "words": s.target_words,
                   "summary": s.text, "word_count": s.word_count,
                   "seconds": None if s.cached else s.seconds, "llm_calls": s.llm_calls,
                   **rouge_f1(scorer, refs[doc_id], s.text)}
            details = pd.concat([details, pd.DataFrame([row])], ignore_index=True)
            details.to_csv(path, index=False)
        log.info("summarised %d / %d test documents", i, len(ids))
    return details


def rouge_table(cfg: Config, details: pd.DataFrame) -> pd.DataFrame:
    """Mean ROUGE F1 per system: lead-3 baseline (same docs, and all test docs) and each length."""
    testset = pd.read_csv(cfg.path("paths.testset_file"))
    docs = pd.read_parquet(cfg.path("paths.documents_file"), columns=["doc_id", "text", "reference_summary"])
    docs = docs.set_index("doc_id")
    scorer = rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=True)

    def lead_rows(ids: list[str]) -> pd.DataFrame:
        recs = []
        for d in ids:
            cand = lead_n(docs.at[d, "text"])
            recs.append({**rouge_f1(scorer, docs.at[d, "reference_summary"], cand),
                         "word_count": len(cand.split())})
        return pd.DataFrame(recs)

    single = details[details["mode"] == "single"]
    ids = single["doc_id"].unique().tolist()
    rows: list[dict[str, object]] = []

    def add(name: str, frame: pd.DataFrame, n: int) -> None:
        rows.append({"system": name, "docs": n,
                     **{c: round(float(frame[c].mean()), 4) for c in ("rouge1_f", "rouge2_f", "rougeL_f")},
                     "mean_words": round(float(frame["word_count"].mean()), 1)})

    add("lead-3 (same docs)", lead_rows(ids), len(ids))
    add("lead-3 (all test docs)", lead_rows(testset["doc_id"].tolist()), len(testset))
    for length in cfg.get("evaluation.summary_lengths"):
        sub = single[single["length"] == length]
        if len(sub):
            add(f"LLM {length} (~{int(sub['words'].iloc[0])} words)", sub, len(sub))
    table = pd.DataFrame(rows)
    table.to_csv(cfg.path("paths.summary_scores_file"), index=False)
    return table


def build_human_eval_sheet(cfg: Config, summarizer: Summarizer | None = None,
                           engine: SearchEngine | None = None) -> pd.DataFrame:
    """Sample test queries, summarise their retrieved documents, and export a scoring sheet."""
    summarizer = summarizer or Summarizer(cfg)
    engine = engine or SearchEngine(cfg)
    testset = pd.read_csv(cfg.path("paths.testset_file"))
    pool = min(int(cfg.get("evaluation.human_eval_pool")), len(testset))
    n = min(int(cfg.get("evaluation.human_eval_samples")), pool)
    # a fixed pool is drawn once; using its first n rows keeps earlier (cached) samples valid
    pick = np.random.default_rng(int(cfg.get("evaluation.seed")) + 1).choice(len(testset), pool, replace=False)
    pick = np.sort(pick)[:n]
    length = cfg.get("evaluation.human_eval_length")
    top = int(cfg.get("evaluation.human_eval_top_docs"))
    path = cfg.path("paths.summary_details_file")
    details = _load_details(path)
    done = set(details.loc[details["mode"] == "multi", "doc_id"])
    rows: list[dict[str, object]] = []
    for k, idx in enumerate(pick.tolist(), 1):
        query, target = testset["query"].iat[idx], testset["doc_id"].iat[idx]
        hits = engine.search(query, top_n=top, method="hybrid", page=1, page_size=top)
        ids = [h.doc_id for h in hits]
        try:
            s = summarizer.summarize_documents(ids, length)
        except LLMUnavailableError:
            raise
        except LLMError as exc:
            log.warning("skipping sample %d: %s", k, exc)
            continue
        key = "+".join(ids)
        if key not in done:
            details = pd.concat([details, pd.DataFrame([{
                "mode": "multi", "doc_id": key, "length": length, "words": s.target_words,
                "summary": s.text, "word_count": s.word_count,
                "seconds": None if s.cached else s.seconds, "llm_calls": s.llm_calls}])], ignore_index=True)
            details.to_csv(path, index=False)
        rows.append({
            "sample_id": k, "query": query,
            "retrieved_docs": "\n".join(f"{h.doc_id} | {h.title}" for h in hits),
            "target_in_results": "yes" if target in ids else "no",
            "source_text": "\n\n".join(f"[{h.doc_id}] {summarizer.store.text(h.doc_id)}" for h in hits),
            "summary": s.text, "summary_words": s.word_count,
            **{c: "" for c in HUMAN_CRITERIA}, "comments": "",
        })
        log.info("human-eval sample %d / %d", k, n)
    sheet = pd.DataFrame(rows)
    sheet.to_csv(cfg.path("paths.human_eval_file"), index=False, encoding="utf-8-sig")
    return sheet


def score_human_eval(sheet_path: Path) -> dict[str, object]:
    """Average the 1-5 ratings in a filled-in human-evaluation sheet."""
    sheet = pd.read_csv(sheet_path)
    out: dict[str, object] = {"rows_total": int(len(sheet))}
    scores = sheet[HUMAN_CRITERIA].apply(pd.to_numeric, errors="coerce")
    bad = scores[(scores < 1) | (scores > 5)].stack()
    if len(bad):
        raise ValueError(f"Ratings must be between 1 and 5; found {len(bad)} out-of-range values")
    out["rows_scored"] = int(scores.notna().all(axis=1).sum())
    out["mean_scores"] = {c: round(float(scores[c].mean()), 2) for c in HUMAN_CRITERIA if scores[c].notna().any()}
    out["overall_mean"] = round(float(scores.stack().mean()), 2) if scores.notna().any().any() else None
    return out


def evaluate_summaries(cfg: Config) -> dict[str, object]:
    """Run ROUGE evaluation and write the human-eval sheet; return a JSON-friendly summary."""
    details = run_single_doc_summaries(cfg)
    table = rouge_table(cfg, details)
    sheet = build_human_eval_sheet(cfg)
    return {"rouge": json.loads(table.to_json(orient="records")), "human_eval_rows": int(len(sheet))}
