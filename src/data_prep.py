"""Corpus preparation: stream CNN/DailyMail, clean, dedupe, chunk and save as parquet."""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.config import Config
from src.utils import get_logger

log = get_logger(__name__)

# --- boilerplate patterns (each is counted separately so cleaning can be audited) ---------------
_DATE = r"\d{1,2}:\d{2}\s*[A-Z]{2,4},\s*\d{1,2}\s+\w+\s+\d{4}"
BOILERPLATE: dict[str, re.Pattern[str]] = {
    "daily_mail_byline": re.compile(r"^\s*By\s*\.\s*[^.]{1,80}?\s*\.\s*(?:and\s*\.\s*[^.]{1,80}?\s*\.\s*)?"),
    "published_updated": re.compile(
        rf"PUBLISHED:\s*\.\s*{_DATE}\s*\.\s*(?:\|\s*\.\s*)?(?:UPDATED:\s*\.\s*{_DATE}\s*\.\s*)?"
        rf"|UPDATED:\s*\.\s*{_DATE}\s*\.\s*"
    ),
    "last_updated": re.compile(
        r"Last updated at \d{1,2}:\d{2} [AP]M on \d{1,2}(?:st|nd|rd|th) \w+ \d{4}\s*\.?\s*"
    ),
    # "LONDON, England (CNN) --", also when a teaser/caption precedes it (after a sentence end)
    "cnn_dateline": re.compile(
        r"(?:^|(?<=[.!?\"] ))(?:[A-Z][A-Za-z'-]*(?: [A-Za-z'-]+){0,3}(?:,\s*[A-Za-z. -]{2,30})?\s)?"
        r"\(CNN[\w ]{0,15}\)\s*(?:--|-)?\s*"
    ),
    # same dateline when no sentence end precedes it (dash required to avoid real "(CNN)" mentions)
    "cnn_dateline_inline": re.compile(
        r"(?:[A-Z][A-Za-z'-]*(?: [A-Za-z'-]+){0,3}(?:,\s*[A-Za-z. -]{2,30})?\s)?"
        r"\(CNN[\w ]{0,15}\)\s*(?:--|-)\s*"
    ),
    # syndicated-partner datelines such as "(Mashable) --", "(EW.com) --"
    "outlet_dateline": re.compile(
        r"^(?:[A-Z][A-Za-z'-]*(?: [A-Za-z'-]+){0,3}(?:,\s*[A-Za-z. -]{2,30})?\s)?"
        r"\((?!CNN|[A-Z]-)[A-Z][^()]{1,30}\)\s*--\s*"
    ),
    "email_to_friend": re.compile(r"E-mail to a friend\s*\.?\s*"),
}
_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(])")
_TOKEN = re.compile(r"[a-z0-9]+")
_NORM = re.compile(r"[^a-z0-9]+")
STOPWORDS = frozenset(ENGLISH_STOP_WORDS)


@dataclass(frozen=True)
class Chunk:
    """A sentence-aligned passage of a source document."""

    doc_id: str
    chunk_id: str
    title: str
    position: int
    n_chunks: int
    text: str
    bm25_text: str
    n_words: int


# --- text helpers ------------------------------------------------------------------------------
def clean_text(text: str, counts: Counter[str] | None = None) -> str:
    """Strip boilerplate, URLs and HTML; normalise unicode and whitespace.

    Original casing is preserved (this is what the LLM sees). ``counts`` accumulates how many
    documents each step modified.
    """
    counts = counts if counts is not None else Counter()

    def step(name: str, new: str, old: str) -> str:
        if new != old:
            counts[name] += 1
        return new

    text = text.strip()
    out = step("html_unescape", html.unescape(text), text)
    out = step("unicode_nfkc", unicodedata.normalize("NFKC", out), out)
    out = step("html_tags", _HTML_TAG.sub(" ", out), out)
    out = step("urls", _URL.sub(" ", out), out)
    for name, pattern in BOILERPLATE.items():
        out = step(name, pattern.sub("", out), out)
    return step("whitespace", _WS.sub(" ", out).strip(), out)


def normalize_for_hash(text: str) -> str:
    """Lowercase alphanumeric-only form used for duplicate detection."""
    return _NORM.sub(" ", text.lower()).strip()


def _digest(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()  # noqa: S324 - not security related


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a light regex splitter (no model download needed)."""
    return [s for s in _SENT_SPLIT.split(text) if s]


def bm25_normalize(text: str) -> str:
    """Lowercase, tokenize and drop stopwords: the lexical-index view of a text."""
    return " ".join(t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS)


def make_title(text: str, max_chars: int) -> str:
    """Use the first sentence as a title, truncated at a word boundary."""
    first = split_sentences(text)[0] if text else ""
    if len(first) <= max_chars:
        return first
    return first[:max_chars].rsplit(" ", 1)[0] + "…"


def chunk_document(doc_id: str, title: str, text: str, size: int, overlap: int) -> list[Chunk]:
    """Split a document into sentence-aware chunks of about ``size`` words.

    Sentences are added until the chunk reaches ``size`` words; the next chunk re-starts from the
    trailing sentences covering about ``overlap`` words so context is not cut mid-thought.
    """
    sents = split_sentences(text)
    lens = [len(s.split()) for s in sents]
    spans: list[tuple[int, int]] = []
    i, n = 0, len(sents)
    while i < n:
        j, words = i, 0
        while j < n and words < size:
            words += lens[j]
            j += 1
        spans.append((i, j))
        if j >= n:
            break
        k, back = j, 0
        while k > i + 1 and back < overlap:
            k -= 1
            back += lens[k]
        i = k
    chunks = []
    for pos, (a, b) in enumerate(spans):
        body = " ".join(sents[a:b])
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_c{pos:02d}",
                title=title,
                position=pos,
                n_chunks=len(spans),
                text=body,
                bm25_text=bm25_normalize(body),
                n_words=sum(lens[a:b]),
            )
        )
    return chunks


# --- streaming + cleaning ----------------------------------------------------------------------
def _stream(cfg: Config, split: str) -> Iterable[dict[str, Any]]:
    """Stream a split from Hugging Face so only the shards actually read are downloaded."""
    from datasets import load_dataset

    return load_dataset(
        cfg.get("corpus.dataset_name"), cfg.get("corpus.config_name"), split=split, streaming=True
    )


def clean_and_filter(
    rows: Iterable[dict[str, Any]],
    target: int,
    prefix: str,
    cfg: Config,
    seen_full: set[str],
    seen_head: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean rows until ``target`` documents survive; return them with an audit of every step."""
    min_words = int(cfg.get("corpus.min_words"))
    head_words = int(cfg.get("corpus.near_dup_head_words"))
    title_max = int(cfg.get("corpus.title_max_chars"))
    modified: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    raw_read = raw_words = clean_words = 0
    kept: list[dict[str, Any]] = []

    for row in rows:
        if len(kept) >= target:
            break
        raw_read += 1
        raw = row["article"] or ""
        raw_words += len(raw.split())
        text = clean_text(raw, modified)
        n_words = len(text.split())
        if n_words < min_words:
            dropped["too_short"] += 1
            continue
        norm = normalize_for_hash(text)
        full = _digest(norm)
        head = _digest(" ".join(norm.split()[:head_words]))
        if full in seen_full:
            dropped["exact_duplicate"] += 1
            continue
        if head in seen_head:
            dropped["near_duplicate"] += 1
            continue
        seen_full.add(full)
        seen_head.add(head)
        clean_words += n_words
        doc_id = f"{prefix}{len(kept):05d}"
        kept.append(
            {
                "doc_id": doc_id,
                "source_id": row["id"],
                "title": make_title(text, title_max),
                "text": text,
                "bm25_text": bm25_normalize(text),
                "reference_summary": _WS.sub(" ", (row["highlights"] or "").replace("\n", " ")).strip(),
                "n_words": n_words,
            }
        )
    audit = {
        "rows_read": raw_read,
        "docs_kept": len(kept),
        "docs_modified_by_step": dict(modified),
        "docs_dropped": dict(dropped),
        "total_words_before_cleaning_of_read_rows": raw_words,
        "total_words_after_cleaning_of_kept_docs": clean_words,
    }
    for name, count in {**modified, **{f"DROP {k}": v for k, v in dropped.items()}}.items():
        log.info("[%s] %-20s %7d docs", prefix, name, count)
    log.info("[%s] read %d rows -> kept %d docs", prefix, raw_read, len(kept))
    return pd.DataFrame(kept), audit


def build_chunks(docs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Chunk every document (sentence-aware, ~chunk_size words, ~chunk_overlap overlap)."""
    size, overlap = int(cfg.get("corpus.chunk_size")), int(cfg.get("corpus.chunk_overlap"))
    rows = [
        asdict(c)
        for d in docs.itertuples(index=False)
        for c in chunk_document(d.doc_id, d.title, d.text, size, overlap)
    ]
    return pd.DataFrame(rows)


def _describe(docs: pd.DataFrame, chunks: pd.DataFrame | None = None) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "documents": int(len(docs)),
        "words_per_doc_mean": round(float(docs["n_words"].mean()), 1),
        "words_per_doc_median": float(docs["n_words"].median()),
        "words_per_doc_min": int(docs["n_words"].min()),
        "words_per_doc_max": int(docs["n_words"].max()),
        "reference_summary_words_mean": round(
            float(docs["reference_summary"].str.split().str.len().mean()), 1
        ),
    }
    if chunks is not None:
        stats.update(
            chunks=int(len(chunks)),
            chunks_per_doc_mean=round(len(chunks) / len(docs), 2),
            words_per_chunk_mean=round(float(chunks["n_words"].mean()), 1),
            words_per_chunk_max=int(chunks["n_words"].max()),
        )
    return stats


def prepare(cfg: Config) -> dict[str, Any]:
    """Run stream -> clean -> dedupe -> chunk and write parquet + stats JSON."""
    seen_full: set[str] = set()
    seen_head: set[str] = set()
    n_train, n_held = int(cfg.get("corpus.train_docs")), int(cfg.get("corpus.heldout_docs"))

    log.info("Preparing %d corpus articles from the train split", n_train)
    docs, audit = clean_and_filter(_stream(cfg, "train"), n_train, "d", cfg, seen_full, seen_head)
    log.info("Preparing %d held-out articles from the test split", n_held)
    held, held_audit = clean_and_filter(_stream(cfg, "test"), n_held, "h", cfg, seen_full, seen_head)

    chunks = build_chunks(docs, cfg)
    for key in ("documents_file", "chunks_file", "heldout_file"):
        cfg.path(f"paths.{key}").parent.mkdir(parents=True, exist_ok=True)
    docs.to_parquet(cfg.path("paths.documents_file"), index=False)
    chunks.to_parquet(cfg.path("paths.chunks_file"), index=False)
    held.to_parquet(cfg.path("paths.heldout_file"), index=False)

    stats = {
        "dataset": f"{cfg.get('corpus.dataset_name')} ({cfg.get('corpus.config_name')})",
        "settings": {
            k: cfg.get(f"corpus.{k}")
            for k in ("train_docs", "heldout_docs", "min_words", "chunk_size", "chunk_overlap")
        },
        "corpus": {**_describe(docs, chunks), "cleaning_audit": audit},
        "heldout": {**_describe(held), "cleaning_audit": held_audit},
    }
    out = cfg.path("paths.stats_file")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Wrote %s", out)
    return stats
