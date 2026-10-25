"""Length-controlled single- and multi-document summarization (map-reduce) with disk caching."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from src.config import Config, load_config
from src.data_prep import split_sentences
from src.llm import LLMClient, get_llm
from src.search import Result
from src.utils import DiskCache, get_logger

log = get_logger(__name__)

LengthLabel = Literal["short", "medium", "long"]
SYSTEM_PROMPT = (
    "You are a careful summarization assistant. Write only from the text you are given: do not add "
    "facts, names, numbers or opinions that are not in it, and do not use outside knowledge. Write "
    "plain, fluent prose in a single paragraph (no bullet points, headings or preamble) and keep "
    "the most important information."
)
_PREAMBLE = re.compile(r"^(?:here(?:'s| is| are)\b[^\n:]*:|summary\s*:|\*\*summary\*\*:?)\s*", re.I)


@dataclass(frozen=True)
class Summary:
    """A generated summary plus how it was produced."""

    text: str
    target_words: int
    word_count: int
    doc_ids: list[str]
    model: str
    provider: str
    per_doc: dict[str, str] = field(default_factory=dict)  # map-step summaries (multi-doc only)
    llm_calls: int = 0
    cached: bool = False
    seconds: float = 0.0


# --- helpers ---------------------------------------------------------------------------------------
def resolve_length(length: str | int, cfg: Config) -> int:
    """Turn ``"short"|"medium"|"long"`` or an exact word count into a target number of words."""
    lengths = cfg.get("summary.lengths")
    if isinstance(length, str) and length.strip().isdigit():
        length = int(length)
    if isinstance(length, str):
        if length not in lengths:
            raise ValueError(f"Unknown length {length!r}; use one of {list(lengths)} or a word count")
        return int(lengths[length])
    if isinstance(length, bool) or not isinstance(length, int):
        raise ValueError(f"length must be a label or an int, got {length!r}")
    lo, hi = int(cfg.get("summary.min_words")), int(cfg.get("summary.max_words"))
    if not lo <= length <= hi:
        raise ValueError(f"Word target {length} outside allowed range {lo}-{hi}")
    return length


def estimate_tokens(text: str, tokens_per_word: float) -> int:
    """Conservative token estimate (words x ratio); avoids shipping a tokenizer per provider."""
    return math.ceil(len(text.split()) * tokens_per_word)


def truncate_to_tokens(text: str, max_tokens: int, tokens_per_word: float) -> str:
    """Cut text to fit ``max_tokens`` (estimated), ending on a sentence boundary when possible."""
    max_words = max(1, int(max_tokens / tokens_per_word))
    words = text.split()
    if len(words) <= max_words:
        return text
    cut = " ".join(words[:max_words])
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return cut[: end + 1] if end > len(cut) * 0.6 else cut


def clean_output(text: str) -> str:
    """Strip preambles/markdown from model output and collapse it to one paragraph."""
    text = re.sub(r"[*_#]+", "", text.strip())
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.M)
    text = _PREAMBLE.sub("", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def trim_to_words(text: str, max_words: int) -> str:
    """Drop trailing sentences until the text has at most ``max_words`` words (keeps >= 1 sentence)."""
    if len(text.split()) <= max_words:
        return text
    kept: list[str] = []
    total = 0
    for sent in split_sentences(text):
        n = len(sent.split())
        if kept and total + n > max_words:
            break
        kept.append(sent)
        total += n
    return " ".join(" ".join(kept).split()[:max_words])  # a single over-long sentence is cut by words


class DocumentStore:
    """Full article text by ``doc_id`` (searchable corpus and held-out pool)."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._texts: dict[str, str] | None = None

    def text(self, doc_id: str) -> str:
        if self._texts is None:
            frames = [
                pd.read_parquet(self.cfg.path(k), columns=["doc_id", "text"])
                for k in ("paths.documents_file", "paths.heldout_file")
                if self.cfg.path(k).exists()
            ]
            all_docs = pd.concat(frames)
            self._texts = dict(zip(all_docs["doc_id"], all_docs["text"], strict=True))
        if doc_id not in self._texts:
            raise KeyError(f"Unknown doc_id {doc_id!r}")
        return self._texts[doc_id]


# --- summarizer ------------------------------------------------------------------------------------
class Summarizer:
    """Summarizes text, documents and sets of retrieved documents with a pluggable LLM."""

    def __init__(
        self,
        cfg: Config | None = None,
        llm: LLMClient | None = None,
        store: DocumentStore | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self._llm = llm
        self.store = store or DocumentStore(self.cfg)
        self.cache = DiskCache(self.cfg.path("paths.cache_dir") / "summaries")
        self.tpw = float(self.cfg.get("summary.tokens_per_word"))
        self.budget = int(self.cfg.get("summary.max_input_tokens"))
        self.map_words = int(self.cfg.get("summary.map_words"))
        self._calls = 0

    @property
    def llm_name(self) -> str:
        """``provider/model`` from config, without creating a client (so no key check yet)."""
        provider = self.cfg.llm_provider
        return f"{provider}/{self.cfg.get(f'llm.{provider}.model')}"

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = get_llm(self.cfg)
        return self._llm

    # -- low level: one LLM call with length control ---------------------------------------------
    def _bounds(self, words: int) -> tuple[int, int]:
        tol = float(self.cfg.get("summary.length_tolerance"))
        return max(1, round(words * (1 - tol))), max(1, round(words * (1 + tol)))

    def _generate(self, prompt: str, words: int) -> str:
        """Call the LLM; if the length is off, re-prompt (config-limited), then trim if too long."""
        lo, hi = self._bounds(words)
        max_tokens = int(words * 2.2) + 60  # generous: avoids cutting mid-sentence
        best = ""
        attempt_prompt = prompt
        for attempt in range(int(self.cfg.get("summary.length_retries")) + 1):
            self._calls += 1
            text = clean_output(self.llm.generate(attempt_prompt, SYSTEM_PROMPT, max_tokens))
            if not best or abs(len(text.split()) - words) < abs(len(best.split()) - words):
                best = text
            n = len(text.split())
            if lo <= n <= hi:
                break
            direction = "too long" if n > hi else "too short"
            attempt_prompt = (
                f"{prompt}\n\nNote: a previous attempt was {n} words, which is {direction}. "
                f"Write about {words} words."
            )
            log.info("summary length %d outside %d-%d (attempt %d)", n, lo, hi, attempt + 1)
        return trim_to_words(best, hi)

    def _single_prompt(self, text: str, words: int) -> str:
        lo, hi = self._bounds(words)
        return (
            f"Summarize the following article in about {words} words (at least {lo}, at most {hi}).\n\n"
            f"ARTICLE:\n{text}\n\nSUMMARY (about {words} words):"
        )

    def _reduce_prompt(self, parts: list[str], words: int) -> str:
        lo, hi = self._bounds(words)
        body = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(parts, 1))
        return (
            f"Below are summaries of {len(parts)} related news articles. Write ONE coherent summary of "
            f"about {words} words (at least {lo}, at most {hi}) that captures the essence of all of "
            "them: merge overlapping information and mention clearly different stories. Use only the "
            f"information below.\n\nSUMMARIES:\n{body}\n\nCOMBINED SUMMARY (about {words} words):"
        )

    # -- map-reduce core -------------------------------------------------------------------------
    def _segments(self, text: str) -> list[str]:
        """Split an over-long text into sentence-aligned pieces that each fit the token budget."""
        segments: list[str] = []
        current: list[str] = []
        used = 0
        for sent in split_sentences(text):
            sent = truncate_to_tokens(sent, self.budget, self.tpw)  # a single huge "sentence"
            n = estimate_tokens(sent, self.tpw)
            if current and used + n > self.budget:
                segments.append(" ".join(current))
                current, used = [], 0
            current.append(sent)
            used += n
        if current:
            segments.append(" ".join(current))
        return segments

    def _reduce(self, parts: list[str], words: int) -> str:
        """Combine partial summaries into one; groups them first if they exceed the budget."""
        joined = "\n\n".join(parts)
        if len(parts) > 1 and estimate_tokens(joined, self.tpw) > self.budget:
            groups: list[list[str]] = [[]]
            used = 0
            for p in parts:
                n = estimate_tokens(p, self.tpw)
                if groups[-1] and used + n > self.budget:
                    groups.append([])
                    used = 0
                groups[-1].append(p)
                used += n
            if len(groups) < len(parts):
                return self._reduce([self._reduce(g, self.map_words) for g in groups], words)
            # no two partial summaries fit together: truncate each so they do (last resort)
            share = max(1, self.budget // len(parts))
            parts = [truncate_to_tokens(p, share, self.tpw) for p in parts]
        return self._generate(self._reduce_prompt(parts, words), words)

    def _summarize_raw(self, text: str, words: int) -> str:
        """Summarize one text; map-reduce inside the document if it exceeds the context budget."""
        if estimate_tokens(text, self.tpw) <= self.budget:
            return self._generate(self._single_prompt(text, words), words)
        log.info("document exceeds %d tokens; summarizing in segments", self.budget)
        partial = [self._generate(self._single_prompt(s, self.map_words), self.map_words)
                   for s in self._segments(text)]
        return self._reduce(partial, words)

    # -- cache -----------------------------------------------------------------------------------
    def _key(self, kind: str, ids: list[str], words: int) -> str:
        return json.dumps(
            {
                "v": self.cfg.get("summary.prompt_version"),
                "kind": kind,
                "docs": ids,
                "words": words,
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.cfg.get("llm.temperature"),
            },
            sort_keys=True,
        )

    def _cached(self, key: str) -> dict | None:  # type: ignore[type-arg]
        return self.cache.get(key) if self.cfg.get("summary.use_cache") else None

    def _store(self, key: str, value: dict) -> None:  # type: ignore[type-arg]
        if self.cfg.get("summary.use_cache"):
            self.cache.set(key, value)

    def _make(self, text: str, words: int, ids: list[str], t0: float, calls0: int,
              per_doc: dict[str, str] | None = None, cached: bool = False) -> Summary:
        return Summary(
            text=text, target_words=words, word_count=len(text.split()), doc_ids=ids,
            model=self.llm.model, provider=self.llm.provider, per_doc=per_doc or {},
            llm_calls=self._calls - calls0, cached=cached, seconds=round(time.perf_counter() - t0, 2),
        )

    # -- public API ------------------------------------------------------------------------------
    def summarize_text(self, text: str, length: str | int = "medium", doc_id: str | None = None) -> Summary:
        """Single-document summary of raw text (used for ROUGE evaluation on held-out articles)."""
        words = resolve_length(length, self.cfg)
        t0, calls0 = time.perf_counter(), self._calls
        ident = doc_id or "text:" + hashlib.sha256(text.encode()).hexdigest()[:16]
        key = self._key("doc", [ident], words)
        hit = self._cached(key)
        if hit:
            return self._make(hit["text"], words, [ident], t0, calls0, cached=True)
        out = self._summarize_raw(text, words)
        self._store(key, {"text": out})
        return self._make(out, words, [ident], t0, calls0)

    def summarize_document(self, doc_id: str, length: str | int = "medium") -> Summary:
        """Single-document summary of an indexed or held-out article, by id."""
        return self.summarize_text(self.store.text(doc_id), length, doc_id=doc_id)

    def summarize_documents(self, doc_ids: list[str], length: str | int = "medium") -> Summary:
        """Multi-document summary: summarize each document (cached), then combine (map-reduce)."""
        ids = list(dict.fromkeys(doc_ids))
        if not ids:
            raise ValueError("No documents to summarize")
        if len(ids) == 1:
            return self.summarize_document(ids[0], length)
        words = resolve_length(length, self.cfg)
        t0, calls0 = time.perf_counter(), self._calls
        key = self._key("multi", ids, words)
        hit = self._cached(key)
        if hit:
            return self._make(hit["text"], words, ids, t0, calls0, hit["per_doc"], cached=True)
        per_doc = {i: self.summarize_document(i, self.map_words).text for i in ids}  # map
        out = self._reduce(list(per_doc.values()), words)  # reduce
        self._store(key, {"text": out, "per_doc": per_doc})
        return self._make(out, words, ids, t0, calls0, per_doc)


_default: Summarizer | None = None


def summarize(docs: list[Result], length: str | int = "medium") -> Summary:
    """Summarize retrieved search results; ``length`` is short | medium | long or a word target."""
    global _default
    if _default is None:
        _default = Summarizer()
    return _default.summarize_documents([d.doc_id for d in docs], length)
