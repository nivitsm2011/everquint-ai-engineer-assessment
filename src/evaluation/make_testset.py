"""5a. Create the evaluation subset and have the LLM write one search query per document."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.config import Config
from src.llm import LLMClient, LLMError, get_llm
from src.utils import DiskCache, get_logger

log = get_logger(__name__)

QUERY_SYSTEM = (
    "You write realistic web search queries. Output only the query text, nothing else: no quotes, "
    "no explanation, no punctuation at the end."
)
_WORD = re.compile(r"[a-z0-9']+")


def longest_copied_run(query: str, text: str) -> int:
    """Longest run of consecutive query words that also appears consecutively in ``text``."""
    q, t = _WORD.findall(query.lower()), _WORD.findall(text.lower())
    if not q or not t:
        return 0
    grams: dict[int, set[tuple[str, ...]]] = {}
    best = 0
    for n in range(1, len(q) + 1):
        grams[n] = {tuple(t[i : i + n]) for i in range(len(t) - n + 1)}
        if any(tuple(q[i : i + n]) in grams[n] for i in range(len(q) - n + 1)):
            best = n
        else:
            break
    return best


def clean_query(raw: str, max_words: int) -> str:
    """Keep the first line, strip quotes/labels/trailing punctuation, cap the length."""
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    line = re.sub(r"^(search query|query)\s*:\s*", "", line, flags=re.I).strip(" \"'`*")
    line = line.rstrip(".?!;:, ")
    return " ".join(line.split()[:max_words])


def _prompt(article: str, note: str = "") -> str:
    return (
        "Read the news article below and write ONE natural search query (3 to 8 words) that a person "
        "who wants to find this article might type into a search engine. Paraphrase: do NOT copy "
        f"sentences or long phrases from the article.{note}\n\nARTICLE:\n{article}\n\nSEARCH QUERY:"
    )


def generate_query(doc_id: str, text: str, cfg: Config, llm: LLMClient, cache: DiskCache) -> str:
    """Generate (and cache) one query for a document; retry once if it copies the article."""
    key = f"{doc_id}|{llm.provider}|{llm.model}|q1"
    hit = cache.get(key)
    if hit:
        return str(hit["query"])
    article = " ".join(text.split()[: int(cfg.get("evaluation.query_doc_words"))])
    max_words, max_run = int(cfg.get("evaluation.query_max_words")), int(cfg.get("evaluation.max_copied_run"))
    query = clean_query(llm.generate(_prompt(article), QUERY_SYSTEM, max_tokens=40), max_words)
    if not query or longest_copied_run(query, text) >= max_run:
        log.info("query for %s copied the article or was empty; retrying", doc_id)
        query = clean_query(
            llm.generate(_prompt(article, " Use different words than the article."), QUERY_SYSTEM, 40),
            max_words,
        ) or query
    cache.set(key, {"query": query})
    return query


def make_testset(cfg: Config, llm: LLMClient | None = None) -> pd.DataFrame:
    """Sample test documents from the indexed corpus (fixed seed) and write ``testset.csv``."""
    docs = pd.read_parquet(cfg.path("paths.documents_file"), columns=["doc_id", "text"])
    rng = np.random.default_rng(int(cfg.get("evaluation.seed")))
    n = int(cfg.get("evaluation.test_docs"))
    sample = docs.iloc[np.sort(rng.choice(len(docs), size=n, replace=False))]
    llm = llm or get_llm(cfg)
    cache = DiskCache(cfg.path("paths.cache_dir") / "queries")
    rows: list[dict[str, str]] = []
    for i, (doc_id, text) in enumerate(zip(sample["doc_id"], sample["text"], strict=True), 1):
        try:
            rows.append({"doc_id": doc_id, "query": generate_query(doc_id, text, cfg, llm, cache)})
        except LLMError as exc:
            log.warning("skipping %s: %s", doc_id, exc)
        if i % 10 == 0:
            log.info("queries generated: %d / %d", i, n)
    out = pd.DataFrame(rows)
    out = out[out["query"].str.strip() != ""]
    path = cfg.path("paths.testset_file")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    log.info("wrote %s (%d queries)", path, len(out))
    return out
