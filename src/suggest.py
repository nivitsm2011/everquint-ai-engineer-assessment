"""Auto-suggestion: prefix index over document titles and frequent corpus n-grams."""

from __future__ import annotations

import pickle
from bisect import bisect_left
from collections import Counter

import pandas as pd

from src.config import Config
from src.utils import get_logger, timed

log = get_logger(__name__)


class SuggestIndex:
    """Sorted phrase list with prefix lookup (binary search); ranked by corpus frequency.

    Phrases are matched case-insensitively from their start, so typing ``tropical st`` suggests
    ``tropical storm``. Recent session queries are offered first.
    """

    def __init__(self, phrases: dict[str, int], display: dict[str, str] | None = None) -> None:
        self._keys = sorted(phrases)
        self._counts = phrases
        self._display = display or {}

    def __len__(self) -> int:
        return len(self._keys)

    def suggest(self, prefix: str, limit: int = 8, recent: list[str] | None = None) -> list[str]:
        """Up to ``limit`` completions for ``prefix`` (recent queries first, then by frequency)."""
        p = " ".join(prefix.lower().split())
        if not p:
            return []
        out: list[str] = []
        for q in reversed(recent or []):
            if q.lower().startswith(p) and q.lower() != p and q not in out:
                out.append(q)
        i = bisect_left(self._keys, p)
        matches = []
        while i < len(self._keys) and self._keys[i].startswith(p):
            matches.append(self._keys[i])
            i += 1
        matches.sort(key=lambda k: (-self._counts[k], len(k)))
        for k in matches:
            if k == p:
                continue
            shown = self._display.get(k, k)
            if shown not in out:
                out.append(shown)
            if len(out) >= limit:
                break
        return out[:limit]


@timed
def build_suggest_index(cfg: Config) -> SuggestIndex:
    """Build phrases from frequent 1-3 word n-grams of the corpus plus all document titles."""
    docs = pd.read_parquet(cfg.path("paths.documents_file"), columns=["title", "bm25_text"])
    thresholds = {1: int(cfg.get("suggest.min_count_unigram")),
                  2: int(cfg.get("suggest.min_count_bigram")),
                  3: int(cfg.get("suggest.min_count_trigram"))}
    counts: dict[int, Counter[str]] = {n: Counter() for n in thresholds}
    for text in docs["bm25_text"]:
        tokens = text.split()
        for n in thresholds:
            counts[n].update(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    phrases: dict[str, int] = {}
    for n, counter in counts.items():
        phrases.update({p: c for p, c in counter.items()
                        if c >= thresholds[n] and all(len(w) > 1 for w in p.split())})  # drops "obama s"
    display: dict[str, str] = {}
    for title in docs["title"]:
        t = title.rstrip(" .…")
        if 8 <= len(t) <= int(cfg.get("suggest.max_title_chars")):
            phrases.setdefault(t.lower(), 1)
            display[t.lower()] = t
    log.info("suggestion index: %d phrases", len(phrases))
    return SuggestIndex(phrases, display)


def load_suggest_index(cfg: Config) -> SuggestIndex:
    """Load the cached index, building and saving it on first use."""
    path = cfg.path("paths.suggest_index")
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)  # noqa: S301 - produced locally
    index = build_suggest_index(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(index, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return index
