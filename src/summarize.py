"""Length-controlled multi-document summarization."""

from __future__ import annotations

from src.config import Config
from src.llm import LLMClient
from src.search import SearchResult


def build_prompt(query: str, results: list[SearchResult], target_words: int) -> str:
    """Compose the summarization prompt from retrieved excerpts."""
    raise NotImplementedError


def summarize(
    query: str, results: list[SearchResult], cfg: Config, llm: LLMClient, length: str = "medium"
) -> str:
    """Summarize ``results`` to the configured word budget for ``length``."""
    raise NotImplementedError
