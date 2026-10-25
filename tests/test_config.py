"""Tests for configuration loading and the LLM provider factory."""

from __future__ import annotations

import pytest

from src.config import PROJECT_ROOT, load_config
from src.llm import OllamaClient, OpenAIClient, get_llm


def test_config_loads() -> None:
    cfg = load_config()
    assert cfg.get("search.top_n") == 5
    assert cfg.get("missing.key", "fallback") == "fallback"


def test_paths_resolve_under_project_root() -> None:
    cfg = load_config()
    assert cfg.path("paths.index_dir") == PROJECT_ROOT / "indexes"


def test_provider_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = load_config()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert isinstance(get_llm(cfg), OpenAIClient)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    assert isinstance(get_llm(cfg), OllamaClient)
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError):
        get_llm(cfg)
