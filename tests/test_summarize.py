"""Summarizer and LLM-client tests using a fake LLM (no network, no model)."""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from src.config import Config, load_config
from src.llm import (
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMUnavailableError,
    OllamaClient,
    OpenAIClient,
    _Retryable,
)
from src.search import Result
from src.summarize import (
    DocumentStore,
    Summarizer,
    clean_output,
    estimate_tokens,
    resolve_length,
    summarize,
    trim_to_words,
    truncate_to_tokens,
)


class FakeLLM(LLMClient):
    """Returns ``n`` words, where ``n`` is read from 'about N words' in the prompt (or fixed)."""

    provider, model = "fake", "fake-1"

    def __init__(self, cfg: Config, words: int | None = None) -> None:
        super().__init__(cfg)
        self.fixed, self.prompts = words, []

    def _call(self, prompt: str, system: str | None, max_tokens: int | None) -> str:
        import re

        self.prompts.append(prompt)
        n = self.fixed or int(re.search(r"about (\d+) words", prompt).group(1))  # type: ignore[union-attr]
        return " ".join(["word."] * n)


@pytest.fixture()
def cfg(tmp_path: Path) -> Config:
    base = load_config().data
    data = {**base, "paths": {**base["paths"], "cache_dir": str(tmp_path / "cache")}}
    data["llm"] = {**base["llm"], "backoff_seconds": 0}
    return Config(data=data)


class StubStore(DocumentStore):
    def __init__(self, cfg: Config, texts: dict[str, str]) -> None:
        super().__init__(cfg)
        self._texts = texts


def make(cfg: Config, llm: FakeLLM | None = None, texts: dict[str, str] | None = None) -> Summarizer:
    texts = texts or {f"d{i}": f"Sentence about topic {i}. " * 30 for i in range(4)}
    return Summarizer(cfg, llm or FakeLLM(cfg), StubStore(cfg, texts))


def test_resolve_length(cfg: Config) -> None:
    assert [resolve_length(x, cfg) for x in ("short", "medium", "long", 77, "88")] == [50, 120, 250, 77, 88]
    for bad in ("huge", 5, 5000, 1.5, True):
        with pytest.raises(ValueError):
            resolve_length(bad, cfg)  # type: ignore[arg-type]


def test_text_helpers() -> None:
    assert clean_output("Here is a summary:\n**Cats** sleep.") == "Cats sleep."
    assert trim_to_words("One two. Three four. Five six.", 4) == "One two. Three four."
    long = "Alpha beta gamma. " * 100
    cut = truncate_to_tokens(long, 140, 1.4)
    assert estimate_tokens(cut, 1.4) <= 140 and cut.endswith(".")


@pytest.mark.parametrize("length,target", [("short", 50), ("medium", 120), (200, 200)])
def test_single_document_hits_target(cfg: Config, length: str | int, target: int) -> None:
    s = make(cfg).summarize_document("d0", length)
    assert (s.target_words, s.word_count, s.llm_calls, s.cached) == (target, target, 1, False)


def test_multi_document_is_map_reduce(cfg: Config) -> None:
    llm = FakeLLM(cfg)
    s = make(cfg, llm).summarize_documents(["d0", "d1", "d2"], "medium")
    assert s.llm_calls == 4  # 3 map calls + 1 reduce
    assert set(s.per_doc) == {"d0", "d1", "d2"}
    assert "SUMMARIES" in llm.prompts[-1] and "[3]" in llm.prompts[-1]
    assert s.word_count == 120


def test_cache_avoids_repeat_llm_calls(cfg: Config) -> None:
    llm = FakeLLM(cfg)
    summ = make(cfg, llm)
    first = summ.summarize_documents(["d0", "d1"], "short")
    n = len(llm.prompts)
    again = make(cfg, llm).summarize_documents(["d0", "d1"], "short")  # fresh object, same disk cache
    assert again.cached and again.llm_calls == 0 and len(llm.prompts) == n and again.text == first.text
    other_len = make(cfg, llm).summarize_documents(["d0", "d1"], "long")
    assert not other_len.cached and other_len.llm_calls == 1  # per-doc map summaries are reused


def test_cache_key_depends_on_model(cfg: Config) -> None:
    a = make(cfg)
    a.summarize_document("d0", "short")
    b = make(cfg)
    b.llm.model = "other-model"
    assert not b.summarize_document("d0", "short").cached


def test_length_retry_then_trim(cfg: Config) -> None:
    llm = FakeLLM(cfg, words=200)  # always far too long for a 50-word target
    s = make(cfg, llm).summarize_document("d0", "short")
    assert len(llm.prompts) == 2 and "too long" in llm.prompts[1]  # one corrective retry
    assert s.word_count <= 63  # trimmed to the upper bound (50 +25%, rounded)


def test_long_document_is_split_and_reduced(cfg: Config) -> None:
    cfg.data["summary"] = {**cfg.data["summary"], "max_input_tokens": 200}
    big = "This is a fairly ordinary sentence about nothing. " * 120  # ~1000 tokens
    llm = FakeLLM(cfg)
    s = make(cfg, llm, {"big": big}).summarize_document("big", "medium")
    assert s.llm_calls >= 4  # several segment summaries + final reduce
    assert all(len(p.split()) < 400 for p in llm.prompts)  # every prompt fits the budget


def test_summarize_wrapper_takes_results(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.summarize as mod

    monkeypatch.setattr(mod, "_default", make(cfg))
    docs = [Result(rank=i, doc_id=f"d{i}", title="t", score=1.0, method="hybrid") for i in range(1, 3)]
    assert summarize(docs, "short").doc_ids == ["d1", "d2"]
    with pytest.raises(ValueError):
        summarize([], "short")


# --- LLM clients ----------------------------------------------------------------------------------
def test_openai_requires_key(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        OpenAIClient(cfg)


def test_ollama_not_running_gives_clear_error(cfg: Config) -> None:
    cfg.data["llm"] = {
        **cfg.data["llm"],
        "ollama": {**cfg.data["llm"]["ollama"], "host": "http://127.0.0.1:9"},
    }
    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        OllamaClient(cfg).generate("hi")


def test_retries_with_backoff_then_succeeds(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    class Resp:
        status_code, ok, text = 200, True, ""

        def json(self) -> dict:  # type: ignore[type-arg]
            return {"message": {"content": " ok "}}

    def fake_post(*a: object, **k: object) -> Resp:
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.Timeout()
        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    assert OllamaClient(cfg).generate("hi") == "ok" and calls["n"] == 3


def test_gives_up_after_max_retries(cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout()))
    with pytest.raises(LLMError, match="after 3 attempts"):
        OllamaClient(cfg).generate("hi")
    assert issubclass(_Retryable, Exception)
