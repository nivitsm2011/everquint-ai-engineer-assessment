"""Pluggable LLM access: OpenAI or Ollama, selected by config, with retries and timeouts."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.config import Config


class LLMClient(ABC):
    """Common interface so providers can be swapped via config."""

    @abstractmethod
    def generate(self, prompt: str, system: str | None = None) -> str:
        """Return the model's text completion for ``prompt``."""


class OpenAIClient(LLMClient):
    """OpenAI chat-completions client; API key is read from the environment."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def generate(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Local Ollama client over its HTTP API."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def generate(self, prompt: str, system: str | None = None) -> str:
        raise NotImplementedError


def get_llm(cfg: Config) -> LLMClient:
    """Factory: build the client for ``cfg.llm_provider``."""
    provider = cfg.llm_provider
    if provider == "openai":
        return OpenAIClient(cfg)
    if provider == "ollama":
        return OllamaClient(cfg)
    raise ValueError(f"Unknown LLM provider: {provider!r} (expected 'openai' or 'ollama')")
