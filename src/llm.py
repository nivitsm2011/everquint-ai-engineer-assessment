"""Pluggable LLM access: OpenAI or Ollama chosen by config, with retries, backoff and timeouts."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar

import requests

from src.config import Config
from src.utils import get_logger

log = get_logger(__name__)
T = TypeVar("T")


class LLMError(RuntimeError):
    """Base class for LLM failures."""


class LLMConfigError(LLMError):
    """The provider is not configured (e.g. missing API key)."""


class LLMUnavailableError(LLMError):
    """The provider cannot be reached (e.g. Ollama is not running)."""


class _Retryable(Exception):
    """Internal marker: timeouts, rate limits and 5xx responses are worth retrying."""


class LLMClient(ABC):
    """Common interface so providers can be swapped via config."""

    provider: str
    model: str

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.temperature = float(cfg.get("llm.temperature"))
        self.timeout = float(cfg.get("llm.timeout_seconds"))
        self.max_retries = int(cfg.get("llm.max_retries"))
        self.backoff = float(cfg.get("llm.backoff_seconds"))

    @abstractmethod
    def _call(self, prompt: str, system: str | None, max_tokens: int | None) -> str:
        """One provider request; raises ``_Retryable`` or an ``LLMError`` subclass."""

    def generate(self, prompt: str, system: str | None = None, max_tokens: int | None = None) -> str:
        """Return the model's completion, retrying transient failures with exponential backoff."""
        return self._retry(lambda: self._call(prompt, system, max_tokens))

    def _retry(self, fn: Callable[[], T]) -> T:
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn()
            except _Retryable as exc:
                last = exc
                if attempt < self.max_retries:
                    wait = min(self.backoff * 2 ** (attempt - 1), 30.0)
                    log.warning("%s attempt %d/%d failed (%s); retrying in %.1fs",
                                self.provider, attempt, self.max_retries, exc, wait)
                    time.sleep(wait)
        raise LLMError(f"{self.provider} request failed after {self.max_retries} attempts: {last}")


class OpenAIClient(LLMClient):
    """OpenAI chat-completions client; the API key is read from the environment (.env)."""

    provider = "openai"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.model = str(cfg.get("llm.openai.model"))
        env_name = str(cfg.get("llm.openai.api_key_env"))
        key = os.environ.get(env_name)
        if not key:
            raise LLMConfigError(
                f"{env_name} is not set. Copy .env.example to .env and add your key, "
                "or set llm.provider to 'ollama' in config/config.yaml."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=key, timeout=self.timeout, max_retries=0)

    def _call(self, prompt: str, system: str | None, max_tokens: int | None) -> str:
        import openai

        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        try:
            resp = self._client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature, max_tokens=max_tokens
            )
        except openai.AuthenticationError as exc:
            raise LLMConfigError(f"OpenAI rejected the API key: {exc}") from exc
        except (openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError,
                openai.InternalServerError) as exc:
            raise _Retryable(type(exc).__name__) from exc
        except openai.APIError as exc:
            raise LLMError(f"OpenAI error: {exc}") from exc
        return (resp.choices[0].message.content or "").strip()


class OllamaClient(LLMClient):
    """Local Ollama client over its HTTP API (free, offline)."""

    provider = "ollama"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self.model = str(cfg.get("llm.ollama.model"))
        self.host = str(cfg.get("llm.ollama.host")).rstrip("/")
        self.num_ctx = int(cfg.get("llm.ollama.num_ctx"))

    def _call(self, prompt: str, system: str | None, max_tokens: int | None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        options: dict[str, float | int] = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        if max_tokens:
            options["num_predict"] = max_tokens
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False, "options": options},
                timeout=self.timeout,
            )
        except requests.ConnectionError as exc:
            raise LLMUnavailableError(
                f"Cannot reach Ollama at {self.host}. Start it with `ollama serve` "
                "(or the Ollama app), or switch llm.provider to 'openai'."
            ) from exc
        except requests.Timeout as exc:
            raise _Retryable(f"timeout after {self.timeout:.0f}s") from exc
        if resp.status_code == 404:
            raise LLMUnavailableError(
                f"Ollama has no model '{self.model}'. Install it with `ollama pull {self.model}`."
            )
        if resp.status_code >= 500:
            raise _Retryable(f"HTTP {resp.status_code}")
        if not resp.ok:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:200]}")
        return str(resp.json()["message"]["content"]).strip()


def get_llm(cfg: Config) -> LLMClient:
    """Factory: build the client for ``cfg.llm_provider`` (config, or LLM_PROVIDER in .env)."""
    provider = cfg.llm_provider
    if provider == "openai":
        return OpenAIClient(cfg)
    if provider == "ollama":
        return OllamaClient(cfg)
    raise ValueError(f"Unknown LLM provider: {provider!r} (expected 'openai' or 'ollama')")
