"""Configuration loader: reads ``config/config.yaml`` and secrets from ``.env``."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Immutable view over the YAML config with dotted-path access."""

    data: dict[str, Any]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Return ``data[a][b]...`` for ``"a.b..."``; ``default`` if any key is missing."""
        node: Any = self.data
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def path(self, dotted: str) -> Path:
        """Resolve a configured path relative to the project root."""
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"Missing path setting: {dotted}")
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def secret(self, env_name: str) -> str | None:
        """Read a secret (e.g. API key) from the environment, never from YAML."""
        return os.environ.get(env_name)

    @property
    def llm_provider(self) -> str:
        """Active LLM provider; the ``LLM_PROVIDER`` env var overrides YAML."""
        return (os.environ.get("LLM_PROVIDER") or self.get("llm.provider", "ollama")).lower()


def with_variant(cfg: Config, variant: str) -> Config:
    """Return a copy of ``cfg`` using the ``base`` or ``finetuned`` embedding model and dense index.

    The BM25/TF-IDF indexes are shared; only the embedding model, embeddings file and FAISS index
    change, so base and fine-tuned results can be compared side by side.
    """
    if variant == "base":
        return cfg
    if variant != "finetuned":
        raise ValueError(f"Unknown embedding variant {variant!r} (base | finetuned)")
    data = copy.deepcopy(cfg.data)
    ft = data["finetune"]
    data["embedding"]["model_name"] = str(PROJECT_ROOT / ft["model_dir"])
    data["paths"]["embeddings_file"] = ft["embeddings_file"]
    data["paths"]["faiss_index"] = ft["faiss_index"]
    return Config(data=data)


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> Config:
    """Load YAML config (cached) after loading ``.env`` into the environment.

    ``finetune.enabled: true`` (or ``EMBEDDING_VARIANT=finetuned`` in the environment) switches the
    whole system to the fine-tuned embedding model.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    cfg = Config(data=data)
    variant = os.environ.get("EMBEDDING_VARIANT") or ("finetuned" if data["finetune"]["enabled"] else "base")
    return with_variant(cfg, variant)
