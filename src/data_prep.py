"""Corpus preparation: download, clean, chunk and save (implemented in the data phase)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import Config


@dataclass(frozen=True)
class Chunk:
    """A passage of a source document."""

    doc_id: str
    chunk_id: str
    title: str
    text: str


def download_corpus(cfg: Config) -> pd.DataFrame:
    """Download a size-capped sample of the corpus into ``paths.raw_dir``."""
    raise NotImplementedError


def clean_text(text: str) -> str:
    """Normalise whitespace and strip markup/noise from a document."""
    raise NotImplementedError


def chunk_document(doc_id: str, title: str, text: str, size: int, overlap: int) -> list[Chunk]:
    """Split a document into overlapping word-window chunks."""
    raise NotImplementedError


def prepare(cfg: Config) -> pd.DataFrame:
    """Run download -> clean -> chunk and save to ``paths.chunks_file``."""
    raise NotImplementedError
