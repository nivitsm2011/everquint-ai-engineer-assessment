"""Create the evaluation subset and generate queries with known relevant documents."""

from __future__ import annotations

import pandas as pd

from src.config import Config


def make_testset(cfg: Config) -> pd.DataFrame:
    """Sample test documents and produce ``(query, relevant_doc_id)`` pairs."""
    raise NotImplementedError
