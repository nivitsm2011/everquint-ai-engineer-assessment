"""Tests for evaluation metrics and helpers (no LLM, no indexes)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from rouge_score import rouge_scorer

from src.config import Config, load_config
from src.evaluation.eval_search import metrics_from_ranks, rank_of
from src.evaluation.eval_summary import HUMAN_CRITERIA, lead_n, rouge_f1, score_human_eval
from src.evaluation.make_testset import clean_query, generate_query, longest_copied_run
from src.llm import LLMClient
from src.utils import DiskCache


def test_rank_of() -> None:
    assert rank_of("b", ["a", "b", "c"]) == 2
    assert rank_of("z", ["a", "b"]) is None


def test_metrics_known_values() -> None:
    m = metrics_from_ranks([1, 2, None, 10, 6], [1, 5, 10])
    assert m["recall@1"] == pytest.approx(1 / 5)
    assert m["recall@5"] == pytest.approx(2 / 5)
    assert m["recall@10"] == pytest.approx(4 / 5)
    assert m["mrr@10"] == pytest.approx((1 + 1 / 2 + 1 / 10 + 1 / 6) / 5)
    assert m["ndcg@10"] == pytest.approx((1 + 1 / math.log2(3) + 1 / math.log2(11) + 1 / math.log2(7)) / 5)
    assert m["precision@5"] == pytest.approx(2 / 5 / 5)


def test_perfect_and_empty_systems() -> None:
    perfect = metrics_from_ranks([1, 1, 1], [1, 5, 10])
    assert perfect["mrr@10"] == perfect["ndcg@10"] == perfect["recall@1"] == 1.0
    assert metrics_from_ranks([None, None], [1, 5, 10])["mrr@10"] == 0.0


def test_copy_detection_and_query_cleaning() -> None:
    doc = "The prime minister announced a new economic plan on Tuesday in London."
    assert longest_copied_run("prime minister announced a new plan", doc) == 5
    assert longest_copied_run("football results today", doc) == 0
    assert clean_query('Query: "government spending plan."\nextra', 8) == "government spending plan"
    assert clean_query("one two three four five", 3) == "one two three"


class ScriptedLLM(LLMClient):
    provider, model = "fake", "fake-1"

    def __init__(self, cfg: Config, outputs: list[str]) -> None:
        super().__init__(cfg)
        self.outputs = outputs
        self.calls = 0

    def _call(self, prompt: str, system: str | None, max_tokens: int | None) -> str:
        self.calls += 1
        return self.outputs.pop(0)


def test_query_retries_when_copied_and_is_cached(tmp_path: Path) -> None:
    cfg = load_config()
    doc = "The prime minister announced a new economic plan on Tuesday in London."
    llm = ScriptedLLM(cfg, ["prime minister announced a new economic plan", "government budget proposal"])
    cache = DiskCache(tmp_path)
    assert generate_query("d1", doc, cfg, llm, cache) == "government budget proposal"
    assert generate_query("d1", doc, cfg, llm, cache) == "government budget proposal"
    assert llm.calls == 2  # second lookup came from the cache


def test_rouge_and_lead() -> None:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    same = rouge_f1(scorer, "the cats are sleeping", "the cat is sleeping")
    assert same["rouge1_f"] == pytest.approx(0.75)  # stemming matches cats/cat; are/is do not match
    assert rouge_f1(scorer, "alpha beta", "gamma delta")["rouge1_f"] == 0.0
    assert lead_n("One is first. Two is second. Three is third. Four is fourth.") == (
        "One is first. Two is second. Three is third."
    )


def test_score_human_eval(tmp_path: Path) -> None:
    sheet = pd.DataFrame({c: [5, 3, None] for c in HUMAN_CRITERIA})
    sheet["comments"] = ""
    path = tmp_path / "s.csv"
    sheet.to_csv(path, index=False)
    out = score_human_eval(path)
    assert out["rows_total"] == 3 and out["rows_scored"] == 2
    assert out["mean_scores"]["coherence"] == 4.0 and out["overall_mean"] == 4.0
    sheet.loc[0, "coherence"] = 9
    sheet.to_csv(path, index=False)
    with pytest.raises(ValueError):
        score_human_eval(path)
