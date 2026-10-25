"""Tests for the prefix-based suggestion index."""

from __future__ import annotations

from src.suggest import SuggestIndex


def make() -> SuggestIndex:
    phrases = {
        "hurricane": 90,
        "hurricane season": 40,
        "hurricane katrina": 70,
        "hunger": 10,
        "tropical storm": 50,
        "the quick brown fox": 1,
    }
    return SuggestIndex(phrases, {"the quick brown fox": "The quick brown fox"})


def test_prefix_ranked_by_frequency() -> None:
    assert make().suggest("hur") == ["hurricane", "hurricane katrina", "hurricane season"]
    assert make().suggest("hu", limit=2) == ["hurricane", "hurricane katrina"]


def test_case_whitespace_and_multiword() -> None:
    assert make().suggest("  Tropical   ST ") == ["tropical storm"]
    assert make().suggest("zzz") == [] and make().suggest("   ") == []


def test_exact_match_is_not_repeated_and_titles_keep_casing() -> None:
    assert "hurricane" not in make().suggest("hurricane")
    assert make().suggest("the quick") == ["The quick brown fox"]


def test_recent_queries_come_first() -> None:
    out = make().suggest("hur", recent=["Hurricane damage costs", "football"])
    assert out[0] == "Hurricane damage costs" and "hurricane" in out
