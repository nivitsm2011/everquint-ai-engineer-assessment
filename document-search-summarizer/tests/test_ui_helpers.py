"""Tests for the pure UI helpers."""

from __future__ import annotations

from src.ui_helpers import clamp_page, md_escape, page_count, slice_page


def test_md_escape_protects_dollars_but_keeps_highlight() -> None:
    assert md_escape("a $41.1 million **fortune** _x_") == r"a \$41.1 million **fortune** \_x\_"


def test_pagination_math() -> None:
    items = list(range(11))
    assert page_count(11, 5) == 3 and page_count(0, 5) == 1
    assert slice_page(items, 1, 5) == [0, 1, 2, 3, 4]
    assert slice_page(items, 3, 5) == [10]
    assert slice_page(items, 9, 5) == []
    assert (clamp_page(0, 11, 5), clamp_page(2, 11, 5), clamp_page(99, 11, 5)) == (1, 2, 3)
