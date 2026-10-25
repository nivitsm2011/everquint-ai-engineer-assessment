"""Pure helpers for the Streamlit app (kept out of the app file so they can be unit-tested)."""

from __future__ import annotations

import math
import re
from typing import TypeVar

T = TypeVar("T")
_SPECIALS = re.compile(r"([\\`$~_\[\]<>#])")


def md_escape(text: str) -> str:
    """Escape Markdown/LaTeX specials (``$`` would start math) but keep ``**highlight**`` markers."""
    return _SPECIALS.sub(r"\\\1", text)


def page_count(total: int, page_size: int) -> int:
    """Number of pages needed for ``total`` items (at least 1)."""
    return max(1, math.ceil(total / page_size))


def slice_page(items: list[T], page: int, page_size: int) -> list[T]:
    """Items on 1-based ``page``."""
    start = (page - 1) * page_size
    return items[start : start + page_size]


def clamp_page(page: int, total: int, page_size: int) -> int:
    """Keep a page number inside ``1..page_count``."""
    return min(max(1, page), page_count(total, page_size))
