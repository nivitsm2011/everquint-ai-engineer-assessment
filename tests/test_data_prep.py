"""Tests for cleaning, dedup and sentence-aware chunking."""

from __future__ import annotations

from src.config import load_config
from src.data_prep import (
    bm25_normalize,
    chunk_document,
    clean_and_filter,
    clean_text,
    make_title,
    split_sentences,
)


def test_cnn_dateline_removed() -> None:
    assert clean_text("LONDON, England (CNN) -- The queen spoke today.") == "The queen spoke today."
    assert clean_text("(CNN) -- Hello there.") == "Hello there."


def test_midtext_and_partner_datelines_removed() -> None:
    assert (
        clean_text("Teaser line here. MIAMI, Florida (CNN) -- The floor.") == "Teaser line here. The floor."
    )
    assert clean_text("Teaser. NEW YORK (CNN) -- Hi there.") == "Teaser. Hi there."
    assert clean_text("(Travel + Leisure) -- It is nice.") == "It is nice."
    assert clean_text("(Mashable) -- Apple did it.") == "Apple did it."
    assert clean_text("LONDON, England (Reuters) -- Prices fell.") == "Prices fell."
    assert clean_text("Kerry (D-Mass.) -- said so.") == "Kerry (D-Mass.) -- said so."
    assert clean_text("Sen. Kerry (D-Mass.) -- said so.") == "Sen. Kerry (D-Mass.) -- said so."


def test_daily_mail_boilerplate_removed() -> None:
    raw = (
        "By . Daily Mail Reporter . PUBLISHED: . 07:53 EST, 25 October 2013 . | . "
        "UPDATED: . 08:22 EST, 25 October 2013 . A man was arrested on Friday."
    )
    assert clean_text(raw) == "A man was arrested on Friday."


def test_urls_html_whitespace() -> None:
    out = clean_text("See <b>this</b> at http://x.com/a?b=1  now &amp; later.")
    assert out == "See this at now & later."


def test_casing_preserved_but_bm25_view_lowercased() -> None:
    assert clean_text("The Queen") == "The Queen"
    assert bm25_normalize("The Queen of England") == "queen england"


def test_title_truncates() -> None:
    assert make_title("Short one. Second.", 50) == "Short one."
    assert len(make_title("word " * 100 + ". Next.", 30)) <= 31


def test_chunks_sentence_aware_with_overlap() -> None:
    sents = [f"Sentence number {i} has exactly seven words." for i in range(40)]
    text = " ".join(sents)
    chunks = chunk_document("d1", "t", text, size=60, overlap=15)
    assert len(chunks) > 1
    assert all(c.text.endswith(".") for c in chunks)
    assert chunks[0].chunk_id == "d1_c00" and chunks[0].n_chunks == len(chunks)
    # consecutive chunks share a sentence (overlap)
    assert set(split_sentences(chunks[0].text)) & set(split_sentences(chunks[1].text))
    # nothing lost
    covered = {s for c in chunks for s in split_sentences(c.text)}
    assert covered == set(sents)


def test_short_doc_is_single_chunk() -> None:
    assert len(chunk_document("d", "t", "One. Two. Three.", 200, 40)) == 1


def test_filter_drops_short_and_duplicates() -> None:
    cfg = load_config()
    long_a = "Alpha story " + "word " * 80 + "ends."
    long_b = "Beta story " + "other " * 80 + "ends."
    rows = [
        {"id": "1", "article": long_a, "highlights": "s"},
        {"id": "2", "article": "tiny article", "highlights": "s"},
        {"id": "3", "article": long_a, "highlights": "s"},
        {"id": "4", "article": long_a + " extra tail", "highlights": "s"},
        {"id": "5", "article": long_b, "highlights": "s"},
    ]
    df, audit = clean_and_filter(rows, 10, "x", cfg, set(), set())
    assert list(df["source_id"]) == ["1", "5"]
    assert audit["docs_dropped"] == {"too_short": 1, "exact_duplicate": 1, "near_duplicate": 1}
