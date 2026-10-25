"""Streamlit UI: search with auto-suggestion, pagination, highlighted excerpts and LLM summaries.

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402
from streamlit_searchbox import st_searchbox  # noqa: E402

from src.config import load_config  # noqa: E402
from src.llm import LLMError  # noqa: E402
from src.search import Result, SearchEngine  # noqa: E402
from src.suggest import SuggestIndex, load_suggest_index  # noqa: E402
from src.summarize import Summarizer  # noqa: E402
from src.ui_helpers import clamp_page, md_escape, page_count, slice_page  # noqa: E402

METHOD_LABELS = {
    "hybrid": "Hybrid (BM25 + embeddings)",
    "bm25": "BM25 (keywords)",
    "tfidf": "TF-IDF (keywords)",
    "dense": "Dense (embeddings)",
}
LENGTH_LABELS = {
    "short": "Short (~50 words)",
    "medium": "Medium (~120 words)",
    "long": "Long (~250 words)",
    "custom": "Custom",
}

st.set_page_config(page_title="Document Search & Summarizer", page_icon="🔎", layout="wide")


# --- resources loaded once per server process ------------------------------------------------------
@st.cache_resource(show_spinner="Loading indexes and models (first run only)…")
def get_resources() -> tuple[SearchEngine, Summarizer, SuggestIndex]:
    cfg = load_config()
    engine = SearchEngine(cfg)
    _ = engine.indexes, engine.chunks, engine.embedder  # load everything now, not on first query
    return engine, Summarizer(cfg), load_suggest_index(cfg)


@st.cache_data(show_spinner=False, max_entries=64)
def run_search(query: str, method: str, top_n: int) -> list[Result]:
    engine, _, _ = get_resources()
    return engine.search(query, top_n=top_n, method=method, page=1, page_size=top_n)  # type: ignore[arg-type]


def suggestions(term: str) -> list[str]:
    """Searchbox callback: the typed text itself first (so Enter searches it), then completions."""
    if not term.strip():
        return []
    _, _, index = get_resources()
    cfg = load_config()
    found = index.suggest(term, int(cfg.get("suggest.limit")), st.session_state.get("recent", []))
    return [term] + [s for s in found if s.lower() != term.lower()]


def go_to(page: int) -> None:
    st.session_state.page = page


# --- page ------------------------------------------------------------------------------------------
cfg = load_config()
st.title("🔎 Document Search & Summarizer")
st.caption(
    "Hybrid keyword + embedding search over ~20,000 news articles, with LLM summaries "
    "that use only the retrieved articles."
)

try:
    engine, summarizer, _ = get_resources()
except FileNotFoundError as exc:
    st.error(
        "The data or indexes are missing. Run `python scripts/prepare_data.py` and then "
        f"`python scripts/build_index.py`, and reload this page.\n\nDetails: {exc}"
    )
    st.stop()
except Exception as exc:  # noqa: BLE001 - show a readable message instead of a stack trace
    st.error(f"Could not start the search engine: {exc}")
    st.stop()

with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Search method", list(METHOD_LABELS), format_func=METHOD_LABELS.get, index=0)
    top_n = st.slider("Top-N documents", 1, 50, int(cfg.get("search.top_n")))
    page_size = st.selectbox("Results per page", [3, 5, 10], index=1)
    st.divider()
    length_choice = st.radio("Summary length", list(LENGTH_LABELS), format_func=LENGTH_LABELS.get, index=1)
    lo, hi = int(cfg.get("summary.min_words")), int(cfg.get("summary.max_words"))
    if length_choice == "custom":
        length: str | int = st.slider("Words", lo, hi, 100, step=10)
    else:
        length = length_choice
    max_docs = int(cfg.get("app.max_summary_docs"))
    st.caption(
        f"LLM: {summarizer.llm_name}  \nSummaries cover the top {max_docs} results at most "
        "(a local CPU model needs about a minute per article)."
    )

st.session_state.setdefault("recent", [])
st.session_state.setdefault("page", 1)
query = st_searchbox(
    suggestions,
    key="searchbox",
    placeholder="Type to search, e.g. “tropical storm flooding”…",
    clear_on_submit=False,
    edit_after_submit="current",
    default_use_searchterm=False,
)

if query and query != st.session_state.get("last_query"):
    st.session_state.last_query = query
    st.session_state.page = 1
    if query not in st.session_state.recent:
        st.session_state.recent.append(query)
        st.session_state.recent = st.session_state.recent[-10:]

query = st.session_state.get("last_query")
if not query:
    st.info(
        "Start typing in the box above. Suggestions come from article titles and common phrases; "
        "press Enter to search."
    )
    st.stop()

try:
    results = run_search(query, method, top_n)
except Exception as exc:  # noqa: BLE001
    st.error(f"Search failed: {exc}")
    st.stop()
if not results:
    st.warning("No documents matched that query. Try different or fewer words.")
    st.stop()

# --- results with pagination -----------------------------------------------------------------------
pages = page_count(len(results), page_size)
page = clamp_page(st.session_state.page, len(results), page_size)
st.session_state.page = page
st.subheader(f"{len(results)} results for “{query}”")
for r in slice_page(results, page, page_size):
    with st.container(border=True):
        st.markdown(f"**{r.rank}. {md_escape(r.title)}**")
        st.caption(f"{r.doc_id} · {METHOD_LABELS[r.method]} · score {r.score:.4f}")
        for ex in r.excerpts:
            st.markdown("> " + md_escape(ex.highlighted).replace("\\*\\*", "**"))
        with st.expander("Read the full article"):
            st.write(md_escape(summarizer.store.text(r.doc_id)))

prev_col, mid_col, next_col = st.columns([1, 2, 1])
prev_col.button("← Previous", disabled=page <= 1, on_click=go_to, args=(page - 1,), use_container_width=True)
mid_col.markdown(
    f"<div style='text-align:center;padding-top:6px'>Page <b>{page}</b> of <b>{pages}</b></div>",
    unsafe_allow_html=True,
)
next_col.button("Next →", disabled=page >= pages, on_click=go_to, args=(page + 1,), use_container_width=True)

# --- summary ---------------------------------------------------------------------------------------
st.divider()
to_summarize = [r.doc_id for r in results[:max_docs]]
run_key = (query, method, top_n, str(length), tuple(to_summarize))
label = (
    f"{length} words" if isinstance(length, int) else f"{length}, ~{cfg.get('summary.lengths')[length]} words"
)
if st.button(f"✨ Summarize results ({label})", type="primary"):
    try:
        with st.spinner(
            f"Summarizing {len(to_summarize)} articles with {summarizer.llm_name}… "
            "this can take a minute or two on CPU."
        ):
            st.session_state.summary = (run_key, summarizer.summarize_documents(to_summarize, length))
    except LLMError as exc:
        st.session_state.summary = None
        st.error(f"Summarization failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        st.session_state.summary = None
        st.error(f"Something went wrong while summarizing: {exc}")

saved = st.session_state.get("summary")
if saved and saved[0] == run_key:
    s = saved[1]
    st.subheader("Summary")
    st.write(md_escape(s.text))
    st.caption(
        f"{s.word_count} words (target {s.target_words}) · {len(s.doc_ids)} articles · "
        f"took {s.seconds:.1f} s{' (from cache)' if s.cached else ''} · {s.provider}/{s.model}"
    )
