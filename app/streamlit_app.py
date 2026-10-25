"""Streamlit UI: search box with suggestions, pagination, adjustable summary length."""

from __future__ import annotations

import streamlit as st

from src.config import load_config


def main() -> None:
    """Render the app shell (features arrive in the UI phase)."""
    cfg = load_config()
    st.title("Document Search & Summarizer")
    st.caption(f"LLM provider: {cfg.llm_provider}")


if __name__ == "__main__":
    main()
