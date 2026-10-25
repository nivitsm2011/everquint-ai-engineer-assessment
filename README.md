# Document Search and Summarization with LLMs

Hybrid search (BM25/TF-IDF + dense embeddings, fused with Reciprocal Rank Fusion) over a news corpus,
followed by length-controlled LLM summaries of the top results. Runs on a CPU laptop; all data, indexes
and models stay local and are git-ignored.

## Setup (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # add OPENAI_API_KEY only if using provider = openai
```

## LLM provider

Set `llm.provider` in `config/config.yaml` (or `LLM_PROVIDER` in `.env`):

- `ollama` (default, free/offline): `llama3.2:3b` via a local Ollama server (`ollama serve`).
- `openai`: `gpt-4o-mini`, key read from `OPENAI_API_KEY` in `.env`.

## Layout

`config/` settings, `src/` library, `scripts/` CLI entry points, `app/` Streamlit UI, `tests/` pytest,
`reports/` results and figures, `data/` and `indexes/` local only (git-ignored).

## Run

```powershell
python scripts/prepare_data.py            # download + clean + chunk (streams ~20k articles)
python scripts/build_index.py             # BM25, TF-IDF, FAISS (embedding takes ~80 min on CPU)
python scripts/query.py "storm flooding" --top_n 5 --method hybrid --summarize --length medium
python scripts/run_eval.py                # evaluation, writes reports/metrics_summary.json
python scripts/score_human_eval.py        # average the filled-in human evaluation sheet
streamlit run app/streamlit_app.py        # web UI (start `ollama serve` first for summaries)
```

## Status

Phases 0-6 complete; see `reports/REPORT.md` for results.

## Checks

```powershell
python -m pytest
python -m ruff check .
python -m black --check .
```
