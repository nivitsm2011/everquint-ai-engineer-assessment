# Document Search and Summarization with LLMs

Hybrid search (BM25/TF-IDF + dense embeddings, fused with RRF) over a text corpus, followed by
length-controlled LLM summaries of the top results. Runs on a CPU laptop; all data, indexes and
models stay local and are git-ignored.

## Setup (Windows / PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # add OPENAI_API_KEY only if using provider = openai
```

## LLM provider

Set `llm.provider` in `config/config.yaml` (or `LLM_PROVIDER` in `.env`):

- `ollama` (default, free/offline): `llama3.2:3b` via a local Ollama server.
- `openai`: `gpt-4o-mini`, key read from `OPENAI_API_KEY` in `.env`.

## Layout

`config/` settings · `src/` library · `scripts/` CLI entry points · `app/` Streamlit UI ·
`tests/` pytest · `reports/` results · `data/`, `indexes/` local only (git-ignored).

## Status

Phase 0 (scaffold) complete. Module bodies are stubs filled in by later phases.
