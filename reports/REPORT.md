# Report

Sections for later phases (indexing, search, summarization, evaluation) are added as they are built.

## 1. Data

### 1.1 Corpus choice

`abisee/cnn_dailymail` (config `3.0.0`) is a sizable, real-world news corpus. Every article comes with
human-written bullet "highlights", which act as reference summaries and make ROUGE evaluation of the
summarizer possible without writing our own gold data.

- **Searchable corpus:** first 20,000 articles of the train split that survive cleaning
  (`corpus.train_docs`, configurable). The split is streamed, so only the parts of the file that are
  needed are downloaded.
- **Held-out pool:** 1,000 articles from the test split (`corpus.heldout_docs`), saved separately in
  `heldout.parquet` and de-duplicated against the corpus, so evaluation never uses texts the system has
  already indexed.
- **Caveat:** the train split is read in file order, so the slice is dominated by CNN-sourced articles.
  Daily Mail boilerplate patterns are implemented but matched nothing in this slice.

### 1.2 Cleaning (all counts from `reports/data_stats.json`, train corpus)

Each step is counted as the number of documents it changed; rows read: 21,490 -> kept 20,000.

| Step | Docs affected | Why (business reason) |
|---|---|---|
| Dateline removal: `LONDON, England (CNN) --`, partner `(Mashable) --`, `(Reuters) --` | ~21,000 | Appears in nearly every article and says nothing about content; left in, it would make every article match queries like "CNN" or "New York". |
| `E-mail to a friend` | 1,273 | Web-page navigation text, not news. |
| URLs / stray HTML / HTML entities | 273 / 1 / 16 | Not readable content; would pollute lexical matching and LLM input. |
| Unicode NFKC + whitespace | 225 / ~6,400 | Same character should be one token, so search is not split by typography. |
| Drop < 50 words | 8 | Too little text to search or summarize. |
| Drop exact duplicates (hash of normalised text) | 1,360 | Duplicates would fill the top-N with the same story. |
| Drop near duplicates (same first 40 normalised words) | 122 | Re-published versions of one story, same effect. |

Words in read rows: 12.8 M before cleaning (raw text of the 21,490 rows read) -> 11.9 M after cleaning
(kept documents only), so cleaning removed little content and mostly boilerplate and duplicates.

Original casing and punctuation are kept in `text` (what the LLM reads). `bm25_text` is lowercased,
tokenised and stopword-removed (scikit-learn English list), used only by BM25/TF-IDF.

### 1.3 Chunking

Articles are split into sentence-aware chunks of about 200 words with about 40 words of overlap, so
retrieval returns a focused excerpt and a sentence is never cut in half. The title is the article's first
sentence (<= 150 characters). Chunks keep `doc_id`, `chunk_id`, `title`, `position` and `n_chunks`.

### 1.4 Statistics

| | Corpus | Held-out |
|---|---|---|
| Documents | 20,000 | 1,000 |
| Words / doc (mean, median) | 593, 528 | 623, 565 |
| Chunks | 77,338 (3.87 per doc) | not chunked |
| Words / chunk (mean) | 192 | |
| Reference-summary words (mean) | 44.5 | 34.5 |

Largest chunk is 548 words: a few articles contain very long sentences that are never split.
Processed files total about 150 MB (git-ignored).

## 2. Indexing

Three indexes are built over the 77,338 chunks and persisted in `indexes/` (git-ignored). Re-running
`python scripts/build_index.py` loads existing files instead of rebuilding; `--rebuild` forces a fresh build.

| Index | Method | Build time | Size on disk |
|---|---|---|---|
| BM25 | `rank_bm25` BM25Okapi (k1=1.5, b=0.75) on stopword-free tokens | 15 s | 76.5 MB |
| TF-IDF | scikit-learn, unigrams, sublinear tf, min_df=2 (lexical baseline) | 12 s | 55.3 MB |
| Embeddings | `all-MiniLM-L6-v2`, 384-dim, L2-normalised, max 256 tokens | 4,659 s (77.7 min, CPU) | 118.8 MB |
| FAISS | `IndexFlatIP` (exact cosine) | 0.2 s | 118.8 MB |
| **Total** | | | **369.4 MB** |

Loading all indexes from disk takes 1.9 s. Embedding is the only expensive step and is a one-off; it is
checkpointed every 5,000 chunks so an interrupted run resumes.

### Scalability path

`index.faiss.type` in `config/config.yaml` switches the dense index without code changes:

- `flat` (default): exact search; fine at this size (77k vectors, a query scans them all in milliseconds).
- `ivf` (`IndexIVFFlat`): clusters vectors into `nlist` cells and visits `nprobe` per query; trades a little
  recall for speed once the corpus reaches millions of chunks (rule of thumb nlist ~ 4*sqrt(N)).
- `hnsw` (`IndexHNSWFlat`): graph index with high recall and fast queries, at the cost of more memory and
  slower builds.

All three are covered by unit tests (each finds a vector's own nearest neighbour). Their recall and speed
at this corpus size are not measured yet; that comparison belongs to the evaluation phase.

## 3. Search

`src/search.py` exposes `search(query, top_n, method, page, page_size)` (defaults from `config.yaml`:
5 documents, `hybrid`, page size 5) and a `SearchEngine` class that loads the indexes once.

- **Query handling:** the query goes through the same preprocessing as the corpus for the lexical path
  (lowercase, tokenise, drop stopwords); an all-stopword query falls back to its plain tokens. The dense
  path embeds the raw query with the same MiniLM model and compares against the FAISS index.
- **Methods:** `bm25`, `tfidf`, `dense`, and `hybrid` = weighted Reciprocal Rank Fusion of the BM25 and
  dense rankings, `score = sum_i w_i / (60 + rank_i)`, with `w` from `search.weights` (0.5 / 0.5). RRF uses
  ranks, not raw scores, so BM25 (unbounded) and cosine (0-1) need no calibration. Each retriever
  contributes its top 100 chunks (`search.candidate_pool`).
- **Document level:** chunk hits are grouped by document; a document's score is its best chunk's score
  and it is returned once. Hybrid scores are small (~0.016 at most with k=60) and are only comparable
  within one query; BM25, TF-IDF and cosine scores are on different scales too.
- **Excerpts:** each result carries up to 2 matching chunks, trimmed to the 60-word window with the most
  query-term hits, with matched words wrapped in `**` (whole-word, case-insensitive, plural-tolerant).
  The Streamlit UI can render this as bold.
- **Pagination:** `top_n` documents are retrieved, then `page`/`page_size` slice them; a page past the end
  returns `[]`.

Sample runs (3 queries x 4 methods, `top_n=3`) are saved in `reports/sample_queries.txt`. Observations
(qualitative only; quantitative comparison comes in the evaluation phase):

- Keyword-heavy query ("Obama health care reform"): all methods agree on the top articles; BM25 and TF-IDF
  favour long articles that repeat the terms, dense favours articles whose opening is about the topic.
- Paraphrase-style query ("tropical storm causes flooding and evacuations"): dense surfaces flood stories
  from Vietnam and the Philippines that BM25 misses, while BM25 surfaces a Florida storm story the dense
  model ranks lower; hybrid mixes both.
- Known weakness: a few CNN Student News items have a date line as their "title" (e.g. "October 29, 2009 .")
  because the title is the first sentence. This is a data-quality issue to revisit, not a search bug.

## 4. Summarization

`src/llm.py` hides the provider behind one interface (`LLMClient.generate`). `llm.provider` in
`config/config.yaml` (or `LLM_PROVIDER` in `.env`) picks `ollama` (default, `llama3.2:3b`, local, free) or
`openai` (`gpt-4o-mini`, key from `OPENAI_API_KEY`). Transient failures (timeouts, rate limits, 5xx) are
retried 3 times with exponential backoff (2 s, 4 s); a missing API key, an unreachable Ollama server or a
missing model fail immediately with a message that says how to fix it. Temperature is 0.2.

`src/summarize.py` exposes `summarize(docs: list[Result], length) -> Summary`, where `length` is `short`
(~50 words), `medium` (~120), `long` (~250) or an exact word count (20-500). It also provides
`Summarizer.summarize_document` / `summarize_text` for single-document summaries (used for ROUGE).

- **Prompting:** the system prompt tells the model to use only the supplied text, add no outside facts,
  and answer in one plain paragraph; the user prompt states the word target and an allowed range (+/-25%).
- **Length control:** small models miss word targets, so if the output is outside the range the request is
  repeated once with a note about the miss; the closest attempt is kept and, if still too long, trimmed at
  a sentence boundary. Too-short outputs cannot be fixed this way, so the achieved word count is always
  returned in `Summary.word_count`.
- **Map-reduce:** several documents are summarized one by one (~100 words each, "map"), then combined into
  one summary at the requested length ("reduce"). An article too long for the context window is split into
  sentence-aligned segments that are mapped and reduced the same way. Token use is estimated as
  1.4 tokens per word (conservative), keeping each call under 2,800 input tokens inside Ollama's 4,096
  context; as a last resort partial summaries are truncated at sentence boundaries.
- **Cache:** results are cached on disk (`data/cache/summaries`) keyed by document ids, word target,
  provider, model, temperature and prompt version, so repeating a query makes no LLM calls. Per-document
  map summaries are cached separately and reused across lengths.

Example (`reports/sample_summaries.txt`: top 3 hybrid results for "tropical storm causes flooding and
evacuations", llama3.2:3b on CPU):

| Length | Target | Actual words | LLM calls | Time |
|---|---|---|---|---|
| short | 50 | 48 | 4 (3 map + 1 reduce) | 108 s |
| medium | 120 | 117 | 1 (map summaries reused from cache) | 25 s |
| long | 250 | 232 | 2 (one length retry) | 53 s |

Quality is only eyeballed so far (the summaries read coherently and cover all three storms); faithfulness
and ROUGE are measured in the evaluation phase.

## 5. Evaluation

Run with `python scripts/run_eval.py` (stages `testset`, `search`, `summary`, `latency`; all resumable).
Raw outputs: `reports/testset.csv`, `search_metrics.csv`, `search_ranks.csv`, `summary_scores.csv`,
`summary_details.csv`, `human_eval_sheet.csv`, `latency.csv`, `metrics_summary.json`.

### 5.1 Test set

200 articles drawn at random (seed 42) from the indexed corpus. `llama3.2:3b` wrote one 3-8 word search
query per article, told to paraphrase; a query sharing 5 or more consecutive words with the article is
regenerated once. Example queries: "A380 maiden flight luxury features", "Argentine president defends grain
export tax hike". Limits: only the source article counts as relevant, so other articles on the same story
count as misses (accuracy is understated); and because an LLM read the article before writing the query,
keyword methods may be favoured. With 200 queries, differences of a few points are within noise
(standard error about 3 points on a 0.6 rate).

### 5.2 Search accuracy (`figures/search_metrics.png`)

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Precision@5 |
|---|---|---|---|---|---|---|
| BM25 | 0.580 | 0.870 | 0.925 | 0.704 | 0.758 | 0.174 |
| TF-IDF | 0.495 | 0.825 | 0.875 | 0.635 | 0.695 | 0.165 |
| Dense | 0.540 | 0.855 | 0.905 | 0.676 | 0.733 | 0.171 |
| **Hybrid (RRF)** | **0.595** | **0.875** | **0.965** | **0.725** | **0.784** | **0.175** |

Hybrid is best on every metric; its clearest gain is Recall@10 (+4 points over BM25, +6 over dense). BM25 is
a strong baseline here because the queries reuse article vocabulary. TF-IDF is weakest. Precision@5 is
capped at 0.2 because each query has one relevant document.

### 5.3 Summary quality

ROUGE F1 (stemmed) against the CNN/DailyMail highlights (mean 44.5 words). 40 test articles were
summarised at each length (`evaluation.summary_eval_docs`; 200 would take about 6 hours on this CPU).
Lead-3 is the first three sentences.

| System | Docs | ROUGE-1 | ROUGE-2 | ROUGE-L | Mean words |
|---|---|---|---|---|---|
| Lead-3 (same 40 docs) | 40 | 0.306 | 0.101 | 0.202 | 56.5 |
| Lead-3 (all 200 test docs) | 200 | 0.346 | 0.140 | 0.224 | 66.8 |
| LLM short (~50 words) | 40 | **0.371** | **0.132** | **0.252** | 49.3 |
| LLM medium (~120 words) | 40 | 0.297 | 0.107 | 0.190 | 123.5 |
| LLM long (~250 words) | 40 | 0.221 | 0.087 | 0.148 | 225.9 |

- The short LLM summary beats lead-3 on the same 40 articles (ROUGE-1 0.371 vs 0.306). Lead-3 scores higher
  on all 200 articles than on this subset, so the subset is a harder sample; the 40-document comparison is
  the fair one, and with n=40 the gap is suggestive, not conclusive.
- Longer summaries score lower because F1 penalises length when the reference is about 45 words: the
  extra words lower precision. ROUGE therefore says little about whether a long summary is better
  for a reader.
- Word targets are met on average (49 / 124 / 226 words for 50 / 120 / 250), with a range of 42-55,
  105-137 and 180-284 words.
- ROUGE rewards overlap with a human-written highlight, not factual faithfulness. Faithfulness is only
  assessed by the human sheet below.

### 5.4 Human evaluation

`reports/human_eval_sheet.csv` has **15 rows** (not the 30 in the brief; time constraint). Each row has a
test query, the top-3 hybrid results, their full text, the medium-length combined summary, and empty
columns to score 1-5 for coherence, coverage, faithfulness and conciseness, plus comments. The 15 rows are the
first 15 of a fixed 30-row pool (`human_eval_pool`); setting `human_eval_samples: 30` and rerunning adds the
rest from the cache. **These ratings have not been filled in yet**; once they are, run
`python scripts/score_human_eval.py` to get averages (`reports/human_eval_scores.json`).

### 5.5 Efficiency (CPU laptop, `latency.csv`)

| Component | n | Mean | p95 |
|---|---|---|---|
| Load indexes (BM25 + TF-IDF + FAISS) | 5 | 1.5 s | 1.5 s |
| Load embedding model | 1 | 0.4 s | n/a |
| Search: BM25 | 200 | 171 ms | 237 ms |
| Search: TF-IDF | 200 | 18 ms | 19 ms |
| Search: dense | 200 | 20 ms | 22 ms |
| Search: hybrid | 200 | 191 ms | 261 ms |
| Summarise 1 doc, short | 40 | 36 s | 58 s |
| Summarise 1 doc, medium | 40 | 41 s | 64 s |
| Summarise 1 doc, long | 40 | 52 s | 76 s |
| Summarise top-3 docs, medium (map-reduce) | 15 | 152 s | 213 s |

Search is interactive. BM25 (and so hybrid) is the slowest retriever because `rank_bm25` scores every chunk in
Python on each query; dense search over 77k vectors takes about 20 ms. Summarisation dominates
the cost: a local 3B model on CPU needs well over a minute for a combined summary, and the disk cache means a
repeated query is instant. Using the OpenAI provider would shorten summaries to seconds.

### 5.6 Limitations

- Single relevant document per query; LLM-written queries; one random sample of 200.
- ROUGE and the human sheet were run on 40 and 15 samples respectively, smaller than planned.
- Summary faithfulness is unmeasured until the human sheet is scored.

## 6. Interface (Streamlit)

Run `streamlit run app/streamlit_app.py` (needs the indexes; `ollama serve` for summaries).

- **Auto-suggestion:** the search box (`streamlit-searchbox`) offers completions as you type, from a prefix
  index (`src/suggest.py`, cached in `indexes/suggest.pkl`, built in about 13 s on first use) over
  about 38,000 phrases: 1-3 word n-grams that occur often in the corpus plus short article titles. Matches are
  ranked by corpus frequency, and the user's recent queries in the session come first. Typing "tropical st"
  offers "tropical storm", "tropical storm warning", "tropical storm force"...
- **Controls (sidebar):** search method (hybrid by default), top-N (1-50), results per page (3/5/10), and
  summary length: short / medium / long or a custom 20-500 word slider.
- **Results:** paginated cards (Previous / Next with a "Page x of y" indicator) showing title, document id,
  method, score, up to 2 excerpts with the query terms in bold, and an expander with the full article.
  Markdown and `$` signs in article text are escaped so money amounts do not render as maths.
- **Summarize results:** summarizes the top results (at most 5, `app.max_summary_docs`, because a local CPU
  model needs about a minute per article) at the chosen length and shows the words produced, the target, the
  articles used and the time taken. Errors (Ollama not running, missing indexes) appear as plain messages.
- **Loading:** indexes, chunk table and embedding model are loaded once with `st.cache_resource`; the first
  page load takes about 30-60 s on this laptop, later searches take well under a second.

Checked manually in a browser (screenshots in `reports/figures/`): `app_1_autosuggest.jpg` (suggestions),
`app_2_pagination_page1.jpg` and `app_3_pagination_page2.jpg` (3 per page, page 1 of 2 and 2 of 2, Next
disabled on the last page), `app_4_custom_length_summary.jpg` (custom 60-word target, 2 articles: 71 words
in 87 s). The Short/Medium/Long radio buttons use the same code path as the custom slider and were exercised in
Phase 4's CLI runs, but were not clicked through in the browser. Known cosmetic issue: after choosing a
suggestion, the box keeps the partially typed text while the results heading shows the chosen phrase.

## 7. Domain adaptation of the embedding model (optional improvement)

`python scripts/finetune_embeddings.py` (stages `pairs`, `train`, `index`, `compare`; resumable).

- **Training data:** 300 (query, passage) pairs from indexed articles that are **not** in the 200-document test
  set (the script asserts this and a unit test checks it). For each, `llama3.2:3b` writes a short search query from
  the first 150 words; the passage is the article's first chunk. Pairs are in `data/processed/finetune_pairs.csv`.
- **Training:** `all-MiniLM-L6-v2` + `MultipleNegativesRankingLoss` (the other passages in a batch of 32 act as
  negatives), 1 epoch = 9 steps, learning rate 2e-5, 10% warm-up. It took 41 s on CPU.
- **Re-indexing:** all 77,338 chunks were re-embedded with the new model into a separate FAISS index
  (`indexes/ft/`); this was the expensive step (about 36 min with the laptop otherwise idle). BM25 and TF-IDF are
  unchanged and shared. Setting `finetune.enabled: true` (or `EMBEDDING_VARIANT=finetuned`) makes search, the
  evaluation and the app use the fine-tuned model; the base index is untouched.

Search accuracy on the same 200 test queries (`reports/embedding_comparison.csv`):

| Method | Metric | Base | Fine-tuned | Change |
|---|---|---|---|---|
| Dense | Recall@1 | 0.540 | 0.575 | +0.035 |
| Dense | Recall@5 | 0.855 | 0.875 | +0.020 |
| Dense | Recall@10 | 0.905 | 0.920 | +0.015 |
| Dense | MRR@10 | 0.676 | 0.705 | +0.028 |
| Dense | nDCG@10 | 0.733 | 0.758 | +0.025 |
| Hybrid | Recall@1 | 0.595 | 0.645 | +0.050 |
| Hybrid | Recall@5 | 0.875 | 0.890 | +0.015 |
| Hybrid | Recall@10 | 0.965 | 0.970 | +0.005 |
| Hybrid | MRR@10 | 0.725 | 0.755 | +0.030 |
| Hybrid | nDCG@10 | 0.784 | 0.807 | +0.024 |
| BM25 (control) | all metrics | unchanged | unchanged | 0.000 |

- Every dense and hybrid metric moved up, and the BM25 control is identical, so the change comes from the
  embedding model alone. The largest gains are at rank 1 (hybrid Recall@1 0.595 -> 0.645) and in MRR (+0.03).
- **Caution:** with 200 queries each point of Recall@k is 0.5 percentage points and the standard error of a
  difference is a few points, so individual changes (especially Recall@10, +0.5 to +1.5 points) are within noise.
  The consistent direction across all metrics is the more convincing evidence, but a significance test or a larger
  test set would be needed to claim a firm improvement. The training queries come from the same LLM and prompt
  style as the test queries, which probably helps; real user queries may benefit less.
- Cost and benefit: 300 LLM-written queries took about 55 min and the re-embedding about 36 min, for roughly
  +3 points of MRR. The fine-tuned model is not enabled by default (`finetune.enabled: false`).

## 8. Quality

- **Tests:** `python -m pytest` runs 64 tests in about 30 s (no network, no LLM, no real indexes; the LLM is a fake
  client and search uses a tiny synthetic corpus with a bag-of-words embedder). Coverage of the required areas:

| Area | Tests |
|---|---|
| Cleaning (datelines, boilerplate, URLs/HTML, duplicates, short docs) | `tests/test_data_prep.py` |
| Chunking (sentence-aware, overlap, nothing lost) | `tests/test_data_prep.py` |
| BM25 / TF-IDF / dense / hybrid return ranked, one-per-document results | `tests/test_search.py` |
| RRF fusion maths and weights | `tests/test_search.py` |
| Pagination (API and UI helpers) | `tests/test_search.py`, `tests/test_ui_helpers.py` |
| Summary length control with a mocked LLM (retry, trim, map-reduce, cache) | `tests/test_summarize.py` |
| ROUGE, lead-3, retrieval metrics, human-sheet scoring | `tests/test_evaluation.py` |
| LLM clients (missing key, Ollama down, retries with backoff) | `tests/test_summarize.py` |
| FAISS flat/IVF/HNSW, embedding-variant switch, suggestions | `tests/test_indexing.py`, `tests/test_finetune.py`, `tests/test_suggest.py` |

- **Style:** `black` (line length 110) and `ruff` (rules E, F, W, I, B, UP and T20, which forbids `print`) both pass
  on the whole repository; settings are in `pyproject.toml`. Library code in `src/` logs through `logging`
  (`src/utils.get_logger`); command-line scripts write output with `sys.stdout`.
- **Dependencies:** every line in `requirements.txt` is pinned with `==`, and `pip check` reports no conflicts.
  Python 3.11 was used.
- Run the checks with `python -m ruff check .`, `python -m black --check .` and `python -m pytest`.
