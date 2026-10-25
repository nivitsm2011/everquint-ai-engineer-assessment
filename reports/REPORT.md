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
