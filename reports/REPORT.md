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
