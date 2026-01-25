# CouncilFlow RAG Architecture

This document describes how the CouncilFlow **Retrieval-Augmented Generation (RAG)** implementation works and how it selects documents from the knowledge base. It also compares this approach to **Google NotebookLM**.

---

## 1. Overview

CouncilFlow uses a **hybrid RAG** pipeline: **BM25** (lexical) + **embedding-based semantic search**, merged via **Reciprocal Rank Fusion (RRF)**. The knowledge base is split into **Core** (always fully loaded) and **Libraries** (chunked, indexed, and selectively retrieved at query time). An **LLM retrieval planner** (optional) chooses which libraries to search and how many chunks to fetch; **query expansion** (optional) and **LLM re-ranking** (optional) improve recall and precision. Planner, expansion, and re-ranking can be toggled in `rag_loader` to reduce API usage.

**Key modules:**

| Module | Role |
|--------|------|
| `librarian` | Google Drive: list Core + Libraries, fetch files, extract text, build Core XML |
| `rag` | Chunking, BM25 index, embeddings, hybrid retrieval, RRF, deduplication, XML output |
| `rag_loader` | Orchestration: load RAG state, plan retrieval, run single/multi-query retrieval, build context |
| `brain` | Embeddings (Gemini), retrieval planner, query expansion, re-ranking |
| `rag_cache` | Disk cache for library indexes (avoid re-embedding on restart) |

---

## 2. Knowledge Base Structure

The knowledge base is backed by a **Google Drive root folder**. Structure:

- **Core**: All **files in the root folder** (non-folder items). These are **never chunked**. Full document text is fetched, extracted, and wrapped in `<document title="..." link="...">...</document>`. Core is **always included** in context.
- **Libraries**: Each **subfolder** of the root is a **library**. Files inside (including nested subfolders) are recursively listed. Library contents are **chunked**, **embedded**, and **BM25-indexed**. Only **retrieved chunks** from selected libraries are added to context.

```
Root folder (Drive)
├── file1.pdf          → Core (full document)
├── file2.docx         → Core (full document)
├── Library A/         → Library "Library A"
│   ├── doc1.pdf
│   └── sub/doc2.docx
└── Library B/         → Library "Library B"
    └── ...
```

`librarian.list_core_and_libraries` discovers this layout; `rag_loader.get_cached_rag_state` builds Core XML and one index per library.

---

## 3. Index Build Flow (Libraries)

When the app loads the knowledge base (or refreshes it):

1. **List Core + Libraries**  
   `librarian.list_core_and_libraries(root_folder_id)` returns `core_files` and `libraries` (each with `id`, `name`, `files`).

2. **Build Core XML**  
   - Core files are passed to `librarian.build_context_xml` (which uses `fetch_and_extract`-style extraction).  
   - Each file → `<document title="..." link="...">extracted text</document>`.  
   - Wrapped in `<knowledge_base name="Core">...</knowledge_base>`.

3. **For each library** (unless a disk-cached index exists):
   - **Fetch & extract**: `librarian.fetch_and_extract_files(files)` → `[{name, link, id, text}, ...]`.
   - **Optional summarization**: `brain.summarize_files_batch` produces a short summary per file.
   - **Optional library description**: `brain.describe_library(library_name, file_descriptors)` produces a 1–2 sentence description of the library (used later by the retrieval planner).
   - **Chunk**: `rag.chunk_text(text)` splits each file’s text into overlapping chunks (~1800 chars, ~220 overlap). Splitting prefers paragraph → line → word boundaries.
   - **Embed**: `brain.embed_documents(chunk_texts)` returns 768‑dim vectors (Gemini `gemini-embedding-001`, `RETRIEVAL_DOCUMENT`). Vectors are L2‑normalized.
   - **BM25**: Chunk texts are tokenized (lowercase, `\w+`, optional Snowball stemmer) and passed to `rank_bm25.BM25Okapi`. The tokenized corpus is stored for BM25 scoring at query time.
   - **Library index** = `{chunks, bm25, tokenized_corpus, name, file_descriptors?, library_description?}`. Each chunk has `chunk_id`, `file_id`, `file_name`, `link`, `text`, `embedding`.
   - **Disk cache**: `rag_cache.save_library_index_to_disk` stores the index (versioned). On subsequent loads, `load_library_index_from_disk` restores it and rebuilds the BM25 object from `tokenized_corpus`.

**Chunking parameters** (`rag.py`):

- `CHUNK_MAX_CHARS = 1800`
- `CHUNK_OVERLAP = 220`

---

## 4. Document Selection Flow (Query Time)

When the user runs an analysis (e.g. “Run Analysis”):

### 4.1 Inputs

- **Task** (name + template/instructions)
- **User content** (uploaded docs, pasted text, etc.)
- **RAG state**: Core XML + list of `{id, name, index}` per library.

### 4.2 Step 1: Retrieval planning (optional)

**Goal:** Decide **which libraries** to search and **how many chunks** (`top_k`) to retrieve per library.

- When **`USE_RETRIEVAL_PLANNER`** is `True`, **`rag_loader.plan_retrieval`** calls **`brain.run_retrieval_planner`** with:
  - Task name, task description, user content.
  - **Library metadata**: for each library, `library_description` and `file_descriptors` (name + summary).
  - **Context budget**: model context limit, reserved tokens for user + prompt, and an approximate **chunk budget** for retrieved KB content. The planner is instructed to keep total chunks at or under this budget.

- The planner prompt (`RETRIEVAL_PLANNER_PROMPT`) asks the LLM to return JSON:

  `{"libraries": [{"name": "<library name>", "top_k": <number>}, ...]}`

- Rules include: `name` must match a library exactly; choose libraries relevant to the task and documents; set `top_k` between 1–100 (often 25–80 for important libraries); prefer 2–4 libraries when relevant; stay within the chunk budget.

- **Output**: `(selected_library_ids, top_k_per_library)`. If the planner fails, all libraries are selected with a default `top_k` (e.g. 35).
- When **`USE_RETRIEVAL_PLANNER`** is `False`, the planning step is skipped; **`get_default_plan`** selects all libraries with **`DEFAULT_TOP_K`** each. No planner UI or LLM call.

### 4.3 Step 2: Query expansion (optional)

**Goal:** Turn the task + user content into **multiple search phrases** to improve recall.

- When **`USE_QUERY_EXPANSION`** is `True`, **`brain.expand_queries`** is called with task name, template text, and user content. An LLM prompt asks for **3–5 short search phrases** (keywords, entities, legal/policy terms) in JSON: `{"phrases": ["...", ...]}`. **Output**: list of phrases (or fallback `[task_name + user_content excerpt]` on failure).
- When **`USE_QUERY_EXPANSION`** is `False`, **`get_fallback_phrases`** returns a single phrase (`[task_name + user_content excerpt]`). No LLM call.

### 4.4 Step 3: Hybrid retrieval (per library, per phrase)

**Goal:** For each **selected library** and each **search phrase**, run **hybrid retrieval** (BM25 + semantic), then merge results across phrases.

- **Single-query path** (`retrieve_and_build_context`): one effective query; `retrieve_hybrid` is called once per library.
- **Multi-query path** (`retrieve_and_build_context_multi`): multiple phrases from query expansion; for each library, `retrieve_hybrid` is run **once per phrase**, then results are merged with **RRF** (see below).

**`rag.retrieve_hybrid(query, library_index, top_k, embed_query_fn)`:**

1. **Tokenize query** for BM25 (same scheme as index: lowercase, `\w+`, optional stemming).
2. **Embed query** via `brain.embed_query` (Gemini `RETRIEVAL_QUERY`, 768‑dim, normalized).
3. **BM25**: Score all chunks in the library with `bm25.get_scores(query_tokens)`, then rank by decreasing score.
4. **Semantic**: Compute cosine similarity between query embedding and each chunk embedding; rank by decreasing similarity.
5. **RRF**: For each chunk, compute  
   `RRF = sum over both rankings of 1 / (60 + rank)`.  
   Sort chunks by RRF descending and take **top_k**.

So each chunk gets two rankings (BM25, semantic); RRF merges them without tuning a separate alpha.

### 4.5 Step 4: Merge (multi-query only), deduplicate, re-rank

- **Multi-query**: For each library, chunks from all phrase runs are aggregated. Each `(chunk_id, file_id)` receives an RRF score summed across phrases. Chunks are ordered by this score; we take `take = k * RERANK_FACTOR` (e.g. `2 * k`) before dedupe/re-rank.
- **Deduplication**: **`rag.dedupe_chunks`** keeps chunks in order and drops a chunk if it’s too similar to any **already selected** chunk. Similarity is **cosine** on embeddings (threshold 0.95), or Jaccard on tokenized text if no embeddings. This keeps diversity while preserving relevance order.
- **Re-ranking**: If **`RERANK_ENABLED`** and there are more chunks than `top_k`, **`brain.rerank_chunks_llm`** is used:
  - Chunks are batched (e.g. 15 per batch).
  - The LLM scores each passage 1–5 for relevance to the query.
  - Chunks are sorted by score descending and **top_k** are kept.

So we **over-fetch** (e.g. `retrieve_k = k * RERANK_FACTOR` or `k * len(queries)`), **dedupe**, then **re-rank** down to `top_k`.

### 4.6 Step 5: Build context XML

- **Core** is always first: `<knowledge_base name="Core">...</knowledge_base>`.
- For each selected library with at least one chunk after dedupe/re-rank, **`rag.build_retrieved_xml`** wraps chunks in:

  ```xml
  <retrieved_library name="...">
    <chunk source="..." file_link="...">chunk text</chunk>
    ...
  </retrieved_library>
  ```

- Final **context** = Core + concatenated `retrieved_library` sections. This is what gets cached (e.g. Gemini context cache) and passed to the main model along with the task template and user content.

### 4.7 Retrieval report

The retrieval layer also returns a **report** per library: `library_name`, `top_k`, `chunks_retrieved`, and `sources` (file name, link, chunk count). The UI uses this to show which files contributed chunks.

---

## 5. End-to-End Flow Summary

```
User runs analysis (task + user content)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. Plan retrieval                                                │
│    LLM → which libraries to search, top_k per library            │
│    (uses library descriptions, file summaries, context budget)   │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Query expansion                                               │
│    LLM → 3–5 search phrases from task + user content             │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. Hybrid retrieval (per library, per phrase)                    │
│    BM25 + semantic (embeddings) → RRF → top retrieve_k chunks    │
│    Multi-query: merge phrase results with RRF, take top 2k etc.  │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Deduplicate                                                   │
│    Drop chunks too similar (cosine ≥ 0.95) to already selected   │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. Re-rank (if enabled)                                          │
│    LLM scores chunks 1–5 → keep top_k                            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. Build context                                                 │
│    Core (full) + <retrieved_library> per library                 │
│    → Gemini cache → model prompt                                 │
└─────────────────────────────────────────────────────────────────┘
```

Steps 1 (plan) and 2 (expand) are optional: when `USE_RETRIEVAL_PLANNER` or `USE_QUERY_EXPANSION` is `False`, the corresponding step is skipped and no planner/expansion UI or LLM call is made. Step 5 (re-rank) is skipped when `RERANK_ENABLED` is `False`.

---

## 6. Configuration and Tunables

| Setting | Location | Default | Description |
|--------|----------|---------|-------------|
| `CHUNK_MAX_CHARS` | `rag` | 1800 | Max characters per chunk |
| `CHUNK_OVERLAP` | `rag` | 220 | Overlap between consecutive chunks |
| `DEFAULT_TOP_K` | `rag_loader` | 35 | Default chunks per library when planner falls back |
| `TOP_K_MIN` / `TOP_K_MAX` | `rag_loader` | 1–100 | Bounds for planner `top_k` |
| `USE_RETRIEVAL_PLANNER` | `rag_loader` | False | Use LLM to select libraries + top_k; if False, use all libraries with `DEFAULT_TOP_K` |
| `USE_QUERY_EXPANSION` | `rag_loader` | True | Use LLM to generate 3–5 search phrases; if False, single fallback phrase |
| `RERANK_ENABLED` | `rag_loader` | False | Use LLM re-ranking; when False, retrieve more chunks (`RETRIEVE_FACTOR`) instead |
| `RERANK_FACTOR` / `RETRIEVE_FACTOR` | `rag_loader` | 2 | Over-fetch factor before re-rank, or when re-rank disabled |
| `PLANNER_MODEL` / `GEMINI_PLANNER_MODEL` | `brain` | `gemini-2.0-flash` | Model used for retrieval planner (env override) |
| RRF `k` | `rag` | 60 | RRF denominator offset: `1 / (k + rank)` |
| Embedding model | `brain` | `gemini-embedding-001` | 768‑dim, document/query‑specific task types |

---

## 7. Comparison: CouncilFlow RAG vs. NotebookLM

NotebookLM is Google’s document-grounded AI product (Gemini + RAG). Below is a concise comparison.

### 7.1 High-level

| Aspect | CouncilFlow | NotebookLM |
|--------|-------------|------------|
| **Retrieval** | Hybrid (BM25 + semantic), RRF | Semantic only (embeddings + cosine similarity) |
| **Chunking** | Paragraph‑aware, ~1800 chars, configurable overlap | “Granular passages” (~hundreds of tokens); exact scheme not public |
| **Library selection** | LLM planner selects libraries + `top_k` per library | All user-added sources searched; no explicit library notion |
| **Query expansion** | LLM generates 3–5 search phrases; multi-query retrieval | Single query; no described multi-query expansion |
| **Re-ranking** | LLM re-ranks over-retrieved chunks (1–5 relevance) | Top‑k semantic only; no described LLM re-rank |
| **Context structure** | Core (full docs) always in context; libraries = retrieved chunks only | All context from retrieved passages |
| **Source layout** | Drive: Core (root files) + Libraries (subfolders) | User uploads (PDF, DOCX, slides, web, YouTube, etc.); flat “Sources” |
| **Caching** | Gemini context cache for built context; library indexes cached on disk | Managed by Google; no user-visible disk cache |
| **Citations** | Chunk‑level `source` / `file_link` in XML; UI report | Inline citations / footnotes to source passages |

### 7.2 Retrieval and ranking

- **CouncilFlow**: BM25 captures exact lexical matches (e.g. statute names, IDs); embeddings capture meaning. RRF combines both. Query expansion + multi-query improve recall; dedupe + LLM re-rank improve precision and diversity.
- **NotebookLM**: Relies on embedding similarity and top‑k retrieval. No described lexical component, query expansion, or LLM re-ranking. Simpler pipeline, less tunable.

### 7.3 Use case fit

- **CouncilFlow**: Designed for **structured knowledge bases** (Core + Libraries), **task-specific** retrieval (planner picks libraries and `top_k`), and **context budget awareness**. Suited to municipal/council workflows, legal/policy docs, and configurable retrieval.
- **NotebookLM**: Oriented to **personal or classroom notebooks**: upload diverse sources, chat, get cited answers. No explicit Core vs Libraries; all sources treated uniformly. Strong for study guides, Q&A, and audio overviews.

### 7.4 Summary

CouncilFlow’s RAG adds **hybrid retrieval**, **library-aware planning**, **query expansion**, and **LLM re-ranking** on top of a **Core + Libraries** layout. NotebookLM keeps a **semantic-only, single-query** retrieval model over a flat set of sources, with rich product features (e.g. citations, audio). CouncilFlow emphasizes **control, structure, and retrieval quality** for domain-specific workflows; NotebookLM emphasizes **ease of use** and **broad source types** in a general-purpose assistant.

---

## 8. References

- **Code**: `rag.py`, `rag_loader.py`, `brain.py`, `librarian.py`, `rag_cache.py`
- **Config**: `config.yaml` (auth); RAG tunables in `rag.py` / `rag_loader.py`
- **NotebookLM**: [Introducing NotebookLM](https://blog.google/technology/ai/notebooklm-google-ai), [Learn about NotebookLM](https://support.google.com/notebooklm/answer/16164461), and related Google AI posts.
