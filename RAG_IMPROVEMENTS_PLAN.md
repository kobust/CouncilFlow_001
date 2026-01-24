# RAG Improvements Implementation Plan

This document outlines the plan to implement retrieval and corpus-understanding improvements for CouncilFlow, in dependency order.

---

## Overview

| Phase | Scope | Cache impact | Dependencies |
|-------|--------|--------------|--------------|
| **1** | Quick wins: BM25 stemming, planner context budget, ordering/dedupe | Index invalidated (stemming) | — |
| **2** | Smarter chunking: increase overlap | Index invalidated | — |
| **3** | Query expansion: LLM-generated search phrases | None | Phase 1 done |
| **4** | Re-ranking: retrieve 2–3× top_k, rerank, keep top_k | None | Phase 1, 3 |

**Cache / deploy:** Phases 1 and 2 change the RAG index (tokenization, chunking). Bump `CACHE_VERSION` in `rag_cache.py` and have users **Refresh Knowledge Base** after deploy.

---

## Phase 1: Quick Wins

### 1.1 BM25 Stemming

**Goal:** Stem tokens for both index and query so variants like "meeting"/"meetings", "adopt"/"adopted" match better.

**Files:** `rag.py`, `rag_cache.py`, `requirements.txt`

**Tasks:**

1. **Add `nltk`** to `requirements.txt` (e.g. `nltk`).
2. **Stemmer setup:** Use `nltk.stem.SnowballStemmer('english')`. Ensure stemmer data is available (Snowball ships with nltk; no extra download typically needed). Add a one-time `nltk.download('punkt')` if you use nltk tokenization; we can keep regex `\w+` and only stem).
3. **Update `_tokenize_bm25` in `rag.py`:**
   - Keep `re.findall(r"\w+", text.lower())` for tokenization.
   - Apply stemmer to each token; return list of stems.
   - Use a module-level stemmer instance (lazy-init) to avoid repeated creation.
4. **BM25 usage:** No API change. `tokenized_corpus` and query tokens both go through `_tokenize_bm25`. Index build and `retrieve_hybrid` already use it.
5. **Bump `CACHE_VERSION`** in `rag_cache.py` (e.g. `2` → `3`). Existing `.pkl` caches will be ignored; indexes rebuild on next load (or after Refresh KB).

**Testing:** Unit test `_tokenize_bm25` with known strings (e.g. "meetings adopted" → stemmed forms). Spot-check that `build_library_index` and `retrieve_hybrid` still run.

---

### 1.2 Planner Context Budget

**Goal:** Pass an explicit context budget into the retrieval planner so it can choose `top_k` per library within the model’s context limit.

**Files:** `brain.py`, `rag_loader.py`

**Tasks:**

1. **Compute budget (tokens):**
   - `max_ctx = model_max_context(DEFAULT_MODEL)` (e.g. 1M).
   - Reserve for user content: `user_tokens = chars_to_tokens(len(user_content))`.
   - Reserve for prompt wrapper: `prompt_tokens = chars_to_tokens(len(template_text) + 80)` (match `app.py` prompt wrapper).
   - Reserve safety margin (e.g. 10% of `max_ctx`) for output and rounding.
   - **KB budget:** `kb_budget_tokens = max(0, max_ctx - user_tokens - prompt_tokens - margin)`.

2. **Translate to chunks:**
   - Approximate tokens per chunk: `CHUNK_MAX_CHARS / 4` (~450). Use `brain.chars_to_tokens(CHUNK_MAX_CHARS)` or a shared constant.
   - `budget_chunks = kb_budget_tokens // tokens_per_chunk`.

3. **Planner API:**
   - `plan_retrieval(rag_state, task_name, template_text, user_content)` already has what we need. Compute `budget_chunks` (and optionally `kb_budget_tokens`) inside `plan_retrieval` (using `brain.model_max_context`, `brain.chars_to_tokens`, and `rag.CHUNK_MAX_CHARS`). Avoid circular imports: `rag_loader` already imports `brain` and `rag`.
   - Add a short "Context budget" section to `RETRIEVAL_PLANNER_PROMPT` in `brain.py`:
     - "Model context limit: X tokens. Reserve Y for user content + prompt. Use approximately Z tokens for retrieved KB chunks (~W chunks). Select libraries and top_k so total retrieved chunks stay within this budget. Prefer staying under the budget; do not exceed it."
   - Pass `budget_chunks` (and optionally `kb_budget_tokens`) into the prompt via new placeholders.

4. **Implement in `rag_loader.plan_retrieval`:**
   - Compute `user_tokens`, `prompt_tokens`, `max_ctx`, `margin`, `kb_budget_tokens`, `budget_chunks`.
   - Call `_build_library_catalog` and build planner prompt as today; add `{{ context_budget }}` (or similar) section with the above text.
   - `run_retrieval_planner` currently takes `task_name`, `task_description`, `user_content`, `library_metadata`. Either:
     - **Option A:** Add optional `context_budget_section: str` to `run_retrieval_planner` and append it to the prompt, or
     - **Option B:** Add `context_budget_section` to a small "scratch" structure passed into the planner (e.g. extend prompt construction in `brain` to accept extra sections).
   - Simplest: **extend the prompt in `brain`** with a `{{ context_budget }}` placeholder and pass `context_budget` from `rag_loader` into `run_retrieval_planner` as an optional kwarg; the function builds the budget paragraph and injects it.

5. **Planner rules:** Keep existing rules (e.g. "total chunks 80–300"); add "never exceed the chunk budget" and "prioritize libraries that clearly matter for the task."

**Testing:** Run a few analyses with small vs large user content; confirm planner selects lower top_k when user content is large, and that context still fits.

---

### 1.3 Ordering / Deduplication

**Goal:** Deduplicate retrieved chunks (e.g. adjacent overlap or near-copies) and preserve a sensible order (relevance first) before building context.

**Files:** `rag.py` (or `rag_loader.py`), `rag_loader.py`

**Tasks:**

1. **Dedupe helper:**
   - Add `dedupe_chunks(chunks: list[dict], *, similarity_threshold: float = 0.95, use_embedding: bool = True) -> list[dict]`.
   - Iterate in order (RRF order). Keep a "selected" list. For each candidate chunk, if it’s too similar to any selected chunk, skip it; otherwise append to selected.
   - Similarity: use cosine similarity between `embedding` vectors if `use_embedding` else lexical (e.g. Jaccard on tokenized text). Threshold tuneable (e.g. 0.92–0.96).
   - Chunks already have `embedding` and `text`. Use `rag._cosine_sim` for embedding similarity; add a small `_jaccard_similarity(tokens_a, tokens_b)` if you support text-based dedupe.
   - Complexity: O(n × k) where k = len(selected). For 2–3× top_k later, n stays modest; optional cap (e.g. compare only to last 50 selected) if needed.

2. **Where to call:**
   - In `retrieve_and_build_context` (rag_loader): after `retrieve_hybrid` per library, run `dedupe_chunks` on the list of chunks for that library, then pass the deduped list into `build_retrieved_xml`. Keep order (already relevance-ordered).

3. **Scope:** Dedupe **per library** (each library’s chunks independently). No cross-library dedupe for now.

4. **Ordering:** Keep current RRF order. No extra sorting step unless we later add "sort by library then score"; for now, dedupe preserves order.

**Testing:** Unit test `dedupe_chunks` with a few hand-crafted chunks (duplicates, near-duplicates, distinct). Integration: run retrieve + build and ensure context XML chunk count sometimes drops when overlaps exist.

---

## Phase 2: Smarter Chunking (Increase Overlap)

**Goal:** Reduce “cut in the middle of a thought” by increasing overlap between consecutive chunks.

**Files:** `rag.py`, `rag_cache.py`

**Tasks:**

1. **Update `rag.py`:**
   - Change `CHUNK_OVERLAP` from `150` to `220` (or `250`). Tune later if needed.
   - Overlap is used in `chunk_text` when splitting by words (`overlap // 8` words carried over). No other code changes.

2. **Bump `CACHE_VERSION`** in `rag_cache.py` if not already bumped in Phase 1 (e.g. ensure we’re at `3`). Chunking changes index structure; caches must be rebuilt.

3. **Optional:** Add a brief comment in `rag.py` that overlap is deliberately larger to improve context continuity.

**Testing:** Unit test `chunk_text` on a long string; check that consecutive chunks overlap by roughly the intended amount. Rebuild a small library and confirm chunk count slightly changes vs old overlap.

---

## Phase 3: Query Expansion (LLM-Generated Search Phrases)

**Goal:** Generate 3–5 search phrases from task + user content, run retrieval for each, merge results (e.g. RRF), then apply existing dedupe and build context.

**Files:** `brain.py`, `rag_loader.py`, optionally `app.py`

**Tasks:**

1. **Query expansion API:**
   - Add `expand_queries(task_name: str, template_text: str, user_content: str, *, model: str | None = None) -> list[str]` in `brain.py`.
   - Single Gemini call, prompt along the lines of:
     - "Given the analysis task and user-provided content below, produce 3–5 short search phrases (each a few words) that would help retrieve relevant passages from a document corpus. Output valid JSON only: {\"phrases\": [\"...\", \"...\", ...]}."
   - Parse JSON, return `phrases` (or default to `[f"{task_name}\n\n{user_content[:2000]}"]` if parsing fails). Dedupe/shorten if needed.

2. **Retrieval pipeline:**
   - **Option A (recommended):** Keep `retrieve_and_build_context(rag_state, query, sel_ids, top_k_map)`. Add `retrieve_and_build_context_multi(rag_state, queries: list[str], sel_ids, top_k_map)` that:
     - For each library in `sel_ids`, runs `retrieve_hybrid` once **per query** with a larger `k` (e.g. `top_k * 2` or `top_k * len(queries)`), collects all chunk lists.
     - Merges per-library rankings via RRF (each chunk can appear from multiple queries; aggregate RRF scores across runs).
     - Takes top `top_k` per library from merged ranking, then **dedupe** (Phase 1), then `build_retrieved_xml`.
   - **Option B:** Reuse single-query flow but replace `query` with a "merged" pseudo-query. That’s harder to justify; multi-query + RRF is clearer.

3. **App flow:**
   - After `plan_retrieval`, call `expand_queries(selected.name, selected.template_text, user_content)`.
   - If we get 2+ phrases, call `retrieve_and_build_context_multi` with `queries`; otherwise keep current `retrieve_and_build_context` with single `query = f"{task_name}\n\n{user_content[:4000]}"`.
   - Alternatively, always use expansion: if expansion returns one phrase, multi-query reduces to single-query behavior.

4. **Place in pipeline:** Plan → **expand** → retrieve (multi-query) → dedupe → build context. Re-ranking (Phase 4) will sit between retrieve and build.

**Testing:** Unit test `expand_queries` (mock Gemini or fixture responses). Integration: run multi-query retrieval on a small corpus and compare context diversity vs single-query.

---

## Phase 4: Re-ranking (Retrieve 2–3×, Rerank, Keep top_k)

**Goal:** Retrieve more chunks (e.g. 2× or 3× `top_k` per library), re-rank them, keep the top `top_k`, then build context.

**Files:** `rag.py` or new `rerank.py`, `rag_loader.py`, `requirements.txt` (if adding deps)

**Tasks:**

1. **Retrieve more:**
   - In the retrieval path (single-query or multi-query), retrieve `retrieve_k = top_k * factor` per library (`factor` configurable, e.g. 2 or 3). Use `retrieve_hybrid(..., top_k=retrieve_k)`.

2. **Re-ranker:**
   - **Option A – LLM re-ranker (no new deps):**
     - Add `rerank_chunks_llm(query: str, chunks: list[dict], top_k: int, *, model: str | None = None) -> list[dict]`.
     - For each chunk, prompt: "Rate relevance of this passage to the query from 1 (irrelevant) to 5 (highly relevant). Query: ... Passage: ... Respond with a single digit."
     - Batch chunks (e.g. 10–20 per request) to limit calls. Sort by score descending, return top `top_k`. Handle parse failures (treat as 1).
   - **Option B – Cross-encoder (e.g. `sentence-transformers`):**
     - Add `rerank_chunks_cross_encoder(query, chunks, top_k, model_name="ms-marco-MiniLM-L-6-v2")`. Score pairs `(query, chunk["text"])`, sort, return top `top_k`. Requires `sentence-transformers` (and torch). Prefer as follow-up if you want to avoid new heavy deps initially.

3. **Integration:**
   - After retrieve (and multi-query merge if Phase 3) and **before** dedupe: run `rerank_chunks_*(query, chunks, top_k)` per library. Use `query = main query` (e.g. `f"{task_name}\n\n{user_content[:4000]}"`) or a representative expansion phrase.
   - Then dedupe (Phase 1) on the re-ranked list, then build context. Alternatively, dedupe first to reduce rerank cost, then rerank. Plan: **retrieve 2–3× → optional dedupe → rerank → keep top_k → build**. Dedupe before rerank keeps rerank cheaper.

4. **Config:**
   - Add `RERANK_FACTOR = 2` (or 3) and `RERANK_ENABLED = True` in `rag_loader` or a small config; allow disabling re-rank for faster runs.

**Testing:** Unit test re-ranker with fake chunks. Integration: run with `RERANK_FACTOR=2`, confirm context uses re-ranked top_k and retrieval reports reflect “retrieved X, reranked to Y”.

---

## Cache and Deployment

- **`CACHE_VERSION`:** Bump in Phase 1 (stemming) and Phase 2 (overlap). Single bump after both is fine.
- **Refresh KB:** Document that users should run **Refresh Knowledge Base** after deploying Phase 1 + 2. Phases 3 and 4 don’t change the index.
- **Backward compatibility:** Old `.pkl` files will be skipped when version changes; no migration.

---

## Testing Summary

| Phase | Unit | Integration |
|-------|------|-------------|
| 1.1 BM25 stemming | `_tokenize_bm25` | Index + retrieve on small corpus |
| 1.2 Planner budget | Budget computation | Plan with small/large user content |
| 1.3 Dedupe | `dedupe_chunks` | Retrieve + build, check chunk counts |
| 2 Chunking | `chunk_text` | Rebuild library, compare chunk stats |
| 3 Query expansion | `expand_queries` | Multi-query retrieve + build |
| 4 Re-ranking | `rerank_chunks_*` | Retrieve 2× → rerank → build |

---

## File Checklist

| File | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|------|---------|---------|---------|---------|
| `requirements.txt` | + nltk | — | — | ± sentence-transformers (optional) |
| `rag_cache.py` | CACHE_VERSION | — | — | — |
| `rag.py` | `_tokenize_bm25`, `dedupe_chunks` | `CHUNK_OVERLAP` | — | — or rerank helpers |
| `brain.py` | Planner prompt + budget | — | `expand_queries` | LLM rerank (if used) |
| `rag_loader.py` | `plan_retrieval` budget, dedupe in retrieve flow | — | `retrieve_and_build_context_multi`, app wiring | Retrieve factor, rerank wiring |
| `app.py` | — | — | Call `expand_queries`, multi-query path | Config, rerank on/off |

---

## Implementation Order

1. **Phase 1.1** – BM25 stemming + `CACHE_VERSION` bump.  
2. **Phase 1.2** – Planner context budget.  
3. **Phase 1.3** – Ordering/dedupe in `retrieve_and_build_context`.  
4. **Phase 2** – Increase `CHUNK_OVERLAP`; ensure `CACHE_VERSION` bumped.  
5. **Phase 3** – `expand_queries` + multi-query retrieve + merge + dedupe.  
6. **Phase 4** – Retrieve 2–3×, rerank, keep top_k, then build.

This order keeps dependencies correct and allows incremental validation after each phase.

---

## Implementation status

All phases have been implemented:

- **Phase 1.1:** BM25 stemming (`rag._tokenize_bm25` + `nltk`), `CACHE_VERSION` → 3.
- **Phase 1.2:** Planner context budget (`brain.RETRIEVAL_PLANNER_PROMPT`, `run_retrieval_planner` + `context_budget_section`, `plan_retrieval` budget computation).
- **Phase 1.3:** `dedupe_chunks` in `rag`, used in `retrieve_and_build_context` and `retrieve_and_build_context_multi`.
- **Phase 2:** `CHUNK_OVERLAP` → 220.
- **Phase 3:** `expand_queries` in `brain`, `retrieve_and_build_context_multi` in `rag_loader`; app always uses expansion + multi-query retrieval.
- **Phase 4:** `rerank_chunks_llm` in `brain`, `RERANK_ENABLED` / `RERANK_FACTOR` in `rag_loader`; retrieve 2×, dedupe, rerank, keep `top_k`.

After deploying, run **Refresh Knowledge Base** so indexes are rebuilt with the new tokenization and chunking.
