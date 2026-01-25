"""
RAG knowledge base loader: Core + library indexes.
Uses librarian (Drive, extract), rag (chunk, BM25, retrieve), brain (embeddings).
Library indexes are cached to disk to avoid re-embedding on restart.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from librarian import (
    build_context_xml,
    fetch_and_extract_files,
    list_core_and_libraries,
)
from brain import (
    chars_to_tokens,
    expand_queries,
    model_max_context,
    rerank_chunks_llm,
    run_retrieval_planner,
)
from brain import get_effective_model, get_planner_model
from rag import (
    CHUNK_MAX_CHARS,
    build_library_index,
    build_retrieved_xml,
    dedupe_chunks,
    retrieve_hybrid,
)
from rag_cache import (
    clear_disk_cache_for_folder,
    load_library_index_from_disk,
    save_library_index_to_disk,
)

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 25
TOP_K_MIN, TOP_K_MAX = 1, 100
RERANK_ENABLED = False  # Disabled: slow. We retrieve more chunks instead (RETRIEVE_FACTOR).
RERANK_FACTOR = 2
RETRIEVE_FACTOR = 2  # When rerank off: retrieve/take this many more chunks (we have context headroom).

# LLM-based query expansion and retrieval planner (set False to reduce Gemini calls).
USE_QUERY_EXPANSION = True
USE_RETRIEVAL_PLANNER = False


def _embed_fn(texts: list[str]):
    from brain import embed_documents
    return embed_documents(texts)


def _embed_query_fn(text: str):
    from brain import embed_query
    return embed_query(text)


def _summarize_fn(extracted: list[dict], progress_callback: callable | None = None):
    from brain import summarize_files_batch
    return summarize_files_batch(extracted, progress_callback=progress_callback)


def _describe_library_fn(library_name: str, file_descriptors: list[dict]):
    from brain import describe_library
    return describe_library(library_name, file_descriptors)


@st.cache_resource
def get_cached_rag_state(root_folder_id: str, _progress_callback: callable | None = None):
    """
    Load Core (root files, full context) + index each library (chunk, embed, BM25).
    Returns: {core_xml, libraries: [{id, name, folder_id, index}]}.
    """
    logger.info(f"Loading RAG state for root folder {root_folder_id}")
    data = list_core_and_libraries(root_folder_id)
    core_files = data["core_files"]
    libraries_meta = data["libraries"]

    # Core: always fully loaded
    def _core_progress(name: str, idx: int, total: int):
        if _progress_callback:
            try:
                _progress_callback("core", name, idx, total)
            except Exception:
                pass
        logger.debug(f"Core file {idx}/{total}: {name}")

    core_xml = ""
    if core_files:
        core_xml = build_context_xml(core_files, progress_callback=_core_progress)
        # Wrap in knowledge_base tag for clarity
        core_xml = f'<knowledge_base name="Core">\n{core_xml}\n</knowledge_base>'
    else:
        core_xml = '<knowledge_base name="Core">\n<document title="(none)" link="">No files in root folder.</document>\n</knowledge_base>'
    logger.info(f"Core context: {len(core_xml)} chars")

    # Index each library (use disk cache when available)
    indexes = []
    for lib in libraries_meta:
        lib_id = lib["id"]
        lib_name = lib["name"]
        files = lib["files"]
        if not files:
            logger.info(f"Library {lib_name}: no files, skip")
            continue

        idx = load_library_index_from_disk(root_folder_id, lib_id)
        if idx is not None:
            indexes.append({"id": lib_id, "name": lib_name, "folder_id": lib_id, "index": idx})
            continue

        def _lib_progress(name: str, idx: int, total: int):
            if _progress_callback:
                try:
                    _progress_callback("library", lib_name, name, idx, total)
                except Exception:
                    pass
            logger.debug(f"Library {lib_name} file {idx}/{total}: {name}")

        extracted = fetch_and_extract_files(files, progress_callback=_lib_progress)
        idx = build_library_index(
            lib_name,
            extracted,
            _embed_fn,
            progress_callback=_lib_progress,
            summarize_fn=_summarize_fn,
            describe_library_fn=_describe_library_fn,
        )
        save_library_index_to_disk(root_folder_id, lib_id, idx)
        indexes.append({
            "id": lib_id,
            "name": lib_name,
            "folder_id": lib_id,
            "index": idx,
        })
        logger.info(f"Indexed library {lib_name}: {len(idx['chunks'])} chunks")

    return {
        "core_xml": core_xml,
        "libraries": indexes,
    }


def plan_retrieval(
    rag_state: dict[str, Any],
    task_name: str,
    template_text: str,
    user_content: str,
) -> tuple[list[str], dict[str, int]]:
    """
    Use LLM to decide which libraries to search and top_k per library.
    Returns (selected_library_ids, top_k_per_library: lib_id -> top_k).
    Falls back to all libraries with DEFAULT_TOP_K each if planner fails.
    """
    libs = rag_state.get("libraries", [])
    if not libs:
        return [], {}

    name_to_lib = {L["name"]: L for L in libs}
    library_metadata: list[dict[str, Any]] = []
    for L in libs:
        idx = L.get("index") or {}
        library_metadata.append({
            "name": L["name"],
            "library_description": idx.get("library_description", ""),
            "file_descriptors": idx.get("file_descriptors", []),
        })

    # Context budget: reserve tokens for user content, prompt, output margin; use rest for KB chunks
    max_ctx = model_max_context(get_effective_model())
    user_tokens = chars_to_tokens(len(user_content or ""))
    prompt_tokens = chars_to_tokens(len(template_text or "") + 80)
    margin = int(0.10 * max_ctx)
    kb_budget_tokens = max(0, max_ctx - user_tokens - prompt_tokens - margin)
    tokens_per_chunk = chars_to_tokens(CHUNK_MAX_CHARS)
    budget_chunks = (kb_budget_tokens // tokens_per_chunk) if tokens_per_chunk else 500
    context_budget_section = (
        "Context budget:\n"
        f"- Model context limit: {max_ctx:,} tokens.\n"
        f"- Reserve {user_tokens:,} for user content, {prompt_tokens:,} for prompt.\n"
        f"- Use approximately {kb_budget_tokens:,} tokens for retrieved KB chunks (~{budget_chunks} chunks).\n"
        "- Select libraries and top_k so total retrieved chunks stay within this budget. Prefer staying under; do not exceed."
    )

    plan = run_retrieval_planner(
        task_name=task_name,
        task_description=template_text,
        user_content=user_content,
        library_metadata=library_metadata,
        context_budget_section=context_budget_section,
        model=get_planner_model(),
    )

    selected: list[str] = []
    top_k_map: dict[str, int] = {}

    if plan and isinstance(plan.get("libraries"), list):
        for item in plan["libraries"]:
            if not isinstance(item, dict):
                continue
            n = item.get("name")
            k = item.get("top_k", DEFAULT_TOP_K)
            if n not in name_to_lib:
                continue
            lib = name_to_lib[n]
            lid = lib["id"]
            try:
                k = max(TOP_K_MIN, min(TOP_K_MAX, int(k)))
            except (TypeError, ValueError):
                k = DEFAULT_TOP_K
            selected.append(lid)
            top_k_map[lid] = k

    if not selected:
        selected = [L["id"] for L in libs]
        top_k_map = {L["id"]: DEFAULT_TOP_K for L in libs}
        logger.info(f"Planner fallback: all {len(libs)} libraries, top_k={DEFAULT_TOP_K}")

    return selected, top_k_map


def get_fallback_phrases(task_name: str, template_text: str, user_content: str) -> list[str]:
    """Single search phrase (no LLM). Use when USE_QUERY_EXPANSION is False."""
    excerpt = (user_content or "")[:2000]
    return [f"{(task_name or '').strip()}\n\n{excerpt}".strip() or "search"]


def get_default_plan(rag_state: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    """All libraries, DEFAULT_TOP_K each. Use when USE_RETRIEVAL_PLANNER is False."""
    libs = rag_state.get("libraries", [])
    ids = [L["id"] for L in libs]
    top_k_map = {L["id"]: DEFAULT_TOP_K for L in libs}
    return ids, top_k_map


def retrieve_and_build_context(
    rag_state: dict[str, Any],
    query: str,
    selected_library_ids: list[str],
    top_k_per_library: int | dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Single-query retrieval: hybrid search over selected libraries, build context = Core + retrieved.
    The app currently uses retrieve_and_build_context_multi only; this is kept for optional use.

    selected_library_ids: list of library folder ids to search.
    top_k_per_library: int (same for all) or dict lib_id -> top_k.

    Returns (context_xml, retrieval_report).
    retrieval_report: list of {
        library_name, top_k, chunks_retrieved,
        sources: [{file_name, link, chunk_count}, ...]
    }.
    """
    core = rag_state["core_xml"]
    parts = [core]
    report: list[dict[str, Any]] = []
    use_map = isinstance(top_k_per_library, dict)
    for lib in rag_state["libraries"]:
        if lib["id"] not in selected_library_ids:
            continue
        name = lib["name"]
        idx = lib["index"]
        k = top_k_per_library.get(lib["id"], DEFAULT_TOP_K) if use_map else top_k_per_library
        retrieve_k = (k * RERANK_FACTOR) if RERANK_ENABLED else (k * RETRIEVE_FACTOR)
        chunks = retrieve_hybrid(query, idx, retrieve_k, _embed_query_fn)
        chunks = dedupe_chunks(chunks, similarity_threshold=0.95, use_embedding=True)
        if RERANK_ENABLED and len(chunks) > k:
            chunks = rerank_chunks_llm(query, chunks, k)
        if chunks:
            parts.append(build_retrieved_xml(name, chunks))
        # Build per-library report: aggregate chunks by (file_name, link)
        by_file: dict[tuple[str, str], int] = {}
        for c in chunks:
            fn = c.get("file_name") or "?"
            link = c.get("link") or ""
            key = (fn, link)
            by_file[key] = by_file.get(key, 0) + 1
        sources = [
            {"file_name": fn, "link": link, "chunk_count": cnt}
            for (fn, link), cnt in sorted(by_file.items(), key=lambda x: -x[1])
        ]
        report.append({
            "library_name": name,
            "top_k": k,
            "chunks_retrieved": len(chunks),
            "sources": sources,
        })
    return "\n\n".join(parts), report


_RRF_K = 60


def retrieve_and_build_context_multi(
    rag_state: dict[str, Any],
    queries: list[str],
    selected_library_ids: list[str],
    top_k_per_library: int | dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Multi-query retrieval: run hybrid retrieval per query, merge via RRF per library,
    dedupe, then build context = Core + retrieved.
    Embeds each phrase once and reuses across libraries to reduce API calls.
    """
    core = rag_state["core_xml"]
    parts = [core]
    report: list[dict[str, Any]] = []
    use_map = isinstance(top_k_per_library, dict)

    # Pre-embed all queries once and reuse across libraries (caching handled in embed_query)
    # Deduplicate queries within the batch to avoid redundant embedding calls
    query_to_embedding: dict[str, list[float]] = {}
    query_embeddings: list[list[float]] = []
    
    for q in queries:
        if q in query_to_embedding:
            # Reuse embedding for duplicate query in same batch
            query_embeddings.append(query_to_embedding[q])
            logger.debug(f"Reusing query embedding for duplicate query in batch: {q[:50]}...")
        else:
            # Embed query (will use cache if available)
            emb = _embed_query_fn(q)
            query_to_embedding[q] = emb
            query_embeddings.append(emb)
    
    unique_queries = len(query_to_embedding)
    if unique_queries < len(queries):
        logger.info(f"Pre-embedded {len(queries)} query phrase(s) ({unique_queries} unique, {len(queries) - unique_queries} duplicates), reusing across libraries")
    else:
        logger.info(f"Pre-embedded {len(queries)} query phrase(s), reusing across libraries")

    for lib in rag_state["libraries"]:
        if lib["id"] not in selected_library_ids:
            continue
        name = lib["name"]
        idx = lib["index"]
        k = top_k_per_library.get(lib["id"], DEFAULT_TOP_K) if use_map else top_k_per_library
        retrieve_k = max(k * 2, k * len(queries))
        if RERANK_ENABLED:
            retrieve_k = max(retrieve_k, k * RERANK_FACTOR)
        else:
            retrieve_k = max(retrieve_k, k * RETRIEVE_FACTOR)
        rrf_scores: dict[tuple[Any, ...], float] = {}
        chunk_ref: dict[tuple[Any, ...], dict[str, Any]] = {}

        for qi, q in enumerate(queries):
            q_emb = query_embeddings[qi] if qi < len(query_embeddings) else None
            run = retrieve_hybrid(
                q, idx, retrieve_k, _embed_query_fn,
                query_embedding=q_emb,
            )
            for rank, c in enumerate(run, 1):
                key = (c.get("chunk_id"), c.get("file_id"))
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
                chunk_ref[key] = c

        take = (k * RERANK_FACTOR) if RERANK_ENABLED else (k * RETRIEVE_FACTOR)
        ordered = sorted(chunk_ref.keys(), key=lambda x: -rrf_scores[x])[:take]
        chunks = [chunk_ref[key] for key in ordered]
        chunks = dedupe_chunks(chunks, similarity_threshold=0.95, use_embedding=True)
        if RERANK_ENABLED and len(chunks) > k:
            main_query = queries[0] if queries else ""
            chunks = rerank_chunks_llm(main_query, chunks, k)
        if chunks:
            parts.append(build_retrieved_xml(name, chunks))
        by_file: dict[tuple[str, str], int] = {}
        for c in chunks:
            fn = c.get("file_name") or "?"
            link = c.get("link") or ""
            by_file[(fn, link)] = by_file.get((fn, link), 0) + 1
        sources = [
            {"file_name": fn, "link": link, "chunk_count": cnt}
            for (fn, link), cnt in sorted(by_file.items(), key=lambda x: -x[1])
        ]
        report.append({
            "library_name": name,
            "top_k": k,
            "chunks_retrieved": len(chunks),
            "sources": sources,
        })
    return "\n\n".join(parts), report
