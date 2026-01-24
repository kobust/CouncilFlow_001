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
from brain import run_retrieval_planner
from rag import build_library_index, build_retrieved_xml, retrieve_hybrid
from rag_cache import (
    clear_disk_cache_for_folder,
    load_library_index_from_disk,
    save_library_index_to_disk,
)

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 35
TOP_K_MIN, TOP_K_MAX = 1, 100


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

    plan = run_retrieval_planner(
        task_name=task_name,
        task_description=template_text,
        user_content=user_content,
        library_metadata=library_metadata,
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


def retrieve_and_build_context(
    rag_state: dict[str, Any],
    query: str,
    selected_library_ids: list[str],
    top_k_per_library: int | dict[str, int],
) -> tuple[str, list[dict[str, Any]]]:
    """
    Run hybrid retrieval over selected libraries, build context = Core + retrieved.
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
        chunks = retrieve_hybrid(query, idx, k, _embed_query_fn)
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
