"""
Hybrid RAG: chunking, embeddings, BM25, and retrieval.
Uses brain.embed_documents / embed_query, rank_bm25, and numpy.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------------

CHUNK_MAX_CHARS = 1800
CHUNK_OVERLAP = 150


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks. Prefer paragraph boundaries."""
    if not text or not text.strip():
        return []
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    # Prefer splitting on double newline, then single newline, then space
    segs = re.split(r"\n\n+", text)
    current = ""
    for s in segs:
        s = s.strip()
        if not s:
            continue
        if len(current) + 1 + len(s) <= max_chars:
            current = f"{current}\n\n{s}".strip() if current else s
            continue
        if current:
            chunks.append(current)
        # s alone might exceed max_chars
        if len(s) <= max_chars:
            current = s
            continue
        # Split s by single newlines
        lines = s.split("\n")
        current = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if len(current) + 1 + len(line) <= max_chars:
                current = f"{current}\n{line}".strip() if current else line
                continue
            if current:
                chunks.append(current)
            if len(line) <= max_chars:
                current = line
                continue
            # Split by spaces
            words = line.split()
            current = ""
            for w in words:
                if len(current) + 1 + len(w) <= max_chars:
                    current = f"{current} {w}".strip() if current else w
                    continue
                if current:
                    chunks.append(current)
                # overlap: keep last few words
                overlap_words = current.split()[-overlap // 8:] if current else []
                current = " ".join(overlap_words + [w]) if overlap_words else w
        if current:
            chunks.append(current)
            current = ""

    if current:
        chunks.append(current)
    return chunks


def _tokenize_bm25(text: str) -> list[str]:
    """Simple tokenization for BM25: lowercase, alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())


# -----------------------------------------------------------------------------
# Library index (chunks + embeddings + BM25)
# -----------------------------------------------------------------------------


def build_library_index(
    library_name: str,
    extracted: list[dict[str, Any]],
    embed_fn: callable,
    progress_callback: callable | None = None,
    *,
    summarize_fn: callable | None = None,
    describe_library_fn: callable | None = None,
) -> dict[str, Any]:
    """
    Build index from extracted files: chunk, embed, BM25.
    Optionally build file_descriptors (name, id, link, summary) and library_description.

    extracted: list of {name, link, id, text} from fetch_and_extract_files.
    embed_fn: callable(list[str]) -> list[list[float]].
    summarize_fn: optional callable(files: list[{name,text}], progress_cb?) -> list[str].
    describe_library_fn: optional callable(lib_name, file_descriptors) -> str.

    Returns: {chunks, bm25, tokenized_corpus, name, file_descriptors?, library_description?}.
    """
    from rank_bm25 import BM25Okapi

    file_descriptors: list[dict[str, Any]] = []
    library_description: str = ""

    if summarize_fn and extracted:
        try:
            summaries = summarize_fn(extracted, progress_callback=None)
        except Exception as e:
            logger.warning(f"Library {library_name}: summarization failed: {e}")
            summaries = []
        if not summaries:
            summaries = ["(no summary)"] * len(extracted)
        for i, rec in enumerate(extracted):
            file_descriptors.append({
                "name": rec.get("name", "?"),
                "id": rec.get("id", ""),
                "link": rec.get("link", ""),
                "summary": summaries[i] if i < len(summaries) else "(no summary)",
            })
        if describe_library_fn and file_descriptors:
            try:
                library_description = describe_library_fn(library_name, file_descriptors)
            except Exception as e:
                logger.warning(f"Library {library_name}: describe_library failed: {e}")
                library_description = f"Library '{library_name}' with {len(file_descriptors)} files."

    all_chunks: list[dict[str, Any]] = []
    chunk_texts: list[str] = []
    chunk_id = 0
    for idx, rec in enumerate(extracted, 1):
        name = rec.get("name", "?")
        link = rec.get("link", "")
        fid = rec.get("id", "")
        text = rec.get("text", "")
        if progress_callback:
            try:
                progress_callback(name, idx, len(extracted))
            except Exception:
                pass
        for c in chunk_text(text):
            if not c.strip():
                continue
            all_chunks.append({
                "chunk_id": chunk_id,
                "file_id": fid,
                "file_name": name,
                "link": link,
                "text": c,
            })
            chunk_texts.append(c)
            chunk_id += 1

    if not chunk_texts:
        logger.warning(f"Library {library_name}: no chunks produced")
        out: dict[str, Any] = {
            "chunks": [],
            "bm25": None,
            "tokenized_corpus": [],
            "name": library_name,
        }
        if file_descriptors:
            out["file_descriptors"] = file_descriptors
            out["library_description"] = library_description or f"Library '{library_name}' (no chunks)."
        return out

    logger.info(f"Library {library_name}: {len(chunk_texts)} chunks, embedding...")
    embeddings = embed_fn(chunk_texts)
    for i, ch in enumerate(all_chunks):
        ch["embedding"] = embeddings[i] if i < len(embeddings) else []

    tokenized = [_tokenize_bm25(t) for t in chunk_texts]
    bm25 = BM25Okapi(tokenized)
    logger.info(f"Library {library_name}: BM25 index built")
    out = {
        "chunks": all_chunks,
        "bm25": bm25,
        "tokenized_corpus": tokenized,
        "name": library_name,
    }
    if file_descriptors:
        out["file_descriptors"] = file_descriptors
        out["library_description"] = library_description or f"Library '{library_name}' with {len(file_descriptors)} files."
    return out


def _cosine_sim(a: list[float], b: list[float]) -> float:
    ax = np.array(a, dtype=float)
    bx = np.array(b, dtype=float)
    n = np.linalg.norm(ax) * np.linalg.norm(bx)
    if n == 0:
        return 0.0
    return float(np.dot(ax, bx) / n)


def _rrf_score(ranks: list[int], k: int = 60) -> float:
    """Reciprocal rank fusion."""
    return sum(1.0 / (k + r) for r in ranks)


def retrieve_hybrid(
    query: str,
    library_index: dict[str, Any],
    top_k: int,
    embed_query_fn: callable,
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval: BM25 + semantic. Merge via RRF, return top_k chunks.
    embed_query_fn: callable(str) -> list[float].
    """
    chunks = library_index.get("chunks", [])
    bm25 = library_index.get("bm25")
    tokenized = library_index.get("tokenized_corpus", [])
    if not chunks:
        return []

    q_tok = _tokenize_bm25(query)
    q_emb = embed_query_fn(query)

    # BM25 scores
    if bm25 and tokenized and q_tok:
        bm25_scores = bm25.get_scores(q_tok)
        bm25_order = np.argsort(-bm25_scores)
    else:
        bm25_order = np.arange(len(chunks))

    # Semantic scores (cosine)
    sem_scores = np.array([_cosine_sim(q_emb, c["embedding"]) for c in chunks])
    sem_order = np.argsort(-sem_scores)

    # RRF: rank by BM25, rank by semantic; combine RRF
    rrf = np.zeros(len(chunks))
    for r, i in enumerate(bm25_order, 1):
        rrf[i] += 1.0 / (60 + r)
    for r, i in enumerate(sem_order, 1):
        rrf[i] += 1.0 / (60 + r)

    top_indices = np.argsort(-rrf)[:top_k]
    return [chunks[i] for i in top_indices]


def build_retrieved_xml(library_name: str, chunks: list[dict[str, Any]]) -> str:
    """Wrap retrieved chunks in <retrieved_library name="..."> with <chunk source="..." file="...">...</chunk>."""
    import html
    parts = [f'<retrieved_library name="{html.escape(library_name)}">']
    for c in chunks:
        name = html.escape(c.get("file_name", "?"))
        link = html.escape(c.get("link", ""))
        text = html.escape((c.get("text") or "").strip())
        parts.append(f'  <chunk source="{name}" file_link="{link}">')
        parts.append(f"    {text}")
        parts.append("  </chunk>")
    parts.append("</retrieved_library>")
    return "\n".join(parts)
