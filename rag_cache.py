"""
Disk cache for RAG library indexes (chunks + embeddings + tokenized corpus).
Avoids re-embedding on service restart. Clear via Refresh Knowledge Base.
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_VERSION = 2
_CACHE_DIR: Path | None = None


def _cache_dir() -> Path:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = Path(__file__).resolve().parent / ".rag_cache"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _safe_id(s: str) -> str:
    """Sanitize folder/library ID for use in filenames."""
    return re.sub(r"[^\w\-.]", "_", s)


def _library_cache_path(folder_id: str, library_id: str) -> Path:
    return _cache_dir() / f"{_safe_id(folder_id)}_{_safe_id(library_id)}.pkl"


def load_library_index_from_disk(folder_id: str, library_id: str) -> dict | None:
    """
    Load cached index from disk. Returns None if missing or invalid.
    Rebuilds BM25 from tokenized_corpus (not stored).
    """
    path = _library_cache_path(folder_id, library_id)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
            logger.warning(f"RAG cache outdated or invalid: {path.name}")
            return None
        from rank_bm25 import BM25Okapi
        tokenized = data.get("tokenized_corpus", [])
        if not tokenized:
            return None
        data["bm25"] = BM25Okapi(tokenized)
        data.setdefault("file_descriptors", [])
        data.setdefault("library_description", "")
        logger.info(f"Loaded library index from cache: {path.name} ({len(data['chunks'])} chunks, {len(data['file_descriptors'])} file descriptors)")
        return data
    except Exception as e:
        logger.warning(f"Failed to load RAG cache {path.name}: {e}")
        return None


def save_library_index_to_disk(folder_id: str, library_id: str, index: dict) -> None:
    """
    Save index to disk. Stores chunks (with embeddings), tokenized_corpus, name.
    BM25 is rebuilt on load from tokenized_corpus.
    """
    path = _library_cache_path(folder_id, library_id)
    try:
        out = {
            "version": CACHE_VERSION,
            "chunks": index.get("chunks", []),
            "tokenized_corpus": index.get("tokenized_corpus", []),
            "name": index.get("name", ""),
            "file_descriptors": index.get("file_descriptors", []),
            "library_description": index.get("library_description", ""),
        }
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_fd = len(out["file_descriptors"])
        logger.info(f"Saved library index to cache: {path.name} ({len(out['chunks'])} chunks, {n_fd} file descriptors)")
    except Exception as e:
        logger.warning(f"Failed to save RAG cache {path.name}: {e}")


def clear_disk_cache_for_folder(folder_id: str) -> int:
    """Delete all cache files for the given root folder. Returns count removed."""
    prefix = _safe_id(folder_id) + "_"
    cache = _cache_dir()
    removed = 0
    for p in cache.glob(f"{prefix}*.pkl"):
        try:
            p.unlink()
            removed += 1
            logger.info(f"Cleared RAG cache: {p.name}")
        except Exception as e:
            logger.warning(f"Failed to remove {p}: {e}")
    return removed
