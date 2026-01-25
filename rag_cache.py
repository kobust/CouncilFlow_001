"""
Disk cache for RAG library indexes (chunks + embeddings + tokenized corpus).
Avoids re-embedding on service restart. Clear via Refresh Knowledge Base.
Also caches query embeddings for faster repeated queries.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import re
from pathlib import Path

from paths import data_path

logger = logging.getLogger(__name__)

CACHE_VERSION = 3
QUERY_EMBED_CACHE_VERSION = 1
QUERY_EXPANSION_CACHE_VERSION = 1
RETRIEVAL_PLANNER_CACHE_VERSION = 1
FILE_SUMMARY_CACHE_VERSION = 1
LIBRARY_DESCRIPTION_CACHE_VERSION = 1
_CACHE_DIR: Path | None = None


def _cache_dir() -> Path:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        _CACHE_DIR = data_path(".rag_cache")
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


# Query embedding cache functions
def _query_embed_cache_path() -> Path:
    """Path to the query embedding cache file."""
    return _cache_dir() / "query_embeddings.pkl"


def load_query_embedding_cache() -> dict[str, list[float]]:
    """
    Load query embedding cache from disk.
    Returns dict mapping query_hash -> embedding vector.
    """
    path = _query_embed_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or data.get("version") != QUERY_EMBED_CACHE_VERSION:
            logger.debug("Query embedding cache outdated or invalid, starting fresh")
            return {}
        cache = data.get("embeddings", {})
        logger.info(f"Loaded {len(cache)} query embeddings from disk cache")
        return cache
    except Exception as e:
        logger.warning(f"Failed to load query embedding cache: {e}")
        return {}


def save_query_embedding_cache(cache: dict[str, list[float]]) -> None:
    """
    Save query embedding cache to disk.
    cache: dict mapping query_hash -> embedding vector.
    """
    path = _query_embed_cache_path()
    try:
        # Limit cache size to prevent unbounded growth (keep most recent 10,000 entries)
        max_size = 10000
        if len(cache) > max_size:
            # Keep most recent entries (dicts maintain insertion order in Python 3.7+)
            items = list(cache.items())
            cache = dict(items[-max_size:])
            logger.info(f"Query embedding cache trimmed to {max_size} entries")
        
        out = {
            "version": QUERY_EMBED_CACHE_VERSION,
            "embeddings": cache,
        }
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Saved {len(cache)} query embeddings to disk cache")
    except Exception as e:
        logger.warning(f"Failed to save query embedding cache: {e}")


def clear_query_embedding_cache() -> bool:
    """Clear the query embedding cache. Returns True if cache was cleared."""
    path = _query_embed_cache_path()
    if path.exists():
        try:
            path.unlink()
            logger.info("Cleared query embedding cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear query embedding cache: {e}")
            return False
    return False


def get_query_hash(query_text: str) -> str:
    """Get SHA256 hash of query text for use as cache key."""
    return hashlib.sha256(query_text.encode('utf-8')).hexdigest()


# File summary cache functions
def _file_summary_cache_path() -> Path:
    """Path to the file summary cache file."""
    return _cache_dir() / "file_summaries.pkl"


def load_file_summary_cache() -> dict[str, str]:
    """
    Load file summary cache from disk.
    Returns dict mapping file_content_hash -> summary.
    """
    path = _file_summary_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or data.get("version") != FILE_SUMMARY_CACHE_VERSION:
            logger.debug("File summary cache outdated or invalid, starting fresh")
            return {}
        cache = data.get("summaries", {})
        logger.info(f"Loaded {len(cache)} file summaries from disk cache")
        return cache
    except Exception as e:
        logger.warning(f"Failed to load file summary cache: {e}")
        return {}


def save_file_summary_cache(cache: dict[str, str]) -> None:
    """
    Save file summary cache to disk.
    cache: dict mapping file_content_hash -> summary.
    """
    path = _file_summary_cache_path()
    try:
        # Limit cache size to prevent unbounded growth (keep most recent 10,000 entries)
        max_size = 10000
        if len(cache) > max_size:
            items = list(cache.items())
            cache = dict(items[-max_size:])
            logger.info(f"File summary cache trimmed to {max_size} entries")
        
        out = {
            "version": FILE_SUMMARY_CACHE_VERSION,
            "summaries": cache,
        }
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Saved {len(cache)} file summaries to disk cache")
    except Exception as e:
        logger.warning(f"Failed to save file summary cache: {e}")


# Library description cache functions
def _library_description_cache_path() -> Path:
    """Path to the library description cache file."""
    return _cache_dir() / "library_descriptions.pkl"


def load_library_description_cache() -> dict[str, str]:
    """
    Load library description cache from disk.
    Returns dict mapping cache_key -> description.
    """
    path = _library_description_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or data.get("version") != LIBRARY_DESCRIPTION_CACHE_VERSION:
            logger.debug("Library description cache outdated or invalid, starting fresh")
            return {}
        cache = data.get("descriptions", {})
        logger.info(f"Loaded {len(cache)} library descriptions from disk cache")
        return cache
    except Exception as e:
        logger.warning(f"Failed to load library description cache: {e}")
        return {}


def save_library_description_cache(cache: dict[str, str]) -> None:
    """
    Save library description cache to disk.
    cache: dict mapping cache_key -> description.
    """
    path = _library_description_cache_path()
    try:
        # Limit cache size to prevent unbounded growth (keep most recent 1,000 entries)
        max_size = 1000
        if len(cache) > max_size:
            items = list(cache.items())
            cache = dict(items[-max_size:])
            logger.info(f"Library description cache trimmed to {max_size} entries")
        
        out = {
            "version": LIBRARY_DESCRIPTION_CACHE_VERSION,
            "descriptions": cache,
        }
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Saved {len(cache)} library descriptions to disk cache")
    except Exception as e:
        logger.warning(f"Failed to save library description cache: {e}")


# Query expansion cache functions
def _query_expansion_cache_path() -> Path:
    """Path to the query expansion cache file."""
    return _cache_dir() / "query_expansion.pkl"


def load_query_expansion_cache() -> dict[str, list[str]]:
    """
    Load query expansion cache from disk.
    Returns dict mapping cache_key -> list of expanded phrases.
    """
    path = _query_expansion_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or data.get("version") != QUERY_EXPANSION_CACHE_VERSION:
            logger.debug("Query expansion cache outdated or invalid, starting fresh")
            return {}
        cache = data.get("expansions", {})
        logger.info(f"Loaded {len(cache)} query expansion results from disk cache")
        return cache
    except Exception as e:
        logger.warning(f"Failed to load query expansion cache: {e}")
        return {}


def save_query_expansion_cache(cache: dict[str, list[str]]) -> None:
    """
    Save query expansion cache to disk.
    cache: dict mapping cache_key -> list of expanded phrases.
    """
    path = _query_expansion_cache_path()
    try:
        # Limit cache size to prevent unbounded growth (keep most recent 5,000 entries)
        max_size = 5000
        if len(cache) > max_size:
            items = list(cache.items())
            cache = dict(items[-max_size:])
            logger.info(f"Query expansion cache trimmed to {max_size} entries")
        
        out = {
            "version": QUERY_EXPANSION_CACHE_VERSION,
            "expansions": cache,
        }
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Saved {len(cache)} query expansion results to disk cache")
    except Exception as e:
        logger.warning(f"Failed to save query expansion cache: {e}")


def clear_query_expansion_cache() -> bool:
    """Clear the query expansion cache. Returns True if cache was cleared."""
    path = _query_expansion_cache_path()
    if path.exists():
        try:
            path.unlink()
            logger.info("Cleared query expansion cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear query expansion cache: {e}")
            return False
    return False


# Retrieval planner cache functions
def _retrieval_planner_cache_path() -> Path:
    """Path to the retrieval planner cache file."""
    return _cache_dir() / "retrieval_planner.pkl"


def load_retrieval_planner_cache() -> dict[str, dict[str, Any]]:
    """
    Load retrieval planner cache from disk.
    Returns dict mapping cache_key -> planner result dict.
    """
    path = _retrieval_planner_cache_path()
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or data.get("version") != RETRIEVAL_PLANNER_CACHE_VERSION:
            logger.debug("Retrieval planner cache outdated or invalid, starting fresh")
            return {}
        cache = data.get("plans", {})
        logger.info(f"Loaded {len(cache)} retrieval planner results from disk cache")
        return cache
    except Exception as e:
        logger.warning(f"Failed to load retrieval planner cache: {e}")
        return {}


def save_retrieval_planner_cache(cache: dict[str, dict[str, Any]]) -> None:
    """
    Save retrieval planner cache to disk.
    cache: dict mapping cache_key -> planner result dict.
    """
    path = _retrieval_planner_cache_path()
    try:
        # Limit cache size to prevent unbounded growth (keep most recent 2,000 entries)
        max_size = 2000
        if len(cache) > max_size:
            items = list(cache.items())
            cache = dict(items[-max_size:])
            logger.info(f"Retrieval planner cache trimmed to {max_size} entries")
        
        out = {
            "version": RETRIEVAL_PLANNER_CACHE_VERSION,
            "plans": cache,
        }
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Saved {len(cache)} retrieval planner results to disk cache")
    except Exception as e:
        logger.warning(f"Failed to save retrieval planner cache: {e}")


def clear_retrieval_planner_cache() -> bool:
    """Clear the retrieval planner cache. Returns True if cache was cleared."""
    path = _retrieval_planner_cache_path()
    if path.exists():
        try:
            path.unlink()
            logger.info("Cleared retrieval planner cache")
            return True
        except Exception as e:
            logger.warning(f"Failed to clear retrieval planner cache: {e}")
            return False
    return False
