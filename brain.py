"""
Gemini-based agent with context caching.
Uses st.secrets['GEMINI_API_KEY'], jinja2 templates, and optional JSON parsing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import streamlit as st
from google import genai
from google.genai import errors
from google.genai import types
from jinja2 import Template

logger = logging.getLogger(__name__)


class CacheExpiredError(RuntimeError):
    """Raised when a cached content is expired or not found (403 PERMISSION_DENIED)."""
    pass

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------

DEFAULT_MODEL = "gemini-3-flash-preview"
CACHE_TTL_SECONDS = 3600  # 60 minutes

# Override via env to reduce quota issues (e.g. GEMINI_MODEL=gemini-2.0-flash).
# Database-stored model takes precedence over env var.
def _effective_model() -> str:
    # First check database (if available)
    try:
        import db
        config = db.get_app_config()
        if config and config.selected_model:
            logger.debug(f"Using model from database: {config.selected_model}")
            return config.selected_model
    except Exception as e:
        logger.debug(f"Could not get model from database (non-fatal): {e}")
    
    # Fallback to env var
    v = (os.environ.get("GEMINI_MODEL") or "").strip()
    if v:
        logger.debug(f"Using model from environment: {v}")
        return v
    
    # Final fallback to default
    logger.debug(f"Using default model: {DEFAULT_MODEL}")
    return DEFAULT_MODEL


# Initialize EFFECTIVE_MODEL at module load (will be refreshed on each call to get_effective_model)
def get_effective_model() -> str:
    """Get the effective model (checks database, then env, then default)."""
    return _effective_model()


EFFECTIVE_MODEL = get_effective_model()  # Initial value, but functions should call get_effective_model()


def _planner_model() -> str:
    """Model for retrieval planner. Checks database, then env, then default."""
    # First check database (if available)
    try:
        import db
        config = db.get_app_config()
        if config and config.planner_model:
            logger.debug(f"Using planner model from database: {config.planner_model}")
            return config.planner_model
    except Exception as e:
        logger.debug(f"Could not get planner model from database (non-fatal): {e}")
    
    # Fallback to env var
    v = (os.environ.get("GEMINI_PLANNER_MODEL") or "").strip()
    if v:
        logger.debug(f"Using planner model from environment: {v}")
        return v
    
    # Final fallback to default
    default_planner = "gemini-2.0-flash"
    logger.debug(f"Using default planner model: {default_planner}")
    return default_planner


def get_planner_model() -> str:
    """Get the planner model (checks database, then env, then default)."""
    return _planner_model()


PLANNER_MODEL = get_planner_model()  # Initial value, but functions should call get_planner_model()


def list_available_models() -> list[str]:
    """
    Query Gemini API for available models.
    Returns list of model names (e.g., ['gemini-3-flash-preview', 'gemini-2.0-flash', ...]).
    """
    try:
        models = []
        # Try using google.generativeai (old SDK) which has list_models()
        try:
            import google.generativeai as genai_old
            api_key = os.environ.get("GEMINI_API_KEY", "").strip()
            if not api_key:
                try:
                    api_key = st.secrets["GEMINI_API_KEY"]
                except Exception:
                    pass
            if api_key:
                genai_old.configure(api_key=api_key)
                for m in genai_old.list_models():
                    if "gemini" in m.name.lower() and "generateContent" in (m.supported_generation_methods if hasattr(m, 'supported_generation_methods') else []):
                        name = m.name
                        # Extract model name (e.g., "models/gemini-3-flash-preview" -> "gemini-3-flash-preview")
                        if '/' in name:
                            name = name.split('/')[-1]
                        models.append(name)
        except Exception as e1:
            logger.debug(f"Could not list models via google.generativeai: {e1}")
            # Try new SDK if available
            try:
                client = _client()
                # The new SDK might have a different API - check what's available
                if hasattr(client, 'models') and hasattr(client.models, 'list'):
                    for model in client.models.list():
                        name = getattr(model, 'name', str(model))
                        if 'gemini' in name.lower():
                            if '/' in name:
                                name = name.split('/')[-1]
                            models.append(name)
            except Exception as e2:
                logger.debug(f"Could not list models via new SDK: {e2}")
        
        # Remove duplicates and sort
        models = sorted(list(set(models)))
        if models:
            logger.info(f"Found {len(models)} available Gemini models")
            return models
        else:
            # Return common models as fallback if API call failed
            logger.warning("Could not fetch models from API, using fallback list")
            return [
                "gemini-3-flash-preview",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.0-flash",
                "gemini-1.5-pro",
                "gemini-1.5-flash",
            ]
    except Exception as e:
        logger.warning(f"Error listing available models: {e}")
        # Return common models as fallback
        return [
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]

# GenerateContent (Gemini) is used by: run_agent (1 per run, main prompt+context),
# run_retrieval_planner (1 when USE_RETRIEVAL_PLANNER), expand_queries (1 when USE_QUERY_EXPANSION),
# rerank_chunks_llm (batches when RERANK_ENABLED), summarize_files_batch + describe_library (index build only).
# Cache creation uses CachedContent API (caches.create), not GenerateContent.

# Optional delay (seconds) before generateContent to spread API burst; helps per-minute quotas.
def _pace_delay_seconds() -> int:
    try:
        v = os.environ.get("GEMINI_PACE_DELAY_SECONDS", "0") or "0"
        return max(0, int(v))
    except Exception:
        return 0


GEMINI_PACE_DELAY_SECONDS = _pace_delay_seconds()

# Rate limiting: track requests per minute to avoid 429 errors
# Gemini typically allows 15-60 requests per minute depending on model and tier
# IMPORTANT: Rate limits are PER API KEY, not per process!
# If running multiple instances (dev + prod), they share the same quota.
# Solution: Use different API keys or reduce GEMINI_RATE_LIMIT_RPM proportionally.
_last_request_times: list[float] = []
_RATE_LIMIT_REQUESTS_PER_MINUTE = 10  # Conservative default (can be overridden via GEMINI_RATE_LIMIT_RPM env var)
_RATE_LIMIT_WINDOW_SECONDS = 60
_rate_limit_warning_shown = False


def _apply_rate_limit() -> None:
    """
    Apply rate limiting before GenerateContent calls to avoid 429 errors.
    Tracks requests per minute and adds delays if needed.
    
    WARNING: This rate limiter is PER-PROCESS. If you're running multiple
    instances (e.g., local dev + production) with the same API key, they will
    share the same quota. You should either:
    1. Use different API keys for each instance, OR
    2. Set GEMINI_RATE_LIMIT_RPM to (total_limit / num_instances)
    """
    import time
    
    # Warn once about multi-instance usage
    global _rate_limit_warning_shown
    if not _rate_limit_warning_shown:
        num_instances = os.environ.get("COUNCILFLOW_INSTANCE_COUNT", "")
        if num_instances:
            try:
                count = int(num_instances)
                if count > 1:
                    logger.warning(
                        f"⚠️  MULTI-INSTANCE RATE LIMITING: {count} instances detected. "
                        f"Rate limits are PER API KEY, not per process. "
                        f"Consider setting GEMINI_RATE_LIMIT_RPM to (total_limit / {count}) "
                        f"or use different API keys for each instance."
                    )
            except ValueError:
                pass
        _rate_limit_warning_shown = True
    
    # Get configured rate limit (default conservative)
    # If multiple instances, this should be set to (total_limit / num_instances)
    rate_limit = int(os.environ.get("GEMINI_RATE_LIMIT_RPM", str(_RATE_LIMIT_REQUESTS_PER_MINUTE)) or str(_RATE_LIMIT_REQUESTS_PER_MINUTE))
    
    now = time.time()
    global _last_request_times
    
    # Remove requests older than 1 minute
    _last_request_times = [t for t in _last_request_times if now - t < _RATE_LIMIT_WINDOW_SECONDS]
    
    # If we're at the limit, wait until the oldest request is 1 minute old
    if len(_last_request_times) >= rate_limit:
        oldest_time = min(_last_request_times)
        wait_time = _RATE_LIMIT_WINDOW_SECONDS - (now - oldest_time) + 0.5  # Add 0.5s buffer
        if wait_time > 0:
            logger.debug(f"Rate limit: {len(_last_request_times)}/{rate_limit} requests in last minute, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
            # Clean up again after waiting
            now = time.time()
            _last_request_times = [t for t in _last_request_times if now - t < _RATE_LIMIT_WINDOW_SECONDS]
    
    # Apply pace delay if configured
    if GEMINI_PACE_DELAY_SECONDS > 0:
        logger.debug(f"Applying pace delay: {GEMINI_PACE_DELAY_SECONDS}s")
        time.sleep(GEMINI_PACE_DELAY_SECONDS)
    
    # Record this request
    _last_request_times.append(time.time())


# Token and context stats (for UI)
CHARS_PER_TOKEN = 4  # rough: ~4 chars per token for English
CHARS_PER_WORD = 5  # rough average including spaces/punctuation
WORDS_PER_PAGE = 300  # average book page density
WORDS_PER_MINUTE = 225  # average silent reading speed
HP_SERIES_PAGES = 4100  # all 7 Harry Potter books, rough combined total
HP_BOOK_PAGES = HP_SERIES_PAGES / 7  # average per book
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.0-flash-001": 1_048_576,
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-3-flash-preview": 1_048_576,
}
DEFAULT_MAX_CONTEXT = 1_048_576


def chars_to_tokens(chars: int) -> int:
    """Rough character-to-token estimate (English)."""
    return max(0, chars // CHARS_PER_TOKEN)


def model_max_context(model: str | None) -> int:
    """Max input context tokens for the given model. Uses DEFAULT_MAX_CONTEXT if unknown."""
    if not model:
        return DEFAULT_MAX_CONTEXT
    model_lower = model.lower().strip()
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if key in model_lower or model_lower in key:
            return limit
    return DEFAULT_MAX_CONTEXT


def format_reading_equivalent(tokens: int) -> str:
    """Return a combined reading equivalent string for UI."""
    if tokens <= 0:
        return "0× Harry Potter book · 0 pages · 0 hours reading"
    # Convert tokens -> words -> pages/hours
    words = tokens * (CHARS_PER_TOKEN / CHARS_PER_WORD)
    pages = words / WORDS_PER_PAGE
    hours = words / (WORDS_PER_MINUTE * 60)
    books = pages / HP_BOOK_PAGES if HP_BOOK_PAGES else 0
    if books >= 1:
        series_str = f"~{books:.1f}× Harry Potter Books"
    else:
        series_str = f"~{books:.2f}× Harry Potter Books"
    return f"{series_str} · ~{pages:,.0f} pages · ~{hours:,.1f} hours reading"


def format_context_usage(tokens: int, max_tokens: int, model: str | None = None) -> str:
    """e.g. '2.3% of model context (24k / 1M)'."""
    max_ctx = model_max_context(model) if max_tokens <= 0 else max_tokens
    if max_ctx <= 0:
        return f"{tokens:,} tokens"
    pct = (tokens / max_ctx) * 100
    return f"{pct:.2f}% of context window ({tokens:,} / {max_ctx:,} tokens)"


def _client() -> genai.Client:
    """Gemini client configured from st.secrets."""
    try:
        logger.debug("Creating Gemini client")
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            api_key = st.secrets["GEMINI_API_KEY"]
        if not api_key or not api_key.strip():
            logger.error("GEMINI_API_KEY is empty or missing")
            raise ValueError("GEMINI_API_KEY is required")
        client = genai.Client(api_key=api_key)
        logger.debug("Gemini client created successfully")
        return client
    except KeyError:
        logger.error("GEMINI_API_KEY not found in st.secrets")
        raise
    except Exception as e:
        logger.error(f"Failed to create Gemini client: {e}", exc_info=True)
        raise


EMBED_MODEL = "gemini-embedding-001"
EMBED_OUTPUT_DIM = 768
EMBED_MAX_RETRIES = 8
EMBED_BATCH_SIZE = 100  # API limit: at most 100 requests per batch
EMBED_INTER_BATCH_DELAY = 0.25


def _parse_429_retry_delay(e: Exception) -> float | None:
    """Extract API-suggested retry delay (seconds) from 429 error, or None."""
    error_str = str(e).lower()
    if "retry in" not in error_str and "retrydelay" not in error_str:
        return None
    try:
        delay_match = re.search(r"retry\s+in\s+([\d.]+)\s*s", error_str, re.IGNORECASE)
        if delay_match:
            return float(delay_match.group(1))
        if hasattr(e, "response") and hasattr(e.response, "json"):
            err = e.response.json().get("error", {})
            for d in err.get("details", []):
                if d.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                    s = d.get("retryDelay", "")
                    if "s" in s:
                        return float(re.sub(r"[^\d.]", "", s) or "0")
    except Exception:
        pass
    return None


def embed_documents(texts: list[str], batch_size: int = EMBED_BATCH_SIZE) -> list[list[float]]:
    """
    Embed texts for retrieval (documents). Uses RETRIEVAL_DOCUMENT task type.
    Returns list of vectors; each vector is normalized when using reduced dims.
    Retries on 429 RESOURCE_EXHAUSTED with API-suggested delay.
    """
    import math
    out: list[list[float]] = []
    client = _client()
    config = types.EmbedContentConfig(
        task_type="RETRIEVAL_DOCUMENT",
        output_dimensionality=EMBED_OUTPUT_DIM,
    )
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        delay = 2.0
        for attempt in range(EMBED_MAX_RETRIES + 1):
            try:
                r = client.models.embed_content(model=EMBED_MODEL, contents=batch, config=config)
                break
            except errors.ClientError as e:
                sc = getattr(e, "status_code", None) or (e.args[0] if e.args else None)
                if sc != 429 and "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e):
                    raise
                retry_sec = _parse_429_retry_delay(e) or delay
                retry_sec = max(retry_sec + 1.0, delay)
                if attempt < EMBED_MAX_RETRIES:
                    logger.warning(
                        f"Embed 429 on batch {i // batch_size + 1}, attempt {attempt + 1}/{EMBED_MAX_RETRIES + 1}. "
                        f"Retrying in {retry_sec:.1f}s..."
                    )
                    time.sleep(retry_sec)
                    delay = retry_sec * 1.5
                else:
                    logger.error(f"Embed quota exceeded after {EMBED_MAX_RETRIES + 1} attempts: {e}")
                    raise RuntimeError(
                        f"Embedding quota exceeded (429). Retry in ~{int(retry_sec)}s or reduce library size. {e}"
                    ) from e
        for e in r.embeddings:
            v = list(e.values)
            norm = math.sqrt(sum(x * x for x in v))
            if norm > 0:
                v = [x / norm for x in v]
            out.append(v)
        if i + batch_size < len(texts):
            time.sleep(EMBED_INTER_BATCH_DELAY)
    return out


# Query embedding cache (module-level fallback if Streamlit not available)
_query_embedding_cache: dict[str, list[float]] = {}
_disk_cache_loaded = False

# Query expansion cache (task + content -> expanded phrases)
_query_expansion_cache: dict[str, list[str]] = {}
_query_expansion_disk_loaded = False

# Retrieval planner cache (task + content + libraries -> plan)
_retrieval_planner_cache: dict[str, dict[str, Any]] = {}
_retrieval_planner_disk_loaded = False

# File summary cache (file content hash -> summary)
_file_summary_cache: dict[str, str] = {}
_file_summary_disk_loaded = False

# Library description cache (library name + file descriptors hash -> description)
_library_description_cache: dict[str, str] = {}
_library_description_disk_loaded = False


def _load_disk_cache_if_needed() -> None:
    """Lazy-load disk cache on first use."""
    global _query_embedding_cache, _disk_cache_loaded
    if _disk_cache_loaded:
        return
    try:
        from rag_cache import load_query_embedding_cache, get_query_hash
        disk_cache = load_query_embedding_cache()
        # Merge disk cache into module cache (disk cache takes precedence for consistency)
        _query_embedding_cache.update(disk_cache)
        _disk_cache_loaded = True
        logger.debug(f"Loaded {len(disk_cache)} query embeddings from disk cache")
    except Exception as e:
        logger.debug(f"Could not load disk cache (non-fatal): {e}")
        _disk_cache_loaded = True  # Mark as loaded to avoid repeated attempts


def _load_all_disk_caches() -> None:
    """Pre-load all disk caches on startup for faster first use."""
    global _query_embedding_cache, _disk_cache_loaded
    global _query_expansion_cache, _query_expansion_disk_loaded
    global _retrieval_planner_cache, _retrieval_planner_disk_loaded
    global _file_summary_cache, _file_summary_disk_loaded
    global _library_description_cache, _library_description_disk_loaded
    
    # Load query embedding cache
    if not _disk_cache_loaded:
        _load_disk_cache_if_needed()
    
    # Load query expansion cache
    if not _query_expansion_disk_loaded:
        try:
            from rag_cache import load_query_expansion_cache
            disk_cache = load_query_expansion_cache()
            _query_expansion_cache.update(disk_cache)
            _query_expansion_disk_loaded = True
            logger.debug(f"Pre-loaded {len(disk_cache)} query expansion results from disk cache")
        except Exception as e:
            logger.debug(f"Could not pre-load query expansion cache (non-fatal): {e}")
            _query_expansion_disk_loaded = True
    
    # Load retrieval planner cache
    if not _retrieval_planner_disk_loaded:
        try:
            from rag_cache import load_retrieval_planner_cache
            disk_cache = load_retrieval_planner_cache()
            _retrieval_planner_cache.update(disk_cache)
            _retrieval_planner_disk_loaded = True
            logger.debug(f"Pre-loaded {len(disk_cache)} retrieval planner results from disk cache")
        except Exception as e:
            logger.debug(f"Could not pre-load retrieval planner cache (non-fatal): {e}")
            _retrieval_planner_disk_loaded = True
    
    # Load file summary cache
    if not _file_summary_disk_loaded:
        try:
            from rag_cache import load_file_summary_cache
            disk_cache = load_file_summary_cache()
            _file_summary_cache.update(disk_cache)
            _file_summary_disk_loaded = True
            logger.debug(f"Pre-loaded {len(disk_cache)} file summaries from disk cache")
        except Exception as e:
            logger.debug(f"Could not pre-load file summary cache (non-fatal): {e}")
            _file_summary_disk_loaded = True
    
    # Load library description cache
    if not _library_description_disk_loaded:
        try:
            from rag_cache import load_library_description_cache
            disk_cache = load_library_description_cache()
            _library_description_cache.update(disk_cache)
            _library_description_disk_loaded = True
            logger.debug(f"Pre-loaded {len(disk_cache)} library descriptions from disk cache")
        except Exception as e:
            logger.debug(f"Could not pre-load library description cache (non-fatal): {e}")
            _library_description_disk_loaded = True


def embed_query(text: str) -> list[float]:
    """
    Embed a single query for retrieval. Uses RETRIEVAL_QUERY task type. Retries on 429.
    
    Multi-level caching strategy (fastest to slowest):
    1. Streamlit session state (fastest, in-memory for current session)
    2. Module-level in-memory cache (persists during process lifetime)
    3. Disk cache (persists across restarts)
    4. API call (if not cached)
    
    Caching improves speed and reduces API costs without affecting quality.
    """
    import hashlib
    import math
    
    # Get query hash for cache key
    query_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    # Level 1: Check Streamlit session state (fastest, session-scoped)
    try:
        if "query_embedding_cache" not in st.session_state:
            st.session_state["query_embedding_cache"] = {}
        session_cache = st.session_state["query_embedding_cache"]
        if query_hash in session_cache:
            logger.debug(f"Query embedding session cache hit: {text[:50]}...")
            return session_cache[query_hash]
    except Exception:
        pass  # Streamlit not available, continue to next level
    
    # Level 2: Check module-level in-memory cache
    if query_hash in _query_embedding_cache:
        logger.debug(f"Query embedding memory cache hit: {text[:50]}...")
        # Also store in session cache if available
        try:
            if "query_embedding_cache" not in st.session_state:
                st.session_state["query_embedding_cache"] = {}
            st.session_state["query_embedding_cache"][query_hash] = _query_embedding_cache[query_hash]
        except Exception:
            pass
        return _query_embedding_cache[query_hash]
    
    # Level 3: Load disk cache if not already loaded
    _load_disk_cache_if_needed()
    if query_hash in _query_embedding_cache:
        logger.debug(f"Query embedding disk cache hit: {text[:50]}...")
        # Also store in session cache if available
        try:
            if "query_embedding_cache" not in st.session_state:
                st.session_state["query_embedding_cache"] = {}
            st.session_state["query_embedding_cache"][query_hash] = _query_embedding_cache[query_hash]
        except Exception:
            pass
        return _query_embedding_cache[query_hash]
    
    # Level 4: API call (cache miss)
    logger.debug(f"Query embedding cache miss, calling API: {text[:50]}...")
    client = _client()
    config = types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY",
        output_dimensionality=EMBED_OUTPUT_DIM,
    )
    delay = 2.0
    for attempt in range(EMBED_MAX_RETRIES + 1):
        try:
            r = client.models.embed_content(model=EMBED_MODEL, contents=text, config=config)
            break
        except errors.ClientError as e:
            sc = getattr(e, "status_code", None) or (e.args[0] if e.args else None)
            if sc != 429 and "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e):
                raise
            retry_sec = _parse_429_retry_delay(e) or delay
            retry_sec = max(retry_sec + 1.0, delay)
            if attempt < EMBED_MAX_RETRIES:
                logger.warning(
                    f"Embed query 429, attempt {attempt + 1}/{EMBED_MAX_RETRIES + 1}. Retrying in {retry_sec:.1f}s..."
                )
                time.sleep(retry_sec)
                delay = retry_sec * 1.5
            else:
                logger.error(f"Embed query quota exceeded after {EMBED_MAX_RETRIES + 1} attempts: {e}")
                raise RuntimeError(
                    f"Embedding quota exceeded (429). Retry in ~{int(retry_sec)}s. {e}"
                ) from e
    v = list(r.embeddings[0].values)
    norm = math.sqrt(sum(x * x for x in v))
    if norm > 0:
        v = [x / norm for x in v]
    
    # Cache the result at all levels
    _query_embedding_cache[query_hash] = v
    
    # Store in session cache if available
    try:
        if "query_embedding_cache" not in st.session_state:
            st.session_state["query_embedding_cache"] = {}
        st.session_state["query_embedding_cache"][query_hash] = v
    except Exception:
        pass
    
    # Save to disk cache periodically (every 10 new entries to avoid excessive I/O)
    try:
        from rag_cache import save_query_embedding_cache
        # Update disk cache periodically (every 10 new entries to avoid excessive I/O)
        if len(_query_embedding_cache) % 10 == 0:
            save_query_embedding_cache(_query_embedding_cache)
    except Exception as e:
        logger.debug(f"Could not save to disk cache (non-fatal): {e}")
    
    logger.debug(f"Cached query embedding (new): {text[:50]}...")
    return v


def save_query_embedding_cache_to_disk() -> None:
    """
    Explicitly save query embedding cache to disk.
    Call this periodically or on shutdown to ensure persistence.
    """
    try:
        from rag_cache import save_query_embedding_cache
        save_query_embedding_cache(_query_embedding_cache)
        logger.debug(f"Query embedding cache saved to disk ({len(_query_embedding_cache)} entries)")
    except Exception as e:
        logger.warning(f"Failed to save query embedding cache: {e}")


def get_query_embedding_cache_stats() -> dict[str, Any]:
    """
    Get statistics about query embedding cache usage.
    Returns dict with cache size, session cache size, etc.
    """
    stats = {
        "memory_cache_size": len(_query_embedding_cache),
        "disk_cache_loaded": _disk_cache_loaded,
        "query_expansion_cache_size": len(_query_expansion_cache),
        "retrieval_planner_cache_size": len(_retrieval_planner_cache),
    }
    try:
        if "query_embedding_cache" in st.session_state:
            stats["session_cache_size"] = len(st.session_state["query_embedding_cache"])
        else:
            stats["session_cache_size"] = 0
    except Exception:
        stats["session_cache_size"] = 0
    return stats


def save_all_caches_to_disk() -> None:
    """
    Save all caches to disk explicitly.
    Call this periodically or on shutdown to ensure persistence.
    """
    save_query_embedding_cache_to_disk()
    try:
        from rag_cache import save_query_expansion_cache, save_retrieval_planner_cache
        save_query_expansion_cache(_query_expansion_cache)
        save_retrieval_planner_cache(_retrieval_planner_cache)
        logger.debug("All caches saved to disk")
    except Exception as e:
        logger.warning(f"Failed to save some caches: {e}")


# -----------------------------------------------------------------------------
# Caching
# -----------------------------------------------------------------------------


def _text_part(text: str) -> types.Part:
    """Build a Part from text (google-genai Part API)."""
    try:
        logger.debug("Creating Part with text parameter")
        # Use constructor directly - Part(text=...) is the standard API
        return types.Part(text=text)
    except Exception as e:
        logger.error(f"Error creating text Part: {e}", exc_info=True)
        raise


def create_gemini_cache(context_xml: str, max_retries: int = 5, initial_delay: float = 3.0, progress_callback: callable | None = None) -> str:
    """
    Create a context cache from context_xml using genai caching (CachedContent.create).
    TTL is 60 minutes. Returns the cache resource name for use in run_agent.
    
    Retries on transient server errors (503, 500, etc.) with exponential backoff.
    
    Args:
        context_xml: XML string containing document context
        max_retries: Maximum number of retry attempts (default: 5)
        initial_delay: Initial delay in seconds before first retry (default: 3.0)
        progress_callback: Optional callback(attempt, max_retries, delay, error_msg) called during retries
    """
    logger.info(f"Creating Gemini cache (context size: {len(context_xml)} chars, TTL: {CACHE_TTL_SECONDS}s)")
    client = _client()
    ttl = f"{CACHE_TTL_SECONDS}s"
    m = get_effective_model()  # Get current model from database
    logger.debug(f"Building cache config with model {m}")
    config = types.CreateCachedContentConfig(
        contents=[
            types.Content(
                role="user",
                parts=[_text_part(context_xml)],
            )
        ],
        ttl=ttl,
        display_name="councilflow-context",
    )
    
    last_error = None
    delay = initial_delay
    all_errors = []
    
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Calling client.caches.create() (attempt {attempt + 1}/{max_retries + 1})")
            cache = client.caches.create(model=m, config=config)
            logger.info(f"Cache created successfully: {cache.name} (after {attempt + 1} attempt(s))")
            return cache.name
        except errors.ServerError as e:
            # Transient server errors (503, 500, etc.) - retry with backoff
            last_error = e
            all_errors.append(str(e))
            # ServerError may have status_code as first arg or as attribute
            status_code = getattr(e, 'status_code', None) or (e.args[0] if e.args else None) or 'unknown'
            error_msg = str(e)
            logger.warning(f"Server error on attempt {attempt + 1}/{max_retries + 1}: {status_code} - {error_msg}")
            
            if attempt < max_retries:
                retry_msg = f"Attempt {attempt + 1}/{max_retries + 1} failed. Retrying in {delay:.1f} seconds..."
                logger.info(retry_msg)
                # Call progress callback if provided (for UI updates)
                if progress_callback:
                    try:
                        progress_callback(attempt + 1, max_retries + 1, delay, error_msg)
                    except Exception as cb_error:
                        logger.debug(f"Progress callback failed: {cb_error}")
                # Break sleep into smaller chunks to allow UI updates
                sleep_chunks = max(1, int(delay))
                for _ in range(sleep_chunks):
                    time.sleep(1.0)
                if delay > sleep_chunks:
                    time.sleep(delay - sleep_chunks)
                delay *= 2  # Exponential backoff (2s, 4s, 8s)
            else:
                error_summary = f"All {max_retries + 1} attempts failed with server errors (503/500). Last error: {e}"
                logger.error(error_summary)
                # Create a more informative error message
                raise RuntimeError(
                    f"Failed to create cache after {max_retries + 1} attempts. "
                    f"Gemini API returned 503 (Service Unavailable) on all attempts. "
                    f"This usually indicates temporary API issues. Please try again in a few minutes. "
                    f"Last error: {e}"
                ) from e
        except errors.ClientError as e:
            # Client errors (400, 401, 403, etc.) - don't retry
            status_code = getattr(e, 'status_code', None) or (e.args[0] if e.args else None) or 'unknown'
            error_str = str(e).lower()
            
            # Check for "too large" cache error
            if "too large" in error_str or "max_total_token_count" in error_str:
                logger.error(f"Cache content too large (non-retryable, status {status_code}): {e}")
                # Extract token count if available
                import re
                token_match = re.search(r'total_token_count[=:]?\s*(\d+)', error_str)
                token_count = token_match.group(1) if token_match else "unknown"
                raise RuntimeError(
                    f"**Cache Content Too Large**\n\n"
                    f"Your knowledge base is too large to cache ({token_count} tokens). "
                    f"Gemini caching has size limits that your current content exceeds.\n\n"
                    f"**Solutions:**\n"
                    f"1. **Use fallback mode** - Enable 'Skip Cache (Fallback Mode)' in sidebar (works but slower)\n"
                    f"2. **Reduce knowledge base size** - Filter or summarize documents in your Drive folder\n"
                    f"3. **Split into multiple caches** - Not currently supported, but could be implemented\n"
                    f"4. **Check API key permissions** - Ensure your API key has caching enabled\n\n"
                    f"**Technical details:** {e}"
                ) from e
            
            logger.error(f"Client error creating cache (non-retryable, status {status_code}): {e}", exc_info=True)
            raise
        except Exception as e:
            # Other errors - log and raise
            logger.error(f"Unexpected error creating Gemini cache: {e}", exc_info=True)
            raise
    
    # Should never reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Failed to create cache after retries")


# -----------------------------------------------------------------------------
# Library / file summarization (for retrieval planner)
# -----------------------------------------------------------------------------

FILE_SUMMARY_MAX_CHARS = 4000
FILE_SUMMARY_BATCH_SIZE = 5
DESCRIBE_LIBRARY_MAX_FILES = 50


def summarize_files_batch(
    files: list[dict[str, Any]],
    *,
    batch_size: int = FILE_SUMMARY_BATCH_SIZE,
    max_chars_per_file: int = FILE_SUMMARY_MAX_CHARS,
    progress_callback: callable | None = None,
) -> list[str]:
    """
    For each file {name, text}, produce a one-sentence summary via LLM.
    Processes in batches. Returns list of summary strings (same order as files).
    
    Caching: Results are cached based on file content hash.
    This avoids redundant LLM calls for identical files during index rebuilds.
    """
    import hashlib
    
    if not files:
        return []
    
    # Load disk cache if not already loaded
    global _file_summary_cache, _file_summary_disk_loaded
    if not _file_summary_disk_loaded:
        try:
            from rag_cache import load_file_summary_cache
            disk_cache = load_file_summary_cache()
            _file_summary_cache.update(disk_cache)
            _file_summary_disk_loaded = True
            logger.debug(f"Pre-loaded {len(disk_cache)} file summaries from disk cache")
        except Exception as e:
            logger.debug(f"Could not load file summary disk cache (non-fatal): {e}")
            _file_summary_disk_loaded = True
    
    # Check cache for each file and collect uncached ones
    all_summaries: list[str] = []
    uncached_files: list[tuple[int, dict[str, Any]]] = []  # (original_index, file)
    
    for idx, f in enumerate(files):
        text = (f.get("text") or "").strip()
        # Use first part of text for hash (consistent with what we send to LLM)
        text_for_hash = text[:max_chars_per_file] if len(text) > max_chars_per_file else text
        file_hash = hashlib.sha256(text_for_hash.encode('utf-8')).hexdigest()
        
        if file_hash in _file_summary_cache:
            all_summaries.append(_file_summary_cache[file_hash])
            logger.debug(f"File summary cache hit for file: {f.get('name', '?')[:50]}...")
        else:
            all_summaries.append(None)  # Placeholder, will be filled later
            uncached_files.append((idx, f))
    
    # If all files were cached, return early
    if not uncached_files:
        logger.info(f"All {len(files)} file summaries retrieved from cache")
        return all_summaries
    
    # Process uncached files in batches
    client = _client()
    m = DEFAULT_MODEL
    cache_updates: dict[str, str] = {}
    
    for b in range(0, len(uncached_files), batch_size):
        batch = uncached_files[b : b + batch_size]
        batch_indices = [idx for idx, _ in batch]
        batch_files = [f for _, f in batch]
        
        if progress_callback:
            try:
                progress_callback("summarize", min(b + batch_size, len(uncached_files)), len(uncached_files), None)
            except Exception:
                pass
        
        parts: list[str] = []
        file_hashes: list[str] = []
        for f in batch_files:
            name = f.get("name") or "Untitled"
            text = (f.get("text") or "").strip()
            if len(text) > max_chars_per_file:
                text = text[:max_chars_per_file] + "..."
            parts.append(f"Document {len(parts) + 1} (filename: {name}):\n\n{text or '(empty)'}")
            # Calculate hash for caching
            text_for_hash = text
            file_hash = hashlib.sha256(text_for_hash.encode('utf-8')).hexdigest()
            file_hashes.append(file_hash)
        
        prompt = (
            "For each of the following documents, provide exactly one short sentence (max 30 words) "
            "summarizing its contents. Respond with valid JSON only:\n"
            '{"summaries": ["summary 1", "summary 2", ...]}\n'
            "Use the same order as the documents. No other text.\n\n"
            + "\n\n---\n\n".join(parts)
        )
        config = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = [types.Content(role="user", parts=[_text_part(prompt)])]
        delay = 1.0
        batch_summaries: list[str] = []
        for attempt in range(3):
            try:
                # Apply rate limiting before GenerateContent call
                _apply_rate_limit()
                
                r = client.models.generate_content(model=m, contents=contents, config=config)
                text = (r.text or "").strip()
                parsed = _parse_json_robust(text)
                summaries = parsed.get("summaries") if isinstance(parsed, dict) else None
                if isinstance(summaries, list) and len(summaries) >= len(batch_files):
                    batch_summaries = summaries[: len(batch_files)]
                    break
                if isinstance(summaries, list):
                    batch_summaries = [(s[:200] if isinstance(s, str) else str(s)) for s in summaries[: len(batch_files)]]
                    while len(batch_summaries) < len(batch_files):
                        batch_summaries.append("(summary unavailable)")
                    break
                logger.warning("Summarizer response invalid, using fallback")
                batch_summaries = ["(summary unavailable)"] * len(batch_files)
                break
            except (errors.ServerError, errors.ClientError) as e:
                if attempt < 2:
                    wait = _parse_429_retry_delay(e) or delay
                    wait = max(wait, delay)
                    logger.warning(f"Summarizer error, retrying in {wait:.0f}s: {e}")
                    time.sleep(wait)
                    delay *= 2
                else:
                    batch_summaries = ["(summary unavailable)"] * len(batch_files)
                    logger.warning(f"Summarizer failed after retries: {e}")
                    break
            except Exception as e:
                batch_summaries = ["(summary unavailable)"] * len(batch_files)
                logger.warning(f"Summarizer error: {e}")
                break
        
        # Update cache and results
        for i, (orig_idx, _) in enumerate(batch):
            if i < len(batch_summaries):
                summary = batch_summaries[i]
                file_hash = file_hashes[i]
                all_summaries[orig_idx] = summary
                _file_summary_cache[file_hash] = summary
                cache_updates[file_hash] = summary
        
        if b + batch_size < len(uncached_files):
            time.sleep(0.3)
    
    # Save cache to disk periodically
    if cache_updates:
        try:
            from rag_cache import save_file_summary_cache
            if len(_file_summary_cache) % 50 == 0:  # Save every 50 new entries
                save_file_summary_cache(_file_summary_cache)
        except Exception as e:
            logger.debug(f"Could not save file summary cache (non-fatal): {e}")
    
    logger.info(f"Generated {len(uncached_files)} new file summaries, {len(files) - len(uncached_files)} from cache")
    return all_summaries


def describe_library(
    library_name: str,
    file_descriptors: list[dict[str, Any]],
    *,
    max_files: int = DESCRIBE_LIBRARY_MAX_FILES,
) -> str:
    """
    Produce a 1–2 sentence description of a library from its file list and per-file summaries.
    file_descriptors: list of {name, summary}.
    
    Caching: Results are cached based on library name + file descriptors hash.
    This avoids redundant LLM calls for identical libraries during index rebuilds.
    """
    import hashlib
    import json
    
    if not file_descriptors:
        return f"Library '{library_name}' has no files."
    
    # Build cache key from library name + file descriptors
    # Use a stable representation of file descriptors (sorted by name, include summary)
    fd_key = sorted([(d.get("name", ""), d.get("summary", "")[:120]) for d in file_descriptors[:max_files]])
    cache_input = f"{library_name}|{json.dumps(fd_key, sort_keys=True)}"
    cache_key = hashlib.sha256(cache_input.encode('utf-8')).hexdigest()
    
    # Load disk cache if not already loaded
    global _library_description_cache, _library_description_disk_loaded
    if not _library_description_disk_loaded:
        try:
            from rag_cache import load_library_description_cache
            disk_cache = load_library_description_cache()
            _library_description_cache.update(disk_cache)
            _library_description_disk_loaded = True
            logger.debug(f"Pre-loaded {len(disk_cache)} library descriptions from disk cache")
        except Exception as e:
            logger.debug(f"Could not load library description disk cache (non-fatal): {e}")
            _library_description_disk_loaded = True
    
    # Check cache
    if cache_key in _library_description_cache:
        logger.debug(f"Library description cache hit: {library_name}")
        return _library_description_cache[cache_key]
    
    # Cache miss - call LLM
    logger.debug(f"Library description cache miss, calling LLM: {library_name}")
    limited = file_descriptors[:max_files]
    lines = []
    for d in limited:
        name = d.get("name", "?")
        summary = (d.get("summary") or "")[:120]
        lines.append(f"- {name}: {summary}")
    if len(file_descriptors) > max_files:
        lines.append(f"... and {len(file_descriptors) - max_files} more files.")
    prompt = (
        f"Describe the following knowledge-base library in 1–2 sentences. "
        f"Library name: {library_name}\n\n"
        "Files (name: short summary):\n" + "\n".join(lines) + "\n\n"
        "Reply with only the 1–2 sentence description, no JSON or prefix."
    )
    client = _client()
    config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[_text_part(prompt)])]
    try:
        # Apply rate limiting before GenerateContent call
        _apply_rate_limit()
        
        r = client.models.generate_content(model=DEFAULT_MODEL, contents=contents, config=config)
        description = (r.text or "").strip() or f"Library '{library_name}' with {len(file_descriptors)} files."
        
        # Cache the result
        _library_description_cache[cache_key] = description
        
        # Save to disk cache periodically
        try:
            from rag_cache import save_library_description_cache
            if len(_library_description_cache) % 20 == 0:  # Save every 20 new entries
                save_library_description_cache(_library_description_cache)
        except Exception as e:
            logger.debug(f"Could not save library description cache (non-fatal): {e}")
        
        return description
    except Exception as e:
        logger.warning(f"describe_library failed: {e}")
        fallback = f"Library '{library_name}' with {len(file_descriptors)} files."
        # Cache fallback too
        _library_description_cache[cache_key] = fallback
        return fallback


# -----------------------------------------------------------------------------
# Retrieval planner (preprocessor)
# -----------------------------------------------------------------------------

RETRIEVAL_PLANNER_PROMPT = """You are a retrieval planner for a RAG system. Given an analysis task and the documents to be analyzed, decide which knowledge-base libraries to search and how many chunks to retrieve from each.

{{ library_catalog }}

{{ context_budget }}

Analysis task name: {{ task_name }}

Task description or instructions (excerpt):
{{ task_description }}

Documents to analyze (excerpt):
{{ user_content }}

Respond with valid JSON only, no other text. Use this exact structure:
{"libraries": [{"name": "<library name>", "top_k": <number>}, ...]}

Rules:
- "name" must exactly match one of the library names listed above.
- Use the library descriptions and file list to pick libraries relevant to the task and documents. Omit irrelevant ones.
- top_k: how many chunks to retrieve from that library (1–100). Weight more important libraries higher; use 25–80 for key libraries.
- If the task or documents clearly need legal/statutory material, prioritize libraries that contain law (e.g. MGL) and use larger top_k (50–100).
- Prefer being inclusive: include 2–4 relevant libraries when applicable, with top_k 30–80 each for important ones.
- Total chunks across all libraries should typically be 80–300 (or more when the task clearly needs broad context). Use the context budget aggressively.
- Never exceed the chunk budget above. Stay at or under the approximate chunk limit. Prioritize libraries that clearly matter for the task.
"""

PLANNER_MAX_FILES_PER_LIB = 25
PLANNER_FILE_SUMMARY_CHARS = 80


def _build_library_catalog(library_metadata: list[dict[str, Any]]) -> str:
    """Format library metadata for the retrieval planner prompt."""
    parts: list[str] = []
    for lib in library_metadata:
        name = lib.get("name") or "Unnamed"
        desc = (lib.get("library_description") or "").strip()
        files = lib.get("file_descriptors") or []
        section = [f"## Library: {name}"]
        if desc:
            section.append(f"Description: {desc}")
        elif not files:
            section.append("Description: (no description available)")
        if files:
            section.append("Files (name: short summary):")
            for fd in files[:PLANNER_MAX_FILES_PER_LIB]:
                fn = (fd.get("name") or "?").replace("\n", " ").strip()
                sm = (fd.get("summary") or "")[:PLANNER_FILE_SUMMARY_CHARS].replace("\n", " ")
                section.append(f"  - {fn}: {sm}")
            if len(files) > PLANNER_MAX_FILES_PER_LIB:
                section.append(f"  ... and {len(files) - PLANNER_MAX_FILES_PER_LIB} more files.")
        parts.append("\n".join(section))
    return "\n\n".join(parts) if parts else "No libraries available."


def run_retrieval_planner(
    task_name: str,
    task_description: str,
    user_content: str,
    library_metadata: list[dict[str, Any]],
    *,
    model: str | None = None,
    max_retries: int = 2,
    context_budget_section: str | None = None,
) -> dict[str, Any] | None:
    """
    LLM call (no KB cache) to decide which libraries to search and top_k per library.
    library_metadata: list of {name, library_description?, file_descriptors?: [{name, summary}]}.
    context_budget_section: optional paragraph describing token/chunk budget for retrieved KB.
    Returns {"libraries": [{"name": str, "top_k": int}, ...]} or None on failure.
    
    Caching: Results are cached based on task_name + task_description + user_content + library_metadata hash.
    This avoids redundant LLM calls for identical or similar planning requests.
    """
    import hashlib
    import json
    
    if not library_metadata:
        logger.debug("No libraries available for retrieval planner")
        return {"libraries": []}
    
    # Build cache key from inputs (include library names for cache key)
    library_names = sorted([lib.get("name", "") for lib in library_metadata])
    cache_input = f"{task_name}|{task_description[:2500]}|{user_content[:3500]}|{context_budget_section or ''}|{','.join(library_names)}"
    cache_key = hashlib.sha256(cache_input.encode('utf-8')).hexdigest()
    
    # Check cache first (memory, then disk)
    if cache_key in _retrieval_planner_cache:
        logger.debug(f"Retrieval planner cache hit: {task_name[:50]}...")
        return _retrieval_planner_cache[cache_key]
    
    # Load disk cache if not already loaded
    global _retrieval_planner_disk_loaded
    if not _retrieval_planner_disk_loaded:
        try:
            from rag_cache import load_retrieval_planner_cache
            disk_cache = load_retrieval_planner_cache()
            _retrieval_planner_cache.update(disk_cache)
            _retrieval_planner_disk_loaded = True
            if cache_key in _retrieval_planner_cache:
                logger.debug(f"Retrieval planner disk cache hit: {task_name[:50]}...")
                return _retrieval_planner_cache[cache_key]
        except Exception as e:
            logger.debug(f"Could not load retrieval planner disk cache (non-fatal): {e}")
            _retrieval_planner_disk_loaded = True
    
    # Cache miss - call LLM
    logger.debug(f"Retrieval planner cache miss, calling LLM: {task_name[:50]}...")
    client = _client()
    m = model or get_planner_model()  # Get current planner model from database
    catalog = _build_library_catalog(library_metadata)
    budget_text = (context_budget_section or "").strip() or "No explicit context budget; use judgment to stay within model limits."
    prompt = (
        RETRIEVAL_PLANNER_PROMPT.strip()
        .replace("{{ library_catalog }}", catalog)
        .replace("{{ context_budget }}", budget_text)
        .replace("{{ task_name }}", task_name)
        .replace("{{ task_description }}", (task_description or "")[:2500])
        .replace("{{ user_content }}", (user_content or "")[:3500])
    )
    config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    contents = [types.Content(role="user", parts=[_text_part(prompt)])]
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            # Apply rate limiting before GenerateContent call
            _apply_rate_limit()
            
            logger.info(f"Running retrieval planner (attempt {attempt + 1}/{max_retries + 1})")
            r = client.models.generate_content(model=m, contents=contents, config=config)
            text = (r.text or "").strip()
            parsed = _parse_json_robust(text)
            if parsed and isinstance(parsed.get("libraries"), list):
                logger.info(f"Planner chose {len(parsed['libraries'])} libraries")
                
                # Cache the result
                _retrieval_planner_cache[cache_key] = parsed
                
                # Save to disk cache periodically (every 10 new entries)
                try:
                    from rag_cache import save_retrieval_planner_cache
                    if len(_retrieval_planner_cache) % 10 == 0:
                        save_retrieval_planner_cache(_retrieval_planner_cache)
                except Exception as e:
                    logger.debug(f"Could not save retrieval planner cache (non-fatal): {e}")
                
                return parsed
            logger.warning("Planner response missing or invalid 'libraries' array")
            return None
        except errors.ServerError as e:
            if attempt < max_retries:
                logger.warning(f"Planner server error, retrying in {delay:.0f}s: {e}")
                time.sleep(delay)
                delay *= 2
            else:
                logger.error(f"Planner failed after retries: {e}")
                raise
        except errors.ClientError as e:
            logger.warning(f"Planner client error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Planner error: {e}")
            return None
    return None


# -----------------------------------------------------------------------------
# Query expansion (LLM-generated search phrases)
# -----------------------------------------------------------------------------

EXPAND_QUERIES_PROMPT = """Given the analysis task and user-provided content below, produce 3–5 short search phrases (each a few words to a short sentence) that would help retrieve relevant passages from a document corpus. Focus on key concepts, entities, and legal or policy terms.

Task: {{ task_name }}

Task description (excerpt): {{ task_description }}

User content (excerpt): {{ user_content }}

Respond with valid JSON only, no other text. Use this exact structure:
{"phrases": ["phrase one", "phrase two", "phrase three", ...]}"""


def expand_queries(
    task_name: str,
    template_text: str,
    user_content: str,
    *,
    model: str | None = None,
) -> list[str]:
    """
    Generate 3–5 search phrases from task + user content via LLM.
    Returns list of phrases, or fallback [task_name + user_content excerpt] on failure.
    
    Caching: Results are cached based on task_name + template_text + user_content hash.
    This avoids redundant LLM calls for identical or similar queries.
    """
    import hashlib
    
    fallback = [f"{task_name}\n\n{(user_content or '')[:2000]}"]
    
    # Build cache key from inputs
    cache_input = f"{task_name}|{template_text[:1500]}|{user_content[:2500]}"
    cache_key = hashlib.sha256(cache_input.encode('utf-8')).hexdigest()
    
    # Check cache first (memory, then disk)
    if cache_key in _query_expansion_cache:
        logger.debug(f"Query expansion cache hit: {task_name[:50]}...")
        return _query_expansion_cache[cache_key]
    
    # Load disk cache if not already loaded
    global _query_expansion_disk_loaded
    if not _query_expansion_disk_loaded:
        try:
            from rag_cache import load_query_expansion_cache
            disk_cache = load_query_expansion_cache()
            _query_expansion_cache.update(disk_cache)
            _query_expansion_disk_loaded = True
            if cache_key in _query_expansion_cache:
                logger.debug(f"Query expansion disk cache hit: {task_name[:50]}...")
                return _query_expansion_cache[cache_key]
        except Exception as e:
            logger.debug(f"Could not load query expansion disk cache (non-fatal): {e}")
            _query_expansion_disk_loaded = True
    
    # Cache miss - call LLM
    logger.debug(f"Query expansion cache miss, calling LLM: {task_name[:50]}...")
    try:
        client = _client()
        m = model or DEFAULT_MODEL
        prompt = (
            EXPAND_QUERIES_PROMPT.strip()
            .replace("{{ task_name }}", (task_name or "")[:500])
            .replace("{{ task_description }}", (template_text or "")[:1500])
            .replace("{{ user_content }}", (user_content or "")[:2500])
        )
        config = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        contents = [types.Content(role="user", parts=[_text_part(prompt)])]
        
        # Apply rate limiting before GenerateContent call
        _apply_rate_limit()
        
        r = client.models.generate_content(model=m, contents=contents, config=config)
        text = (r.text or "").strip()
        parsed = _parse_json_robust(text)
        if not parsed or not isinstance(parsed.get("phrases"), list):
            logger.warning("expand_queries: invalid or missing phrases, using fallback")
            return fallback
        phrases = [str(p).strip() for p in parsed["phrases"] if p]
        if not phrases:
            return fallback
        # Dedupe and cap length
        seen: set[str] = set()
        out: list[str] = []
        for p in phrases:
            key = p[:200]
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
            if len(out) >= 5:
                break
        logger.info("expand_queries: %d phrases", len(out))
        
        # Cache the result
        _query_expansion_cache[cache_key] = out
        
        # Save to disk cache periodically (every 20 new entries)
        try:
            from rag_cache import save_query_expansion_cache
            if len(_query_expansion_cache) % 20 == 0:
                save_query_expansion_cache(_query_expansion_cache)
        except Exception as e:
            logger.debug(f"Could not save query expansion cache (non-fatal): {e}")
        
        return out
    except Exception as e:
        logger.warning("expand_queries failed: %s, using fallback", e)
        return fallback


# -----------------------------------------------------------------------------
# Re-ranking (LLM-based)
# -----------------------------------------------------------------------------

RERANK_BATCH_SIZE = 15
RERANK_MAX_CHUNK_CHARS = 800


def rerank_chunks_llm(
    query: str,
    chunks: list[dict[str, Any]],
    top_k: int,
    *,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Re-rank chunks by LLM relevance scoring (1–5). Returns top_k chunks in descending
    relevance order. On parse failure, treats score as 1. Falls back to original order
    if all batches fail.
    """
    if not chunks or top_k <= 0:
        return chunks[:top_k]
    if len(chunks) <= top_k:
        return chunks

    logger.info("rerank_chunks_llm: starting re-rank of %d chunks -> top_k %d", len(chunks), top_k)
    client = _client()
    m = model or DEFAULT_MODEL
    scored: list[tuple[float, dict[str, Any]]] = []
    prompt_tpl = (
        "Rate each passage's relevance to the query from 1 (irrelevant) to 5 (highly relevant). "
        "Reply with valid JSON only: {\"scores\": [n1, n2, ...]} in the same order as the passages.\n\n"
        "Query: {{ query }}\n\nPassages:\n{{ passages }}"
    )

    total_batches = (len(chunks) + RERANK_BATCH_SIZE - 1) // RERANK_BATCH_SIZE
    rerank_config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    for start in range(0, len(chunks), RERANK_BATCH_SIZE):
        batch_idx = start // RERANK_BATCH_SIZE + 1
        batch = chunks[start : start + RERANK_BATCH_SIZE]
        logger.info("rerank batch %d/%d (%d chunks)", batch_idx, total_batches, len(batch))
        passages = []
        for i, c in enumerate(batch, 1):
            t = (c.get("text") or "")[:RERANK_MAX_CHUNK_CHARS]
            if len((c.get("text") or "")) > RERANK_MAX_CHUNK_CHARS:
                t += "..."
            passages.append(f"{i}. {t}")
        q = (query or "")[:1500]
        prompt = prompt_tpl.replace("{{ query }}", q).replace(
            "{{ passages }}", "\n\n".join(passages)
        )
        contents = [types.Content(role="user", parts=[_text_part(prompt)])]
        try:
            # Apply rate limiting before GenerateContent call
            _apply_rate_limit()
            
            r = client.models.generate_content(model=m, contents=contents, config=rerank_config)
            text = (r.text or "").strip()
            parsed = _parse_json_robust(text)
            scores = parsed.get("scores") if isinstance(parsed, dict) else None
            if isinstance(scores, list) and len(scores) >= len(batch):
                for i, c in enumerate(batch):
                    try:
                        sc = int(scores[i])
                    except (TypeError, ValueError):
                        sc = 1
                    scored.append((min(5, max(1, sc)), c))
            else:
                for c in batch:
                    scored.append((1.0, c))
        except Exception as e:
            logger.warning("rerank batch failed: %s, using score 1", e)
            for c in batch:
                scored.append((1.0, c))

    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:top_k]]


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------


def extract_legal_questions(output: str) -> tuple[str, list[str]]:
    """
    Extract legal questions from prompt output and return (main_content, legal_questions).
    
    The output should contain a markdown section with the exact title:
    "## Legal Questions Requiring Expert Review"
    
    We look for this specific section title and extract bullet-pointed questions below it.
    
    Returns (main_content, legal_questions_list).
    If no legal questions found, returns (original_output, []).
    """
    import re
    
    if not output or not output.strip():
        return output, []
    
    # Look for the specific markdown section title: "## Legal Questions Requiring Expert Review"
    # This is case-insensitive and allows for variations in spacing
    md_pattern = r'(?i)##\s*Legal\s+Questions\s+Requiring\s+Expert\s+Review\s*\n(.*?)(?=\n##|\Z)'
    match = re.search(md_pattern, output, re.DOTALL)
    
    if match:
        questions_text = match.group(1).strip()
        questions = []
        
        # Extract questions from bullet points or numbered lists
        for line in questions_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Remove markdown list markers: -, *, •, or numbered lists
            line = re.sub(r'^[-*•]\s+', '', line)
            line = re.sub(r'^\d+[.)]\s+', '', line)
            line = line.strip()
            
            # Only include lines that look like questions (minimum length, ends with ? or is substantial)
            if line and len(line) > 10 and (line.endswith('?') or len(line) > 20):
                questions.append(line)
        
        if questions:
            # Extract main content (everything before the legal questions section)
            main = output[:match.start()].strip()
            logger.info(f"Extracted {len(questions)} legal questions from 'Legal Questions Requiring Expert Review' section")
            return main, questions
    
    # Fallback: Try other formats for backwards compatibility
    # Try JSON format
    try:
        parsed = _parse_json_robust(output)
        if parsed and isinstance(parsed, dict):
            questions = parsed.get("legal_questions")
            main = parsed.get("main_content", output)
            if isinstance(questions, list) and questions:
                valid_questions = [q.strip() for q in questions if q and str(q).strip()]
                if valid_questions:
                    logger.info(f"Extracted {len(valid_questions)} legal questions from JSON format (fallback)")
                    return str(main).strip() or output, valid_questions
    except Exception:
        pass
    
    # Try generic "## Legal Questions" section (backwards compatibility)
    try:
        md_pattern_generic = r'(?i)##\s*Legal\s+Questions?\s*\n(.*?)(?=\n##|\Z)'
        match = re.search(md_pattern_generic, output, re.DOTALL)
        if match:
            questions_text = match.group(1).strip()
            questions = []
            for line in questions_text.split('\n'):
                line = line.strip()
                line = re.sub(r'^[-*•]\s+', '', line)
                line = re.sub(r'^\d+[.)]\s+', '', line)
                if line and len(line) > 10:
                    questions.append(line)
            
            if questions:
                main = output[:match.start()].strip()
                logger.info(f"Extracted {len(questions)} legal questions from generic Legal Questions section (fallback)")
                return main, questions
    except Exception:
        pass
    
    # No legal questions found
    return output, []


def _parse_json_robust(raw: str) -> dict[str, Any] | None:
    """
    Try to extract and parse JSON from model output. Handles markdown fences,
    trailing commas, and common glitches. Returns None if not valid JSON.
    """
    logger.debug(f"Parsing JSON from {len(raw)} char string")
    s = raw.strip()
    # Strip ```json ... ``` or ``` ... ```
    for pattern in (r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", r"^```\s*\n?(.*?)\n?```\s*$"):
        m = re.search(pattern, s, re.DOTALL)
        if m:
            logger.debug("Stripped markdown code fence")
            s = m.group(1).strip()
            break
    # Find first { ... } span (balance braces)
    start = s.find("{")
    if start == -1:
        logger.debug("No opening brace found")
        return None
    depth = 0
    end = -1
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        logger.debug("No matching closing brace found")
        return None
    s = s[start:end]
    # Common repairs: trailing commas
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    try:
        result = json.loads(s)
        logger.debug("Successfully parsed JSON on first attempt")
        return result
    except json.JSONDecodeError as e:
        logger.debug(f"First JSON parse attempt failed: {e}")
    # Try replacing single-quoted keys/strings (simple pattern)
    try:
        repaired = re.sub(r"('\w+')\s*:", lambda m: f'"{m.group(1)[1:-1]}":', s)
        result = json.loads(repaired)
        logger.debug("Successfully parsed JSON after repair")
        return result
    except (json.JSONDecodeError, Exception) as e:
        logger.debug(f"Repaired JSON parse also failed: {e}")
    logger.warning("Could not parse response as JSON")
    return None


def run_agent(
    prompt_template: str,
    transient_data: dict[str, Any],
    cache_name: str | None,
    *,
    model: str | None = None,
    expect_json: bool = True,
    max_retries: int = 7,
    initial_delay: float = 1.0,
    context_xml: str | None = None,
    input_schema_json: str | None = None,
    output_schema_json: str | None = None,
) -> str | dict[str, Any]:
    """
    Render prompt_template with transient_data (jinja2), call Gemini with
    cached_content=cache_name (if provided), return generated text or parsed Dict if JSON.

    If expect_json is True and the response parses as JSON, returns a dict;
    otherwise returns the raw text string.
    
    If cache_name is None, will include context_xml directly in the prompt (fallback mode).
    This is slower and more expensive but works when caching fails.
    
    Retries on transient server errors with exponential backoff.
    """
    logger.info(f"Running agent (cache: {cache_name}, expect_json: {expect_json}, fallback_mode: {cache_name is None})")
    client = _client()
    logger.debug("Rendering Jinja2 template (first 200 chars): %r", prompt_template[:200])
    try:
        t = Template(prompt_template)
        prompt = t.render(**transient_data)
    except Exception as e:
        # Log full template for debugging
        logger.error("Template has invalid syntax. Dumping template text for debugging. Position 1312 may contain an unescaped backslash.\n%s", prompt_template)
        logger.error("Template error occurred at (around) char %d or later.\n%s", min(1312, len(prompt_template)), prompt_template[1300:1360])
        raise RuntimeError("Invalid template syntax in prompt. Check template text for backslashes or other special characters that may need escaping.") from e
    logger.debug(f"Rendered prompt length: {len(prompt)} chars")
    m = model or get_effective_model()  # Use function to get current model from DB
    logger.debug(f"Using model: {m}")
    
    # Build content - either use cache or include context directly.
    # If JSON Schema sidecars are provided, append them as additional Parts so that
    # the model can see and follow them. The prompt itself can reference these
    # schemas explicitly.
    no_afc = types.AutomaticFunctionCallingConfig(disable=True)

    # If we have an output schema and it's valid JSON, configure Gemini for
    # structured JSON output.
    response_schema = None
    if output_schema_json:
        try:
            response_schema = json.loads(output_schema_json)
            logger.debug("Parsed output_schema_json for structured JSON response")
        except Exception as e:
            logger.warning(f"Could not parse output_schema_json as JSON; falling back to free-form output: {e}")
            response_schema = None

    if cache_name:
        # Use cached content (preferred - faster and cheaper)
        parts: list[types.Part] = [_text_part(prompt)]
        if input_schema_json:
            parts.append(
                _text_part(
                    "\n\n[INPUT_JSON_SCHEMA]\n"
                    f"{input_schema_json.strip()}\n"
                    "[/INPUT_JSON_SCHEMA]"
                )
            )
        if output_schema_json:
            parts.append(
                _text_part(
                    "\n\n[OUTPUT_JSON_SCHEMA]\n"
                    f"{output_schema_json.strip()}\n"
                    "[/OUTPUT_JSON_SCHEMA]"
                )
            )

        config_kwargs: dict[str, Any] = {
            "cached_content": cache_name,
            "automatic_function_calling": no_afc,
        }
        if response_schema is not None:
            # Tell Gemini to return strict JSON following this schema.
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = response_schema
        config = types.GenerateContentConfig(**config_kwargs)

        prompt_content = types.Content(
            role="user",
            parts=parts,
        )
        contents = [prompt_content]
    else:
        # Fallback: include context directly (works but slower/expensive)
        if not context_xml:
            raise ValueError("context_xml is required when cache_name is None (fallback mode)")
        logger.warning("Running without cache - including full context in prompt (slower and more expensive)")
        # Combine context and prompt
        full_prompt = f"{context_xml}\n\n---\n\n{prompt}"

        full_parts: list[types.Part] = [_text_part(full_prompt)]
        if input_schema_json:
            full_parts.append(
                _text_part(
                    "\n\n[INPUT_JSON_SCHEMA]\n"
                    f"{input_schema_json.strip()}\n"
                    "[/INPUT_JSON_SCHEMA]"
                )
            )
        if output_schema_json:
            full_parts.append(
                _text_part(
                    "\n\n[OUTPUT_JSON_SCHEMA]\n"
                    f"{output_schema_json.strip()}\n"
                    "[/OUTPUT_JSON_SCHEMA]"
                )
            )

        config_kwargs = {
            "automatic_function_calling": no_afc,
        }
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_json_schema"] = response_schema
        config = types.GenerateContentConfig(**config_kwargs)

        prompt_content = types.Content(
            role="user",
            parts=full_parts,
        )
        contents = [prompt_content]
    
    last_error = None
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
            # Apply rate limiting before each GenerateContent call
            _apply_rate_limit()
            
            logger.debug(f"Calling client.models.generate_content() (attempt {attempt + 1}/{max_retries + 1})")
            response = client.models.generate_content(
                model=m,
                contents=contents,
                config=config,
            )
            text = (response.text or "").strip()
            logger.info(f"Received response: {len(text)} chars")
            
            if expect_json:
                logger.debug("Attempting to parse response as JSON")
                parsed = _parse_json_robust(text)
                if parsed is not None:
                    logger.info("Successfully parsed JSON response")
                    return parsed
                else:
                    logger.warning("JSON parsing failed, returning raw text")
            return text
        except errors.ServerError as e:
            # Transient server errors (503, 500, etc.) - retry with backoff
            last_error = e
            # ServerError may have status_code as first arg or as attribute
            status_code = getattr(e, 'status_code', None) or (e.args[0] if e.args else None) or 'unknown'
            error_msg = str(e)
            logger.warning(f"Server error on attempt {attempt + 1}/{max_retries + 1}: {status_code} - {error_msg}")
            
            if attempt < max_retries:
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
                raise
        except errors.ClientError as e:
            # Client errors - handle 429 (quota) specially, others don't retry
            status_code = getattr(e, 'status_code', None) or (e.args[0] if e.args else None) or 'unknown'
            
            # 429 RESOURCE_EXHAUSTED - quota exceeded, retry with API-suggested delay
            if status_code == 429 or '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e):
                error_str = str(e).lower()
                
                # Try to extract retry delay from error message
                retry_delay = None
                if 'retry in' in error_str or 'retrydelay' in error_str:
                    try:
                        # Look for patterns like "Please retry in 14.14s" or retryDelay: '14s'
                        import re
                        delay_match = re.search(r'retry\s+in\s+([\d.]+)\s*s', error_str, re.IGNORECASE)
                        if delay_match:
                            retry_delay = float(delay_match.group(1))
                        else:
                            # Try to find retryDelay in JSON details
                            if hasattr(e, 'response') and hasattr(e.response, 'json'):
                                error_json = e.response.json()
                                if 'details' in error_json.get('error', {}):
                                    for detail in error_json['error']['details']:
                                        if detail.get('@type') == 'type.googleapis.com/google.rpc.RetryInfo':
                                            retry_delay_str = detail.get('retryDelay', '')
                                            if 's' in retry_delay_str:
                                                retry_delay = float(retry_delay_str.replace('s', ''))
                    except Exception:
                        pass
                
                # Use API-suggested delay or fallback; enforce minimum for per-minute quotas
                QUOTA_MIN_DELAY = 15.0
                QUOTA_MAX_DELAY = 60.0
                if retry_delay is None:
                    retry_delay = max(delay, QUOTA_MIN_DELAY)
                else:
                    retry_delay = max(retry_delay + 5.0, delay, QUOTA_MIN_DELAY)
                retry_delay = min(retry_delay, QUOTA_MAX_DELAY)

                if attempt < max_retries:
                    logger.warning(
                        f"Quota exceeded (429) on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Retrying in {retry_delay:.1f}s (min {QUOTA_MIN_DELAY:.0f}s for per‑minute limits)..."
                    )
                    time.sleep(retry_delay)
                    delay = min(retry_delay * 1.5, QUOTA_MAX_DELAY)
                else:
                    logger.error(f"Quota exceeded after {max_retries + 1} attempts. Last error: {e}")
                    ctx_str = f"{len(context_xml):,} chars" if context_xml else "unknown"
                    quota_msg = (
                        f"**Quota Exceeded (429 RESOURCE_EXHAUSTED)**\n\n"
                        f"You've exceeded your per-minute token limit for **{m}**.\n\n"
                        f"**Solutions:**\n"
                        f"1. **Wait 1–2 minutes** – Quotas reset per minute; then try again.\n"
                        f"2. **Try another model** – Set env `GEMINI_MODEL=gemini-2.0-flash` (often higher quotas), then restart.\n"
                        f"3. **Add pace delay** – Set env `GEMINI_PACE_DELAY_SECONDS=30` to wait before each generate call; helps spread usage.\n"
                        f"4. **Reduce context size** – Your knowledge base is very large ({ctx_str}).\n"
                        f"5. **Request quota increase** – Google Cloud Console → APIs & Services → Quotas.\n"
                        f"6. **Check usage**: https://ai.dev/rate-limit\n\n"
                        f"**Technical details:** {e}"
                    )
                    raise RuntimeError(quota_msg) from e
            # 403 PERMISSION_DENIED with "CachedContent not found" - cache expired or deleted
            elif status_code == 403 or '403' in str(e):
                error_str = str(e).lower()
                if 'cachedcontent not found' in error_str or 'permission denied' in error_str:
                    logger.warning(f"Cache expired or not found (403) on attempt {attempt + 1}/{max_retries + 1}: {e}")
                    # Raise special exception that app.py can catch to recreate cache
                    raise CacheExpiredError(
                        f"Cache expired or not found. The cached content '{cache_name}' is no longer available. "
                        f"This usually happens after the cache TTL expires (60 minutes) or if the cache was deleted."
                    ) from e
                else:
                    # Other 403 errors - don't retry
                    logger.error(f"Client error running agent (non-retryable, status {status_code}): {e}", exc_info=True)
                    raise
            else:
                # Other client errors (400, 401, etc.) - don't retry
                logger.error(f"Client error running agent (non-retryable, status {status_code}): {e}", exc_info=True)
                raise
        except Exception as e:
            # Other errors - log and raise
            logger.error(f"Unexpected error running agent: {e}", exc_info=True)
            raise
    
    # Should never reach here, but just in case
    if last_error:
        raise last_error
    raise RuntimeError("Failed to run agent after retries")
