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

# Token and context stats (for UI)
CHARS_PER_TOKEN = 4  # rough: ~4 chars per token for English
TOKENS_PER_BIBLE = 800_000  # typical Bible ~800k tokens
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


def format_bible_equivalent(tokens: int) -> str:
    """e.g. '0.12 Bibles' or '~12% of a Bible'."""
    if tokens <= 0:
        return "0 Bibles"
    ratio = tokens / TOKENS_PER_BIBLE
    if ratio >= 1:
        return f"~{ratio:.1f} Bibles"
    pct = ratio * 100
    if pct >= 1:
        return f"~{pct:.0f}% of a Bible"
    return f"~{pct:.1f}% of a Bible"


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


def embed_query(text: str) -> list[float]:
    """Embed a single query for retrieval. Uses RETRIEVAL_QUERY task type. Retries on 429."""
    import math
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
    return v


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
    logger.debug(f"Building cache config with model {DEFAULT_MODEL}")
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
            cache = client.caches.create(model=DEFAULT_MODEL, config=config)
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
    """
    if not files:
        return []
    client = _client()
    m = DEFAULT_MODEL
    all_summaries: list[str] = []
    for b in range(0, len(files), batch_size):
        batch = files[b : b + batch_size]
        if progress_callback:
            try:
                progress_callback("summarize", min(b + batch_size, len(files)), len(files), None)
            except Exception:
                pass
        parts: list[str] = []
        for i, f in enumerate(batch):
            name = f.get("name") or "Untitled"
            text = (f.get("text") or "").strip()
            if len(text) > max_chars_per_file:
                text = text[:max_chars_per_file] + "..."
            parts.append(f"Document {i + 1} (filename: {name}):\n\n{text or '(empty)'}")
        prompt = (
            "For each of the following documents, provide exactly one short sentence (max 30 words) "
            "summarizing its contents. Respond with valid JSON only:\n"
            '{"summaries": ["summary 1", "summary 2", ...]}\n'
            "Use the same order as the documents. No other text.\n\n"
            + "\n\n---\n\n".join(parts)
        )
        config = types.GenerateContentConfig()
        contents = [types.Content(role="user", parts=[_text_part(prompt)])]
        delay = 1.0
        for attempt in range(3):
            try:
                r = client.models.generate_content(model=m, contents=contents, config=config)
                text = (r.text or "").strip()
                parsed = _parse_json_robust(text)
                summaries = parsed.get("summaries") if isinstance(parsed, dict) else None
                if isinstance(summaries, list) and len(summaries) >= len(batch):
                    all_summaries.extend(summaries[: len(batch)])
                    break
                if isinstance(summaries, list):
                    for s in summaries[: len(batch)]:
                        all_summaries.append((s[:200] if isinstance(s, str) else str(s)))
                    while len(all_summaries) < b + len(batch):
                        all_summaries.append("(summary unavailable)")
                    break
                logger.warning("Summarizer response invalid, using fallback")
                for _ in batch:
                    all_summaries.append("(summary unavailable)")
                break
            except (errors.ServerError, errors.ClientError) as e:
                if attempt < 2:
                    wait = _parse_429_retry_delay(e) or delay
                    wait = max(wait, delay)
                    logger.warning(f"Summarizer error, retrying in {wait:.0f}s: {e}")
                    time.sleep(wait)
                    delay *= 2
                else:
                    for _ in batch:
                        all_summaries.append("(summary unavailable)")
                    logger.warning(f"Summarizer failed after retries: {e}")
                    break
            except Exception as e:
                for _ in batch:
                    all_summaries.append("(summary unavailable)")
                logger.warning(f"Summarizer error: {e}")
                break
        if b + batch_size < len(files):
            time.sleep(0.3)
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
    """
    if not file_descriptors:
        return f"Library '{library_name}' has no files."
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
    config = types.GenerateContentConfig()
    contents = [types.Content(role="user", parts=[_text_part(prompt)])]
    try:
        r = client.models.generate_content(model=DEFAULT_MODEL, contents=contents, config=config)
        return (r.text or "").strip() or f"Library '{library_name}' with {len(file_descriptors)} files."
    except Exception as e:
        logger.warning(f"describe_library failed: {e}")
        return f"Library '{library_name}' with {len(file_descriptors)} files."


# -----------------------------------------------------------------------------
# Retrieval planner (preprocessor)
# -----------------------------------------------------------------------------

RETRIEVAL_PLANNER_PROMPT = """You are a retrieval planner for a RAG system. Given an analysis task and the documents to be analyzed, decide which knowledge-base libraries to search and how many chunks to retrieve from each.

{{ library_catalog }}

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
) -> dict[str, Any] | None:
    """
    LLM call (no KB cache) to decide which libraries to search and top_k per library.
    library_metadata: list of {name, library_description?, file_descriptors?: [{name, summary}]}.
    Returns {"libraries": [{"name": str, "top_k": int}, ...]} or None on failure.
    """
    if not library_metadata:
        logger.debug("No libraries available for retrieval planner")
        return {"libraries": []}
    client = _client()
    m = model or DEFAULT_MODEL
    catalog = _build_library_catalog(library_metadata)
    prompt = (
        RETRIEVAL_PLANNER_PROMPT.strip()
        .replace("{{ library_catalog }}", catalog)
        .replace("{{ task_name }}", task_name)
        .replace("{{ task_description }}", (task_description or "")[:2500])
        .replace("{{ user_content }}", (user_content or "")[:3500])
    )
    config = types.GenerateContentConfig()
    contents = [types.Content(role="user", parts=[_text_part(prompt)])]
    delay = 1.0
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Running retrieval planner (attempt {attempt + 1}/{max_retries + 1})")
            r = client.models.generate_content(model=m, contents=contents, config=config)
            text = (r.text or "").strip()
            parsed = _parse_json_robust(text)
            if parsed and isinstance(parsed.get("libraries"), list):
                logger.info(f"Planner chose {len(parsed['libraries'])} libraries")
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
# Agent
# -----------------------------------------------------------------------------


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
    max_retries: int = 3,
    initial_delay: float = 1.0,
    context_xml: str | None = None,
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
    m = model or DEFAULT_MODEL
    logger.debug(f"Using model: {m}")
    
    # Build content - either use cache or include context directly
    if cache_name:
        # Use cached content (preferred - faster and cheaper)
        config = types.GenerateContentConfig(cached_content=cache_name)
        prompt_content = types.Content(
            role="user",
            parts=[_text_part(prompt)],
        )
        contents = [prompt_content]
    else:
        # Fallback: include context directly (works but slower/expensive)
        if not context_xml:
            raise ValueError("context_xml is required when cache_name is None (fallback mode)")
        logger.warning("Running without cache - including full context in prompt (slower and more expensive)")
        # Combine context and prompt
        full_prompt = f"{context_xml}\n\n---\n\n{prompt}"
        config = types.GenerateContentConfig()
        prompt_content = types.Content(
            role="user",
            parts=[_text_part(full_prompt)],
        )
        contents = [prompt_content]
    
    last_error = None
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
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
                
                # Use API-suggested delay or fallback to exponential backoff
                if retry_delay is None:
                    retry_delay = delay
                else:
                    # Add a small buffer to the API-suggested delay
                    retry_delay = max(retry_delay + 1.0, delay)
                
                if attempt < max_retries:
                    logger.warning(
                        f"Quota exceeded (429) on attempt {attempt + 1}/{max_retries + 1}. "
                        f"Retrying in {retry_delay:.1f} seconds (API suggested delay)..."
                    )
                    time.sleep(retry_delay)
                    delay = retry_delay * 1.5  # Increase delay for next attempt if needed
                else:
                    logger.error(f"Quota exceeded after {max_retries + 1} attempts. Last error: {e}")
                    # Provide helpful error message
                    quota_msg = (
                        f"**Quota Exceeded (429 RESOURCE_EXHAUSTED)**\n\n"
                        f"You've exceeded your per-minute token limit for gemini-2.0-flash.\n\n"
                        f"**Solutions:**\n"
                        f"1. **Wait a few minutes** - Quotas reset per minute\n"
                        f"2. **Reduce context size** - Your knowledge base is very large ({len(context_xml) if context_xml else 'unknown'} chars)\n"
                        f"3. **Use caching** - Enable cache mode (disable fallback mode) to reduce token usage\n"
                        f"4. **Request quota increase** - Go to Google Cloud Console → APIs & Services → Quotas\n"
                        f"5. **Check usage**: https://ai.dev/rate-limit\n\n"
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
