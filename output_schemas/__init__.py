"""
Registry for JSON output schemas and their Markdown transformers.

Schema bundles (schema + transformers) are defined in code and registered by key.
Transformers are pure functions (data: Any) -> str that return Markdown.
Composite transformer key format: "schema_key::display_name".
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

JsonToMarkdown = Callable[[Any], str]

# schema_key -> { "schema_json": str, "transformers": [(display_name, fn), ...] }
SCHEMA_REGISTRY: dict[str, dict[str, Any]] = {}

# Composite key "schema_key::display_name" -> (schema_key, fn)
_TRANSFORMER_BY_KEY: dict[str, tuple[str, JsonToMarkdown]] = {}


def register_schema(
    key: str,
    schema_json: dict | str,
    transformers: list[tuple[str, JsonToMarkdown]],
) -> None:
    """Register a schema bundle. schema_json can be dict (will be JSON-stringified for Gemini)."""
    schema_str = json.dumps(schema_json) if isinstance(schema_json, dict) else schema_json
    SCHEMA_REGISTRY[key] = {
        "schema_json": schema_str,
        "transformers": list(transformers),
    }
    for display_name, fn in transformers:
        composite = f"{key}::{display_name}"
        _TRANSFORMER_BY_KEY[composite] = (key, fn)
    logger.debug("Registered schema bundle %s with %d transformers", key, len(transformers))


def get_schema_json(key: str) -> str | None:
    """Return JSON Schema string for Gemini, or None if key unknown."""
    bundle = SCHEMA_REGISTRY.get(key)
    if not bundle:
        return None
    return bundle.get("schema_json")


def get_transformers(key: str) -> list[tuple[str, JsonToMarkdown]]:
    """Return list of (display_name, fn) for this schema key. Empty if key unknown."""
    bundle = SCHEMA_REGISTRY.get(key)
    if not bundle:
        return []
    return list(bundle.get("transformers", []))


def get_all_transformer_keys() -> list[tuple[str, str]]:
    """Return all (composite_key, display_label) for UI dropdowns. display_label = schema_key :: display_name."""
    out: list[tuple[str, str]] = []
    for key, bundle in SCHEMA_REGISTRY.items():
        for display_name, _ in bundle.get("transformers", []):
            composite = f"{key}::{display_name}"
            out.append((composite, f"{key} :: {display_name}"))
    return sorted(out, key=lambda x: x[1].lower())


def run_transformer(composite_key: str, data: Any) -> str:
    """
    Run a transformer by composite key on parsed JSON. Returns Markdown or an error message.
    Never raises; on failure returns a short error string.
    """
    entry = _TRANSFORMER_BY_KEY.get(composite_key)
    if not entry:
        return f"Unknown transformer: {composite_key}"
    _, fn = entry
    try:
        return fn(data)
    except Exception as e:
        logger.warning("Transformer %s failed: %s", composite_key, e)
        return f"Transformer failed: {e}"


def get_registry_schema_keys() -> list[str]:
    """Return all registered schema keys (for schema editor dropdown)."""
    return sorted(SCHEMA_REGISTRY.keys(), key=str.lower)


def _load_bundles() -> None:
    """Import all bundle modules so they call register_schema. Called once at app startup."""
    from output_schemas import motions  # noqa: F401

    motions.load()


_loaded = False


def ensure_registry_loaded() -> None:
    """Call from app when schemas/transformers are needed. Idempotent."""
    global _loaded
    if not _loaded:
        _load_bundles()
        _loaded = True
