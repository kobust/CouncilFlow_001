"""
Deep-link share viewer: render a shared run without requiring login.
Used by app.py when URL has ?v=TOKEN and token is valid.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import streamlit as st

import db
import runs_db

if TYPE_CHECKING:
    from runs_db import AnalysisRun

logger = logging.getLogger(__name__)


def _markdown_with_copy(md: str, key_suffix: str) -> None:
    """Render markdown and show a Copy block."""
    st.markdown(md)
    with st.expander(f"📋 Copy markdown ({key_suffix})", expanded=False):
        st.caption("Use the copy icon in the code block below to copy the markdown.")
        st.code(md, language="markdown", line_numbers=False)


def _get_json_view_options(schema_key: str | None = None) -> list[tuple[str, str]]:
    """Return [(value, label), ...] for JSON view selector."""
    options: list[tuple[str, str]] = [("saved", "Saved"), ("raw", "Raw JSON")]
    if not schema_key:
        return options
    try:
        import output_schemas
        output_schemas.ensure_registry_loaded()
        for display_name, _ in output_schemas.get_transformers(schema_key):
            composite = f"{schema_key}::{display_name}"
            options.append((composite, f"{schema_key} :: {display_name}"))
    except Exception as e:
        logger.debug("get_json_view_options: %s", e)
    return options


def _render_json_view(
    output_json_str: str | None,
    output_text_saved: str,
    view_choice: str,
) -> str:
    """Return the string to display for the chosen view."""
    if view_choice == "saved":
        return output_text_saved or "(no output)"
    if view_choice == "raw":
        if not output_json_str:
            return "(raw JSON not stored for this run)"
        try:
            parsed = json.loads(output_json_str)
            return json.dumps(parsed, indent=2)
        except Exception:
            return output_json_str
    if not output_json_str:
        return "(raw JSON not stored; cannot re-run transformer)"
    try:
        import output_schemas
        output_schemas.ensure_registry_loaded()
        data = json.loads(output_json_str)
        return output_schemas.run_transformer(view_choice, data)
    except Exception as e:
        return f"Transformer failed: {e}"


def render_share_viewer(run: AnalysisRun) -> None:
    """Render the public deep-link viewer for a shared run (no login required)."""
    st.markdown(
        '<meta name="robots" content="noindex, nofollow">',
        unsafe_allow_html=True,
    )
    title = (getattr(run, "share_title", None) or "").strip() or run.task_name
    st.subheader(f"Shared analysis: {title}")
    st.caption("Anyone with this link can view this content. Do not share if it contains confidential or PII.")
    st.markdown("---")
    pre_qa = getattr(run, "pre_qa_output", None)
    qa_out = getattr(run, "qa_output", None)
    if pre_qa or qa_out:
        tab_labels = ["Output"]
        if pre_qa:
            tab_labels.append("Pre-QA")
        if qa_out:
            tab_labels.append("Post-QA")
        tabs = st.tabs(tab_labels)
        with tabs[0]:
            _markdown_with_copy(run.output_text or "(no output)", f"share_{run.id}_output")
        idx = 1
        if pre_qa:
            with tabs[idx]:
                _markdown_with_copy(pre_qa, f"share_{run.id}_pre_qa")
            idx += 1
        if qa_out:
            with tabs[idx]:
                _markdown_with_copy(qa_out, f"share_{run.id}_post_qa")
    else:
        st.markdown("#### Output")
        out_json = getattr(run, "output_json", None)
        if out_json and run.prompt_template_id:
            try:
                prompt = db.get_prompt_by_id(run.prompt_template_id)
                schema_key = getattr(prompt, "output_schema_key", None) if prompt else None
            except Exception:
                schema_key = None
            if schema_key:
                view_opts = _get_json_view_options(schema_key=schema_key)
                view_labels = [lbl for _, lbl in view_opts]
                view_keys = [val for val, _ in view_opts]
                sh_key = f"share_view_{run.id}"
                idx = view_keys.index(st.session_state.get(sh_key, "saved")) if st.session_state.get(sh_key, "saved") in view_keys else 0
                view_choice = st.selectbox("View as", options=view_labels, index=idx, key=sh_key + "_select")
                st.session_state[sh_key] = view_keys[view_labels.index(view_choice)]
                display_str = _render_json_view(out_json, run.output_text or "", st.session_state[sh_key])
                _markdown_with_copy(display_str, f"share_{run.id}_output")
            else:
                _markdown_with_copy(run.output_text or "(no output)", f"share_{run.id}_output")
        else:
            _markdown_with_copy(run.output_text or "(no output)", f"share_{run.id}_output")
    if st.session_state.get("authentication_status"):
        st.sidebar.markdown("#### **Navigation**")
        if st.sidebar.button("▶️ Run Analysis", key="share_nav_run", use_container_width=True):
            st.query_params.clear()
            st.session_state["current_page"] = "runner"
            st.rerun()
        if st.session_state.get("username") == "admin":
            if st.sidebar.button("📋 Run history", key="share_nav_history", use_container_width=True):
                st.query_params.clear()
                st.session_state["current_page"] = "run_history"
                st.rerun()
        st.sidebar.divider()
