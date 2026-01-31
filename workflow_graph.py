"""
Phase 3: LangGraph-based analysis workflow.

Same flow as workflow.py but orchestrated by a StateGraph:
plan_retrieval → retrieve_context → create_cache → main_agent → extract_legal_questions
→ [conditional: legal_questions?] → legal_agent → integrate_legal  OR  integrate_legal
→ follow_on_chain → finalize → END.

State is wrapped as {"data": workflow_state_dict} so existing workflow steps can be reused.
WorkflowError from any step propagates to the caller (app).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None  # type: ignore[misc, assignment]

import db
from brain import (
    chars_to_tokens,
    get_effective_model,
    model_max_context,
)
from rag_loader import get_cached_rag_state
from workflow import (
    WorkflowError,
    create_cache_step,
    extract_legal_questions_step,
    integrate_legal_step,
    plan_retrieval_step,
    retrieve_context_step,
    run_follow_on_chain_step,
    run_legal_expert_step,
    run_main_agent_step,
    run_qa_step,
)
import runs_db

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Graph state: single key "data" holding the workflow state dict (Phase 2 schema).
# -----------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    data: dict[str, Any]


def _data(state: GraphState) -> dict[str, Any]:
    return state["data"]


# -----------------------------------------------------------------------------
# Config: callbacks and build_prompt_variables passed via config (not state) for serializability.
# Resolve selected_prompt and rag_state from prompt_template_id / folder_id in each node.
# -----------------------------------------------------------------------------

_NON_SERIALIZABLE_KEYS = ("_callbacks", "selected_prompt", "rag_state", "build_prompt_variables")


def _configurable(config: Any) -> dict[str, Any]:
    """Extract configurable dict from LangGraph config (dict or RunnableConfig)."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return config.get("configurable") or {}
    return getattr(config, "configurable", {})


def _default_build_prompt_variables(username: str = "", user_name: str = "") -> str:
    """Fallback when config does not provide build_prompt_variables (e.g. after checkpoint restore)."""
    return "\n\n**Context Variables**\n\n- **Analysis Performed By**: " + (user_name or username or "unknown")


def _prepare_step_state(d: dict[str, Any], config: Any) -> dict[str, Any]:
    """Inject callbacks, build_prompt_variables, selected_prompt, rag_state into d from config/DB. Returns callbacks."""
    conf = _configurable(config)
    callbacks = conf.get("callbacks") or {}
    build_prompt_variables = conf.get("build_prompt_variables")
    # Ensure steps never see None (config can omit callables when checkpointing strips non-serializable data)
    d["_callbacks"] = callbacks
    d["build_prompt_variables"] = build_prompt_variables if callable(build_prompt_variables) else _default_build_prompt_variables
    pid = d.get("prompt_template_id")
    d["selected_prompt"] = db.get_prompt_by_id(pid) if pid else None
    fid = d.get("folder_id")
    d["rag_state"] = get_cached_rag_state(fid, _progress_callback=None) if fid else None
    return callbacks


def _strip_non_serializable(d: dict[str, Any]) -> None:
    """Remove keys that must not be persisted (callables, ORM, heavy RAG state)."""
    for k in _NON_SERIALIZABLE_KEYS:
        d.pop(k, None)


def _log_run_event(config: Any, step_name: str, event_type: str, payload: str | None = None) -> None:
    """P5.3: Log node_started/node_completed via configurable callback (avoids importing runs_db in graph)."""
    conf = _configurable(config)
    fn = conf.get("log_run_event")
    if fn and callable(fn):
        try:
            fn(step_name, event_type, payload=payload)
        except Exception as e:
            logger.debug("log_run_event failed: %s", e)


# -----------------------------------------------------------------------------
# Node wrappers: each calls the corresponding workflow step; steps mutate state in place.
# -----------------------------------------------------------------------------


def _status_write(d: dict, msg: str) -> None:
    cbs = d.get("_callbacks") or {}
    fn = cbs.get("write")
    if fn and callable(fn):
        try:
            fn(msg)
        except Exception:
            pass


def _node_plan_retrieval(state: GraphState, config: RunnableConfig | None = None) -> GraphState:
    _log_run_event(config, "plan_retrieval", "node_started")
    d = _data(state)
    callbacks = _prepare_step_state(d, config)
    d["status"] = "running"
    _status_write(d, "📋 Planning retrieval…")
    plan_retrieval_step(d, callbacks)
    _status_write(d, "✅ Plan retrieval done.")
    _strip_non_serializable(d)
    _log_run_event(config, "plan_retrieval", "node_completed")
    return {"data": d}


def _node_retrieve_context(state: GraphState, config: RunnableConfig | None = None) -> GraphState:
    _log_run_event(config, "retrieve_context", "node_started")
    d = _data(state)
    callbacks = _prepare_step_state(d, config)
    _status_write(d, "📚 Retrieving context from knowledge base…")
    retrieve_context_step(d, callbacks)
    _status_write(d, "✅ Context retrieved.")
    _strip_non_serializable(d)
    _log_run_event(config, "retrieve_context", "node_completed")
    return {"data": d}


def _node_create_cache(state: GraphState, config: Any = None) -> GraphState:
    _log_run_event(config, "create_cache", "node_started")
    d = _data(state)
    callbacks = _prepare_step_state(d, config)
    _status_write(d, "💾 Creating/reusing Gemini context cache…")
    create_cache_step(d, callbacks)
    _status_write(d, "✅ Cache ready.")
    _strip_non_serializable(d)
    _log_run_event(config, "create_cache", "node_completed")
    return {"data": d}


def _node_main_agent(state: GraphState, config: RunnableConfig | None = None) -> GraphState:
    _log_run_event(config, "main_agent", "node_started")
    d = _data(state)
    callbacks = _prepare_step_state(d, config)
    _status_write(d, "🤖 Running main agent…")
    run_main_agent_step(d, callbacks)
    _status_write(d, "✅ Main agent done.")
    _strip_non_serializable(d)
    _log_run_event(config, "main_agent", "node_completed")
    return {"data": d}


def _node_extract_legal_questions(state: GraphState, config: Any = None) -> GraphState:
    _log_run_event(config, "extract_legal_questions", "node_started")
    d = _data(state)
    callbacks = _prepare_step_state(d, config)
    _status_write(d, "⚖️ Extracting legal questions…")
    extract_legal_questions_step(d, callbacks)
    _status_write(d, "✅ Legal questions extracted.")
    _strip_non_serializable(d)
    _log_run_event(config, "extract_legal_questions", "node_completed")
    return {"data": d}


def _node_legal_agent(state: GraphState, config: RunnableConfig | None = None) -> GraphState:
    _log_run_event(config, "legal_agent", "node_started")
    d = _data(state)
    callbacks = _prepare_step_state(d, config)
    _status_write(d, "⚖️ Running legal expert…")
    run_legal_expert_step(d, callbacks)
    _status_write(d, "🔗 Integrating legal output…")
    integrate_legal_step(d, callbacks)
    _status_write(d, "✅ Legal agent done.")
    _strip_non_serializable(d)
    _log_run_event(config, "legal_agent", "node_completed")
    return {"data": d}


def _node_integrate_legal(state: GraphState, config: Any = None) -> GraphState:
    _log_run_event(config, "integrate_legal", "node_started")
    d = _data(state)
    callbacks = _prepare_step_state(d, config)
    _status_write(d, "🔗 Integrating legal (no expert run).")
    integrate_legal_step(d, callbacks)
    _strip_non_serializable(d)
    _log_run_event(config, "integrate_legal", "node_completed")
    return {"data": d}


def _node_follow_on_chain(state: GraphState, config: RunnableConfig | None = None) -> GraphState:
    _log_run_event(config, "follow_on_chain", "node_started")
    d = _data(state)
    callbacks = _prepare_step_state(d, config)
    _status_write(d, "🔁 Running follow-on chain…")
    run_follow_on_chain_step(d, callbacks)
    _status_write(d, "✅ Follow-on chain done.")
    _strip_non_serializable(d)
    _log_run_event(config, "follow_on_chain", "node_completed")
    return {"data": d}


def _node_qa_agent(state: GraphState, config: Any = None) -> GraphState:
    _log_run_event(config, "qa_agent", "node_started")
    d = _data(state)
    callbacks = _prepare_step_state(d, config)
    _status_write(d, "📝 Running QA agent (review and polish)…")
    run_qa_step(d, callbacks)
    _status_write(d, "✅ QA agent done.")
    _strip_non_serializable(d)
    _log_run_event(config, "qa_agent", "node_completed")
    return {"data": d}


def _node_finalize(state: GraphState, config: RunnableConfig | None = None) -> GraphState:
    """Set last_run_context_stats and status=completed (same as workflow.run_workflow)."""
    _log_run_event(config, "finalize", "node_started")
    d = _data(state)
    context_xml = d.get("context_xml") or ""
    user_content = d.get("user_content") or ""
    template_text = d.get("template_text") or ""
    final_output = d.get("final_output") or ""
    kb_tokens = chars_to_tokens(len(context_xml))
    transient_tokens = chars_to_tokens(len(user_content))
    prompt_tokens = chars_to_tokens(len(template_text) + 80)
    total_input_tokens = kb_tokens + transient_tokens + prompt_tokens
    d["last_run_context_stats"] = {
        "total_input_tokens": total_input_tokens,
        "kb_tokens": kb_tokens,
        "transient_tokens": transient_tokens,
        "prompt_tokens": prompt_tokens,
        "max_context": model_max_context(get_effective_model()),
        "output_tokens": chars_to_tokens(len(final_output)),
        "model": get_effective_model(),
        "timings": d.get("timings", {}),
        "gemini_calls": d.get("gemini_call_count", 0),
    }
    d["status"] = "completed"
    _log_run_event(config, "finalize", "node_completed")
    return {"data": d}


# -----------------------------------------------------------------------------
# Routing: after extract_legal_questions, route to legal_agent or integrate_legal.
# -----------------------------------------------------------------------------


def _route_after_legal_extract(state: GraphState) -> Literal["legal_agent", "integrate_legal"]:
    d = _data(state)
    legal_questions = d.get("legal_questions") or []
    pid = d.get("prompt_template_id")
    selected = db.get_prompt_by_id(pid) if pid else None
    legal_expert_prompt_id = getattr(selected, "legal_expert_prompt_id", None) if selected else None
    if legal_questions and legal_expert_prompt_id:
        return "legal_agent"
    return "integrate_legal"


def _route_after_follow_on(state: GraphState) -> Literal["qa_agent", "finalize"]:
    """Phase 4: if task has use_qa_agent, run QA; else go to finalize."""
    d = _data(state)
    pid = d.get("prompt_template_id")
    selected = db.get_prompt_by_id(pid) if pid else None
    if selected and getattr(selected, "use_qa_agent", False):
        return "qa_agent"
    return "finalize"


# -----------------------------------------------------------------------------
# Build and compile graph (D6 = A: checkpointer to council_runs.db when serializable).
# -----------------------------------------------------------------------------


# Hold context manager and saver for process lifetime (from_conn_string returns a context manager).
_sqlite_checkpointer_context = None
_sqlite_checkpointer_instance = None


def _get_checkpointer():
    """SqliteSaver on council_runs.db for audit/resume (D6 = A). Returns None if unavailable."""
    global _sqlite_checkpointer_context, _sqlite_checkpointer_instance
    if SqliteSaver is None:
        return None
    if _sqlite_checkpointer_instance is not None:
        return _sqlite_checkpointer_instance
    try:
        conn_str = runs_db.RUNS_SQLITE_URL
        # from_conn_string returns a context manager; enter once and reuse the saver.
        _sqlite_checkpointer_context = SqliteSaver.from_conn_string(conn_str)
        _sqlite_checkpointer_instance = _sqlite_checkpointer_context.__enter__()
        return _sqlite_checkpointer_instance
    except Exception as e:
        logger.warning("SqliteSaver unavailable, using no checkpointer: %s", e)
        return None


def _compile_analysis_graph(use_sqlite_checkpointer: bool = True):
    """Build and compile the analysis StateGraph. Returns compiled graph."""
    builder = StateGraph(GraphState)

    builder.add_node("plan_retrieval", _node_plan_retrieval)
    builder.add_node("retrieve_context", _node_retrieve_context)
    builder.add_node("create_cache", _node_create_cache)
    builder.add_node("main_agent", _node_main_agent)
    builder.add_node("extract_legal_questions", _node_extract_legal_questions)
    builder.add_node("legal_agent", _node_legal_agent)
    builder.add_node("integrate_legal", _node_integrate_legal)
    builder.add_node("follow_on_chain", _node_follow_on_chain)
    builder.add_node("qa_agent", _node_qa_agent)
    builder.add_node("finalize", _node_finalize)

    builder.add_edge(START, "plan_retrieval")
    builder.add_edge("plan_retrieval", "retrieve_context")
    builder.add_edge("retrieve_context", "create_cache")
    builder.add_edge("create_cache", "main_agent")
    builder.add_edge("main_agent", "extract_legal_questions")
    builder.add_conditional_edges(
        "extract_legal_questions",
        _route_after_legal_extract,
        {"legal_agent": "legal_agent", "integrate_legal": "integrate_legal"},
    )
    builder.add_edge("legal_agent", "follow_on_chain")
    builder.add_edge("integrate_legal", "follow_on_chain")
    builder.add_conditional_edges(
        "follow_on_chain",
        _route_after_follow_on,
        {"qa_agent": "qa_agent", "finalize": "finalize"},
    )
    builder.add_edge("qa_agent", "finalize")
    builder.add_edge("finalize", END)

    checkpointer = _get_checkpointer() if use_sqlite_checkpointer else None
    return builder.compile(checkpointer=checkpointer)


# Compiled graph singleton (lazy). Keyed by (workflow_key, use_sqlite_checkpointer).
_analysis_graph_cache: dict[tuple[str, bool], Any] = {}

DEFAULT_WORKFLOW_KEY = "default"


def get_analysis_graph(
    workflow_key: str = DEFAULT_WORKFLOW_KEY,
    use_sqlite_checkpointer: bool = True,
):
    """Return the compiled analysis graph for the given workflow (cached per workflow_key + checkpointer).
    Phase 5: workflow_key selects which graph to run (e.g. 'default', 'agenda_review').
    For now 'agenda_review' uses the same graph as 'default'; can be split later.
    """
    global _analysis_graph_cache
    key = (workflow_key, use_sqlite_checkpointer)
    if key not in _analysis_graph_cache:
        # Same graph for all workflow keys for now; can branch on workflow_key later
        _analysis_graph_cache[key] = _compile_analysis_graph(
            use_sqlite_checkpointer=use_sqlite_checkpointer
        )
    return _analysis_graph_cache[key]


def _use_graph_checkpointer_env() -> bool:
    """True unless COUNCILFLOW_USE_GRAPH_CHECKPOINTER is set to 0/false/no (D6 = A). Default: on."""
    v = (os.environ.get("COUNCILFLOW_USE_GRAPH_CHECKPOINTER") or "").strip().lower()
    if v in ("0", "false", "no"):
        return False
    return True


def run_analysis_graph(
    initial_state: dict[str, Any],
    *,
    workflow_key: str = DEFAULT_WORKFLOW_KEY,
    thread_id: str | None = None,
    use_sqlite_checkpointer: bool | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run the LangGraph analysis workflow. Raises WorkflowError on failure.
    Returns the final workflow state dict (same shape as workflow.run_workflow).
    workflow_key: which workflow graph to run (Phase 5; from prompt's workflow).
    use_sqlite_checkpointer: if True, use council_runs.db for checkpoints (D6).
    If None, follows env COUNCILFLOW_USE_GRAPH_CHECKPOINTER (1/true = on).
    config: optional run config. configurable.callbacks, build_prompt_variables, log_run_event
    are passed to nodes for serializable checkpointing.
    """
    if use_sqlite_checkpointer is None:
        use_sqlite_checkpointer = _use_graph_checkpointer_env()
    graph = get_analysis_graph(
        workflow_key=workflow_key,
        use_sqlite_checkpointer=use_sqlite_checkpointer,
    )
    run_config: dict[str, Any] = dict(config) if config else {}
    run_config.setdefault("configurable", {})
    if thread_id:
        run_config["configurable"]["thread_id"] = thread_id

    inputs: GraphState = {"data": initial_state}
    result = graph.invoke(inputs, config=run_config)
    return result["data"]


# -----------------------------------------------------------------------------
# Verify graph compiles (run: python -m workflow_graph)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    g = get_analysis_graph(workflow_key=DEFAULT_WORKFLOW_KEY, use_sqlite_checkpointer=False)
    print("Graph compiled OK (no checkpointer)")
    if SqliteSaver is not None:
        cp = _get_checkpointer()
        if cp is not None:
            _analysis_graph_cache.clear()
            _analysis_graph_cache[(DEFAULT_WORKFLOW_KEY, True)] = _compile_analysis_graph(use_sqlite_checkpointer=True)
            print("Graph compiled OK (with SqliteSaver)")
        else:
            print("SqliteSaver requested but unavailable (e.g. DB path); graph uses no checkpointer.")
    print("Done.")
