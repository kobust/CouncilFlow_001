"""
Analysis workflow: plan → retrieve → cache → main agent → legal expert (optional) → follow-on chain.

Phase 2: Single workflow definition executed by a small runner. State is a dict passed through steps.
No Streamlit dependency; app passes optional callbacks for progress. Steps raise WorkflowError on failure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from typing import Any, Callable

import db
from brain import (
    CacheExpiredError,
    chars_to_tokens,
    create_gemini_cache,
    extract_legal_questions,
    get_effective_model,
    model_max_context,
    run_agent,
)
from rag_loader import (
    USE_QUERY_EXPANSION,
    USE_RETRIEVAL_PLANNER,
    get_default_plan,
    get_fallback_phrases,
    plan_retrieval,
    retrieve_and_build_context_multi,
)

logger = logging.getLogger(__name__)

MIN_CONTEXT_SIZE = 16000  # Same as app: too small for Gemini cache

# Collapse 3+ consecutive newlines to 2 (paragraph break); avoids model output with repeated \n.
_NL_COLLAPSE = re.compile(r"\n{3,}")


def _collapse_repeated_newlines(obj: Any) -> Any:
    """Recursively collapse 3+ consecutive newlines to \\n\\n in string values (in-place for dict/list)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = _collapse_repeated_newlines(v)
        return obj
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = _collapse_repeated_newlines(v)
        return obj
    if isinstance(obj, str):
        return _NL_COLLAPSE.sub("\n\n", obj)
    return obj


def _get_input_schema_json(prompt: Any) -> str | None:
    """Resolve input schema JSON from prompt's input_schema_key (code registry only)."""
    sk = getattr(prompt, "input_schema_key", None)
    if sk:
        try:
            import output_schemas
            output_schemas.ensure_registry_loaded()
            return output_schemas.get_schema_json(sk)
        except Exception:
            return None
    return None


def _get_output_schema_json(prompt: Any) -> str | None:
    """Resolve output schema JSON from prompt's output_schema_key (code registry only)."""
    sk = getattr(prompt, "output_schema_key", None)
    if sk:
        try:
            import output_schemas
            output_schemas.ensure_registry_loaded()
            return output_schemas.get_schema_json(sk)
        except Exception:
            return None
    return None

# Env-based delays (same as app). Default 0 for faster runs; set to 5-10 if hitting rate limits.
PIPELINE_STEP_DELAY_SECONDS = int(os.environ.get("PIPELINE_STEP_DELAY_SECONDS", "0") or "0")
LEGAL_EXPERT_DELAY_SECONDS = int(os.environ.get("LEGAL_EXPERT_DELAY_SECONDS", "0") or "0")
GEMINI_PACE_DELAY_SECONDS = 0
try:
    from brain import GEMINI_PACE_DELAY_SECONDS
except ImportError:
    pass


class WorkflowError(Exception):
    """Raised when a workflow step fails. Message is shown to the user."""

    def __init__(self, message: str, details: str | None = None):
        self.message = message
        self.details = details
        super().__init__(message)


# -----------------------------------------------------------------------------
# State schema (dict keys; all optional except inputs set by app)
# -----------------------------------------------------------------------------
# Inputs: task_name, template_text, user_content, folder_id, prompt_template_id, username, user_name,
#         rag_state, selected_prompt (ORM object), prompt_variables (str), main_full_template (str),
#         input_schema_json, output_schema_json, build_prompt_variables (callable),
#         session_cache_name, session_cache_model, session_cache_folder_id, session_run_cache_key
# RAG: selected_library_ids, top_k_map, context_xml, cache_name, retrieval_report, query_phrases,
#      legal_context_xml, legal_cache_name (for legal expert step)
# Outputs: main_output, legal_questions, legal_expert_output, final_output, chain (list of (name, output)),
#          chain_outputs (list of {step_name, output}), output_mode, timings,
#          pipeline_step_results, legal_questions_by_step, last_run_context_stats, chain_timings
#          pre_qa_output (Phase 4: final_output before QA), qa_output (Phase 4: QA result)
# Control: error, status ('running'|'completed'|'failed')
# Session update (for app): run_cache_key, cache_name, cache_folder_id, cache_model
# -----------------------------------------------------------------------------

Callbacks = dict[str, Any]  # write, update_label, status_container, etc.; all optional


def _cb(state: dict, key: str, *args: Any, **kwargs: Any) -> None:
    cbs = state.get("_callbacks") or {}
    fn = cbs.get(key)
    if fn and callable(fn):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.debug(f"Callback {key} failed: {e}")


# -----------------------------------------------------------------------------
# Legal tracking instructions (same text as app for main and follow-on)
# -----------------------------------------------------------------------------

LEGAL_TRACKING_MAIN = """

---

**IMPORTANT - Legal Question Tracking:** 

You must actively review this analysis for unresolved legal questions that are **substantive, relevant, and would materially influence your recommendations or conclusions**.

**Quality Over Quantity:** Only identify legal questions that meet ALL of the following criteria:
1. **Substantive**: The question addresses a real legal issue, not a minor procedural detail or obvious matter
2. **Relevant**: The question directly relates to the subject matter being analyzed
3. **Material**: The answer to the question would meaningfully change your analysis, recommendations, or conclusions

**What to Look For:**
- Ambiguous statutory language that requires interpretation for this specific situation
- Compliance obligations or deadlines that are unclear or potentially applicable
- Legal risks or liabilities that could significantly impact recommendations
- Regulatory requirements that may conflict or need clarification
- Procedural requirements that could invalidate or delay proposed actions
- Legal precedents or case law that might affect the analysis

**What NOT to Include:**
- Questions that are purely informational or already answered in your analysis
- Hypothetical scenarios unrelated to the current situation
- Minor procedural details that don't affect the substance of recommendations
- Questions that can be answered with general legal knowledge

**Output Format:** At the very end of your response, if you identified ANY legal questions that meet the criteria above, add a section with this exact title:

## Legal Questions Requiring Expert Review

Then list each question as a clear, specific bullet point:
- [Your first substantive legal question, phrased as a specific question that would influence the analysis]
- [Your second substantive legal question, if applicable]

Each question should be:
- Specific and actionable (not vague or general)
- Directly relevant to the analysis
- Formulated such that an answer would materially inform your recommendations

If you found NO legal questions that meet these criteria, do NOT include this section. Simply end your analysis normally.

Legal questions will be automatically forwarded to a legal expert who will perform a targeted knowledge base search and provide expert guidance that will be integrated into your analysis.
"""

LEGAL_TRACKING_FOLLOWON = """

---

**IMPORTANT - Legal Question Tracking:** 

You must actively review this analysis for unresolved legal questions that are **substantive, relevant, and would materially influence your recommendations or conclusions**.

**Quality Over Quantity:** Only identify legal questions that meet ALL of the following criteria:
1. **Substantive**: The question addresses a real legal issue, not a minor procedural detail or obvious matter
2. **Relevant**: The question directly relates to the subject matter being analyzed
3. **Material**: The answer to the question would meaningfully change your analysis, recommendations, or conclusions
4. **Unresolved**: The question cannot be definitively answered from the knowledge base or general legal knowledge

**What to Look For:**
- Ambiguous statutory language that requires interpretation for this specific situation
- Compliance obligations or deadlines that are unclear or potentially applicable
- Legal risks or liabilities that could significantly impact recommendations
- Regulatory requirements that may conflict or need clarification
- Procedural requirements that could invalidate or delay proposed actions
- Legal precedents or case law that might affect the analysis

**What NOT to Include:**
- Questions that are purely informational or already answered in your analysis
- Hypothetical scenarios unrelated to the current situation
- Minor procedural details that don't affect the substance of recommendations
- Questions that can be answered with general legal knowledge

**Think Critically:** Before listing a legal question, ask yourself:
- "Would the answer to this question change my recommendations?"
- "Is this a real legal uncertainty, or can I infer the answer from context?"
- "Is this question specific enough to be actionable by a legal expert?"

**Output Format:** At the very end of your response, if you identified ANY substantive legal questions that meet the criteria above, add a section with this exact title:

## Legal Questions Requiring Expert Review

Then list each question as a clear, specific bullet point:
- [Your first substantive legal question, phrased as a specific question that would influence the analysis]
- [Your second substantive legal question, if applicable]

Each question should be:
- Specific and actionable (not vague or general)
- Directly relevant to the analysis
- Formulated such that an answer would materially inform your recommendations

If you found NO legal questions that meet these criteria, do NOT include this section. Simply end your analysis normally.

Legal questions will be automatically forwarded to a legal expert who will perform a targeted knowledge base search and provide expert guidance that will be integrated into your analysis.
"""


# -----------------------------------------------------------------------------
# Step 1: Plan retrieval
# -----------------------------------------------------------------------------

def plan_retrieval_step(state: dict, callbacks: Callbacks | None = None) -> None:
    state["_callbacks"] = callbacks or {}
    _rs = state["rag_state"]
    task_name = state["task_name"]
    template_text = state["template_text"]
    user_content = state["user_content"]
    timings = state.setdefault("timings", {})

    if USE_RETRIEVAL_PLANNER:
        _t0 = time.perf_counter()
        sel_ids, top_k_map = plan_retrieval(_rs, task_name, template_text, user_content)
        timings["plan_retrieval_s"] = time.perf_counter() - _t0
        _cb(state, "write", f"✅ Planning done in {timings['plan_retrieval_s']:.2f}s")
    else:
        sel_ids, top_k_map = get_default_plan(_rs)
        timings["plan_retrieval_s"] = 0.0

    state["selected_library_ids"] = sel_ids
    state["top_k_map"] = top_k_map


# -----------------------------------------------------------------------------
# Step 2: Retrieve context
# -----------------------------------------------------------------------------

def retrieve_context_step(state: dict, callbacks: Callbacks | None = None) -> None:
    state["_callbacks"] = callbacks or {}
    _rs = state["rag_state"]
    user_content = state["user_content"]
    template_text = state["template_text"]
    task_name = state["task_name"]
    sel_ids = state["selected_library_ids"]
    top_k_map = state["top_k_map"]
    timings = state.setdefault("timings", {})

    if USE_QUERY_EXPANSION:
        from brain import expand_queries
        query_phrases = expand_queries(task_name, template_text, user_content)
    else:
        query_phrases = get_fallback_phrases(task_name, template_text, user_content)
    state["query_phrases"] = query_phrases

    _cb(state, "write", "Retrieving context from knowledge base…")
    _t0 = time.perf_counter()
    context_xml, retrieval_report = retrieve_and_build_context_multi(_rs, query_phrases, sel_ids, top_k_map)
    timings["build_context_s"] = time.perf_counter() - _t0
    _cb(state, "write", f"✓ Context built in {timings['build_context_s']:.2f}s")

    state["context_xml"] = context_xml
    state["retrieval_report"] = retrieval_report or []

    if len(context_xml) < MIN_CONTEXT_SIZE:
        raise WorkflowError(
            "Context is too small for the AI cache. Add more core documents or use additional libraries, then try again.",
            details="Context too small for cache",
        )


# -----------------------------------------------------------------------------
# Step 3: Create cache (or reuse)
# -----------------------------------------------------------------------------

def create_cache_step(state: dict, callbacks: Callbacks | None = None) -> None:
    state["_callbacks"] = callbacks or {}
    context_xml = state["context_xml"]
    folder_id = state["folder_id"]
    user_content = state["user_content"]
    prompt_template_id = state.get("prompt_template_id")
    timings = state.setdefault("timings", {})

    session_cache_name = state.get("session_cache_name")
    session_cache_model = state.get("session_cache_model")
    session_cache_folder_id = state.get("session_cache_folder_id")
    session_run_cache_key = state.get("session_run_cache_key")
    current_model = get_effective_model()

    run_cache_key = hashlib.sha256(
        f"{folder_id}|{prompt_template_id}|{(user_content or '')}".encode()
    ).hexdigest()[:32]

    reuse_cache = (
        session_run_cache_key == run_cache_key
        and session_cache_name
        and session_cache_folder_id == folder_id
        and session_cache_model == current_model
    )

    if reuse_cache:
        state["cache_name"] = session_cache_name
        timings["cache_create_s"] = 0.0
        state["cache_reused"] = True
        _cb(state, "write", "✓ Reused cache (same task + input)")
        state["run_cache_key"] = run_cache_key
        return

    state["cache_reused"] = False
    _cb(state, "write", "Creating Gemini context cache…")
    _t0 = time.perf_counter()
    try:
        def retry_progress(attempt: int, max_attempts: int, delay: float, error_msg: str):
            _cb(state, "write", f"Attempt {attempt}/{max_attempts} failed. Retrying in {delay:.0f}s…")
            _cb(state, "update_label", "Building context + cache… (retry)", "running")

        cache_name = create_gemini_cache(context_xml, progress_callback=retry_progress)
        timings["cache_create_s"] = time.perf_counter() - _t0
    except Exception as e:
        err_lower = str(e).lower()
        if "too small" in err_lower or "min_total_token_count" in err_lower:
            raise WorkflowError(
                "Knowledge base too small for Gemini caching. The knowledge base needs at least ~16KB of text (4096 tokens).",
                details=f"Knowledge base too small: {e}",
            )
        if "too large" in err_lower or "max_total_token_count" in err_lower:
            raise WorkflowError(
                "Cache Content Too Large. Your knowledge base is too large to cache.",
                details=str(e),
            )
        if "503" in err_lower or "unavailable" in err_lower or "server" in err_lower or "failed to create cache after" in err_lower:
            raise WorkflowError(
                "Gemini API temporarily unavailable. Wait a few minutes and try again.",
                details=str(e),
            )
        raise WorkflowError(f"Failed to create cache: {e}", details=str(e))

    state["cache_name"] = cache_name
    state["run_cache_key"] = run_cache_key


# -----------------------------------------------------------------------------
# Step 4: Run main agent (with retry on cache expiry)
# -----------------------------------------------------------------------------

def run_main_agent_step(state: dict, callbacks: Callbacks | None = None) -> None:
    state["_callbacks"] = callbacks or {}
    full_template = state["main_full_template"]
    user_content = state["user_content"]
    input_schema_json = state.get("input_schema_json")
    output_schema_json = state.get("output_schema_json")
    cache_name = state["cache_name"]
    context_xml = state["context_xml"]
    timings = state.setdefault("timings", {})

    _cb(state, "write", "Running main agent…")
    max_retries = 1
    for retry_attempt in range(max_retries + 1):
        try:
            if GEMINI_PACE_DELAY_SECONDS > 0:
                time.sleep(GEMINI_PACE_DELAY_SECONDS)
            _t0 = time.perf_counter()
            main_transient_data = {
                "content": user_content,
                "input_schema_json": input_schema_json or "",
                "output_schema_json": output_schema_json or "",
            }
            expect_json = bool(output_schema_json)
            result = run_agent(
                full_template,
                main_transient_data,
                cache_name,
                expect_json=expect_json,
                input_schema_json=input_schema_json,
                output_schema_json=output_schema_json,
            )
            state["gemini_call_count"] = state.get("gemini_call_count", 0) + 1
            timings["model_run_s"] = time.perf_counter() - _t0
            _cb(state, "write", f"✓ Main agent done in {timings['model_run_s']:.2f}s")
            if expect_json and isinstance(result, dict):
                _collapse_repeated_newlines(result)
                state["main_output"] = json.dumps(result, indent=2)
                state["main_output_dict"] = result
            else:
                state["main_output"] = str(result)
                state["main_output_dict"] = None
            return
        except CacheExpiredError as cache_expired:
            if retry_attempt < max_retries:
                _cb(state, "write", "Cache expired. Recreating…")
                try:
                    cache_name = create_gemini_cache(context_xml)
                    state["cache_name"] = cache_name
                except Exception as recreate_error:
                    raise WorkflowError(
                        f"Failed to recreate cache after expiration: {recreate_error}",
                        details=str(recreate_error),
                    )
            else:
                raise WorkflowError(
                    "Cache expired and could not be recreated. Please try again.",
                    details=str(cache_expired),
                )


# -----------------------------------------------------------------------------
# Step 5: Extract legal questions
# -----------------------------------------------------------------------------

def extract_legal_questions_step(state: dict, callbacks: Callbacks | None = None) -> None:
    state["_callbacks"] = callbacks or {}
    _cb(state, "write", "Extracting legal questions from main output…")
    main_output = state["main_output"]
    main_content, legal_questions = extract_legal_questions(main_output)
    state["main_content"] = main_content
    state["legal_questions"] = legal_questions or []


# -----------------------------------------------------------------------------
# Step 6: Run legal expert (conditional)
# -----------------------------------------------------------------------------

def run_legal_expert_step(state: dict, callbacks: Callbacks | None = None) -> None:
    state["_callbacks"] = callbacks or {}
    selected = state["selected_prompt"]
    legal_expert_prompt_id = getattr(selected, "legal_expert_prompt_id", None)
    legal_questions = state.get("legal_questions") or []

    _cb(state, "write", "Running legal expert (retrieval + cache + model)…")
    if not legal_questions or not legal_expert_prompt_id:
        state["legal_expert_output"] = None
        state["legal_expert_report"] = None
        return

    if LEGAL_EXPERT_DELAY_SECONDS > 0:
        time.sleep(LEGAL_EXPERT_DELAY_SECONDS)

    legal_expert_p = db.get_prompt_by_id(legal_expert_prompt_id)
    if not legal_expert_p:
        logger.warning(f"Legal expert prompt {legal_expert_prompt_id} not found")
        state["legal_expert_output"] = None
        state["legal_expert_report"] = None
        return

    _rs = state["rag_state"]
    main_content = state["main_content"]
    legal_query_text = "\n\n".join([f"Q{i+1}: {q}" for i, q in enumerate(legal_questions)])

    if USE_RETRIEVAL_PLANNER:
        legal_sel_ids, legal_top_k_map = plan_retrieval(
            _rs, legal_expert_p.name, legal_expert_p.template_text, legal_query_text
        )
    else:
        legal_sel_ids, legal_top_k_map = get_default_plan(_rs)

    if USE_QUERY_EXPANSION:
        from brain import expand_queries
        legal_query_phrases = expand_queries(
            legal_expert_p.name, legal_expert_p.template_text, legal_query_text
        )
    else:
        legal_query_phrases = get_fallback_phrases(
            legal_expert_p.name, legal_expert_p.template_text, legal_query_text
        )

    legal_context_xml, legal_expert_report = retrieve_and_build_context_multi(
        _rs, legal_query_phrases, legal_sel_ids, legal_top_k_map
    )
    state["legal_expert_report"] = legal_expert_report

    legal_cache_name = None
    if len(legal_context_xml) >= MIN_CONTEXT_SIZE:
        legal_cache_name = create_gemini_cache(legal_context_xml)

    legal_expert_output = None
    if legal_cache_name:
        legal_expert_variables = state["build_prompt_variables"](
            state["username"], state.get("user_name", "unknown")
        )
        legal_expert_template = (
            legal_expert_variables + legal_expert_p.template_text
            + "\n\n---\n\nLegal questions to answer:\n{{ legal_questions }}\n\n---\n\nOriginal analysis context:\n{{ original_output }}"
        )
        try:
            le_input_schema_json = None
            le_output_schema_json = None
            le_input_schema_json = _get_input_schema_json(legal_expert_p)
            le_output_schema_json = _get_output_schema_json(legal_expert_p)
            le_transient_data = {
                "legal_questions": legal_query_text,
                "original_output": main_content,
                "input_schema_json": le_input_schema_json or "",
                "output_schema_json": le_output_schema_json or "",
            }
            legal_expert_output = run_agent(
                legal_expert_template,
                le_transient_data,
                legal_cache_name,
                expect_json=False,
                input_schema_json=le_input_schema_json,
                output_schema_json=le_output_schema_json,
            )
            state["gemini_call_count"] = state.get("gemini_call_count", 0) + 1
        except Exception as legal_err:
            logger.warning(f"Legal expert consultation failed: {legal_err}")
            legal_expert_output = None

    state["legal_expert_output"] = str(legal_expert_output) if legal_expert_output else None


# -----------------------------------------------------------------------------
# Step 7: Integrate legal
# -----------------------------------------------------------------------------

def integrate_legal_step(state: dict, callbacks: Callbacks | None = None) -> None:
    state["_callbacks"] = callbacks or {}
    _cb(state, "write", "Integrating legal output into final response…")
    main_content = state["main_content"]
    legal_expert_output = state.get("legal_expert_output")
    if legal_expert_output:
        state["final_output"] = f"{main_content}\n\n---\n\n## Legal Expert Consultation\n\n{legal_expert_output}"
    else:
        state["final_output"] = main_content


# -----------------------------------------------------------------------------
# Step 8: Follow-on chain
# -----------------------------------------------------------------------------

def run_follow_on_chain_step(state: dict, callbacks: Callbacks | None = None) -> None:
    state["_callbacks"] = callbacks or {}
    _cb(state, "write", "Running follow-on verification chain…")
    selected = state.get("selected_prompt")
    if selected is None:
        _cb(state, "write", "⚠️ No prompt selected; skipping follow-on chain.")
        logger.warning("run_follow_on_chain_step: selected_prompt is None, skipping")
        return
    _rs = state.get("rag_state")
    if _rs is None:
        _cb(state, "write", "⚠️ RAG state not loaded; skipping follow-on chain.")
        logger.warning("run_follow_on_chain_step: rag_state is None, skipping")
        return
    build_prompt_variables = state.get("build_prompt_variables")
    if not callable(build_prompt_variables):
        build_prompt_variables = lambda u, n: "\n\n**Context Variables**\n\n- **Analysis Performed By**: " + (n or u or "unknown")
    _sep = "\n\n---\n\n"
    accumulated = state["final_output"]
    chain: list[tuple[str, str]] = [(state["task_name"], accumulated)]
    first_step_result = state["main_content"]
    if state.get("legal_expert_output"):
        first_step_result = state["final_output"]
    pipeline_step_results = [{
        "step_number": 1,
        "step_name": state["task_name"],
        "output": state["main_content"],
        "has_legal_expert": bool(state.get("legal_expert_output")),
        "legal_expert_output": state.get("legal_expert_output"),
        "full_output": first_step_result,
    }]
    legal_questions_by_step = []
    if state.get("legal_questions"):
        legal_questions_by_step.append({
            "step_name": state["task_name"],
            "questions": state["legal_questions"],
            "expert_output": state.get("legal_expert_output"),
            "expert_report": state.get("legal_expert_report"),
            "has_legal_expert": bool(getattr(selected, "legal_expert_prompt_id", None)),
        })
    elif getattr(selected, "legal_expert_prompt_id", None):
        legal_questions_by_step.append({
            "step_name": state["task_name"],
            "questions": None,
            "expert_output": None,
            "expert_report": None,
            "has_legal_expert": True,
        })
    chain_timings: list[dict] = []
    state["last_chain_error"] = None

    current = selected
    seen: set[int] = {selected.id}

    while getattr(current, "verifier_id", None):
        fid = current.verifier_id
        if fid in seen:
            logger.warning(f"Follow-on cycle detected (prompt id={fid}), stopping chain")
            break
        next_p = db.get_prompt_by_id(fid)
        if not next_p:
            logger.warning(f"Follow-on prompt id {fid} not found, stopping chain")
            break
        seen.add(next_p.id)

        _cb(state, "write", f"📋 Follow-on step: **{next_p.name}** — planning retrieval…")
        step_t0 = time.perf_counter()
        if PIPELINE_STEP_DELAY_SECONDS > 0:
            logger.info("Pipeline step delay: sleeping %ds before follow-on «%s»", PIPELINE_STEP_DELAY_SECONDS, next_p.name)
            _cb(state, "write", f"⏳ Waiting {PIPELINE_STEP_DELAY_SECONDS}s (rate-limit pacing)…")
            time.sleep(PIPELINE_STEP_DELAY_SECONDS)

        followon_variables = build_prompt_variables(state["username"], state.get("user_name", "unknown"))
        followon_legal_tracking = LEGAL_TRACKING_FOLLOWON if getattr(next_p, "legal_expert_prompt_id", None) else ""
        followon_template = (
            followon_variables + next_p.template_text + followon_legal_tracking
            + "\n\n---\n\nOutput from previous step(s):\n{{ previous_output }}"
        )

        if USE_RETRIEVAL_PLANNER:
            sel_ids, top_k_map = plan_retrieval(_rs, next_p.name, next_p.template_text, accumulated)
        else:
            sel_ids, top_k_map = get_default_plan(_rs)
        if USE_QUERY_EXPANSION:
            _cb(state, "write", f"🔍 **{next_p.name}** — expanding queries (LLM)…")
            from brain import expand_queries
            step_phrases = expand_queries(next_p.name, next_p.template_text, accumulated)
        else:
            step_phrases = get_fallback_phrases(next_p.name, next_p.template_text, accumulated)

        _cb(state, "write", f"📚 **{next_p.name}** — retrieving context…")
        step_context_xml, step_report = retrieve_and_build_context_multi(_rs, step_phrases, sel_ids, top_k_map)
        if len(step_context_xml) < MIN_CONTEXT_SIZE:
            state["last_chain_error"] = f"Follow-on «{next_p.name}»: context too small for cache"
            break

        _cb(state, "write", f"📦 **{next_p.name}** — creating context cache…")
        step_cache_name = create_gemini_cache(step_context_xml)
        if GEMINI_PACE_DELAY_SECONDS > 0:
            time.sleep(GEMINI_PACE_DELAY_SECONDS)

        step_input_schema_json = _get_input_schema_json(next_p)
        step_output_schema_json = _get_output_schema_json(next_p)

        _cb(state, "write", f"🤖 **{next_p.name}** — running agent…")
        retry = 0
        while True:
            try:
                step_transient_data = {
                    "previous_output": accumulated,
                    "input_schema_json": step_input_schema_json or "",
                    "output_schema_json": step_output_schema_json or "",
                }
                out = run_agent(
                    followon_template,
                    step_transient_data,
                    step_cache_name,
                    expect_json=False,
                    input_schema_json=step_input_schema_json,
                    output_schema_json=step_output_schema_json,
                )
                state["gemini_call_count"] = state.get("gemini_call_count", 0) + 1
                break
            except CacheExpiredError:
                if retry < 1:
                    step_cache_name = create_gemini_cache(step_context_xml)
                    retry += 1
                else:
                    raise WorkflowError(f"Follow-on «{next_p.name}»: cache expired", details=next_p.name)

        step_main_content, step_legal_questions = extract_legal_questions(str(out))
        step_legal_expert_output = None
        step_output_with_legal = step_main_content
        if step_legal_questions and getattr(next_p, "legal_expert_prompt_id", None):
            step_legal_expert_p = db.get_prompt_by_id(next_p.legal_expert_prompt_id)
            if step_legal_expert_p:
                step_legal_query_text = "\n\n".join([f"Q{i+1}: {q}" for i, q in enumerate(step_legal_questions)])
                if USE_RETRIEVAL_PLANNER:
                    step_legal_sel_ids, step_legal_top_k_map = plan_retrieval(
                        _rs, step_legal_expert_p.name, step_legal_expert_p.template_text, step_legal_query_text
                    )
                else:
                    step_legal_sel_ids, step_legal_top_k_map = get_default_plan(_rs)
                if USE_QUERY_EXPANSION:
                    from brain import expand_queries
                    step_legal_phrases = expand_queries(
                        step_legal_expert_p.name, step_legal_expert_p.template_text, step_legal_query_text
                    )
                else:
                    step_legal_phrases = get_fallback_phrases(
                        step_legal_expert_p.name, step_legal_expert_p.template_text, step_legal_query_text
                    )
                step_legal_context_xml, _ = retrieve_and_build_context_multi(
                    _rs, step_legal_phrases, step_legal_sel_ids, step_legal_top_k_map
                )
                if len(step_legal_context_xml) >= MIN_CONTEXT_SIZE:
                    step_legal_cache = create_gemini_cache(step_legal_context_xml)
                    step_legal_vars = build_prompt_variables(state["username"], state.get("user_name", "unknown"))
                    step_legal_template = (
                        step_legal_vars + step_legal_expert_p.template_text
                        + "\n\n---\n\nLegal questions to answer:\n{{ legal_questions }}\n\n---\n\nOriginal analysis context:\n{{ original_output }}"
                    )
                    try:
                        sle_input = _get_input_schema_json(step_legal_expert_p)
                        sle_output = _get_output_schema_json(step_legal_expert_p)
                        sle_data = {
                            "legal_questions": step_legal_query_text,
                            "original_output": step_main_content,
                            "input_schema_json": sle_input or "",
                            "output_schema_json": sle_output or "",
                        }
                        step_legal_expert_output = run_agent(
                            step_legal_template, sle_data, step_legal_cache,
                            expect_json=False, input_schema_json=sle_input, output_schema_json=sle_output,
                        )
                        state["gemini_call_count"] = state.get("gemini_call_count", 0) + 1
                    except Exception:
                        step_legal_expert_output = None
                if step_legal_expert_output:
                    step_output_with_legal = f"{step_main_content}\n\n---\n\n## Legal Expert Consultation\n\n{str(step_legal_expert_output)}"

        step_elapsed = time.perf_counter() - step_t0
        chain_timings.append({"step_name": next_p.name, "elapsed_s": round(step_elapsed, 2)})
        logger.info("Follow-on «%s» completed in %.2fs", next_p.name, step_elapsed)

        accumulated = accumulated + _sep + step_output_with_legal
        chain.append((next_p.name, accumulated))
        pipeline_step_results.append({
            "step_number": len(pipeline_step_results) + 1,
            "step_name": next_p.name,
            "output": step_main_content,
            "has_legal_expert": bool(step_legal_expert_output),
            "legal_expert_output": str(step_legal_expert_output) if step_legal_expert_output else None,
            "full_output": step_output_with_legal,
        })
        if step_legal_questions:
            legal_questions_by_step.append({
                "step_name": next_p.name,
                "questions": step_legal_questions,
                "expert_output": step_legal_expert_output,
                "expert_report": None,
                "has_legal_expert": bool(getattr(next_p, "legal_expert_prompt_id", None)),
            })
        current = next_p

    state["final_output"] = accumulated
    state["chain"] = chain
    state["pipeline_step_results"] = pipeline_step_results
    state["legal_questions_by_step"] = legal_questions_by_step
    state["chain_timings"] = chain_timings
    state["chain_outputs"] = [{"step_name": n, "output": o} for n, o in chain]
    state["output_mode"] = "markdown"


# -----------------------------------------------------------------------------
# Step 9: QA agent (Phase 4) — optional full QA analysis/review of full chain output
# -----------------------------------------------------------------------------

QA_PROMPT = """You are a QA reviewer for municipal council and committee materials. Your job is **only** to (1) fix mistakes and (2) integrate legal expert answers into the narrative where needed. You must **preserve the format, structure, and purpose** of the document exactly as the original task requested.

**What you MUST preserve (do not change):**
- The **type** of document (e.g. constituent feedback, committee report, memo, summary). If the draft is constituent feedback, your output must remain constituent feedback in the same format.
- The **structure** and **section layout** (headings, bullets, paragraphs) unless they are wrong or inconsistent.
- The **tone and audience** (e.g. direct reply to a resident vs. internal committee memo).
- The **substance and intent** of every section. Do not condense, expand, or reframe the content; only correct errors and weave in legal answers.

**What you MUST do:**
1. **Fix mistakes**
   - Correct typos, grammar, and obvious factual errors (wrong dates, names, figures).
   - Fix internal contradictions (e.g. a recommendation that conflicts with the legal section or with another part of the draft).
   - Ensure facts, dates, and names are consistent throughout.

2. **Integrate legal answers (if the draft includes a legal expert section)**
   - If the draft has a separate "Legal Expert" or "Legal consultation" section, weave the relevant legal guidance into the main narrative where it affects conclusions or recommendations. Do not leave legal answers as an isolated add-on when they should inform the body of the document.
   - Ensure the narrative reflects how legal answers affect (or do not affect) recommendations. Keep any necessary legal caveats or conditions clear.
   - If there is no legal section, skip this; do not add legal content.

**What you must NOT do:**
- Do **not** rewrite the document for style, "polish," or "professional tone" in a way that changes its purpose or format.
- Do **not** add new sections, remove sections, or change the document type (e.g. do not turn constituent feedback into a formal report).
- Do **not** include a QA report, checklist, or meta-commentary. Output **only** the corrected document.

---
Draft to review:

{{ draft }}
"""


def run_qa_step(state: dict, callbacks: Callbacks | None = None) -> None:
    """
    Run QA agent on the current final_output (full chain output).
    Overwrites final_output with the QA result. No RAG; uses only the combined text.
    """
    state["_callbacks"] = callbacks or {}
    _cb(state, "write", "Running QA agent (review and polish)…")
    draft = state.get("final_output") or ""
    if not draft.strip():
        _cb(state, "write", "No draft to review; skipping QA.")
        return
    state["pre_qa_output"] = draft
    try:
        # QA does not use RAG; pass minimal context so run_agent fallback path is satisfied
        qa_result = run_agent(
            QA_PROMPT,
            {"draft": draft},
            cache_name=None,
            expect_json=False,
            context_xml=" ",
            fallback_expected=True,
        )
        state["gemini_call_count"] = state.get("gemini_call_count", 0) + 1
        state["final_output"] = str(qa_result).strip() if qa_result else draft
        state["qa_output"] = state["final_output"]
        _cb(state, "write", "✅ QA agent done.")
    except Exception as e:
        logger.warning(f"QA agent failed: {e}; keeping pre-QA output")
        state["final_output"] = draft
        state["qa_output"] = None
        _cb(state, "write", f"⚠️ QA failed ({e}); keeping original output.")


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


def _run_retrieve_and_cache(state: dict, callbacks: Callbacks | None = None) -> None:
    """Retrieve context then create cache (combined for one status in UI)."""
    retrieve_context_step(state, callbacks)
    create_cache_step(state, callbacks)


def _run_legal_and_integrate(state: dict, callbacks: Callbacks | None = None) -> None:
    """Extract legal questions, run legal expert if needed, integrate."""
    extract_legal_questions_step(state, callbacks)
    run_legal_expert_step(state, callbacks)
    integrate_legal_step(state, callbacks)


def run_workflow(state: dict, callbacks: Callbacks | None = None) -> dict:
    """
    Run the full analysis workflow: plan → retrieve → cache → main agent → legal (optional) → integrate → follow-on chain.
    Mutates state; raises WorkflowError on failure.
    If callbacks has "with_step"(label, state, step_fn), each step is run inside that callback (e.g. st.status).
    """
    state["_callbacks"] = callbacks or {}
    state["status"] = "running"
    state["error"] = None

    with_step = (callbacks or {}).get("with_step")
    if with_step:
        with_step("🔍 Planning retrieval…", state, lambda: plan_retrieval_step(state, state.get("_callbacks")))
        with_step("🧠 Building context + cache…", state, lambda: _run_retrieve_and_cache(state, state.get("_callbacks")))
        with_step("🚀 Running main agent…", state, lambda: run_main_agent_step(state, state.get("_callbacks")))
        with_step("⚖️ Legal + integrate…", state, lambda: _run_legal_and_integrate(state, state.get("_callbacks")))
        with_step("📎 Follow-on chain…", state, lambda: run_follow_on_chain_step(state, state.get("_callbacks")))
    else:
        plan_retrieval_step(state, callbacks)
        retrieve_context_step(state, callbacks)
        create_cache_step(state, callbacks)
        run_main_agent_step(state, callbacks)
        extract_legal_questions_step(state, callbacks)
        run_legal_expert_step(state, callbacks)
        integrate_legal_step(state, callbacks)
        run_follow_on_chain_step(state, callbacks)

    # Token stats for UI (same shape as app's last_run_context_stats)
    context_xml = state.get("context_xml") or ""
    user_content = state.get("user_content") or ""
    template_text = state.get("template_text") or ""
    final_output = state.get("final_output") or ""
    kb_tokens = chars_to_tokens(len(context_xml))
    transient_tokens = chars_to_tokens(len(user_content))
    prompt_tokens = chars_to_tokens(len(template_text) + 80)
    total_input_tokens = kb_tokens + transient_tokens + prompt_tokens
    state["last_run_context_stats"] = {
        "total_input_tokens": total_input_tokens,
        "kb_tokens": kb_tokens,
        "transient_tokens": transient_tokens,
        "prompt_tokens": prompt_tokens,
        "max_context": model_max_context(get_effective_model()),
        "output_tokens": chars_to_tokens(len(final_output)),
        "model": get_effective_model(),
        "timings": state.get("timings", {}),
        "gemini_calls": state.get("gemini_call_count", 0),
    }

    state["status"] = "completed"
    return state
