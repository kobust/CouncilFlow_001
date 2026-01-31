# Implementation Plan: Audit, Run History UX, PromptVersion UI, and Read-Only Tools

This document plans the implementation of four areas:

1. **Enable checkpointer (default or env) and add RunEvent (P5.3) for audit**
2. **Run history filters + Pre-QA/Post-QA + event log for better governance and UX**
3. **PromptVersion "View history" / "Restore" for safer prompt changes**
4. **Establish read-only tools (search_ordinances, search_mgl)**

---

## 1. Enable checkpointer and add RunEvent (P5.3)

### 1.1 Enable checkpointer by default (or env)

**Goal:** Use LangGraph SqliteSaver by default so runs can be resumed and audited; keep env override for opt-out.

**Touchpoints:**

| File | Change |
|------|--------|
| `workflow_graph.py` | Change default: `get_analysis_graph(..., use_sqlite_checkpointer=True)` when not explicitly passed. In `_use_graph_checkpointer_env()`, treat *unset* as True (checkpointer on) and require explicit `0`/`false` to disable. Or: add `COUNCILFLOW_USE_GRAPH_CHECKPOINTER` default to `1` in docstring; keep `run_analysis_graph(..., use_sqlite_checkpointer=None)` so None → read env; if env unset, default to True. |
| `workflow_graph.py` | In `run_analysis_graph`, when `use_sqlite_checkpointer is None`, set `use_sqlite_checkpointer = _use_graph_checkpointer_env()`. Update `_use_graph_checkpointer_env()` so that when the env var is **unset**, return `True` (checkpointer on). When set to `0`, `false`, `no`, return `False`. |
| `app.py` | When calling `run_analysis_graph`, pass a stable `thread_id` for the run (e.g. `thread_id=str(run_id)` after inserting a "running" row, or `thread_id=f"run_{run_started_at.timestamp()}_{username}"`) so that if we add interrupt/resume later, the same thread can resume. Optional for this phase: if we don't insert the run row before starting the graph, we can use a UUID or timestamp-based thread_id and then set the run id on the AnalysisRun row after completion (current design inserts run only on success/failure at end). So: either (a) insert a "running" run row first, get its id, use as thread_id, then update on completion, or (b) keep current insert-only-on-done and use a generated thread_id for checkpointer only (no DB link until run is persisted). Recommendation: (b) for this phase; thread_id can be `f"run_{run_started_at.isoformat()}_{username}"` or UUID. |

**Implementation details:**

- `_use_graph_checkpointer_env()`:  
  - If `COUNCILFLOW_USE_GRAPH_CHECKPOINTER` is unset or empty → return `True`.  
  - If set to `0`, `false`, `no` (case-insensitive) → return `False`.  
  - Otherwise (e.g. `1`, `true`, `yes`) → return `True`.
- Ensure `_get_checkpointer()` is called when building the graph with checkpointer=True, and that `council_runs.db` exists and is writable (already the case; SqliteSaver uses `runs_db.RUNS_SQLITE_URL`).

### 1.2 RunEvent table and logging

**Goal:** Step-level audit trail: which node ran, when, and optional payload (small summary).

**Schema (runs_db.py):**

- New table **RunEvent** in `council_runs.db`:
  - `id` (PK, auto)
  - `run_id` (int, nullable for now—events may be written before we have a run row if we add "running" row first; or we can link by thread_id later)
  - `thread_id` (str, nullable)—LangGraph thread_id so we can correlate with checkpointer even when run_id is not yet set
  - `step_name` (str)—e.g. `plan_retrieval`, `main_agent`, `finalize`
  - `event_type` (str)—e.g. `node_started`, `node_completed`, `interrupt_requested`, `human_responded`
  - `payload` (str, nullable)—optional JSON or short text (e.g. hash or summary; avoid storing full output)
  - `created_at` (datetime, default UTC)

**Touchpoints:**

| File | Change |
|------|--------|
| `runs_db.py` | Add `RunEvent` SQLModel; add migration for new table; add `insert_run_event(run_id=None, thread_id=None, step_name, event_type, payload=None)`; optionally `list_run_events_by_run_id(run_id)` and `list_run_events_by_thread_id(thread_id)` for UI. |
| `workflow_graph.py` | In each node wrapper (e.g. `_node_plan_retrieval`): at entry, log RunEvent `node_started`; at exit (before return), log `node_completed`. Get `thread_id` from `config["configurable"].get("thread_id")` and `run_id` from `config["configurable"].get("run_id")` if we pass it. Pass `run_id` in config only after we have it (e.g. if we insert "running" run first). For minimal change: pass only `thread_id` in config; store `run_id` in RunEvent when we persist the run (e.g. backfill run_id on events with matching thread_id when we insert_analysis_run). Simpler: don't set run_id in RunEvent at node time; when app calls insert_analysis_run, also call an optional `runs_db.update_run_events_run_id(thread_id, run_id)` to set run_id on events that have that thread_id. So: RunEvent has thread_id (and run_id nullable). Graph nodes log with thread_id from config. App: before run, generate thread_id; pass in config; after successful insert_analysis_run, call update_run_events_run_id(thread_id, run.id). |
| `app.py` | When building `run_config`, set `run_config["configurable"]["thread_id"] = thread_id` (generate once at start, e.g. uuid.uuid4().hex or timestamp-based). After `insert_analysis_run` on success or failure, call `runs_db.update_run_events_run_id(thread_id, run.id)` so events get linked. |

**Event types to log (this phase):**

- `node_started` — when entering a node (step_name = node name)
- `node_completed` — when leaving a node (step_name = node name)

Payload for `node_completed` can be empty or a tiny summary (e.g. `{"status":"ok"}` or chunk count for retrieve_context) to keep DB small.

**Order of work:**

1. Add RunEvent table and CRUD in `runs_db.py`.
2. In `workflow_graph.py`, inject a logger function via config (e.g. `config["configurable"]["log_run_event"]`) so nodes don't import runs_db directly (avoids circular deps and keeps graph testable). The callable receives (step_name, event_type, payload=None). App provides a function that calls `runs_db.insert_run_event(..., thread_id=..., step_name=..., event_type=..., payload=...)`.
3. In each node: at start, call `log_run_event(step_name, "node_started")`; before return, call `log_run_event(step_name, "node_completed")`.
4. In app: generate thread_id; pass thread_id and log_run_event in config; after insert_analysis_run, update RunEvent rows with that thread_id to set run_id.

---

## 2. Run history filters + Pre-QA/Post-QA + event log

### 2.1 AnalysisRun: pre_qa_output and qa_output

**Goal:** Persist pre-QA and post-QA text so Run history can show "Pre-QA" / "Post-QA" for governance and comparison.

**Touchpoints:**

| File | Change |
|------|--------|
| `runs_db.py` | Add columns `pre_qa_output` (TEXT, nullable) and `qa_output` (TEXT, nullable) to AnalysisRun. Migration: add columns if missing. Extend `insert_analysis_run(..., pre_qa_output=None, qa_output=None)` and model. |
| `app.py` | When calling `insert_analysis_run` after successful run, pass `pre_qa_output=state.get("pre_qa_output")`, `qa_output=state.get("qa_output")`. Workflow already sets these in state when QA runs. |
| `app.py` (Run history detail) | When displaying a run that has `pre_qa_output` or `qa_output`, add an expander or tabs: "Output" (current, = final output), "Pre-QA" (if present), "Post-QA" (if present). Use same copy-markdown behavior for each. |

### 2.2 Run history filters

**Goal:** Filter runs by task name, date range, status, prompt version (and optionally username, already supported).

**Touchpoints:**

| File | Change |
|------|--------|
| `runs_db.py` | Extend `list_analysis_runs(limit, username_filter=None, task_name_filter=None, status_filter=None, prompt_version_filter=None, date_from=None, date_to=None)`. Build query with optional `.where(...)` for each. Use `started_at` for date range. |
| `app.py` (Run history list) | Add filter UI above the list: dropdown or multiselect for task name (from distinct task_name in runs or from prompts), status (running/completed/failed), optional prompt version, and date range (date_from, date_to). Store filter state in session_state so they persist. Call `list_analysis_runs(..., **filters)` and render the result. |

**Filter semantics:**

- **Task name:** Exact match or "all"; options can be derived from `db.get_all_prompts()` names plus "All".
- **Status:** All / Completed / Failed / Running.
- **Prompt version:** Optional int or "Any".
- **Date range:** Optional start/end date (use run's `started_at` in UTC, compare with user-selected dates in local or UTC as consistent with existing timezone handling).

### 2.3 Event log in Run detail

**Goal:** Show a step-level event log for the run in Run history detail.

**Touchpoints:**

| File | Change |
|------|--------|
| `runs_db.py` | Add `list_run_events(run_id)` returning list of RunEvent rows for that run_id, ordered by created_at. |
| `app.py` (Run history detail) | After "Output" and chain steps, add an expander "Event log" that calls `list_run_events(run_detail.id)` and displays a table or list: timestamp (formatted in local time), step_name, event_type, optional payload. If no events, show "No event log for this run." (e.g. old runs before RunEvent existed). |

---

## 3. PromptVersion "View history" / "Restore"

**Goal:** In the prompt editor, show version history (list of PromptVersion rows for the current prompt) and allow "Restore to version N" to load that version into the form (user can then save as a new version).

### 3.1 DB and API

| File | Change |
|------|--------|
| `db.py` | Add `list_prompt_versions(prompt_template_id: int)` returning list of PromptVersion rows for that template, ordered by version desc (or saved_at desc). Add `get_prompt_version_by_id(version_id: int)` or `get_prompt_version(prompt_template_id, version: int)` to fetch one version for restore. |

### 3.2 Prompt editor UI

| File | Change |
|------|--------|
| `app.py` (Edit prompts / prompt editor) | When an existing prompt is selected: add a section "Version history" with a button "View history". When "View history" is toggled or a dedicated subview: list versions from `db.list_prompt_versions(existing.id)` showing version number, saved_at (formatted), optionally name/template_text excerpt. Each row has a "Restore" button. On "Restore" for version N: load that PromptVersion row; set session state (and form) to that version's name, template_text, verifier_id, follow_on_only, legal_expert_prompt_id, input_schema_id, output_schema_id, use_qa_agent, workflow_id; do not change the prompt's id (we're editing the same prompt). Show a message "Restored to version N. You can edit and Save to create a new version." User can then click Save to persist (which will create a new PromptVersion snapshot and increment current_version). |

**Implementation details:**

- "View history" can be an expander or a separate "tab" / subsection on the same page. List: table or st.dataframe with columns Version, Saved at, (optional) Name; actions: Restore.
- Restore: `pv = db.get_prompt_version(prompt_template_id, version)` or by id; then set `st.session_state["crud_name"] = pv.name`, `crud_template` = pv.template_text, etc., and set `st.session_state["crud_restored_version"] = pv.version` so the UI can show "Restored to vN". Clear crud_restored_version on next load or on Save.
- No DB write on Restore; only on Save.

---

## 4. Read-only tools: search_ordinances and search_mgl

**Goal:** Expose two read-only tools to the model so it can pull additional context during a run: `search_ordinances` (general ordinance/knowledge-base search) and `search_mgl` (Massachusetts General Laws, or a designated "MGL" library). Both return text (or structured summary) for the model to use; no write operations.

### 4.1 Tool semantics

- **search_ordinances(query: str, max_chunks?: int)**  
  - Runs retrieval over the **current run’s** knowledge base (same folder_id / rag_state as the run).  
  - Uses the same RAG path as the main flow: hybrid search over Core + libraries (or libraries only, depending on product choice).  
  - Returns a string: e.g. top chunks formatted as XML or markdown (similar to build_retrieved_xml), truncated to a safe token budget.  
  - Implementation: reuse `rag_loader.retrieve_and_build_context` or `retrieve_and_build_context_multi` with a single query; get rag_state from run state (folder_id → get_cached_rag_state) and use get_default_plan or a fixed top_k.

- **search_mgl(query: str, max_chunks?: int)**  
  - Same as search_ordinances but restricted to a designated "MGL" library.  
  - Options: (a) a library whose name contains "MGL" or matches a configurable name; (b) a dedicated folder_id or library id in config.  
  - Implementation: same retrieval as above but filter `selected_library_ids` to only the MGL library (by name or id). If no MGL library exists, return a short message "No MGL library configured."

### 4.2 Where tools are invoked

- Tools are invoked by the **Gemini model** during `run_agent()` when the model issues a function call. So we need to:
  - Define Gemini **FunctionDeclaration** (or equivalent) for `search_ordinances` and `search_mgl` (name, description, parameters schema).
  - In `brain.run_agent()`, enable **automatic function calling** or a manual tool loop: when the model returns a function call, execute the corresponding Python function (with access to run context: folder_id, rag_state), then append the result to the conversation and call again until the model returns a final text response.

### 4.3 Context for tools

- Tools need **folder_id** (and thus rag_state) and optionally **run_id** for audit. Today `run_agent` is called from workflow steps with state that has folder_id and rag_state. So the tool implementations can be functions that receive (query, max_chunks, folder_id) or (query, max_chunks, rag_state). We don’t have rag_state in the serializable state anymore; we resolve it in nodes via get_cached_rag_state(folder_id). So when the model is in a node (e.g. main_agent), we have folder_id in state. We can pass a **tool runner** in config (e.g. a callable `run_tool(name, args)` that has access to folder_id and runs the right tool). So:
  - **Option A:** Pass `folder_id` (and optionally run_id) in config to the graph; in the node that calls run_agent, also pass a `tools` list and a `tool_runner(tool_name, args)` that uses folder_id and get_cached_rag_state(folder_id) to run search_ordinances / search_mgl.
  - **Option B:** Implement tools inside `brain.run_agent()` by adding a `tools` parameter and a `tool_runner(state_or_context)` that receives the minimal context (folder_id) and runs the tool. The runner is provided by the caller (workflow step).

Recommendation: **Option B** with a small twist: add optional `tools` and `tool_context` to `run_agent`. When tools are provided, enable Gemini function calling for those tools; when the model returns a function call, the caller (workflow step) is responsible for executing it—but we can pass a single `tool_runner(tool_name, args, context)` from the workflow step that has access to state (folder_id, etc.). So:

- `run_agent(..., tools=[search_ordinances_decl, search_mgl_decl], tool_runner=...)`.  
- Inside `run_agent`, use Gemini with `tools` and when response has a function call, call `tool_runner(tool_name, args)` and pass result back, loop until model returns text.  
- `tool_runner` is implemented in workflow.py or workflow_graph and has access to state (folder_id); it calls a small helper that does get_cached_rag_state(folder_id), then retrieve_and_build_context with query, then return formatted string.

### 4.4 Concrete steps

| File | Change |
|------|--------|
| `brain.py` | Add optional `tools: list[types.FunctionDeclaration] | None = None` and `tool_runner: Callable[[str, dict], str] | None = None` to `run_agent`. When both are provided: enable function calling (remove or override no_afc); in the request/response loop, if response contains a function call, call `tool_runner(function_name, args_dict)`, append result as a part, and call generate_content again until response is text-only. Define schema for search_ordinances: query (string), max_chunks (optional int, default 10). Same for search_mgl. |
| `workflow.py` or new `tools.py` | Implement `search_ordinances_impl(rag_state, query, max_chunks)` and `search_mgl_impl(rag_state, query, max_chunks)` (MGL: filter libraries by name "MGL" or config). Both return a string (XML or markdown). Implement `make_tool_runner(state)` that returns a callable `(tool_name, args) -> str` that dispatches to these impls using state["folder_id"] → get_cached_rag_state. |
| `workflow.py` | In `run_main_agent_step` (and optionally `run_legal_expert_step` or follow-on steps): build list of FunctionDeclaration for search_ordinances and search_mgl; build tool_runner from state; call run_agent(..., tools=..., tool_runner=...). Only enable for main agent (or also legal/follow-on) as desired. |
| `rag_loader.py` | No change required if we use existing retrieve_and_build_context; we may add a helper `retrieve_for_query(rag_state, query, selected_library_ids, top_k)` that returns (context_xml, report) for use by tools. Or use get_default_plan + retrieve_and_build_context with single query. |

### 4.5 Tool declarations (Gemini)

- **search_ordinances**  
  - Description: "Search the council knowledge base (ordinances, policies, documents) with a natural language query. Returns relevant excerpts."  
  - Parameters: query (string, required), max_chunks (integer, optional, default 10).  

- **search_mgl**  
  - Description: "Search the Massachusetts General Laws (MGL) library with a natural language query. Returns relevant MGL excerpts."  
  - Parameters: query (string, required), max_chunks (integer, optional, default 10).  

If no MGL library is configured, search_mgl can return a single sentence: "No MGL library is configured for this knowledge base."

### 4.6 MGL library identification

- Use a naming convention: any library whose **name** (case-insensitive) contains `"mgl"` or equals a configured value (e.g. in AppConfig or env `COUNCILFLOW_MGL_LIBRARY_NAME`) is treated as the MGL library. When building selected_library_ids for search_mgl, filter rag_state["libraries"] to that name; if none, return the "No MGL library configured" message.

---

## 5. Suggested implementation order

1. **RunEvent + checkpointer default**  
   - Add RunEvent table and CRUD; add event logging in graph nodes via config; app generates thread_id and updates run_id on events after persist. Then enable checkpointer by default (env unset → True).

2. **Pre-QA / Post-QA and Run history event log**  
   - Add pre_qa_output and qa_output to AnalysisRun and UI in run detail; add list_run_events and "Event log" in run detail.

3. **Run history filters**  
   - Extend list_analysis_runs with filters; add filter UI on Run history list page.

4. **PromptVersion View history / Restore**  
   - Add list_prompt_versions and get_prompt_version in db; add "View history" and "Restore" in prompt editor.

5. **Read-only tools**  
   - Implement search_ordinances and search_mgl (impl + tool_runner); add tools + tool_runner to run_agent; wire in run_main_agent_step (and optionally legal/follow-on).

---

## 6. Success criteria

- **Checkpointer:** With env unset or true, graph runs with SqliteSaver; state is persisted per thread_id.  
- **RunEvent:** Every node execution produces node_started and node_completed events; run detail shows event log.  
- **Pre-QA/Post-QA:** Runs that used QA have pre_qa_output and qa_output stored and visible in run detail.  
- **Filters:** Run history list can be filtered by task, status, date range, prompt version.  
- **PromptVersion:** User can view version list and restore a version into the editor, then save as new version.  
- **Tools:** Main agent (and optionally others) can call search_ordinances and search_mgl and receive relevant excerpts during the run.
