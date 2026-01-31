# CouncilFlow → Agentic Municipal Governance: Design Ideas

This document analyzes the CouncilFlow codebase and proposes design directions to evolve it into an **effective agentic solution for municipal governance**. It is intended to guide architecture decisions and to be consumed by LLM-based processing tools for implementation.

---

## Document outline (for LLM and human readers)

| Section | Purpose |
|--------|---------|
| **Design Decisions** | Resolved choices (D1–D8) and decision log. |
| **1. Current State Summary** | Architecture, existing behavior, strengths to preserve. |
| **1b. Post–Phase 3 Holistic Review** | Actual state after Phases 0–3; new considerations and opportunities for Phase 4+. |
| **2. Gap Analysis** | Current vs. agentic target. |
| **3. Design Ideas** | Multi-agent roles, workflow, tools, persistence, incremental path. |
| **4. Technical Considerations** | Rate limits, security, compatibility. |
| **5. Summary** | Resolved plan and end state. |
| **6. Agentic Libraries and Tooling** | LangGraph, Gemini, persistence, tools. |
| **7. Detailed Progression Plan** | Phase 0 (pre-work), Phases 1–5, success criteria; Phase 4 refined with current-state touchpoints. |
| **8. LangGraph-Specific Considerations** | Opportunities and constraints with LangGraph. |
| **9. Context Caching Strategy** | How context is built, shared, and optimized. |
| **10. Final Review and Phase 0** | Pre-work before Phase 1; migration effectiveness; doc readiness. |
| **11. Phase 5 Holistic Analysis** | Current state post–Phase 4; gaps and blockers; agentic best practices; prioritized recommendations for HITL and generic flows. |

---

**Goal for this plan:** Move from the current state to the **same user-visible functionality** (single analysis, optional legal review, integration of legal results, optional follow-on steps) but implemented with a **modern agentic framework** (LangGraph, resolved D2) that handles workflow orchestration, conditional branching, state/checkpointing, and sharing. We preserve the ability to run existing workflows; the framework is replaced, not the behavior. Later we can add an optional QA agent and human-in-the-loop (HITL).

---

## Design Decisions for Human-in-the-Loop

Before implementing Phase 1 and later phases, the following decisions should be made. Each is listed with options, implications, and a recommendation. **Resolved decisions** are recorded in the table below; update the doc as choices are made so the rest of the design stays consistent.

### Decisions to Make

| ID | Decision | Options | Implications | Recommendation |
|----|----------|----------|---------------|----------------|
| **D1** | **Where to store run/analysis data (Phase 1)** | **A:** Separate DB file (`council_runs.db`) for runs; keep `council.db` for config only. Export/import only copy `council.db`. **B:** Same DB; add run tables to `council.db` and change export/import to **config-only** (dump/restore only PromptTemplate, JsonSchema, AppConfig). | A: No change to export/import behavior (they keep copying one file); run data never in that file. Two files to back up. B: One DB; export/import logic must become table-selective (e.g. SQL dump of config tables only). | **A** — simpler, export/import stay “copy this file”; run data is naturally isolated. |
| **D2** | **Agentic framework (Phase 3)** | **A:** LangGraph. **B:** CrewAI. **C:** Stay with in-process workflow runner only (no framework). | A: Graph, state, checkpointer, interrupts; most control. B: Role-based, simpler API; less low-level control. C: No new deps; no framework features (resume, HITL) later. | **A (LangGraph)** — matches need for conditional edges, state, and future HITL. |
| **D3** | **Context caching in framework (Phase 3)** | **A:** Keep creating Gemini caches in nodes; pass `cache_name` in state. **B:** Use framework checkpointer to store state (including large context); single “context” node builds and caches once; later nodes read from state. | A: Minimal change; current `brain.create_gemini_cache` stays. B: Framework holds context; may duplicate large payloads in checkpointer. | **A** — reuse existing cache logic; state stays small. |
| **D4** | **QA agent configuration (Phase 4)** | **A:** Per-task (flag on PromptTemplate: “Use QA for this task”). **B:** Global (flag on AppConfig: “Use QA agent”). **C:** Both (global default + per-task override). | A: Some tasks get QA, others don’t. B: All or nothing. C: Flexible; more UI. | **A or C** — per-task allows “QA only for heavy analyses”; C if you want a global default. |
| **D5** | **QA agent placement (Phase 4)** | **A:** QA after `integrate_legal`, **before** follow-on chain. **B:** QA after follow-on chain (QA reviews full chain output). | A: QA integrates legal into main output; follow-ons then refine that. B: Follow-ons run first; QA polishes the final combined output. | **A** — “integrate legal, then QA, then optional follow-ons” matches “final output” semantics. |
| **D6** | **Run / framework state storage (Phase 3)** | **A:** Same file as run data (e.g. `council_runs.db`) for run rows and framework state/checkpoints. **B:** Separate SQLite file for framework state (e.g. `council_checkpoints.db`). | A: One file for runs + state; simpler backup. B: Isolate framework state from run records. | **A** — one “runs” DB for run rows and framework state; keep config DB separate. |
| **D7** | **Run table name** | **A:** `AnalysisRun`. **B:** `Run`. | Naming only; `AnalysisRun` is more explicit. | **A (AnalysisRun)**. |
| **D8** | **Recent runs UI location** | **A:** Sidebar (collapsible list). **B:** Run Analysis page (section above or below the runner). **C:** Dedicated “Run history” page (link from sidebar). | A: Always visible in sidebar. B: On same page as run. C: Clean separation; one more click. | **A or B** — A for quick access; B to keep run and history on one page. |

### Resolved Decisions Log

Record choices here so the rest of the design can be updated accordingly.

| ID | Decision | Resolved choice | Date |
|----|----------|-----------------|------|
| D1 | Run storage | **A** — Separate DB file (`council_runs.db`) for runs; export/import only `council.db`. | 2026-01-31 |
| D2 | Agentic framework | **A** — LangGraph. (Re-evaluated 2026-01-31: LangGraph preferred for conditional edges, checkpointer, HITL; see Framework re-evaluation below.) | 2026-01-31 |
| D3 | Context caching in framework | **B** — Framework holds state/context; single “context” build, later steps read from state. | 2026-01-31 |
| D4 | QA agent configuration | **A** — Per-task (flag on PromptTemplate: “Use QA for this task”). | 2026-01-31 |
| D5 | QA agent placement | **B** — QA after follow-on chain (QA reviews full chain output). | 2026-01-31 |
| D6 | Run/checkpoint storage | **A** — Same file as run data (`council_runs.db`) for run rows and framework state. | 2026-01-31 |
| D7 | Run table name | **A** — `AnalysisRun`. | 2026-01-31 |
| D8 | Recent runs UI location | **C** — Dedicated “Run history” page (link from sidebar); **admin users only**. | 2026-01-31 |

**How to use:** When you decide each item, fill in the “Resolved choice” column (and optionally the date). Then update any phase deliverables in Sections 3 and 7 that depend on that choice (e.g. if D1 = B, Phase 1 deliverables should say “config-only export/import” instead of “separate run DB”).

### Framework re-evaluation (pre-Phase 3): LangGraph vs CrewAI

Before Phase 3, we re-evaluated CrewAI vs LangGraph for CouncilFlow's needs (optional legal review, dynamic follow-on chain, future HITL, audit).

**CrewAI:** Supports **conditional tasks** (`ConditionalTask` with a `condition(output)`; if true the task runs, if false it is skipped). So "if legal questions → run Legal task" is possible via a conditional task or wrapper code. **Flows** offer router/conditional routing. Constraints: (1) Multiple conditional tasks in sequence can have output-flow bugs (task_outputs[-1]). (2) No built-in checkpointer; resume/audit requires manual persist to e.g. `council_runs.db`. (3) HITL ("Human Input on Execution" / "Human Feedback in Flows") exists but is less first-class than LangGraph's `interrupt()`.

**LangGraph:** **Conditional edges** are native: after the main agent node, a routing function reads state (e.g. `legal_questions`) and returns the next node ("legal_agent" or "integrate"). No wrapper needed. **Dynamic follow-on chain** maps to a graph loop: conditional edge back to a "follow_on" node until no more `verifier_id`. **Checkpointer** (e.g. `SqliteSaver`) gives resume and audit without custom persist logic. **`interrupt()`** (Phase 5 approval) is built-in: graph pauses, state checkpointed; Streamlit shows approval UI; resume with `Command(resume=...)`. **State** is an explicit TypedDict; our Phase 2 workflow state maps 1:1 to graph state.

**Conclusion:** LangGraph is the **better fit** for CouncilFlow: optional legal and follow-on branching are native; checkpointing and HITL are first-class; Phase 2 state schema becomes the graph state with no redesign. CrewAI can achieve the same behavior with conditional tasks or wrapper code, but we would be working around the framework for branching and checkpointing. **D2 is set to A (LangGraph).**

---

## 1. Current State Summary

### 1.1 Architecture Overview

| Layer | Components | Role |
|-------|------------|------|
| **UI** | `app.py` (Streamlit) | Auth, Drive connection, task selection, Run Analysis, prompt CRUD (with version display), model/settings, Run history (timezone + prompt version + copy markdown) |
| **Orchestration** | `workflow_graph.py` (LangGraph) + `workflow.py` | StateGraph: plan_retrieval → retrieve_context → create_cache → main_agent → extract_legal_questions → [conditional] legal_agent or integrate_legal → follow_on_chain → finalize. Steps live in `workflow.py`; graph nodes wrap them and pass callbacks for status. |
| **LLM / “Brain”** | `brain.py` | Gemini client, embeddings, retrieval planner, query expansion, re-ranking, `run_agent()` (single shot), `extract_legal_questions()`, `create_gemini_cache()` |
| **RAG** | `rag_loader.py`, `rag.py`, `librarian.py`, `rag_cache.py` | Core + Libraries (Drive), hybrid retrieval (BM25 + semantic, RRF), optional planner/expansion/rerank, context XML + Gemini cache |
| **Persistence (config)** | `db.py` | PromptTemplate (current_version, chains via `verifier_id`, legal via `legal_expert_prompt_id`), **PromptVersion** (snapshots on save), JsonSchema, AppConfig |
| **Persistence (runs)** | `runs_db.py` | AnalysisRun in `council_runs.db`: output, chain_steps, **prompt_version**, **stored_timezone** (UTC), retrieval_report_summary, model_used, etc. Never exported/imported. |

### 1.2 What Exists Today

- **Single main agent**: One prompt + one RAG context + one `run_agent()` call per “Run Analysis.”
- **Specialized sub-calls**: Legal expert (separate RAG plan/retrieve + separate prompt) triggered by parsing “Legal Questions Requiring Expert Review” from main output.
- **Chained follow-ons**: Sequential follow-on prompts (`verifier_id`), each with its own plan/retrieve/cache/run; output of step N is `previous_output` for step N+1.
- **No tools**: No function calling, no search/API/tool use inside the model.
- **No multi-turn**: Each `run_agent()` is a single request/response; no conversational loop.
- **No explicit workflow engine**: Order is hardcoded (main → legal expert → follow-on chain).
- **Run/audit persistence (Phase 1):** Every run is persisted to `AnalysisRun` in `council_runs.db`; Run history page (admin-only) lists runs and shows detail with timezone-aware times, prompt version, and copy-markdown.

### 1.3 Strengths to Preserve

- **RAG pipeline**: Core + Libraries, hybrid retrieval, optional planner/expansion/rerank, and Gemini context caching are solid foundations for grounding agents in municipal documents.
- **Prompt templates and chains**: Reusable tasks and follow-on steps map well to “analysis → verification → formatting” workflows.
- **Legal expert pattern**: Routing to a dedicated legal prompt with its own RAG context is a good precedent for role-specific agents.
- **Structured output**: JSON Schema sidecars and optional structured output support future tool/event schemas.

---

## 1b. Post–Phase 3 Holistic Review (Current State and Phase 4 Readiness)

This section summarizes the **actual state** after Phases 0–3 and the **post–Phase 3 enhancements** (prompt versioning, run timezone, status callbacks, copy markdown, prompt editor version display). It then lists **new considerations and opportunities** for Phase 4 and beyond.

### 1b.1 What Was Delivered (Phases 0–3 + Enhancements)

| Area | Delivered |
|------|-----------|
| **Run persistence** | `runs_db.py`: `AnalysisRun` in `council_runs.db`; insert on success/failure; Run history page (admin-only). |
| **Run metadata** | `prompt_version` (PromptTemplate.current_version at run time); `stored_timezone` (e.g. UTC); migrations for existing DBs. |
| **Workflow** | `workflow.py`: dict-based state, step functions (plan → retrieve → cache → main_agent → extract_legal → legal_expert → integrate → follow_on_chain), optional callbacks for progress. |
| **LangGraph** | `workflow_graph.py`: StateGraph with nodes wrapping workflow steps; conditional edge after extract_legal_questions (legal_agent vs integrate_legal); single follow_on_chain node; finalize node sets last_run_context_stats and status=completed. State wrapped as `{"data": workflow_state_dict}`. |
| **Context** | Context built in retrieve_context + create_cache nodes; `context_xml` and `cache_name` in state; legal_agent builds its own legal context (same as design). No single “context node” owning everything—nodes build and pass cache_name (D3 in practice aligns with “build in nodes, pass in state”). |
| **Checkpointer** | SqliteSaver on `council_runs.db` available via `COUNCILFLOW_USE_GRAPH_CHECKPOINTER=1`; **default off** because state contains non-serializable objects (ORM `selected_prompt`, callables like `build_prompt_variables`). |
| **UI progress** | App passes `_callbacks` (write, update_label) from `st.status` into state; graph nodes and workflow steps call `_cb(state, "write", ...)` for granular status (planning, retrieval, cache, main agent, legal, follow-on). |
| **Prompt versioning** | `db.py`: `PromptTemplate.current_version`; `PromptVersion` table (snapshot on every save); `save_prompt()` snapshots then increments version. Prompt editor shows “Current version: N”; Run history and results show prompt version; runs record `prompt_version`. Revert-to-version tooling deferred. |
| **Run history UX** | Stored times in UTC, displayed in local time; prompt version in list and detail; copy markdown for run output. |
| **Export/import** | Unchanged: only `council.db` (config); run data never exported or imported. |

### 1b.2 New Considerations After Phases 0–3

1. **Checkpointer and serialization**  
   Enabling the LangGraph checkpointer (D6) for resume/audit requires **serializable state**. Today state holds `selected_prompt` (ORM object) and `build_prompt_variables` (callable). Options for Phase 5 (HITL): (a) store only `prompt_template_id` and resolve `selected_prompt` at node entry; (b) pass callbacks outside state (e.g. config or closure); (c) keep checkpointer off until state is refactored. Phase 4 (QA) does not require the checkpointer.

2. **D3 (context in framework)**  
   Resolved D3 = B said “framework holds context; single context build.” In practice we kept building context **inside nodes** (retrieve_context, create_cache) and passing `cache_name` in state—i.e. context is “in” the graph state but built by nodes, not by a single dedicated context node. This is sufficient and matches “context built once, later nodes read from state.” No change needed for Phase 4.

3. **PromptVersion and revert**  
   All prompt versions are stored in `PromptVersion`. Revert tooling (e.g. “Restore to version N”) is not yet implemented; Phase 4 does not depend on it. When adding revert, the editor would load a `PromptVersion` row and write its fields back into the prompt (and optionally create a new version).

4. **Run history and QA (Phase 4)**  
   If Phase 4 stores `pre_qa_output` and `qa_output` on `AnalysisRun` for audit, Run history can show “Pre-QA” / “Post-QA” in the detail view (e.g. expanders or tabs). The existing Run history timezone and copy-markdown UX applies to whichever output we show as primary.

5. **Status callbacks**  
   Any new node (e.g. QA agent) should receive the same `_callbacks` from state and use `_cb(state, "write", ...)` (and optionally `update_label`) so progress appears in the same `st.status` container. The pattern is established in workflow_graph node wrappers and workflow steps.

### 1b.3 Opportunities for Phase 4 and Beyond

| Opportunity | Description |
|-------------|-------------|
| **QA as PromptTemplate** | Phase 4 can implement QA with a **fixed system prompt** (e.g. “Review and polish the following…”) or a **dedicated PromptTemplate** (e.g. “QA Review”) for reusable, editable QA instructions. The latter allows per-environment tuning and versioning via existing PromptVersion. |
| **pre_qa_output / qa_output** | Store pre-QA and post-QA text on `AnalysisRun` (optional columns or JSON) for audit and Run history “before/after” view. |
| **RunEvent-style logging** | Phase 4 can add lightweight “QA started” / “QA completed” events (e.g. in a JSON column or a small RunEvent table) to prepare for Phase 5 audit and HITL. |
| **Filter Run history** | Run history already shows prompt version and timezone; later: filter by prompt version, date range, task name, or status. |
| **PromptVersion UI** | Expose version history in the prompt editor (e.g. “View history” → list of versions with saved_at; “Restore” → load that version into the form). |
| **Serializable state for checkpointer** | Refactor state so that by the time it is passed to the graph, it contains only serializable data (e.g. prompt_template_id, no ORM; callbacks passed via invoke config). Then enable SqliteSaver by default for resume and Phase 5 HITL. |

---

## 2. Gap Analysis: Current vs. Agentic Municipal Governance

| Dimension | Current | Agentic Target |
|----------|---------|----------------|
| **Agent model** | Single “analyst” + one optional “legal” sub-call | Multiple roles (analyst, legal, clerk, chair, etc.) that can be composed and delegated |
| **Control flow** | Fixed sequence in app.py | Explicit workflows (graphs/state machines) with branching and human gates |
| **Tools** | None | Lookups (ordinances, agendas), compliance checks, draft creation, notifications |
| **Autonomy** | One shot per step | Plan → act (tools) → observe → replan until done or human step |
| **Human-in-the-loop** | Implicit (user clicks Run) | Explicit approval/escalation steps and audit trail |
| **Memory / state** | Session only | Persistent runs, context, and decisions for accountability |
| **Routing** | User picks task; legal is triggered by text parsing | Router or orchestrator assigns tasks and sub-agents by intent/content |

---

## 3. Design Ideas

### 3.1 Multi-Agent Roles (Municipal Personas)

Introduce **role-based agents** that share the same RAG backend but have different prompts, tools, and responsibilities.

- **Analyst**: Current main prompt—summarize, analyze, recommend; emit legal questions when needed.
- **Legal**: Current legal expert—answer legal questions with targeted RAG (already a separate RAG + prompt).
- **Clerk**: Agenda formatting, motion numbering, deadline tracking (could use tools: “list upcoming deadlines”, “format motion”).
- **Chair / Facilitator**: Synthesize multiple inputs, suggest next steps, draft chair script (optional).

**Implementation direction**:  
- Keep one “brain” (Gemini) but multiple **agent configs** (name, system prompt, allowed tools, default RAG plan).  
- Store agent configs in DB (e.g. `AgentConfig`: name, prompt_template_id, tools[], role_description).  
- Orchestrator chooses which agent(s) to invoke and in what order (see 3.2).

### 3.2 Workflow Engine (Explicit Process)

Move from **hardcoded pipeline** to **declarative workflows** so municipal processes (e.g. “Agenda packet review”, “Motion analysis with legal check”) are first-class.

- **Nodes**: Start, End, Human Approval, “Run Agent” (agent_id + task inputs), “Run RAG only” (e.g. fetch context for human), “Condition” (e.g. “has legal questions?”).
- **Edges**: Next node, optional conditions (e.g. if legal_questions → Legal Agent node).
- **State**: Per-run workflow state (current node, inputs, outputs, who approved what).

**Implementation direction**:  
- Add `Workflow` and `WorkflowRun` (or equivalent) to DB; store graph as JSON (nodes/edges) or use a small DSL.  
- A **workflow runner** in Python (or a dedicated module) executes the graph: evaluates conditions, calls the right agent/RAG step, and pauses at Human Approval nodes.  
- Current “Run Analysis” becomes one default workflow: Main Agent → [if legal questions] Legal Agent → Follow-on chain (or map follow-on chain to a linear sub-graph).

### 3.3 Tool Use (Function Calling)

Give agents **tools** so they can look up and act on municipal data instead of only reading a static RAG context.

- **Read-only tools**: “Search ordinances by keyword”, “Get agenda item by id”, “List deadlines for next 30 days” (could query Drive, DB, or external APIs).
- **Draft/assist tools**: “Create draft motion”, “Format section for packet” (write to a staging area or return structured text for human copy/paste).
- **Compliance tools**: “Check quorum rules for this meeting”, “Verify posting requirements” (wrap RAG or rule engine).

**Implementation direction**:  
- Define tool schemas (name, description, parameters as JSON Schema); register per agent.  
- Use Gemini **automatic function calling** (or structured tool calls) in `run_agent()`: when the model requests a tool, execute it (in a sandbox), inject result into conversation, and continue until the model returns a final answer.  
- Tools can call existing RAG (e.g. “search_ordinances” → run retrieval, return top chunks) or external services.  
- Start with 2–3 read-only tools to avoid complexity; add draft/compliance tools once the loop is stable.

### 3.4 Orchestrator / Router

Add a **router** that decides which agent(s) to run and with what inputs, instead of the user always picking one task and the app always running the same sequence.

- **Input**: User request (e.g. “Analyze this memo and flag legal and procedural issues”) + optional document(s).  
- **Router**: Lightweight LLM or rule-based classifier: “analysis + legal” → run Analyst then Legal; “format only” → Clerk only.  
- **Output**: Ordered list of (agent_id, input_spec) or a workflow_id + initial payload.

**Implementation direction**:  
- Single “router” prompt or small model: given user message + doc metadata, output JSON e.g. `{ "workflow_id": "motion_analysis", "params": {} }` or `{ "agents": ["analyst", "legal"] }`.  
- Orchestrator (in app or workflow engine) then runs the chosen workflow or agent list.  
- Keeps backward compatibility: “Run Analysis” with a selected task can be “router says: run this task only.”

### 3.5 Human-in-the-Loop and Audit

Make **approval and audit** explicit so governance is traceable.

- **Approval steps**: In a workflow, mark certain nodes as “Human Approval”: show output to user, wait for Approve/Reject/Edit; then continue.  
- **Audit log**: For each run, store: user_id, timestamp, workflow_id, node, agent_id, input hashes, output summary, approval result (if any).  
- **Versioning**: When a prompt or workflow is updated, either retain “run used prompt version X” or snapshot the config at run time.

**Implementation direction**:  
- Add `Run` (or `WorkflowRun`) table: run_id, user_id, workflow_id, started_at, status, current_node, payload (JSON).  
- Add `RunEvent`: run_id, node_id, agent_id, event_type (agent_started, agent_completed, human_approval_requested, human_approved), payload, timestamp.  
- UI: “Pending approvals” list; when user approves, workflow runner resumes and logs the event.

### 3.6 Persistent Run History and Context

Move from **session-only results** to **stored runs** so users can revisit and so future agents can reference prior work.

- **Run history**: List of past runs (who, when, which task/workflow, status); click to see full output and retrieval report.  
- **Context for future runs**: Optional “prior run id” or “prior run summary” as input to the next run (e.g. “Continue from run #42”).
- **Data isolation**: Run/analysis results must **not** be exported or imported. Export/import is used primarily to **sync prompts and app config** between environments; analysis results stay local to each instance.

**Implementation direction**:  
- Save to DB after each successful run: run_id, user, task/workflow, inputs (hashes or refs), outputs (or link to blob/store), RAG report summary.  
- Expose in UI: dedicated **Run history** page (link from sidebar), **admin users only**; list runs and “Open run #id” so that output and chain steps are visible after refresh (resolved: D8 = C).  
- **Exclude run/analysis data from export/import**: Use a separate DB file for runs, `council_runs.db`; export/import only `council.db` (config). Run data is never in the exported file (resolved: D1 = A).

### 3.7 Incremental Path (Suggested Order)

See **Section 7 (Detailed Progression Plan)** for the full phase-by-phase plan. In brief (with resolved decisions applied):

0. **Phase 0:** Pre-work — run DB path, `runs_db.py` (or equivalent), export/import scope, Phase 1 touchpoints; optional `.gitignore` for `council_runs.db`.  
1. **Phase 1:** Persist runs in **`AnalysisRun`** in **`council_runs.db`**; exclude run data from export/import; dedicated **Run history** page (admin only).  
2. **Phase 2:** Extract pipeline into workflow runner + state (in-process).  
3. **Phase 3:** Introduce **LangGraph**; same workflow as crew; context built once and shared via framework state; run/state in `council_runs.db`.  
4. **Phase 4:** Optional QA agent (per-task flag), placed **after** follow-on chain, for final integrated output.  
5. **Phase 5 (later):** Human approval, RunEvent, more workflows, tools.

---

## 4. Technical Considerations

- **Rate limits and cost**: Multi-agent and tool loops increase Gemini calls. Keep pacing, caching, and optional “fast path” (single agent, no tools) as in current design.  
- **Security**: Tools that write or send data must be sandboxed and reviewed; prefer “draft” outputs for human copy/paste before any automated publish.  
- **Streamlit**: The current UI can host “Run”, “Approval”, and “History”; for complex workflow UIs, consider a dedicated “workflow run” page with node status and approval buttons.  
- **Backward compatibility**: Default workflow should mirror current behavior so existing deployments keep working while new features are opt-in.

---

## 5. Summary

CouncilFlow already provides **strong RAG, context caching, prompt chains, and a legal expert pattern**. The **detailed progression plan** (Section 7) takes us from the current state to the **same functionality** delivered by an agentic framework. Resolved decisions (Section “Design Decisions for Human-in-the-Loop”):

- **Phase 0 (pre-work):** Run DB path (`data_path("council_runs.db")`), separate module `runs_db.py`, export/import confirmed config-only, Phase 1 touchpoints documented; optional `.gitignore` for `council_runs.db`.  
- **Phase 1:** Persist analysis results in **`AnalysisRun`** in **`council_runs.db`**; dedicated **Run history** page (sidebar link), **admin users only**; run data **excluded from export/import** (only `council.db` is exported/imported).  
- **Phase 2:** Extract the current pipeline into a single workflow runner and state (no new framework).  
- **Phase 3:** Replace the runner with **LangGraph**; same workflow as a crew; context built once and shared via framework state; run/state in `council_runs.db`.  
- **Phase 4:** Optional **QA agent** (per-task flag on PromptTemplate), placed **after** the follow-on chain, to review full chain output and write the final document.  
- **Phase 5 (later):** Human-in-the-loop, RunEvent audit, more workflows, tools.

End state: single analysis, optional legal review, integration of legal results, optional follow-on chain, optional per-task QA (after follow-ons), and persistent run history (admin-only Run history page, timezone + prompt version + copy markdown), all orchestrated by **LangGraph**—without changing what users can do today. Export/import continues to sync only prompts and app config; run/analysis data is never exported or imported. See **Section 1b** for post–Phase 3 holistic review and Phase 4 opportunities.

---

## 6. Agentic Libraries and Tooling (Don’t Reinvent the Wheel)

Below are **concrete libraries and tools** that fit CouncilFlow’s stack (Python, Gemini, Streamlit) and the design ideas above. Use them instead of building workflow engines, tool loops, or human-in-the-loop from scratch.

### 6.1 Orchestration and Workflow

| Library | What it gives you | Fit for CouncilFlow |
|--------|--------------------|----------------------|
| **LangGraph** | Graph-based workflows: nodes = steps/agents, edges = control flow, state = shared TypedDict. Checkpointing, persistence, **interrupts** for human-in-the-loop, streaming. | **Strong.** Explicit workflows, conditional edges (e.g. “has legal questions?” → Legal node), approval nodes, audit via thread_id + checkpointer. |
| **LangGraph** | Role-based “crew” of agents (Analyst, Legal, Clerk) with goals and tools; sequential or hierarchical flows. | **Good.** Maps to municipal “roles”; simpler API than LangGraph; less low-level control over state and branching. |
| **AutoGen** (Microsoft) | Multi-agent **conversational** orchestration (agents chat with each other). | **Narrower.** Best for brainstorm/collab; less suited to deterministic “run analyst → then legal → then approve” pipelines. |

**Recommendation:**  
- **LangGraph** if you want **explicit workflows**, **human approval steps**, and **state/audit** (persistence, replay). It has first-class `interrupt()` and `Command(resume=...)` for “pause for approval, then continue.”  
- **CrewAI** if you prefer a **role-based team** metaphor and a simpler API; it supports Gemini and is documented by Google.

### 6.2 Gemini Integration

| Option | Use case |
|--------|----------|
| **google-genai** (current) | You already use it. Native **function calling**: declare tools via `types.Tool(function_declarations=[...])`, pass to `GenerateContentConfig`; model returns function calls; you execute and pass results back. No extra framework needed for “agent with tools.” |
| **langchain-google-genai** | `ChatGoogleGenerativeAI` for Gemini. Use this **if you adopt LangGraph**: LangChain/LangGraph expect an LCEL chat model; `ChatGoogleGenerativeAI` plugs in and supports `.bind_tools(tools)` for tool use inside the graph. |
| **LangGraph + Gemini** | LangGraph’s `Agent` accepts an LLM; use `gemini/gemini-2.5-pro` (or similar) as the model. Official examples: [Gemini + LangGraph](https://ai.google.dev/gemini-api/docs/crewai-example). |

So: **tools-only** → stay with `google-genai` and add function calling. **Workflow + tools + HITL** → LangGraph + `langchain-google-genai`, or CrewAI + Gemini.

### 6.3 Human-in-the-Loop and Approval

| Tool | What it gives you |
|------|--------------------|
| **LangGraph `interrupt()`** | In any node, call `interrupt({"question": "Approve?", "details": state})`. Graph pauses, state is checkpointed; you show UI; on “Approve” you run `graph.invoke(Command(resume=True), config=config)`. Same `thread_id` resumes. Built-in, no extra service. |
| **LangGraph `interrupt_before`** | Compile with `interrupt_before=["node_name"]` to pause *before* that node. Simpler but less flexible than `interrupt()`. |
| **Temporal** | Durable workflows and “Signals” for human approval (workflow waits for signal). Heavier: separate Temporal server and worker; use if you need **cross-process, durable** approval and retries at scale. |

For most municipal governance flows (single app, Streamlit UI), **LangGraph’s `interrupt()` + checkpointer** is enough; Temporal is optional for larger or multi-service deployments.

### 6.4 Persistence and Audit

| Approach | What it gives you |
|----------|--------------------|
| **LangGraph checkpointer** | `MemorySaver()` (dev) or `SqliteSaver(conn)` (production). Persists graph state per `thread_id`; you can resume, inspect, and replay. Fits “run history” and “resume after approval.” |
| **Your own DB (Run / RunEvent)** | Still valuable: store run_id, user_id, workflow_id, timestamps, approval decisions, and high-level payloads. Gives you **cross-framework** audit (e.g. “all runs by user X”) and reporting. Use in addition to LangGraph’s checkpointer. |

### 6.5 Tools (Function Calling)

| Approach | Use case |
|----------|----------|
| **Gemini API** | Define tools as `FunctionDeclaration` (name, description, parameters as JSON Schema). In `run_agent()` (or equivalent), enable `automatic_function_calling` or parse `response.candidates[0].content.parts` for `FunctionCall`; execute; append `FunctionResponse`; call again until model returns text. No new lib. |
| **LangChain `@tool`** | Decorate Python functions; LangChain turns them into a schema for the LLM. Use with `ChatGoogleGenerativeAI.bind_tools(tools)`. Nice if you’re already in LangGraph/LangChain. |
| **Instructor** | Pydantic-based structured output from LLMs (e.g. “extract legal questions” as a typed object). You already have `extract_legal_questions`; Instructor can replace ad-hoc parsing if you want stronger typing. |

So: **minimal change** → add Gemini function calling in `brain.py`. **With LangGraph** → use LangChain tools + `bind_tools` in the graph.

### 6.6 Suggested Stack (Concrete)

- **Option A – Minimal (no new framework)**  
  - Keep current pipeline in `app.py`.  
  - Add **Gemini function calling** in `brain.run_agent()`: 1–2 read-only tools (e.g. “search_ordinances” that calls your RAG).  
  - Add **Run / RunEvent** tables and save each run; add a “Recent runs” UI.  
  - **Libraries:** none new; `google-genai` only.

- **Option B – Workflow + HITL (recommended)**  
  - Introduce **LangGraph** for the “Run Analysis” flow: nodes = plan_retrieval, retrieve_context, main_agent, legal_agent (conditional), approval (optional), follow_ons.  
  - Use **langchain-google-genai** (`ChatGoogleGenerativeAI`) as the model in the graph; your existing RAG stays as a **tool** (“get_context”) or as pre-built context injected into the first agent node.  
  - Use **interrupt()** (or `interrupt_before`) for “Human Approval” node; **SqliteSaver** (or Postgres) for checkpointer.  
  - Keep **Streamlit** for UI: on interrupt, show approval form; on submit, call `graph.invoke(Command(resume=...), config=config)`.  
  - **Libraries:** `langgraph`, `langchain-google-genai`, `langchain-core` (and optionally `langchain` for tools).

- **Option C – Role-based team**  
  - Use **CrewAI**: define Analyst, Legal, Clerk agents with Gemini; one Crew or Flow = “motion analysis” (Analyst → Legal → output).  
  - Simpler than LangGraph; less explicit control over state and branching.  
  - **Libraries:** `crewai`, `crewai-tools`; Gemini via CrewAI’s Gemini integration.

---

## 7. Detailed Progression Plan: Current State → Agentic Same-Functionality

This section provides a **phase-by-phase plan** to move from the current hardcoded pipeline to an agentic framework while **preserving the same functionality**: single analysis, optional legal review, integration of legal results, and optional follow-on prompts. Optional QA agent and more complex workflows come later.

**Principles:**
- **Parity first:** Each phase keeps the app usable; we do not remove behavior before replacing it.
- **Run data isolation:** Analysis/run results are persisted for history and audit but **never** exported or imported (export/import = prompt/config sync only).
- **Framework handles context:** The chosen agentic framework (**LangGraph**, resolved D2) will own context caching and sharing; we migrate from ad-hoc cache handling in `app.py`/`brain.py` to the framework’s state (resolved D3 = B: context built once, later steps read from state). Run data and framework state use `council_runs.db` (resolved D6 = A).

---

### Phase 0: Pre-work (Before Phase 1)

**Objective:** Make Phase 1 and the later LangGraph migration smoother by establishing paths, run DB location, and code touchpoints. No user-facing behavior change.

**Deliverables:**

1. **Run database path and module**  
   - **Path:** Use `data_path("council_runs.db")` (same directory as `council.db`; `paths.py` already provides `data_path`). Document in this design and in code that `council_runs.db` is the run store; `council.db` remains config only.  
   - **Module:** Either (a) add a separate module `runs_db.py` that owns the `council_runs.db` engine and `AnalysisRun` model/CRUD, or (b) extend `db.py` with a second engine and `AnalysisRun` in the same file but with a distinct `RUNS_DB_PATH` and engine. **Recommendation:** Separate `runs_db.py` so config DB (`db.py`) and run DB stay clearly separated; export/import in `db.py` never touch runs.  
   - **.gitignore:** Add `council_runs.db` to `.gitignore` if run data should not be committed (recommended for local/dev); document in README.

2. **Export/import scope**  
   - Confirm that `export_database` and `import_database` in `db.py` use only `DB_PATH` (i.e. `council.db`). They must not reference `council_runs.db`. No code change if they already use `DB_PATH` only; add a one-line comment that export/import are config-only and run data lives in `council_runs.db`.

3. **Phase 1 touchpoints (for implementation)**  
   - **New file:** `runs_db.py` — `RUNS_DB_PATH`, engine, `AnalysisRun` model, `init_runs_db()`, `insert_analysis_run()`, `get_analysis_run_by_id()`, `list_analysis_runs(limit, username_filter=None)`.  
   - **app.py:** After successful Run Analysis (and on failure), call `runs_db.insert_analysis_run(...)` with the same data you currently put in session state. Add sidebar link “Run history” (visible only when `is_admin`) that navigates to a dedicated Run history page. New page: list runs via `runs_db.list_analysis_runs(20)`, click to view single run via `runs_db.get_analysis_run_by_id(id)`.  
   - **paths.py:** No change required; use `data_path("council_runs.db")` in `runs_db.py`.

4. **Dependencies**  
   - No new dependencies in Phase 0. Phase 1 uses existing SQLModel/SQLAlchemy (same as `db.py`).

**Success criteria:** Path and module strategy documented and decided; export/import confirmed config-only; Phase 1 implementer can create `runs_db.py` and wire app.py without ambiguity.

---

### Phase 1: Persist Analysis Results (No Framework Change)

**Objective:** Store every analysis run in the database and expose “Recent runs” in the UI. Run data is excluded from export/import.

**Deliverables:**

1. **New table: `AnalysisRun`** (resolved: D7 = A)  
   - Stored in a **separate SQLite file** `council_runs.db` (resolved: D1 = A) so that existing `export_database` / `import_database` (which copy `council.db`) continue to sync only prompts and app config; run data is never in the exported file.  
   - Fields (minimal): `id`, `username`, `task_name`, `prompt_template_id`, `folder_id` (KB), `started_at`, `completed_at`, `status` (`running` | `completed` | `failed`), `input_summary` (e.g. content length or hash), `output_text` (full result), `output_mode`, `has_legal_review`, `legal_questions` (JSON or text), `legal_expert_output` (optional), `chain_steps` (JSON: list of step names + outputs for follow-ons), `retrieval_report_summary` (optional), `model_used`, `error_message` (if failed).

2. **Exclude run data from export/import**  
   - Runs live in `council_runs.db`; `export_database` / `import_database` operate on `council.db` only. No run data in the exported file.

3. **App changes**  
   - After a successful “Run Analysis” (main + optional legal + follow-ons), insert one row into `AnalysisRun` (in `council_runs.db`) with the full output, chain steps, and metadata.  
   - On failure, insert a row with `status=failed` and `error_message`.  
   - Add a **“Run history”** page (resolved: D8 = C): dedicated page linked from the sidebar, listing runs (e.g. last 20) with task name, date, user; click to open a read-only view of that run’s output and chain steps. **Access restricted to admin users only.**

4. **No change** to the current orchestration (still inline in `app.py`), RAG, or caching logic.

**Success criteria:** Every run is persisted in `council_runs.db`; admin users can open the Run history page and view any run; export/import does not include run data.

---

### Phase 2: Extract Current Pipeline into a Single “Workflow” (Still In-Process)

**Objective:** Move the current sequence (plan → retrieve → cache → main agent → extract legal questions → [if any] legal expert → integrate → follow-on chain) into a **single workflow definition** executed by a small runner (e.g. a Python function that takes state and runs steps in order). No new framework yet; just a clear abstraction so the same flow can later be mapped to **LangGraph** nodes (resolved D2 = A).

**Deliverables:**

1. **Workflow state type**  
   - Define a state object (e.g. TypedDict or Pydantic) used across steps. **State schema (all keys used by Phase 2 and needed by Phase 3/4):**  
     - **Inputs:** `task_name`, `template_text`, `user_content`, `folder_id`, `prompt_template_id`, `username`, `selected_prompt` (or id).  
     - **RAG:** `rag_state`, `selected_library_ids`, `top_k_map`, `context_xml`, `cache_name`, `retrieval_report`; for legal: `legal_context_xml`, `legal_cache_name` (optional).  
     - **Outputs:** `main_output`, `legal_questions`, `legal_expert_output`, `final_output`, `chain_outputs` (list of `{step_name, output}`), `output_mode`.  
     - **Control:** `current_step`, `error`, `status`.  
   - Use this same schema (or a superset) when mapping to LangGraph in Phase 3 so only the runner is replaced, not the state shape.

2. **Step functions**  
   - Refactor `app.py` so that each logical step is a function that takes state and returns updated state: e.g. `plan_retrieval_step(state)`, `retrieve_context_step(state)`, `create_cache_step(state)`, `run_main_agent_step(state)`, `extract_legal_questions_step(state)`, `run_legal_expert_step(state)` (conditional), `integrate_legal_step(state)`, `run_follow_on_chain_step(state)`.  
   - Runner: a simple loop or function that calls these in order and branches on “has legal questions?” and “has follow-on prompt?”.

3. **Persistence**  
   - At the end of the workflow, persist to `AnalysisRun` as in Phase 1 (unchanged).

4. **UI**  
   - “Run Analysis” still triggers the same flow; it now goes through the workflow runner. No UX change.

**Success criteria:** Behavior identical to Phase 1; orchestration lives in a single workflow runner and state object; ready to map steps to LangGraph nodes (Phase 3).

**Phase 2 executed (2026-01-31):** Added `workflow.py` with dict-based state, step functions (`plan_retrieval_step`, `retrieve_context_step`, `create_cache_step`, `run_main_agent_step`, `extract_legal_questions_step`, `run_legal_expert_step`, `integrate_legal_step`, `run_follow_on_chain_step`), and `run_workflow(state, callbacks)`. App calls `run_workflow` with a `with_step` callback that wraps each step in `st.status`. Persistence and Run history unchanged (Phase 1).

**Phase 2 holistic review (2026-01-31):**

- **State schema:** Implemented in `workflow.py` as a single mutable dict. Keys align with the design: inputs (`task_name`, `template_text`, `user_content`, `folder_id`, `prompt_template_id`, `username`, `selected_prompt`, `rag_state`, `build_prompt_variables`, `main_full_template`, schema refs, session cache fields); RAG (`selected_library_ids`, `top_k_map`, `context_xml`, `cache_name`, `retrieval_report`, `query_phrases`, `legal_context_xml`, `legal_cache_name`); outputs (`main_output`, `main_content`, `legal_questions`, `legal_expert_output`, `final_output`, `chain`, `chain_outputs`, `pipeline_step_results`, `legal_questions_by_step`, `chain_timings`, `output_mode`, `last_run_context_stats`); control (`error`, `status`, `_callbacks`). Session cache keys (`run_cache_key`, `cache_name`, etc.) are updated for reuse. No Streamlit dependency; callbacks are optional. **Phase 3:** This dict can become a LangGraph TypedDict (or equivalent) with the same keys; only the runner is replaced.

- **Step coverage:** All eight logical steps are implemented. Legal is **conditional** inside `run_legal_expert_step` (no-op when no legal questions or no legal expert prompt). Follow-on is a **loop** in `run_follow_on_chain_step` (while `verifier_id`, with cycle detection). The runner groups steps for UI: plan → (retrieve + cache) → main_agent → (extract_legal + legal_expert + integrate) → follow_on_chain. Branching is implicit (legal step no-ops; follow-on loop); Phase 3 will make these **explicit** conditional edges and a follow-on subgraph/loop.

- **Error handling:** Steps raise `WorkflowError(message, details)` on failure. App catches `WorkflowError`, shows `st.error(e.message)` and optional `st.caption(e.details)`, persists a failed run to `AnalysisRun` with `status=failed` and `error_message`, then `st.stop()`. Failures covered: context too small (retrieve step); cache too small/too large/503 (create_cache_step); cache expiry and retry (main_agent, follow-on); legal expert or follow-on step errors. No silent failures; all paths either complete or raise.

- **UI integration:** App builds initial state from session (selected prompt, RAG state, schema, session cache keys). Callback `with_step(label, state, step_fn)` wraps each group in `st.status(label)`; step callbacks (`write`, `update_label`) feed progress. On success, app syncs state to `st.session_state` (last_result, pipeline_step_results, legal fields, chain, cache keys, run_cache_key). Persistence (Phase 1) runs after sync: `runs_db.insert_analysis_run(...)` with full output and metadata.

- **Persistence:** Success: one `AnalysisRun` row with `status=completed`, output_text, chain_steps, legal fields, retrieval_report_summary, model_used. Failure: one row with `status=failed`, error_message, no output_text. Run history page (admin-only) unchanged; reads from `council_runs.db`.

- **Phase 3 readiness:** State shape is stable; steps are side-effect-only on state (no return value). Each step can map 1:1 to a LangGraph node. Conditional “run legal” becomes a **conditional edge** after main_agent (route to `legal_agent` or `integrate_legal` based on `legal_questions`). Follow-on chain becomes a **loop**: conditional edge from integrate to `follow_on` node; from follow_on, conditional edge back to follow_on until no `verifier_id`, then to END. Checkpointer (D6) and optional HITL (Phase 5) can be added without changing the state schema.

---

### Phase 3: Introduce Agentic Framework (LangGraph) and Map Workflow

**Objective:** Replace the in-process workflow runner with **LangGraph** (resolved: D2 = A). The same steps become **graph nodes**; conditional execution (“has legal questions?”, “has follow-on?”) is handled by **conditional edges**. Context is built once and stored in graph state; later nodes read from state (resolved: D3 = B). Run data and framework state use the same store (resolved: D6 = A; e.g. `council_runs.db` for `AnalysisRun` and LangGraph checkpointer).

**Deliverables:**

1. **Dependencies**  
   - Add `langgraph`, `langchain-google-genai`, `langchain-core`. Use `ChatGoogleGenerativeAI` (or equivalent) as the LLM for agent nodes.  
   - Keep `google-genai` for RAG/cache logic that runs **inside** nodes (retrieval planner, embeddings, `create_gemini_cache`, `run_agent`), or pass cache_name/context via state.

2. **Graph definition**  
   - **State:** TypedDict (or equivalent) matching Phase 2 state schema so existing step logic can be reused inside nodes.  
   - **Nodes:** `plan_retrieval`, `retrieve_context`, `create_cache`, `main_agent`, `extract_legal_questions`, `legal_agent`, `integrate_legal`, `follow_on_chain` (or a single `follow_on` node invoked in a loop). Each node receives state, updates it, returns state updates.  
   - **Edges:** Start → plan_retrieval → retrieve_context → create_cache → main_agent → extract_legal_questions → **conditional**: if `legal_questions` → `legal_agent` → integrate_legal, else → integrate_legal. Then integrate_legal → **conditional**: if next follow-on prompt exists → follow_on node (loop back until no more verifier_id) → END, else → END.  
   - **Context (D3 = B):** Build context once in retrieve_context + create_cache; store `context_xml`, `cache_name` in state. Main_agent and (per-step) follow-on nodes read from state; legal_agent builds its own legal context and cache (same as today).

3. **Checkpointer (D6 = A)**  
   - Use LangGraph’s `SqliteSaver` (e.g. connection to `council_runs.db` or a dedicated table) so runs can be resumed and audited. Persist `thread_id` (e.g. run_id) in app so “Run Analysis” invokes with that config.

4. **Streamlit**  
   - “Run Analysis” invokes the graph with initial state (task, user content, folder_id, etc.). Optionally stream node updates into `st.status`-style UI. On completion, sync final state to session and persist to `AnalysisRun` as in Phase 1/2.

5. **Persistence**  
   - After the graph run completes, write to `AnalysisRun` in `council_runs.db` as before. Run data excluded from export/import. Framework checkpointer state lives in the same DB (or same file) per D6.

**Success criteria:** Same user flow (single analysis, legal review when needed, integration, follow-ons); orchestration and branching are handled by LangGraph; context is built once and consumed from graph state; ready for Phase 4 (QA node) and Phase 5 (HITL/interrupt).

**Phase 3 executed (2026-01-31):** Added `workflow_graph.py`: LangGraph `StateGraph` with nodes `plan_retrieval`, `retrieve_context`, `create_cache`, `main_agent`, `extract_legal_questions`, `legal_agent`, `integrate_legal`, `follow_on_chain`, `finalize`. Conditional edge after `extract_legal_questions`: route to `legal_agent` (then `follow_on_chain`) or `integrate_legal` (then `follow_on_chain`) based on `legal_questions` and `legal_expert_prompt_id`. State wrapped as `{"data": workflow_state_dict}`; nodes reuse workflow step functions. `run_analysis_graph(initial_state)` invokes the graph and returns final state; `WorkflowError` propagates. App calls `workflow_graph.run_analysis_graph(state)` inside a single `st.status("Running analysis…")`. Dependencies: `langgraph`, `langgraph-checkpoint-sqlite`, `langchain-core`. SqliteSaver (D6) available via `use_sqlite_checkpointer=True`; default False because state contains non-serializable objects (ORM, callables). Run history and persistence unchanged (Phase 1).

---

### Phase 4: Optional QA Agent and Final Output

**Objective:** Add an **optional** QA agent that reviews the **full chain output** (analysis + legal + follow-ons) and produces a final, polished document. QA runs **after** the follow-on chain (resolved: D5 = B). Configuration is **per-task** via a flag on `PromptTemplate`: “Use QA for this task” (resolved: D4 = A). Default: off.

**Current-state context (post–Phase 3):** The graph is `plan_retrieval → … → follow_on_chain → finalize → END`. QA will sit **between** `follow_on_chain` and `finalize`: `follow_on_chain → [if QA enabled] qa_agent → finalize → END`. State already has `final_output` (combined output); QA reads it and writes a new `final_output`. Callbacks in state support status messages (“Running QA agent…”) without new wiring.

**Deliverables:**

1. **QA agent (node + step)**  
   - **When enabled (per-task flag on PromptTemplate):** After the follow-on chain completes, run a QA node.  
   - **Inputs (from state):** `final_output` (full combined output: main + legal + follow-ons). No new RAG retrieval; QA receives only the combined text (Section 9.3: QA gets combined output only to save tokens).  
   - **Prompt:** Either (a) a fixed system prompt (“Review the following analysis… integrate legal answers… produce a single clear final output suitable for [council/committee] use”), or (b) a **dedicated PromptTemplate** (e.g. “QA Review”) for reusable, versioned QA instructions (opportunity in 1b.3).  
   - **Output:** `final_output` is overwritten with the QA result; optionally keep `pre_qa_output` in state (and persist both) for audit.  
   - **Implementation:** New step `run_qa_step(state, callbacks)` in `workflow.py` (calls `run_agent` with QA prompt + `final_output` as transient input; no cache required or reuse main cache from state). New node `_node_qa_agent` in `workflow_graph.py` that calls `run_qa_step(d, callbacks)` and uses `_status_write` for progress.

2. **Configuration**  
   - Add a boolean on **PromptTemplate**: `use_qa_agent` (default: False). Migration in `db.py` to add the column. Prompt editor: checkbox “Use QA agent for this task.” No global QA switch (D4 = A).

3. **Graph flow**  
   - **Conditional edge** after `follow_on_chain`: if `selected_prompt.use_qa_agent` → `qa_agent` node, else → `finalize`. From `qa_agent` → `finalize`.  
   - **Touchpoints:** `workflow_graph.py`: add node `qa_agent`, add conditional edge `follow_on_chain → (qa_agent | finalize)`, edge `qa_agent → finalize`. `workflow.py`: add `run_qa_step(state, callbacks)`.

4. **Persistence**  
   - `AnalysisRun` stores the final output (after QA if present). **Optional for Phase 4:** add `pre_qa_output` and `qa_output` columns (or a JSON blob) so Run history can show “Pre-QA” / “Post-QA” (opportunity in 1b.3). If not added now, a later phase can add them.

5. **UI**  
   - No new page; “Run Analysis” and Run history unchanged except that when QA ran, the stored output is the QA-polished one. Status area already shows “Running QA agent…” via callbacks if we add `_cb(state, "write", …)` in `run_qa_step`.

**Success criteria:** With QA disabled (default), behavior unchanged. With QA enabled for a task, one extra step (after follow-ons) produces a reviewed, integrated final document; run history shows the final (post-QA) output; prompt versioning and timezone display continue to apply.

---

### Phase 5: Human-in-the-Loop and Future Work (Later)

**Objective:** Add optional approval steps and prepare for more complex workflows. Not required for “same functionality” parity.

**See Section 11 (Phase 5 Holistic Analysis)** for current-state recap, gaps and blockers, agentic best practices, and **prioritized recommendations** (P5.1–P5.5). In brief:

**Deliverables (later), in suggested order:**

1. **P5.1 — Serializable state + checkpointer:** Refactor state so it contains only serializable data (prompt_template_id instead of selected_prompt; resolve ORM/callables at node entry; pass callbacks via config or clear before checkpoint). Enable SqliteSaver so interrupt/resume works.  
2. **P5.2 — Approval node + interrupt:** Add a human_approval node that calls LangGraph `interrupt()`; Streamlit shows “Approve / Reject / Edit”; resume with `Command(resume=...)` and same thread_id.  
3. **P5.3 — RunEvent:** Optional event log (run_id, step, event_type, payload, timestamp) for fine-grained audit.  
4. **P5.4 — More workflows:** Second graph (e.g. “Agenda packet review”) or workflow selector in UI.  
5. **P5.5 — Tools:** Add 1–2 read-only tools (e.g. “search_ordinances”) and bind to main or legal agent.

---

### Summary: Progression at a Glance

| Phase | Focus | Run data | Export/import | Framework |
|-------|--------|----------|---------------|-----------|
| **0** | Pre-work: run DB path, module, export scope, Phase 1 touchpoints | — | Confirm config-only | — |
| **1** | Persist every run; Run history page (admin only) | `AnalysisRun` in `council_runs.db`; never in export | `council.db` only | None |
| **2** | Extract pipeline into workflow runner + state | Same as Phase 1 | Same | In-process runner |
| **3** | LangGraph: graph nodes + conditional edges; context in state | Same; run/state in `council_runs.db` | Same | LangGraph |
| **4** | Optional QA agent (per-task, after follow-on chain); see Section 1b for opportunities | Same | Same | LangGraph |
| **5** | Optional HITL, RunEvent, more workflows | Same | Same | LangGraph |

**End state (after Phase 4):** Same functionality as today—single analysis, legal review when needed, integration of legal results, optional follow-on chain—plus optional per-task QA (after follow-ons) and persistent run history (admin Run history page, timezone + prompt version + copy markdown), all orchestrated by LangGraph with context built in nodes and shared via state. Export/import continues to sync only prompts and config (`council.db`); run data stays in `council_runs.db`. See **Section 1b** for post–Phase 3 holistic review and Phase 4 opportunities.

---

## 8. LangGraph-Specific Considerations and Opportunities

Now that LangGraph is the chosen framework (D2 = A), the following considerations and opportunities apply.

### 8.1 State and node flow

- **State:** Within a single `kickoff()`, LangGraph passes each task’s output as context to the next task. Use **sequential process** (`Process.sequential`) so: main analysis → (optional legal) → integrate → follow-ons → (optional QA) run in order with outputs flowing forward.
- **Shared RAG context:** LangGraph does not automatically inject a “global” KB; we must pass it. **Approach:** Build the main RAG context once (plan → retrieve → cache or payload); pass it as **crew inputs** (e.g. `crew.kickoff(inputs={"kb_context": context_xml, "user_content": user_content})`). In the first task’s description, reference the crew input (e.g. “Use the provided knowledge base context and user content to …”). Subsequent tasks receive previous task outputs automatically; they can also receive the same `kb_context` via task context if the framework allows, or we inject it once into the first task and later tasks only see prior task outputs (acceptable if follow-ons are refinements of the combined output).
- **Legal agent:** The legal step uses a **separate** RAG retrieval (different query and possibly different libraries). Run it outside the crew as a conditional step, or as a separate “legal” task that receives legal questions and has its own context (we build legal context once and pass it to that task). Either way: one context build for main + follow-ons, one for legal (already current behavior).

### 8.2 LangGraph features to leverage

- **Role-based agents:** Analyst and Legal map naturally to LangGraph agents (role, goal, backstory). Use the same prompt text from `PromptTemplate` as the agent’s goal/backstory so behavior stays aligned.
- **Tasks and expected_output:** Define tasks with clear `expected_output` so the framework can validate or structure outputs. Use `expected_output` to describe “markdown analysis with optional Legal Questions Requiring Expert Review section” for the main task.
- **Gemini integration:** Use LangGraph’s official Gemini LLM (e.g. `gemini/gemini-2.5-pro`). Set `GEMINI_API_KEY` in env; configure the LLM once and assign to all agents that use Gemini.
- **Tools (Phase 5+):** LangGraph supports tools per agent. When adding read-only tools (e.g. “search_ordinances”), attach them to the Analyst or Legal agent; the framework handles the tool loop.

### 8.3 LangGraph constraints to work around

- **Conditional branching:** LangGraph’s sequential process does not natively support “if legal questions, run Legal task.” Implement in wrapper code: run the main task → parse output for legal questions → if any, run a second “legal” crew or a single Legal task with legal context → then integrate and run follow-ons. Alternatively, one crew with Legal task that receives “N/A” when there are no questions (simpler but wastes a call).
- **Large context:** Passing a very large RAG payload in crew inputs may hit token limits or slow the run. Prefer **Gemini context caching**: build context once, create a Gemini cache (existing `brain.create_gemini_cache`), and pass the **cache name** (or a ref) in crew inputs; then use a custom Gemini LLM wrapper for LangGraph that supports `cached_content` so each agent call uses the cache. If LangGraph’s Gemini integration does not support cached content, keep passing the context string and rely on Gemini’s implicit caching where applicable.
- **No built-in checkpointer:** LangGraph does not have LangGraph-style checkpoints. For “resume after approval,” implement manually: persist state (e.g. to `council_runs.db`) after each major step; on resume, reload state and run the remainder. Phase 5 can add this.

### 8.4 Summary

- Use **StateGraph** with Phase 2 state schema; **conditional edges** for legal and follow-on branching; **loop** for follow-on chain.
- **Checkpointer** in `council_runs.db` (or same file) per D6; **interrupt()** for Phase 5 approval.
- Build main context once in retrieve/cache nodes; legal context once inside legal_agent node; QA (Phase 4) receives only combined output from state.

---

## 9. Context Caching Strategy and Optimizations

### 9.1 Current behavior (to preserve)

- **RAG:** Plan retrieval → query expansion → retrieve_and_build_context_multi → build context XML (Core + retrieved libraries). One context build for main analysis; a **second** context build for legal expert (different plan/query/retrieval).
- **Gemini cache:** `brain.create_gemini_cache(context_xml)` creates a cached content; `run_agent(..., cache_name=cache_name)` uses it so the model sees the KB without re-sending tokens. TTL 60 minutes.

### 9.2 Resolved approach (D3 = B)

- **Framework holds context:** Build context once (or twice: main + legal) and store in graph state. Later nodes read from state; no per-step RAG re-call for the same context.
- **Where to build:** In retrieve_context + create_cache nodes: run existing plan_retrieval → retrieve_and_build_context_multi → create_gemini_cache. Store `context_xml` and `cache_name` in state for main_agent and follow-on nodes; legal_agent node builds its own legal context.

### 9.3 Optimizations to consider

1. **Gemini explicit caching with LangGraph**  
   - If the LangGraph Gemini LLM wrapper can accept a “cached content” reference (e.g. cache name or session), use it so each agent call reuses the same cached tokens (cheaper, faster). If not, pass the context string and rely on Gemini’s **implicit** caching (automatic for repeated prefixes; no guarantee).

2. **One cache for main, one for legal**  
   - Main RAG context → one Gemini cache (or one payload). Legal RAG context → separate cache (or payload). Do not re-build main context for legal; legal has its own retrieval. This matches current behavior.

3. **QA agent context (Phase 4)**  
   - QA task only needs the **combined text output** (main + legal + follow-ons) to review and polish. It does **not** need the full RAG context again. Pass only `combined_output` to the QA task to save tokens and time. Optionally allow a “full context” mode for QA if we want it to re-ground in the KB later.

4. **Retrieval planner and query expansion caches**  
   - Keep existing disk/memory caches for retrieval planner, query expansion, and embeddings (`rag_cache.py`, `brain` caches). They reduce API calls and stay independent of LangGraph.

5. **Context size and truncation**  
   - If context exceeds model limits, truncate or summarize (e.g. keep Core full, cap library chunks per library). Document a max context size and behavior when exceeded (e.g. “retrieve fewer chunks” or “summarize oldest chunks”).

### 9.4 Summary

- Build main context once, legal context once; pass both via graph state as needed.
- Prefer Gemini explicit cache for main (and legal) context when calling from LangGraph if supported; else pass context string.
- QA task receives only combined output (no full RAG re-send). Keep existing RAG and embedding caches.

---

## 10. Final Review and Readiness for Implementation

### 10.1 Migration effectiveness (before Phase 1)

- **Phase 0 (pre-work)** is added so that run DB path, module (`runs_db.py`), and export/import scope are decided and documented before Phase 1. Complete Phase 0 first: define `RUNS_DB_PATH`, add `council_runs.db` to `.gitignore`, confirm export/import use only `council.db`, and document Phase 1 touchpoints (`runs_db.py`, `app.py`, Run history page).
- **Phase 1 touchpoints** are explicit: new `runs_db.py` with `AnalysisRun` and CRUD; `app.py` inserts after run (success/failure) and adds admin-only “Run history” page. No orchestration change.
- **Phase 2** state schema should include every field that Phase 3 LangGraph (and Phase 4 QA) need so that when we replace the runner with LangGraph, we only swap the runner, not the state shape. Consider adding a short “state schema” subsection in Phase 2 listing all keys (task_name, template_text, user_content, folder_id, rag_state, context_xml, cache_name, main_output, legal_questions, legal_expert_output, final_output, chain_outputs, retrieval_report, etc.).

### 10.2 LangGraph opportunities (recap)

- StateGraph + conditional edges for legal and follow-on; checkpointer in council_runs.db; Analyst and Legal as nodes; Gemini in nodes; tools later (Phase 5).
- Conditional legal and follow-on loop as native edges; large context via cache_name in state; interrupt() for Phase 5 approval.

### 10.3 Context caching (recap)

- One build for main, one for legal; pass into framework; QA gets combined output only; keep existing caches for planner/expansion/embeddings; consider Gemini explicit cache with LangGraph.

### 10.4 Document readiness for LLM processing

- **Outline** at the top gives a clear section map.
- **Resolved decisions** are in one table with IDs (D1–D8) and dates.
- **Phases** are numbered 0–5 with objective, deliverables, and success criteria each.
- **Terminology** is consistent: `AnalysisRun`, `council_runs.db`, `council.db`, LangGraph, “Run history” (admin only), “context built once,” “per-task QA after follow-on chain.”
- **Cross-references** use section names and decision IDs (e.g. “resolved D2”, “Section 7 Phase 1”).
- **Code touchpoints** are named (e.g. `runs_db.py`, `app.py`, `db.py`, `paths.py`, `brain.py`).

No further structural changes are required for handoff to LLM-based implementation tools. Proceed with **Phase 0** (pre-work), then **Phase 1** (persist analysis results and Run history page).

---

## 11. Phase 5 Holistic Analysis: HITL, Generic Flows, and Agentic Best Practices

This section provides a **holistic analysis** of CouncilFlow’s current state after Phases 0–4, identifies **gaps and blockers** for Human-in-the-Loop (HITL) and more generic flows, aligns with **agentic best practices**, and gives **prioritized recommendations** to make the system more robust and effective.

### 11.1 Current State (Post–Phase 4)

| Area | State |
|------|--------|
| **Orchestration** | Single LangGraph `StateGraph`: plan_retrieval → retrieve_context → create_cache → main_agent → extract_legal_questions → [conditional] legal_agent or integrate_legal → follow_on_chain → [conditional] qa_agent or finalize → finalize → END. |
| **State** | Wrapped as `{"data": workflow_state_dict}`. Dict holds inputs (task_name, template_text, user_content, folder_id, prompt_template_id, **selected_prompt** (ORM), **build_prompt_variables** (callable), main_full_template, …), RAG (context_xml, cache_name, …), outputs (final_output, chain, …), **\_callbacks** (write, update_label). |
| **Checkpointer** | SqliteSaver available; **default off** because state contains **non-serializable** values: `selected_prompt` (SQLModel instance), `build_prompt_variables` (function), `rag_state` (complex object), `_callbacks` (Streamlit callables). |
| **Persistence** | AnalysisRun in council_runs.db (output, chain_steps, prompt_version, stored_timezone, …). No step-level event log. |
| **UI** | Run Analysis: select task (prompt), add content, Run; single `st.status` with callbacks for progress. Run history (admin): list/detail, local time, prompt version, copy markdown. No approval or resume UI. |
| **Workflows** | One fixed graph (“analysis + optional legal + follow-ons + optional QA”). No second workflow or workflow selector. |
| **Tools** | None; no function calling in agents. |

### 11.2 Gaps and Blockers for Phase 5

1. **HITL (approval / resume)**  
   - **Blocker:** Checkpointer is off, so graph state is not persisted and **resume** is not possible. LangGraph’s `interrupt()` requires a **checkpointer** so that after pause, state is saved and can be restored when the user resumes.  
   - **Blocker:** State is **not serializable** (ORM, callables, rag_state, _callbacks). SqliteSaver cannot persist it; even if we persisted manually, we could not restore `selected_prompt` or callbacks from JSON.  
   - **Gap:** No “approval” node in the graph; no Streamlit UI for “Approve / Reject / Edit” or for resuming with `Command(resume=...)`.

2. **Audit and traceability**  
   - **Gap:** We persist only the **final** run (AnalysisRun row). We do not persist **per-step events** (e.g. “main_agent completed”, “legal_agent started”, “human_approval requested”, “human_approved”). For governance and debugging, a RunEvent (or equivalent) table would support “who approved what, when” and step-level replay.

3. **More generic flows**  
   - **Gap:** The graph is **hardcoded** in Python. “Agenda packet review” or “Motion + legal + chair script” would require either a second graph, or a **workflow definition** (e.g. graph as data / DSL) and a generic runner. Today there is no workflow selector in the UI and no notion of multiple “flow types.”

4. **Tools**  
   - **Gap:** Main and legal agents use only static RAG context (and QA uses only combined text). There are no **tools** (e.g. search_ordinances, get_agenda_item). Adding tools would allow agents to pull additional context or perform compliance checks within the same run.

### 11.3 Agentic Best Practices and How We Compare

| Practice | Recommendation | CouncilFlow today |
|----------|----------------|-------------------|
| **Serializable state** | Keep graph state JSON-serializable so checkpointer (and optional replay/audit) works. Resolve ORM/callables at node entry or pass them outside state (e.g. config). | ❌ State holds ORM, callables, rag_state, _callbacks. Checkpointer off. |
| **Interrupt in a dedicated node** | Use a single “human_approval” (or similar) node that calls `interrupt(...)`; do not wrap interrupt in try/except; do not return complex objects from interrupt. Resume with `Command(resume=value)`. | ❌ No approval node; no interrupt. |
| **Checkpointer for HITL** | Enable checkpointer when using interrupt so state is persisted at pause and can be restored on resume (same or different process). | ❌ Checkpointer off (state not serializable). |
| **Audit trail** | Log step-level events (node entered, node completed, interrupt requested, human response) for accountability and debugging. | ⚠️ Only final run row; no RunEvent. |
| **Idempotent side effects before interrupt** | Any side effects (e.g. DB writes) before an interrupt should be idempotent so resume does not duplicate them. | ⚠️ We persist only at end of run; if we add “persist before approval,” make it idempotent. |
| **Multi-workflow** | Support multiple flows (e.g. “analysis only”, “analysis + legal + QA”, “agenda review”) via graph variants or a workflow registry + generic runner. | ❌ Single hardcoded graph. |
| **Tools** | Start with read-only tools; bind to agents; keep writes in a sandbox or as “draft” for human copy/paste. | ❌ No tools. |
| **Structured errors** | Use a consistent error type (e.g. WorkflowError) and propagate to UI with message and optional details. | ✅ WorkflowError; app shows message + details. |

### 11.4 Prioritized Phase 5 Recommendations

**Foundation (required for HITL)**

1. **P5.1 — Serializable state and checkpointer**  
   - **Goal:** State passed to the graph (and stored in checkpointer) contains only **serializable** data.  
   - **Actions:**  
     - Store **prompt_template_id** (int) in state; **resolve** `selected_prompt` at the start of each node that needs it (e.g. `selected_prompt = db.get_prompt_by_id(state["prompt_template_id"])`).  
     - Do **not** put `build_prompt_variables` in state. Either (a) build `main_full_template` (and any other derived strings) in the app **before** invoking the graph and pass only strings, or (b) resolve a “build_prompt_variables” behavior from config (e.g. a well-known function name) inside the node.  
     - Pass **callbacks** via invoke **config** (e.g. `config["callbacks"] = {"write": ..., "update_label": ...}`) and have nodes read from config instead of state; or keep callbacks out of persisted state by clearing `_callbacks` before checkpointing (if the framework allows injecting them on resume).  
     - **rag_state:** Either (a) do not put it in state—load RAG state inside the first node that needs it using `folder_id`—or (b) keep a minimal serializable handle (e.g. folder_id) and re-load rag_state in each node that needs it. Re-loading may have a performance cost; document the tradeoff.  
   - **Outcome:** Enable SqliteSaver by default (or via env). Graph state can be checkpointed and resumed; same thread_id can resume after interrupt.

2. **P5.2 — Approval node(s) and interrupt**  
   - **Goal:** Optional human approval step(s) in the graph; pause with LangGraph `interrupt()`; resume with `Command(resume=...)`.  
   - **Actions:**  
     - Add a **human_approval** (or **approval**) node that receives state, presents a **reason** (e.g. “Review main analysis before legal step?” or “Approve final output before persisting?”), and calls `interrupt({"reason": ..., "summary": state.get("final_output", "")[:N]})`. Do not wrap the interrupt call in try/except.  
     - **Configuration:** Per-task or global “require approval before X” (e.g. before legal, before persist). Use conditional edges so the approval node is only used when the flag is set.  
     - **Streamlit:** When the run returns “interrupted,” show the approval UI (reason, preview, Approve / Reject / Edit). On Approve (or Edit then Approve), call `graph.invoke(Command(resume={"approved": True, "edited_output": ...}), config=config)` with the same **thread_id** and **checkpointer**.  
     - **Reject:** Resume with `Command(resume={"approved": False})`; route to a “rejected” end or retry path (e.g. back to main_agent or stop).  
   - **Outcome:** Users can approve or edit at configured points; runs can resume after pause; state is restored from checkpointer.

**Audit and robustness**

3. **P5.3 — RunEvent (or event log)**  
   - **Goal:** Step-level audit trail for accountability and debugging.  
   - **Actions:**  
     - Add **RunEvent** table (or equivalent) in council_runs.db: run_id, step_name, event_type (e.g. `node_started`, `node_completed`, `interrupt_requested`, `human_responded`), payload (JSON or text), timestamp, optional user_id.  
     - In the graph: at node entry/exit (or in a wrapper), append an event to RunEvent (or a in-memory list that is flushed at persist). Keep payload small (e.g. hashes or summaries for large outputs).  
     - **UI:** Run history detail can show “Event log” (expandable) for that run.  
   - **Outcome:** “Who approved what, when” and step order are queryable; supports compliance and replay.

**More generic flows**

4. **P5.4 — Workflow selector or second workflow**  
   - **Goal:** Support more than one “flow” (e.g. current analysis flow vs. “agenda packet review” or “motion + legal + chair script”).  
   - **Options:**  
     - **A. Multiple graphs in code:** Define a second (and optionally third) graph in workflow_graph.py (e.g. `get_agenda_review_graph()`). App selects graph by “Workflow” dropdown (e.g. “Analysis (default)”, “Agenda packet review”). Same state shape or adapter.  
     - **B. Workflow as data:** Store workflow definitions (nodes, edges, conditions) in DB (e.g. JSON); a **generic runner** builds a LangGraph from that JSON and invokes it. Larger design; more flexible.  
   - **Recommendation:** Start with **A** (second graph + dropdown) for a second flow; move to B if you need many user-defined flows.  
   - **Outcome:** Users can choose “Analysis” vs. “Agenda review” (or similar); both use the same run persistence and Run history.

**Tools (later)**

5. **P5.5 — Read-only tools for main/legal agent**  
   - **Goal:** Agents can call 1–2 read-only tools (e.g. “search_ordinances”, “get_agenda_item”) during a run.  
   - **Actions:** Define tool schemas (e.g. Gemini FunctionDeclaration); implement tools (e.g. search_ordinances runs RAG and returns top chunks); bind to the model in `run_agent` (or in a LangChain/LangGraph agent node) and run the tool loop until the model returns a final answer.  
   - **Outcome:** Richer, on-demand context without changing the overall flow; foundation for future write/draft tools.

### 11.5 Suggested Phase 5 Order

1. **P5.1 (serializable state + checkpointer)** — Unblocks HITL and resume.  
2. **P5.2 (approval node + interrupt + Streamlit resume UI)** — Delivers visible HITL.  
3. **P5.3 (RunEvent)** — Audit trail; can be implemented in parallel or right after P5.2.  
4. **P5.4 (workflow selector / second workflow)** — Generic flows; can be done after or in parallel with P5.2/P5.3.  
5. **P5.5 (tools)** — Optional; can follow once HITL and flows are stable.

### 11.6 Summary

CouncilFlow is in strong shape after Phases 0–4: LangGraph graph, optional QA, prompt versioning, run persistence, and local-time Run history. To make Phase 5 (HITL and more generic flows) **robust and effective**:

- **Make state serializable** and enable the checkpointer so interrupt/resume works.  
- **Add an approval node** using LangGraph `interrupt()` and a Streamlit resume UI with `Command(resume=...)`.  
- **Add RunEvent** (or equivalent) for step-level audit.  
- **Introduce a second workflow** (or a workflow selector) for more generic flows.  
- **Consider read-only tools** next for stronger agentic behavior.

This aligns with agentic best practices: serializable state, dedicated interrupt node, checkpointer for HITL, audit trail, and optional multi-workflow and tools.

---

### 6.7 References

- [LangGraph: Interrupts (human-in-the-loop)](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)  
- [ReAct agent with Gemini and LangGraph](https://ai.google.dev/gemini-api/docs/langgraph-example)  
- [LangGraph + Gemini example](https://ai.google.dev/gemini-api/docs/crewai-example)  
- [Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling)  
- [LangChain ChatGoogleGenerativeAI](https://python.langchain.com/docs/integrations/chat/google_generative_ai)
