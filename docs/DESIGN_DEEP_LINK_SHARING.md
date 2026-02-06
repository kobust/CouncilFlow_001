# Design: Deep Link Sharing for Inquiry Results

## Summary

Allow sharing the results of an analysis run via a **deep link** that can be viewed **without logging in**. Sharing is **opt-in** (enable/disable per run); existing runs are never shared until the user explicitly enables sharing. The viewer page shows markdown (or JSON with a selectable renderer), supports copying markdown, and remains navigable when the user is logged in; unauthenticated users can optionally log in from the viewer. Users can **edit the title** of a shared entity (the label shown on the viewer and when copying the link).

---

## 1. Requirements (from spec)

| Requirement | Notes |
|-------------|--------|
| Share results via deep link | URL contains a **share token** (not raw run id). |
| View without login | Public viewer route bypasses auth when a valid token is present. |
| Display markdown for that item | Primary content is the run’s output (markdown or rendered JSON). |
| Items not publicly viewable by default | Only when “deep link” is explicitly enabled. |
| Enable/disable deep link | User can turn sharing on and off later. |
| Viewer navigable when logged in | If user is logged in, sidebar/nav still work (e.g. back to Run Analysis, Run history). |
| Optional login for unauthenticated viewers | e.g. “Log in” button on viewer so they can then navigate. |
| Copy markdown | Viewer includes copy-markdown (same pattern as run history detail). |
| JSON output: select renderer | When run has `output_json`, viewer offers same “View as” options (Saved, Raw JSON, schema transformers). |
| Editable title | User can set or change the display title for a shared run (shown on viewer and when sharing). |

---

## 2. Data Model

### Option A: Columns on `AnalysisRun` (recommended)

Add to `runs_db.AnalysisRun`:

- **`share_token`** (str, nullable, unique): Unguessable token (e.g. 32-byte hex or URL-safe random). `NULL` = never shared or sharing disabled.
- **`share_enabled`** (bool, default False): When True, `share_token` is valid for access. When False, link returns “link invalid or disabled”.
- **`share_title`** (str, nullable): User-editable display title for the shared run. Shown on the viewer and when referring to the link. When NULL or empty, fall back to `task_name` (e.g. “Run: {task_name}”).

Lookup: “resolve run by share token” → single query: `WHERE share_token = :token AND share_enabled = 1`.

### Option B: Separate `ShareLink` table

- `id`, `run_id` (FK), `token` (unique), `enabled` (bool), `created_at`, optional `expires_at`.

Use if you want multiple links per run or expiry without touching `AnalysisRun`. For “one share link per run, toggle on/off”, Option A is simpler.

**Recommendation:** Option A. Add a migration in `runs_db.py` for `share_token`, `share_enabled`, and `share_title`; add `get_analysis_run_by_share_token(token: str) -> AnalysisRun | None` (returns run only when `share_enabled` is True) and `update_run_share_title(run_id, title)` (and optionally `update_run_share_enabled`, etc.).

---

## 3. URL and Routing

- **URL shape:** Single-page Streamlit app, so use query params. For example:
  - `?v=TOKEN` or `?share=TOKEN`
- **Entry flow in `app.py`:**
  1. **Before** `authenticator.login()` and the “if not authenticated then st.stop()” block:
     - Read `st.query_params.get("v")` (or `"share"`). If absent, continue to normal auth.
     - If present: call `get_analysis_run_by_share_token(token)`. If no run or not enabled, clear the param (or show “Link invalid or disabled”) and continue to normal auth.
     - If valid: set a flag (e.g. `st.session_state["viewing_shared_run"] = run_id` or store minimal run info) and **skip the login gate** for this request; render only the **viewer layout** (see below), then `st.stop()`.
  2. **Viewer layout:** Minimal shell: optional sidebar with “Log in” (if not logged in) or normal nav (if logged in), main area = shared run viewer (markdown + copy, JSON renderer if applicable). No Run Analysis / Edit Prompts / etc. in main area unless you want a “Back to app” that goes to runner.

So: **one code path** that checks query params first; if valid share token, render viewer and stop; otherwise require login and show full app.

---

## 4. Viewer Page Behavior

- **Content:** Same as run history detail for that run:
  - **Title:** Show the shared entity’s display title: use `share_title` when set, otherwise fall back to `task_name` (e.g. “Shared analysis: {share_title or task_name}”). Do not expose run id in the viewer.
  - Output: `output_text` as markdown. If `output_json` and prompt has `output_schema_key`, show the same “View as” dropdown as in run history (`_get_json_view_options`, `_render_json_view`) and render accordingly.
- **Copy markdown:** Reuse `_markdown_with_copy` (or a shared component) so users can copy the markdown (and for JSON, the currently selected view’s text).
- **Logged in:** Sidebar shows normal nav (Run Analysis, Run history, etc.). “Back to Run Analysis” or “Run history” is enough to stay navigable.
- **Not logged in:** Sidebar (or top of viewer) shows “Log in” only. No access to Run history or other pages until they log in; after login, same sidebar as above.
- **Invalid/disabled link:** Don’t leak existence of a run. Show a generic message (“This link is invalid or has been disabled”) and optionally a link to the main app (which will then show login).

---

## 5. Enabling / Disabling Sharing and Editable Title (Run history)

- **Run history list:** No change needed for list view.
- **Run detail (when viewing a run):**
  - **Editable title:** When the run is shared (or when the user is about to enable sharing), show an editable “Share title” field. Default value: current `share_title` or `task_name`. On save, persist `share_title` via `update_run_share_title(run_id, title)`. This title is used on the deep-link viewer and can be updated anytime from run detail.
  - If `share_enabled` is False: button “Enable sharing” → show PII/sensitivity warning (see §7), then generate `share_token` if null, set `share_enabled = True`, persist, then show “Sharing enabled”, “Copy link”, and “Disable sharing”. Optionally prompt for share title before enabling.
  - If `share_enabled` is True: show “Share title” (editable), “Sharing enabled”, “Copy link”, and “Disable sharing” → set `share_enabled = False`, persist.
- **Copy link:** Full URL: `{base_url}?v={share_token}`. **Base URL** must come from configuration (see §7.3), not from the browser, so “Copy link” works from any environment (local, staging, prod).

---

## 6. Token Generation and Security

- **Token:** Use `secrets.token_urlsafe(32)` (or 32-byte hex) so it’s unguessable. Store in `AnalysisRun.share_token`.
- **No enumeration:** Resolve only by token; never expose “run id” in the public URL. Invalid token → same generic error as disabled link.
- **HTTPS:** In production, serve over HTTPS so the link isn’t sniffed.

---

## 7. Required: Privacy, Discovery, Base URL, Audit, and Backwards Compatibility

The following are **required** parts of the design, not optional.

### 7.1 PII / sensitivity warning

- **When enabling sharing:** Before turning on the deep link, show a clear warning, e.g. “Only enable sharing if this analysis does not contain confidential information or personally identifiable information (PII). Anyone with the link will be able to view the content without logging in.”
- **Placement:** Modal, expander, or inline message on the run detail page when the user clicks “Enable sharing”; require acknowledgment (e.g. checkbox “I understand” or “Continue”) before enabling.
- **Viewer:** Do not show viewer username or run id to unauthenticated viewers; use only the share title (or task name) and the analysis content.

### 7.2 Disable search engines

- **Requirement:** Shared links must not be indexed by search engines.
- **Implementation:** When the page is rendered with `?v=...` (valid share token), send `X-Robots-Tag: noindex, nofollow` in the HTTP response. In Streamlit this may require a custom response header (e.g. via `st.set_page_config` or server hook if available) or configuration in the reverse proxy (e.g. nginx, Cloud Run) that serves the app: for requests whose URL contains the share param, add `X-Robots-Tag: noindex, nofollow`.
- **Documentation:** Note in deployment docs that the proxy or app must set this header for viewer URLs.

### 7.3 Base URL (required for “Copy link”)

- **Requirement:** “Copy link” must produce a URL that works when pasted elsewhere (email, chat, another device). The link must not depend on the current browser host/port (e.g. localhost).
- **Implementation:** Store **base_url** in configuration (e.g. `config.yaml` under a key like `app.base_url`, or environment variable `COUNCILFLOW_BASE_URL`). Examples: `https://councilflow.example.com`, `https://councilflow-staging.example.com`. Use this value when building the share link: `{base_url}?v={share_token}`. If `base_url` is missing, “Copy link” can be disabled or show a warning (“Configure base_url to copy share link”).
- **No fallback to request host:** Do not derive base URL from the current request in production; that would break when the user is on localhost or a different domain than the one recipients use.

### 7.4 Audit of people viewing

- **Requirement:** Record when a shared link is viewed (audit trail).
- **Data to log:** At minimum: run_id (or share_token hash), timestamp (UTC), and optionally IP or request identifier. Do **not** store PII (e.g. do not log viewer identity unless they log in; if they log in, you may log username for that view).
- **Storage:** New table or append-only log, e.g. `ShareViewEvent`: `id`, `run_id`, `viewed_at` (UTC), `ip_hash` or `request_id` (optional), `viewer_username` (nullable, set only if the viewer was logged in). Or write to a simple audit log file/table used only for this.
- **Access:** Only admins (or a dedicated “Share audit” view) can list view events for a run (e.g. “This link was viewed 3 times: …”). Run history detail can show “Viewed N times” when sharing is enabled.

### 7.5 Backwards compatibility — explicit sharing only

- **Requirement:** Existing runs and any run created before or after the feature must **never** be publicly viewable unless the user explicitly enables sharing.
- **Defaults:** New columns: `share_token` = NULL, `share_enabled` = False, `share_title` = NULL. No migration that sets `share_enabled = True` for any existing row.
- **No “share by default”:** Even for new runs, sharing is off until the user clicks “Enable sharing” and acknowledges the PII warning. No automatic or bulk sharing.

---

## 8. What Else to Consider (optional)

### 8.1 Expiration and revocation

- **Expiry:** Optional `share_expires_at` on the run. Viewer checks it and shows “Link has expired” instead of content.
- **Revoke:** “Disable sharing” is immediate revoke.

### 8.2 Rate limiting

- **Rate limit** requests that carry `?v=...` (per IP or per token) to avoid scraping or brute force.

### 8.3 Embedding and framing

- **X-Frame-Options / CSP:** Keep default framing restrictions unless you need embedding for known origins.

### 8.4 Accessibility and mobile

- **Viewer:** Same semantic structure as run history; test “Copy link” / “Log in” on small screens.

### 8.5 Streamlit specifics

- **Query params:** Use `st.query_params` for `v`; keep param in URL so refresh keeps the viewer.
- **Rerun:** On “Enable/Disable sharing” or “Save” share title, rerun to reflect state.

---

## 9. Implementation Checklist (high level)

1. **DB:** Add `share_token` (nullable unique), `share_enabled` (bool, default False), and `share_title` (nullable string) to `AnalysisRun`; migrations in `runs_db.py`. Add `get_analysis_run_by_share_token(token)`, `update_run_share_title(run_id, title)`, and helpers to enable/disable sharing and generate token. Add **ShareViewEvent** (or equivalent) table and `insert_share_view_event(run_id, ...)` for audit.
2. **Config:** Add `base_url` to `config.yaml` (or env) and read it for “Copy link”. Require it when sharing is used (or disable “Copy link” if missing).
3. **App entry:** At top of `app.py`, before auth: if `st.query_params.get("v")` and run found and enabled, record audit event (§7.4), set session state, render viewer (see 4), send noindex header (§7.2), then `st.stop()`. Else continue to login.
4. **Viewer UI:** Title = `share_title` or `task_name`. Reuse run-detail logic (markdown, `_markdown_with_copy`, JSON view options). Sidebar: if authenticated, normal nav; if not, “Log in” only.
5. **Run history detail:** Editable “Share title” when run is (or will be) shared. “Enable sharing” only after PII warning (§7.1). “Disable sharing” and “Copy link” (using `base_url`). Persist title and share state via `runs_db`.
6. **Error handling:** Invalid or disabled token → generic message, no leak of run existence.
7. **Audit:** On each successful viewer load with `?v=...`, call `insert_share_view_event(run_id, ...)`. Expose “Viewed N times” (or list of view events) to admins on run detail.

This design keeps sharing explicit and opt-in, includes PII warning, noindex, base URL, audit of viewers, and editable titles, and remains backwards compatible.
