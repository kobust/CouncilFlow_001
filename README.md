# CouncilFlow

CouncilFlow is a **Streamlit app** for document-backed analysis using a **RAG** (Retrieval-Augmented Generation) pipeline. It uses a **Google Drive** folder as the knowledge base (Core + Libraries), **Gemini** for embeddings and the main LLM, and **hybrid retrieval** (BM25 + semantic search) to build context for each run.

---

## Requirements

- **Python 3.10+**
- **Google Cloud**: A **Gemini API key** and a **Google Drive**-backed knowledge base. Drive access uses a **Service Account** with read-only access to the root folder.

---

## Quick start

### 1. Clone and install

```bash
git clone <repo-url>
cd CouncilFlow
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration

- **Auth (Streamlit login):** Edit `config.yaml`. Change all passwords and `cookie.key` before production. Regenerate password hashes:
  ```bash
  python -c "import streamlit_authenticator as stauth; print(stauth.Hasher.hash('YOUR_PASSWORD'))"
  ```
- **Secrets:** The app needs:
  - **`GEMINI_API_KEY`** – from [Google AI Studio](https://aistudio.google.com/apikey).
  - **Google Drive access** – Service Account JSON (read-only) for the Drive folder that holds your knowledge base.

  You can provide them either via **environment variables** or **`.streamlit/secrets.toml`**:

  **Option A – Environment variables (local)**

  ```bash
  export GEMINI_API_KEY="your-gemini-api-key"
  export GCP_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}'   # full JSON string
  ```

  **Option B – `.streamlit/secrets.toml`**

  Create `.streamlit/secrets.toml` (this path is gitignored):

  ```toml
  GEMINI_API_KEY = "your-gemini-api-key"

  [gcp_service_account]
  type = "service_account"
  project_id = "..."
  private_key_id = "..."
  private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
  client_email = "..."
  client_id = "..."
  # auth_uri, token_uri, etc. as in your Service Account JSON
  ```

  For **Docker/Cloud Run**, use `docker_secrets.py`: it builds `secrets.toml` from `GEMINI_API_KEY` and `GCP_SERVICE_ACCOUNT_JSON` env vars, then starts Streamlit.

### 3. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal. Log in with a user from `config.yaml`, connect a Drive root folder, refresh the knowledge base, then run analyses.

---

## Optional: Environment variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Gemini API key (required if not in `secrets.toml`) |
| `GCP_SERVICE_ACCOUNT_JSON` | Full Service Account JSON string for Drive (if not in `secrets.toml`) |
| `GEMINI_MODEL` | Main model for the agent (default: `gemini-3-flash-preview`). e.g. `gemini-2.0-flash` |
| `GEMINI_PLANNER_MODEL` | Model for the retrieval planner (default: `gemini-2.0-flash`) |
| `GEMINI_PACE_DELAY_SECONDS` | Seconds to wait before each GenerateContent call; helps with rate limits (default: `0`) |
| `GEMINI_RATE_LIMIT_RPM` | Requests per minute limit for rate limiting (default: `10`). **If running multiple instances with same API key, set to `total_limit / num_instances`** |
| `COUNCILFLOW_INSTANCE_COUNT` | Number of instances sharing the API key (for rate limit warnings, optional) |
| `COUNCILFLOW_DATA_DIR` | Data directory on Linux/Docker (default: `/app/data`) |
| `COUNCILFLOW_USE_GRAPH_CHECKPOINTER` | Set to `1` or `true` to persist LangGraph checkpoints in `council_runs.db` (D6). Default off; state may contain non-serializable objects. |

---

## Optional: Verify LangGraph (Phase 3)

After installing requirements, you can confirm the analysis graph compiles:

```bash
python -m workflow_graph
```

Expected: `Graph compiled OK (no checkpointer)` and, if `langgraph-checkpoint-sqlite` is installed, `Graph compiled OK (with SqliteSaver)`.

---

## Optional: List Gemini models

```bash
export GEMINI_API_KEY="your-key"
python check_models.py
```

---

## Docker

The Dockerfile runs `docker_secrets.py`, which expects `GEMINI_API_KEY` and `GCP_SERVICE_ACCOUNT_JSON` at runtime. Build and run:

```bash
docker build -t councilflow .
docker run -p 8080:8080 -e GEMINI_API_KEY="..." -e GCP_SERVICE_ACCOUNT_JSON='{"type":"service_account",...}' councilflow
```

---

## Project structure

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit UI, auth, runner, RAG orchestration |
| `brain.py` | Gemini client, embeddings, planner, query expansion, re-ranking, agent |
| `rag_loader.py` | RAG state, retrieval planning, multi-query retrieval, context build |
| `rag.py` | Chunking, BM25, hybrid retrieval, RRF, deduplication |
| `librarian.py` | Drive client, file fetch, text extraction |
| `rag_cache.py` | Disk cache for library indexes |
| `db.py` | Config persistence: prompts, JSON schemas, app config (`council.db`) |
| `runs_db.py` | Run/analysis persistence: `AnalysisRun` table (`council_runs.db`) |
| `workflow.py` | Analysis workflow steps (Phase 2); used by `workflow_graph` | 
| `workflow_graph.py` | LangGraph StateGraph: plan → retrieve → cache → main agent → [conditional legal] → integrate → follow-on chain (Phase 3) |
| `config.yaml` | Auth (streamlit-authenticator) and cookie config |
| `RAG_ARCHITECTURE.md` | RAG design, document selection flow, config tunables |

**Databases:** `council.db` holds prompts, JSON schemas, and app config (export/import sync this file only). `council_runs.db` holds analysis run history and is not exported or imported; it is gitignored. See `AGENTIC_DESIGN_IDEAS.md` for the migration plan.

---

## Documentation

- **`RAG_ARCHITECTURE.md`** – RAG pipeline, knowledge base layout, retrieval flow, configuration, and comparison with NotebookLM.
