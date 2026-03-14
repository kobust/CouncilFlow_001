"""
SQLModel-backed persistence for prompt templates.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel, create_engine, select

from paths import data_path

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DB_PATH = str(data_path("council.db"))
SQLITE_URL = f"sqlite:///{DB_PATH}"

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        logger.info(f"Creating database engine: {DB_PATH}")
        _engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
        logger.debug("Database engine created")
    return _engine


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class Workflow(SQLModel, table=True):
    """
    Phase 5: Workflow definition. Prompts that are not follow-on-only can be assigned
    to a workflow (which graph to run when this prompt is selected at run start).
    """
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str  # display name, e.g. "Default (Analysis)"
    graph_key: str  # key used to select graph, e.g. "default", "agenda_review"


class PromptTemplate(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    template_text: str
    output_mode: str  # 'markdown' (legacy: 'table' | 'text')
    verifier_id: Optional[int] = None
    follow_on_only: bool = Field(default=False, nullable=False)  # follow-on only
    legal_expert_prompt_id: Optional[int] = None  # prompt to use for legal questions
    # Code-defined JSON schemas (from output_schemas registry). No DB schema table.
    input_schema_key: Optional[str] = Field(default=None, nullable=True)
    output_schema_key: Optional[str] = Field(default=None, nullable=True)
    # Auto-incremented on each save; used when recording which prompt version produced a run.
    current_version: Optional[int] = Field(default=1, nullable=True)
    # Phase 4: when True, run QA agent after follow-on chain to review and polish final output.
    use_qa_agent: bool = Field(default=False, nullable=False)
    # Phase 5: which workflow (graph) runs when this prompt is selected at start. N/A for follow_on_only.
    workflow_id: Optional[int] = Field(default=None, nullable=True)
    # JSON transformers: when output_schema_key is set, optional default transformer (e.g. "mayors_communication::Table").
    output_transformer_key: Optional[str] = Field(default=None, nullable=True)


class PromptVersion(SQLModel, table=True):
    """
    One row per saved version of a prompt. Created on each save (update or create).
    Enables recording which prompt version generated a run; revert tools can be added later.
    """
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    prompt_template_id: int  # FK to PromptTemplate.id
    version: int  # version number (1, 2, 3, ...)
    name: str
    template_text: str
    output_mode: str
    verifier_id: Optional[int] = None
    follow_on_only: bool = Field(default=False, nullable=False)
    legal_expert_prompt_id: Optional[int] = None
    input_schema_key: Optional[str] = None
    output_schema_key: Optional[str] = None
    use_qa_agent: bool = Field(default=False, nullable=False)
    workflow_id: Optional[int] = None
    output_transformer_key: Optional[str] = None
    saved_at: Optional[datetime] = None  # when this version was saved


class JsonSchema(SQLModel, table=True):
    """
    Reusable JSON Schemas that can be attached to one or more prompt templates.

    The schema payload is stored as raw JSON text so it can be edited directly
    in the admin UI and sent to Gemini as a sidecar without needing a specific
    Python type.
    """
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    # Optional short description for humans (how / where this schema is used)
    description: Optional[str] = None
    # Raw JSON Schema document (Draft-07+). Stored as TEXT in SQLite.
    schema_json: str
    # Optional: reference to code-defined schema bundle (output_schemas registry key).
    schema_key: Optional[str] = Field(default=None, nullable=True)
    # Optional: JSON array of transformer composite keys (e.g. ["constituent_reply::Summary", "constituent_reply::Full"]).
    transformer_keys: Optional[str] = Field(default=None, nullable=True)


class AppConfig(SQLModel, table=True):
    """Application configuration stored in database (single row)."""
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=1, primary_key=True)  # Always ID 1 (singleton)
    selected_model: str = Field(default="gemini-3.1-pro-preview")  # Selected Gemini model
    planner_model: Optional[str] = Field(default="gemini-2.0-flash")  # Planner model (optional override)


# -----------------------------------------------------------------------------
# Init & seed
# -----------------------------------------------------------------------------


def _migrate_follow_on_only() -> None:
    """Add follow_on_only column if missing (existing DBs)."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(prompttemplate)")).fetchall()
        columns = [r[1] for r in rows] if rows else []
        if "follow_on_only" not in columns:
            logger.info("Adding follow_on_only column to prompttemplate")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE prompttemplate ADD COLUMN follow_on_only BOOLEAN DEFAULT 0"))
                conn.commit()
    except Exception as e:
        logger.warning(f"Migration follow_on_only: {e}")


def _migrate_legal_expert_prompt_id() -> None:
    """Add legal_expert_prompt_id column if missing (existing DBs)."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(prompttemplate)")).fetchall()
        columns = [r[1] for r in rows] if rows else []
        if "legal_expert_prompt_id" not in columns:
            logger.info("Adding legal_expert_prompt_id column to prompttemplate")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE prompttemplate ADD COLUMN legal_expert_prompt_id INTEGER"))
                conn.commit()
    except Exception as e:
        logger.warning(f"Migration legal_expert_prompt_id: {e}")


def _migrate_prompt_schema_ids() -> None:
    """
    Add input_schema_id and output_schema_id columns to prompttemplate if missing.
    This keeps existing databases working without manual migrations.
    """
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(prompttemplate)")).fetchall()
        columns = [r[1] for r in rows] if rows else []
        with engine.connect() as conn:
            if "input_schema_id" not in columns:
                logger.info("Adding input_schema_id column to prompttemplate")
                conn.execute(text("ALTER TABLE prompttemplate ADD COLUMN input_schema_id INTEGER"))
            if "output_schema_id" not in columns:
                logger.info("Adding output_schema_id column to prompttemplate")
                conn.execute(text("ALTER TABLE prompttemplate ADD COLUMN output_schema_id INTEGER"))
            conn.commit()
    except Exception as e:
        logger.warning(f"Migration prompt schema ids: {e}")


def _migrate_current_version() -> None:
    """Add current_version column to prompttemplate if missing; set to 1 for existing rows."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(prompttemplate)")).fetchall()
        columns = [r[1] for r in rows] if rows else []
        if "current_version" not in columns:
            logger.info("Adding current_version column to prompttemplate")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE prompttemplate ADD COLUMN current_version INTEGER"))
                conn.execute(text("UPDATE prompttemplate SET current_version = 1 WHERE current_version IS NULL"))
                conn.commit()
    except Exception as e:
        logger.warning(f"Migration current_version: {e}")


def _migrate_workflow_id() -> None:
    """Add workflow_id column to prompttemplate and promptversion if missing (Phase 5)."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            rows_pt = conn.execute(text("PRAGMA table_info(prompttemplate)")).fetchall()
        columns_pt = [r[1].lower() for r in rows_pt] if rows_pt else []
        if "workflow_id" not in columns_pt:
            logger.info("Adding workflow_id column to prompttemplate")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE prompttemplate ADD COLUMN workflow_id INTEGER"))
                conn.commit()
        try:
            with engine.connect() as conn:
                rows_pv = conn.execute(text("PRAGMA table_info(promptversion)")).fetchall()
        except Exception:
            rows_pv = []
        columns_pv = [r[1].lower() for r in rows_pv] if rows_pv else []
        if rows_pv and "workflow_id" not in columns_pv:
            logger.info("Adding workflow_id column to promptversion")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE promptversion ADD COLUMN workflow_id INTEGER"))
                conn.commit()
    except Exception as e:
        logger.warning(f"Migration workflow_id: {e}")


def _migrate_use_qa_agent() -> None:
    """Add use_qa_agent column to prompttemplate and promptversion if missing (Phase 4)."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            rows_pt = conn.execute(text("PRAGMA table_info(prompttemplate)")).fetchall()
        columns_pt = [r[1] for r in rows_pt] if rows_pt else []
        if "use_qa_agent" not in columns_pt:
            logger.info("Adding use_qa_agent column to prompttemplate")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE prompttemplate ADD COLUMN use_qa_agent BOOLEAN DEFAULT 0"))
                conn.commit()
        # PromptVersion table (may not exist in very old DBs; create_all creates it)
        try:
            with engine.connect() as conn:
                rows_pv = conn.execute(text("PRAGMA table_info(promptversion)")).fetchall()
        except Exception:
            rows_pv = []
        columns_pv = [r[1] for r in rows_pv] if rows_pv else []
        if rows_pv and "use_qa_agent" not in columns_pv:
            logger.info("Adding use_qa_agent column to promptversion")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE promptversion ADD COLUMN use_qa_agent BOOLEAN DEFAULT 0"))
                conn.commit()
    except Exception as e:
        logger.warning(f"Migration use_qa_agent: {e}")


def _migrate_output_transformer_key() -> None:
    """Add output_transformer_key to prompttemplate and promptversion if missing (JSON transformers)."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            rows_pt = conn.execute(text("PRAGMA table_info(prompttemplate)")).fetchall()
        columns_pt = [r[1].lower() for r in rows_pt] if rows_pt else []
        if "output_transformer_key" not in columns_pt:
            logger.info("Adding output_transformer_key column to prompttemplate")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE prompttemplate ADD COLUMN output_transformer_key VARCHAR(128)"))
                conn.commit()
        try:
            with engine.connect() as conn:
                rows_pv = conn.execute(text("PRAGMA table_info(promptversion)")).fetchall()
        except Exception:
            rows_pv = []
        columns_pv = [r[1].lower() for r in rows_pv] if rows_pv else []
        if rows_pv and "output_transformer_key" not in columns_pv:
            logger.info("Adding output_transformer_key column to promptversion")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE promptversion ADD COLUMN output_transformer_key VARCHAR(128)"))
                conn.commit()
    except Exception as e:
        logger.warning(f"Migration output_transformer_key: {e}")


def _migrate_schema_keys() -> None:
    """Add input_schema_key and output_schema_key to prompttemplate and promptversion (code-defined schemas only)."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            rows_pt = conn.execute(text("PRAGMA table_info(prompttemplate)")).fetchall()
        columns_pt = [r[1].lower() for r in rows_pt] if rows_pt else []
        with engine.connect() as conn:
            if "output_schema_key" not in columns_pt:
                logger.info("Adding output_schema_key column to prompttemplate")
                conn.execute(text("ALTER TABLE prompttemplate ADD COLUMN output_schema_key VARCHAR(64)"))
            if "input_schema_key" not in columns_pt:
                logger.info("Adding input_schema_key column to prompttemplate")
                conn.execute(text("ALTER TABLE prompttemplate ADD COLUMN input_schema_key VARCHAR(64)"))
            conn.commit()
        try:
            with engine.connect() as conn:
                rows_pv = conn.execute(text("PRAGMA table_info(promptversion)")).fetchall()
        except Exception:
            rows_pv = []
        columns_pv = [r[1].lower() for r in rows_pv] if rows_pv else []
        if rows_pv:
            with engine.connect() as conn:
                if "output_schema_key" not in columns_pv:
                    logger.info("Adding output_schema_key column to promptversion")
                    conn.execute(text("ALTER TABLE promptversion ADD COLUMN output_schema_key VARCHAR(64)"))
                if "input_schema_key" not in columns_pv:
                    logger.info("Adding input_schema_key column to promptversion")
                    conn.execute(text("ALTER TABLE promptversion ADD COLUMN input_schema_key VARCHAR(64)"))
                conn.commit()
    except Exception as e:
        logger.warning(f"Migration schema_keys: {e}")


def _migrate_jsonschema_transformers() -> None:
    """Add schema_key and transformer_keys to jsonschema if missing (JSON transformers)."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(jsonschema)")).fetchall()
        columns = [r[1].lower() for r in rows] if rows else []
        with engine.connect() as conn:
            if "schema_key" not in columns:
                logger.info("Adding schema_key column to jsonschema")
                conn.execute(text("ALTER TABLE jsonschema ADD COLUMN schema_key VARCHAR(64)"))
            if "transformer_keys" not in columns:
                logger.info("Adding transformer_keys column to jsonschema")
                conn.execute(text("ALTER TABLE jsonschema ADD COLUMN transformer_keys TEXT"))
            conn.commit()
    except Exception as e:
        logger.warning(f"Migration jsonschema transformers: {e}")


def _migrate_default_model_to_3_1() -> None:
    """Migrate existing AppConfig rows that still use the old default model to Gemini 3.1 Pro."""
    try:
        engine = _get_engine()
        with Session(engine) as s:
            config = s.get(AppConfig, 1)
            if config and getattr(config, "selected_model", None) == "gemini-3-flash-preview":
                logger.info("Migrating AppConfig.selected_model from gemini-3-flash-preview to gemini-3.1-pro-preview")
                config.selected_model = "gemini-3.1-pro-preview"
                s.add(config)
                s.commit()
    except Exception as e:
        logger.warning(f"Migration default model to 3.1 failed (non-fatal): {e}")


def _seed_workflows() -> None:
    """Seed Workflow table with default workflow(s) if empty (Phase 5)."""
    try:
        engine = _get_engine()
        with Session(engine) as s:
            existing = list(s.exec(select(Workflow)).all())
            if existing:
                return
            logger.info("Seeding Workflow table")
            s.add(Workflow(name="Default (Analysis)", graph_key="default"))
            s.add(Workflow(name="Agenda packet review", graph_key="agenda_review"))
            s.commit()
            logger.info("Workflow table seeded")
    except Exception as e:
        logger.warning(f"Error seeding workflows: {e}")


def _init_app_config() -> None:
    """Initialize AppConfig table with default values if it doesn't exist."""
    try:
        engine = _get_engine()
        with Session(engine) as s:
            existing = s.get(AppConfig, 1)
            if existing is None:
                logger.info("Initializing AppConfig with default model")
                config = AppConfig(
                    id=1,
                    selected_model="gemini-3.1-pro-preview",
                    planner_model="gemini-2.0-flash",
                )
                s.add(config)
                s.commit()
                logger.info("AppConfig initialized")
            else:
                logger.debug("AppConfig already exists")
    except Exception as e:
        logger.warning(f"Error initializing AppConfig: {e}")


def init_db() -> None:
    """Create tables and seed default prompt templates."""
    logger.info("Initializing database")
    try:
        engine = _get_engine()
        logger.debug("Creating database tables")
        SQLModel.metadata.create_all(engine)
        _migrate_follow_on_only()
        _migrate_legal_expert_prompt_id()
        _migrate_prompt_schema_ids()
        _migrate_current_version()
        _migrate_use_qa_agent()
        _migrate_workflow_id()
        _migrate_output_transformer_key()
        _migrate_schema_keys()
        _migrate_jsonschema_transformers()
        _migrate_default_model_to_3_1()
        _init_app_config()
        _seed_workflows()
        with Session(engine) as s:
            existing = s.exec(select(PromptTemplate)).first()
            if existing is not None:
                logger.info("Database already seeded, skipping")
                return
            logger.info("Seeding default prompt templates")
            s.add(
                PromptTemplate(
                    name="MC Analysis",
                    template_text="Analyze this Mayor's Communication. Summarize key points, stakeholders, and recommendations.",
                    output_mode="markdown",
                    verifier_id=None,
                    follow_on_only=False,
                )
            )
            s.add(
                PromptTemplate(
                    name="Constituent Reply",
                    template_text="Draft a reply to this constituent message. Be professional, empathetic, and actionable.",
                    output_mode="markdown",
                    verifier_id=None,
                    follow_on_only=False,
                )
            )
            s.commit()
            logger.info("Database seeded successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        raise


# -----------------------------------------------------------------------------
# CRUD
# -----------------------------------------------------------------------------


def get_all_prompts() -> list[PromptTemplate]:
    """Return all prompt templates."""
    logger.debug("Fetching all prompts")
    try:
        engine = _get_engine()
        with Session(engine) as s:
            prompts = list(s.exec(select(PromptTemplate)).all())
            logger.info(f"Fetched {len(prompts)} prompts")
            return prompts
    except Exception as e:
        logger.error(f"Error fetching prompts: {e}", exc_info=True)
        raise


def get_prompt_by_id(prompt_id: int) -> PromptTemplate | None:
    """Get a prompt template by its ID."""
    logger.debug(f"Fetching prompt with id {prompt_id}")
    try:
        engine = _get_engine()
        with Session(engine) as s:
            prompt = s.get(PromptTemplate, prompt_id)
            if prompt:
                logger.debug(f"Found prompt: {prompt.name} (id: {prompt_id})")
            else:
                logger.warning(f"Prompt with id {prompt_id} not found")
            return prompt
    except Exception as e:
        logger.error(f"Error fetching prompt {prompt_id}: {e}", exc_info=True)
        raise


def list_prompt_versions(prompt_template_id: int) -> list[PromptVersion]:
    """Return version history for a prompt template, newest first (by version desc)."""
    try:
        engine = _get_engine()
        with Session(engine) as s:
            return list(
                s.exec(
                    select(PromptVersion)
                    .where(PromptVersion.prompt_template_id == prompt_template_id)
                    .order_by(PromptVersion.version.desc())
                ).all()
            )
    except Exception as e:
        logger.error(f"Error listing prompt versions for {prompt_template_id}: {e}", exc_info=True)
        return []


def get_prompt_version(prompt_template_id: int, version: int) -> PromptVersion | None:
    """Return one saved version of a prompt template, or None."""
    try:
        engine = _get_engine()
        with Session(engine) as s:
            return s.exec(
                select(PromptVersion).where(
                    PromptVersion.prompt_template_id == prompt_template_id,
                    PromptVersion.version == version,
                )
            ).first()
    except Exception as e:
        logger.error(f"Error getting prompt version {prompt_template_id} v{version}: {e}", exc_info=True)
        return None


def get_all_workflows() -> list[Workflow]:
    """Return all workflows (Phase 5)."""
    try:
        engine = _get_engine()
        with Session(engine) as s:
            return list(s.exec(select(Workflow).order_by(Workflow.id)).all())
    except Exception as e:
        logger.error(f"Error fetching workflows: {e}", exc_info=True)
        raise


def get_workflow_by_id(workflow_id: int) -> Workflow | None:
    """Get a workflow by its ID (Phase 5)."""
    try:
        engine = _get_engine()
        with Session(engine) as s:
            return s.get(Workflow, workflow_id)
    except Exception as e:
        logger.error(f"Error fetching workflow {workflow_id}: {e}", exc_info=True)
        raise


def save_prompt(
    name: str,
    template_text: str,
    output_mode: str,
    verifier_id: Optional[int] = None,
    follow_on_only: bool = False,
    legal_expert_prompt_id: Optional[int] = None,
    *,
    input_schema_key: Optional[str] = None,
    output_schema_key: Optional[str] = None,
    use_qa_agent: bool = False,
    workflow_id: Optional[int] = None,
    output_transformer_key: Optional[str] = None,
    id: Optional[int] = None,
) -> PromptTemplate:
    """Insert or update a prompt template. If id is given, update; else insert.
    On each save, a snapshot is written to PromptVersion and current_version is set/incremented.
    Schemas are code-defined only (from output_schemas registry); no DB schema table.
    """
    follow_on_only = bool(follow_on_only)
    logger.info(
        "Saving prompt: %s (id: %s, mode: %s, follow_on_only: %s, legal_expert: %s, "
        "input_schema_key: %s, output_schema_key: %s)",
        name,
        id,
        output_mode,
        follow_on_only,
        legal_expert_prompt_id,
        input_schema_key,
        output_schema_key,
    )
    try:
        engine = _get_engine()
        with Session(engine) as s:
            if id is not None:
                logger.debug(f"Updating existing prompt with id {id}")
                existing = s.get(PromptTemplate, id)
                if existing is None:
                    logger.warning(f"Prompt id {id} not found, creating new")
                    p = PromptTemplate(
                        name=name,
                        template_text=template_text,
                        output_mode=output_mode,
                        verifier_id=verifier_id,
                        follow_on_only=follow_on_only,
                        legal_expert_prompt_id=legal_expert_prompt_id,
                        input_schema_key=input_schema_key,
                        output_schema_key=output_schema_key,
                        current_version=1,
                        use_qa_agent=use_qa_agent,
                        workflow_id=workflow_id if not follow_on_only else None,
                        output_transformer_key=output_transformer_key,
                    )
                    s.add(p)
                    s.flush()  # get p.id
                    s.add(
                        PromptVersion(
                            prompt_template_id=p.id,
                            version=1,
                            name=p.name,
                            template_text=p.template_text,
                            output_mode=p.output_mode,
                            verifier_id=p.verifier_id,
                            follow_on_only=p.follow_on_only,
                            legal_expert_prompt_id=p.legal_expert_prompt_id,
                            input_schema_key=p.input_schema_key,
                            output_schema_key=p.output_schema_key,
                            use_qa_agent=p.use_qa_agent,
                            workflow_id=p.workflow_id,
                            output_transformer_key=output_transformer_key,
                            saved_at=datetime.now(timezone.utc),
                        )
                    )
                else:
                    logger.debug(f"Updating prompt: {existing.name} -> {name}")
                    # Snapshot current state to PromptVersion before updating
                    cur_ver = existing.current_version if existing.current_version is not None else 1
                    s.add(
                        PromptVersion(
                            prompt_template_id=existing.id,
                            version=cur_ver,
                            name=existing.name,
                            template_text=existing.template_text,
                            output_mode=existing.output_mode,
                            verifier_id=existing.verifier_id,
                            follow_on_only=existing.follow_on_only,
                            legal_expert_prompt_id=existing.legal_expert_prompt_id,
                            input_schema_key=existing.input_schema_key,
                            output_schema_key=existing.output_schema_key,
                            use_qa_agent=getattr(existing, "use_qa_agent", False),
                            workflow_id=getattr(existing, "workflow_id", None),
                            output_transformer_key=getattr(existing, "output_transformer_key", None),
                            saved_at=datetime.now(timezone.utc),
                        )
                    )
                    new_ver = cur_ver + 1
                    existing.name = name
                    existing.template_text = template_text
                    existing.output_mode = output_mode
                    existing.verifier_id = verifier_id
                    existing.follow_on_only = follow_on_only
                    existing.legal_expert_prompt_id = legal_expert_prompt_id
                    existing.input_schema_key = input_schema_key
                    existing.output_schema_key = output_schema_key
                    existing.use_qa_agent = use_qa_agent
                    existing.workflow_id = workflow_id if not follow_on_only else None
                    existing.output_transformer_key = output_transformer_key
                    existing.current_version = new_ver
                    p = existing
            else:
                logger.debug("Creating new prompt")
                p = PromptTemplate(
                    name=name,
                    template_text=template_text,
                    output_mode=output_mode,
                    verifier_id=verifier_id,
                    follow_on_only=follow_on_only,
                        legal_expert_prompt_id=legal_expert_prompt_id,
                        input_schema_key=input_schema_key,
                        output_schema_key=output_schema_key,
                        current_version=1,
                    use_qa_agent=use_qa_agent,
                    workflow_id=workflow_id if not follow_on_only else None,
                    output_transformer_key=output_transformer_key,
                )
                s.add(p)
                s.flush()
                s.add(
                    PromptVersion(
                        prompt_template_id=p.id,
                        version=1,
                        name=p.name,
                        template_text=p.template_text,
                        output_mode=p.output_mode,
                        verifier_id=p.verifier_id,
                        follow_on_only=p.follow_on_only,
                        legal_expert_prompt_id=p.legal_expert_prompt_id,
                        input_schema_key=p.input_schema_key,
                        output_schema_key=p.output_schema_key,
                        use_qa_agent=p.use_qa_agent,
                        workflow_id=p.workflow_id,
                        output_transformer_key=output_transformer_key,
                        saved_at=datetime.now(timezone.utc),
                    )
                )
            s.commit()
            s.refresh(p)
            logger.info(f"Prompt saved successfully: {p.name} (id: {p.id}, version: {p.current_version})")
            return p
    except Exception as e:
        logger.error(f"Error saving prompt {name}: {e}", exc_info=True)
        raise


def delete_prompt(prompt_id: int) -> bool:
    """
    Delete a prompt template by id. Any prompt that used this as a follow-on
    (verifier_id) is updated to have verifier_id=None. Returns True if deleted, False if not found.
    """
    logger.info(f"Deleting prompt id {prompt_id}")
    try:
        engine = _get_engine()
        with Session(engine) as s:
            existing = s.get(PromptTemplate, prompt_id)
            if existing is None:
                logger.warning(f"Prompt id {prompt_id} not found")
                return False
            # Clear verifier_id from any prompt that referenced this one
            for p in s.exec(select(PromptTemplate)).all():
                if p.verifier_id == prompt_id:
                    p.verifier_id = None
                    logger.debug(f"Cleared verifier_id from prompt {p.name} (id={p.id})")
            s.delete(existing)
            s.commit()
            logger.info(f"Deleted prompt {existing.name} (id={prompt_id})")
            return True
    except Exception as e:
        logger.error(f"Error deleting prompt {prompt_id}: {e}", exc_info=True)
        raise


# -----------------------------------------------------------------------------
# JSON Schema CRUD
# -----------------------------------------------------------------------------


def get_all_schemas() -> list[JsonSchema]:
    """Return all stored JSON Schemas."""
    logger.debug("Fetching all JSON Schemas")
    try:
        engine = _get_engine()
        with Session(engine) as s:
            schemas = list(s.exec(select(JsonSchema)).all())
            logger.info("Fetched %d JSON Schemas", len(schemas))
            return schemas
    except Exception as e:
        logger.error(f"Error fetching JSON Schemas: {e}", exc_info=True)
        raise


def get_schema_by_id(schema_id: int) -> JsonSchema | None:
    """Get a JSON Schema by its ID."""
    logger.debug("Fetching JSON Schema with id %s", schema_id)
    try:
        engine = _get_engine()
        with Session(engine) as s:
            schema = s.get(JsonSchema, schema_id)
            if schema:
                logger.debug("Found JSON Schema: %s (id: %s)", schema.name, schema_id)
            else:
                logger.warning("JSON Schema with id %s not found", schema_id)
            return schema
    except Exception as e:
        logger.error(f"Error fetching JSON Schema {schema_id}: {e}", exc_info=True)
        raise


def save_schema(
    name: str,
    schema_json: str,
    description: Optional[str] = None,
    *,
    id: Optional[int] = None,
    schema_key: Optional[str] = None,
    transformer_keys: Optional[str] = None,
) -> JsonSchema:
    """
    Insert or update a JSON Schema. If id is given, update; else insert.
    The schema_json string should already be valid JSON (we do not enforce it here).
    schema_key: optional code-defined bundle key. transformer_keys: optional JSON array of composite keys.
    """
    logger.info("Saving JSON Schema: %s (id: %s)", name, id)
    try:
        engine = _get_engine()
        with Session(engine) as s:
            if id is not None:
                existing = s.get(JsonSchema, id)
                if existing is None:
                    logger.warning("JSON Schema id %s not found, creating new", id)
                    schema = JsonSchema(
                        name=name,
                        description=description,
                        schema_json=schema_json,
                        schema_key=schema_key,
                        transformer_keys=transformer_keys,
                    )
                    s.add(schema)
                else:
                    existing.name = name
                    existing.description = description
                    existing.schema_json = schema_json
                    existing.schema_key = schema_key
                    existing.transformer_keys = transformer_keys
                    schema = existing
            else:
                schema = JsonSchema(
                    name=name,
                    description=description,
                    schema_json=schema_json,
                    schema_key=schema_key,
                    transformer_keys=transformer_keys,
                )
                s.add(schema)
            s.commit()
            s.refresh(schema)
            logger.info("JSON Schema saved successfully: %s (id: %s)", schema.name, schema.id)
            return schema
    except Exception as e:
        logger.error(f"Error saving JSON Schema {name}: {e}", exc_info=True)
        raise


def delete_schema(schema_id: int) -> bool:
    """Delete a JSON Schema by id. Returns True if deleted, False if not found."""
    logger.info("Deleting JSON Schema id %s", schema_id)
    try:
        engine = _get_engine()
        with Session(engine) as s:
            existing = s.get(JsonSchema, schema_id)
            if existing is None:
                logger.warning("JSON Schema id %s not found", schema_id)
                return False
            s.delete(existing)
            s.commit()
            logger.info("Deleted JSON Schema %s (id=%s)", existing.name, schema_id)
            return True
    except Exception as e:
        logger.error(f"Error deleting JSON Schema {schema_id}: {e}", exc_info=True)
        raise


# -----------------------------------------------------------------------------
# App Config CRUD
# -----------------------------------------------------------------------------


def get_app_config() -> AppConfig:
    """Get application configuration (singleton, always ID 1)."""
    logger.debug("Fetching app config")
    try:
        engine = _get_engine()
        with Session(engine) as s:
            config = s.get(AppConfig, 1)
            if config is None:
                # Initialize if missing
                _init_app_config()
                config = s.get(AppConfig, 1)
            return config
    except Exception as e:
        logger.error(f"Error fetching app config: {e}", exc_info=True)
        raise


def update_selected_model(model_name: str) -> AppConfig:
    """Update the selected Gemini model. Returns updated config."""
    logger.info(f"Updating selected model to: {model_name}")
    try:
        engine = _get_engine()
        with Session(engine) as s:
            config = s.get(AppConfig, 1)
            if config is None:
                config = AppConfig(id=1, selected_model=model_name)
                s.add(config)
            else:
                config.selected_model = model_name
            s.commit()
            s.refresh(config)
            logger.info(f"Model updated successfully: {config.selected_model}")
            return config
    except Exception as e:
        logger.error(f"Error updating selected model: {e}", exc_info=True)
        raise


def update_planner_model(model_name: str | None) -> AppConfig:
    """Update the planner model (optional override). Returns updated config."""
    logger.info(f"Updating planner model to: {model_name}")
    try:
        engine = _get_engine()
        with Session(engine) as s:
            config = s.get(AppConfig, 1)
            if config is None:
                config = AppConfig(id=1, planner_model=model_name)
                s.add(config)
            else:
                config.planner_model = model_name
            s.commit()
            s.refresh(config)
            logger.info(f"Planner model updated successfully: {config.planner_model}")
            return config
    except Exception as e:
        logger.error(f"Error updating planner model: {e}", exc_info=True)
        raise


# -----------------------------------------------------------------------------
# Database Import/Export (config only: council.db; run data lives in council_runs.db)
# -----------------------------------------------------------------------------


def export_database(export_path: str | Path) -> str:
    """
    Export the config database (council.db) to a file by copying the database file.
    Run/analysis data is not exported; it lives in council_runs.db (runs_db.py).
    Returns the path to the exported file.
    """
    export_path = Path(export_path)
    source_path = Path(DB_PATH)
    
    if not source_path.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")
    
    # Ensure export directory exists
    export_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy database file
    shutil.copy2(source_path, export_path)
    logger.info(f"Database exported to: {export_path}")
    return str(export_path)


def import_database(import_path: str | Path, backup_existing: bool = True) -> None:
    """
    Import the config database (council.db) from a file by replacing the current database.
    Run/analysis data in council_runs.db is never modified by import.

    Args:
        import_path: Path to the database file to import
        backup_existing: If True, create a backup of the current database before importing

    Raises:
        FileNotFoundError: If import_path doesn't exist
        ValueError: If import_path is not a valid SQLite database
    """
    import_path = Path(import_path)
    target_path = Path(DB_PATH)
    
    if not import_path.exists():
        raise FileNotFoundError(f"Import file not found: {import_path}")
    
    # Verify it's a valid SQLite database (check magic bytes)
    with open(import_path, "rb") as f:
        magic = f.read(16)
        if not magic.startswith(b"SQLite format 3"):
            raise ValueError(f"File does not appear to be a valid SQLite database: {import_path}")
    
    # Backup existing database if requested
    if backup_existing and target_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = target_path.parent / f"council.db.backup_{timestamp}"
        shutil.copy2(target_path, backup_path)
        logger.info(f"Backed up existing database to: {backup_path}")
    
    # Ensure target directory exists
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy imported database to target location
    shutil.copy2(import_path, target_path)
    
    # Reset the engine so it picks up the new database
    global _engine
    _engine = None
    
    logger.info(f"Database imported from: {import_path}")
    
    # Reinitialize to ensure schema is up to date
    init_db()


def get_database_info() -> dict:
    """
    Get information about the current database.
    Returns dict with file path, size, and record counts.
    """
    db_path = Path(DB_PATH)
    info = {
        "path": str(db_path),
        "exists": db_path.exists(),
        "size_bytes": db_path.stat().st_size if db_path.exists() else 0,
    }
    
    if db_path.exists():
        try:
            engine = _get_engine()
            with Session(engine) as s:
                # Count prompts
                prompt_count = len(list(s.exec(select(PromptTemplate)).all()))
                info["prompt_count"] = prompt_count
                # Count JSON Schemas (if table exists)
                try:
                    schema_count = len(list(s.exec(select(JsonSchema)).all()))
                    info["schema_count"] = schema_count
                except Exception:
                    # Older databases may not yet have the JsonSchema table
                    info["schema_count"] = 0
                
                # Get app config
                config = s.get(AppConfig, 1)
                if config:
                    info["selected_model"] = config.selected_model
                    info["planner_model"] = config.planner_model
                else:
                    info["selected_model"] = None
                    info["planner_model"] = None
        except Exception as e:
            logger.warning(f"Error getting database info: {e}")
            info["error"] = str(e)
    
    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()