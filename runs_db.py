"""
Run/analysis persistence: AnalysisRun table in a separate SQLite file.

This module owns council_runs.db only. Config (prompts, schemas, app config)
lives in council.db (db.py). Export/import in db.py operate on council.db
only; run data is never exported or imported.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel, create_engine, select

from paths import data_path

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

# Same directory as council.db; run data is isolated from config (db.py).
# Use forward slashes so SqliteSaver/checkpointer can open the file on Windows.
RUNS_DB_PATH = str(data_path("council_runs.db"))
RUNS_SQLITE_URL = f"sqlite:///{RUNS_DB_PATH.replace(os.sep, '/')}"

_runs_engine = None


def _get_runs_engine():
    global _runs_engine
    if _runs_engine is None:
        logger.info(f"Creating runs database engine: {RUNS_DB_PATH}")
        _runs_engine = create_engine(
            RUNS_SQLITE_URL, connect_args={"check_same_thread": False}
        )
        logger.debug("Runs database engine created")
    return _runs_engine


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class AnalysisRun(SQLModel, table=True):
    """
    One row per analysis run. Stored in council_runs.db only.
    Not exported or imported (export/import in db.py are config-only).
    """
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = ""
    task_name: str = ""
    prompt_template_id: Optional[int] = None
    folder_id: str = ""  # KB root folder id
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "running"  # 'running' | 'completed' | 'failed'
    input_summary: str = ""  # e.g. content length or hash
    output_text: str = ""  # full result
    output_mode: str = "markdown"
    has_legal_review: bool = False
    legal_questions: Optional[str] = None  # JSON or plain text
    legal_expert_output: Optional[str] = None
    chain_steps: Optional[str] = None  # JSON: list of {step_name, output}
    retrieval_report_summary: Optional[str] = None
    model_used: Optional[str] = None
    error_message: Optional[str] = None
    prompt_version: Optional[int] = None  # PromptTemplate.current_version that produced this run
    stored_timezone: Optional[str] = None  # IANA name for stored datetimes (e.g. 'UTC')
    pre_qa_output: Optional[str] = None  # Phase 4: output before QA agent (if QA ran)
    qa_output: Optional[str] = None  # Phase 4: output after QA agent (if QA ran)
    output_json: Optional[str] = None  # JSON transformers: raw JSON when prompt had output schema


class RunEvent(SQLModel, table=True):
    """
    Phase 5.3: Step-level audit events for a run.
    Logged by graph nodes (node_started, node_completed); run_id backfilled after persist.
    """
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: Optional[int] = None  # FK to AnalysisRun; set after run is persisted
    thread_id: Optional[str] = None  # LangGraph thread_id; links events before run_id exists
    step_name: str = ""
    event_type: str = ""  # e.g. node_started, node_completed, interrupt_requested, human_responded
    payload: Optional[str] = None  # optional JSON or short text
    created_at: Optional[datetime] = None


# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------


def _migrate_prompt_version() -> None:
    """Add prompt_version column to analysisrun if missing (existing DBs)."""
    try:
        engine = _get_runs_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(analysisrun)")).fetchall()
        columns = [r[1].lower() for r in rows] if rows else []
        if "prompt_version" not in columns:
            logger.info("Adding prompt_version column to analysisrun")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE analysisrun ADD COLUMN prompt_version INTEGER"))
                conn.commit()
    except Exception as e:
        logger.warning(f"Migration prompt_version: {e}")


def _migrate_stored_timezone() -> None:
    """Add stored_timezone column to analysisrun if missing (existing DBs)."""
    try:
        engine = _get_runs_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(analysisrun)")).fetchall()
        columns = [r[1].lower() for r in rows] if rows else []
        if "stored_timezone" not in columns:
            logger.info("Adding stored_timezone column to analysisrun")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE analysisrun ADD COLUMN stored_timezone VARCHAR(64)"))
                conn.execute(text("UPDATE analysisrun SET stored_timezone = 'UTC' WHERE stored_timezone IS NULL"))
                conn.commit()
    except Exception as e:
        logger.warning(f"Migration stored_timezone: {e}")


def _migrate_pre_qa_qa_output() -> None:
    """Add pre_qa_output and qa_output columns to analysisrun if missing (existing DBs)."""
    try:
        engine = _get_runs_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(analysisrun)")).fetchall()
        columns = [r[1].lower() for r in rows] if rows else []
        with engine.connect() as conn:
            if "pre_qa_output" not in columns:
                logger.info("Adding pre_qa_output column to analysisrun")
                conn.execute(text("ALTER TABLE analysisrun ADD COLUMN pre_qa_output TEXT"))
            if "qa_output" not in columns:
                logger.info("Adding qa_output column to analysisrun")
                conn.execute(text("ALTER TABLE analysisrun ADD COLUMN qa_output TEXT"))
            conn.commit()
    except Exception as e:
        logger.warning(f"Migration pre_qa/qa_output: {e}")


def _migrate_output_json() -> None:
    """Add output_json column to analysisrun if missing (JSON transformers)."""
    try:
        engine = _get_runs_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("PRAGMA table_info(analysisrun)")).fetchall()
        columns = [r[1].lower() for r in rows] if rows else []
        if "output_json" not in columns:
            logger.info("Adding output_json column to analysisrun")
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE analysisrun ADD COLUMN output_json TEXT"))
                conn.commit()
    except Exception as e:
        logger.warning(f"Migration output_json: {e}")


def init_runs_db() -> None:
    """Create AnalysisRun and RunEvent tables if they do not exist. Only these tables are in council_runs.db."""
    logger.info("Initializing runs database")
    try:
        engine = _get_runs_engine()
        AnalysisRun.__table__.create(engine, checkfirst=True)
        RunEvent.__table__.create(engine, checkfirst=True)
        _migrate_prompt_version()
        _migrate_stored_timezone()
        _migrate_pre_qa_qa_output()
        _migrate_output_json()
        logger.debug("Runs database initialized")
    except Exception as e:
        logger.error(f"Error initializing runs database: {e}", exc_info=True)
        raise


# -----------------------------------------------------------------------------
# CRUD
# -----------------------------------------------------------------------------


def insert_analysis_run(
    username: str,
    task_name: str,
    status: str,
    *,
    prompt_template_id: Optional[int] = None,
    prompt_version: Optional[int] = None,
    folder_id: str = "",
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    input_summary: str = "",
    output_text: str = "",
    output_mode: str = "markdown",
    has_legal_review: bool = False,
    legal_questions: Optional[str] = None,
    legal_expert_output: Optional[str] = None,
    chain_steps: Optional[str] = None,
    retrieval_report_summary: Optional[str] = None,
    model_used: Optional[str] = None,
    error_message: Optional[str] = None,
    stored_timezone: Optional[str] = None,
    pre_qa_output: Optional[str] = None,
    qa_output: Optional[str] = None,
    output_json: Optional[str] = None,
) -> AnalysisRun:
    """Insert one analysis run. Returns the created row with id set.
    Datetimes are stored in UTC; stored_timezone records that (default 'UTC').
    """
    if started_at is None:
        started_at = datetime.now(timezone.utc)
    if stored_timezone is None:
        stored_timezone = "UTC"
    run = AnalysisRun(
        username=username,
        task_name=task_name,
        prompt_template_id=prompt_template_id,
        prompt_version=prompt_version,
        folder_id=folder_id,
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        input_summary=input_summary,
        output_text=output_text,
        output_mode=output_mode,
        has_legal_review=has_legal_review,
        legal_questions=legal_questions,
        legal_expert_output=legal_expert_output,
        chain_steps=chain_steps,
        retrieval_report_summary=retrieval_report_summary,
        model_used=model_used,
        error_message=error_message,
        stored_timezone=stored_timezone,
        pre_qa_output=pre_qa_output,
        qa_output=qa_output,
        output_json=output_json,
    )
    try:
        engine = _get_runs_engine()
        with Session(engine) as session:
            session.add(run)
            session.commit()
            session.refresh(run)
        logger.info(f"Inserted analysis run id={run.id} status={run.status}")
        return run
    except Exception as e:
        logger.error(f"Error inserting analysis run: {e}", exc_info=True)
        raise


def get_analysis_run_by_id(run_id: int) -> AnalysisRun | None:
    """Return the analysis run with the given id, or None."""
    try:
        engine = _get_runs_engine()
        with Session(engine) as session:
            return session.get(AnalysisRun, run_id)
    except Exception as e:
        logger.error(f"Error getting analysis run {run_id}: {e}", exc_info=True)
        raise


def list_analysis_runs(
    limit: int = 20,
    username_filter: Optional[str] = None,
    task_name_filter: Optional[str] = None,
    status_filter: Optional[str] = None,
    prompt_version_filter: Optional[int] = None,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> list[AnalysisRun]:
    """Return the most recent analysis runs, newest first. Optional filters."""
    # Normalize date bounds: accept date for start/end of day in UTC
    if date_from is not None and isinstance(date_from, date) and not isinstance(date_from, datetime):
        date_from = datetime.combine(date_from, time.min)
    if date_to is not None and isinstance(date_to, date) and not isinstance(date_to, datetime):
        date_to = datetime.combine(date_to, time.max)
    try:
        engine = _get_runs_engine()
        with Session(engine) as session:
            query = select(AnalysisRun).order_by(AnalysisRun.started_at.desc())
            if username_filter:
                query = query.where(AnalysisRun.username == username_filter)
            if task_name_filter:
                query = query.where(AnalysisRun.task_name == task_name_filter)
            if status_filter:
                query = query.where(AnalysisRun.status == status_filter)
            if prompt_version_filter is not None:
                query = query.where(AnalysisRun.prompt_version == prompt_version_filter)
            if date_from is not None:
                query = query.where(AnalysisRun.started_at >= date_from)
            if date_to is not None:
                query = query.where(AnalysisRun.started_at <= date_to)
            query = query.limit(limit)
            return list(session.exec(query).all())
    except Exception as e:
        logger.error(f"Error listing analysis runs: {e}", exc_info=True)
        raise


# -----------------------------------------------------------------------------
# RunEvent (P5.3)
# -----------------------------------------------------------------------------


def insert_run_event(
    step_name: str,
    event_type: str,
    *,
    thread_id: Optional[str] = None,
    run_id: Optional[int] = None,
    payload: Optional[str] = None,
) -> RunEvent:
    """Insert one run event. run_id can be set later via update_run_events_run_id."""
    ev = RunEvent(
        thread_id=thread_id,
        run_id=run_id,
        step_name=step_name,
        event_type=event_type,
        payload=payload,
        created_at=datetime.now(timezone.utc),
    )
    try:
        engine = _get_runs_engine()
        with Session(engine) as session:
            session.add(ev)
            session.commit()
            session.refresh(ev)
        return ev
    except Exception as e:
        logger.error(f"Error inserting run event: {e}", exc_info=True)
        raise


def update_run_events_run_id(thread_id: str, run_id: int) -> int:
    """Set run_id on all RunEvent rows with the given thread_id. Returns count updated."""
    try:
        from sqlalchemy import update
        engine = _get_runs_engine()
        with Session(engine) as session:
            stmt = update(RunEvent).where(RunEvent.thread_id == thread_id).values(run_id=run_id)
            result = session.execute(stmt)
            session.commit()
            return result.rowcount
    except Exception as e:
        logger.warning(f"Error updating run events run_id: {e}")
        return 0


def list_run_events(run_id: int) -> list[RunEvent]:
    """Return run events for the given run_id, ordered by created_at."""
    try:
        engine = _get_runs_engine()
        with Session(engine) as session:
            query = select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.created_at.asc())
            return list(session.exec(query).all())
    except Exception as e:
        logger.error(f"Error listing run events: {e}", exc_info=True)
        return []
