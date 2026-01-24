"""
SQLModel-backed persistence for prompt templates.
"""

from __future__ import annotations

import logging
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


class PromptTemplate(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    template_text: str
    output_mode: str  # 'markdown' (legacy: 'table' | 'text')
    verifier_id: Optional[int] = None
    follow_on_only: bool = Field(default=False, nullable=False)  # follow-on only


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


def init_db() -> None:
    """Create tables and seed default prompt templates."""
    logger.info("Initializing database")
    try:
        engine = _get_engine()
        logger.debug("Creating database tables")
        SQLModel.metadata.create_all(engine)
        _migrate_follow_on_only()
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


def save_prompt(
    name: str,
    template_text: str,
    output_mode: str,
    verifier_id: Optional[int] = None,
    follow_on_only: bool = False,
    *,
    id: Optional[int] = None,
) -> PromptTemplate:
    """Insert or update a prompt template. If id is given, update; else insert."""
    follow_on_only = bool(follow_on_only)
    logger.info(f"Saving prompt: {name} (id: {id}, mode: {output_mode}, follow_on_only: {follow_on_only})")
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
                    )
                    s.add(p)
                else:
                    logger.debug(f"Updating prompt: {existing.name} -> {name}")
                    existing.name = name
                    existing.template_text = template_text
                    existing.output_mode = output_mode
                    existing.verifier_id = verifier_id
                    existing.follow_on_only = follow_on_only
                    p = existing
            else:
                logger.debug("Creating new prompt")
                p = PromptTemplate(
                    name=name,
                    template_text=template_text,
                    output_mode=output_mode,
                    verifier_id=verifier_id,
                    follow_on_only=follow_on_only,
                )
                s.add(p)
            s.commit()
            s.refresh(p)
            logger.info(f"Prompt saved successfully: {p.name} (id: {p.id})")
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()