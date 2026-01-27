"""
SQLModel-backed persistence for prompt templates.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
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


class PromptTemplate(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    template_text: str
    output_mode: str  # 'markdown' (legacy: 'table' | 'text')
    verifier_id: Optional[int] = None
    follow_on_only: bool = Field(default=False, nullable=False)  # follow-on only
    legal_expert_prompt_id: Optional[int] = None  # prompt to use for legal questions
    # Optional reusable JSON Schemas (sidecars) for this prompt:
    # - input_schema_id: schema describing the expected transient input shape
    # - output_schema_id: schema describing the model's output shape
    input_schema_id: Optional[int] = Field(default=None, nullable=True)
    output_schema_id: Optional[int] = Field(default=None, nullable=True)


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


class AppConfig(SQLModel, table=True):
    """Application configuration stored in database (single row)."""
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=1, primary_key=True)  # Always ID 1 (singleton)
    selected_model: str = Field(default="gemini-3-flash-preview")  # Selected Gemini model
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
                    selected_model="gemini-3-flash-preview",
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
        _init_app_config()
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
    legal_expert_prompt_id: Optional[int] = None,
    *,
    input_schema_id: Optional[int] = None,
    output_schema_id: Optional[int] = None,
    id: Optional[int] = None,
) -> PromptTemplate:
    """Insert or update a prompt template. If id is given, update; else insert."""
    follow_on_only = bool(follow_on_only)
    logger.info(
        "Saving prompt: %s (id: %s, mode: %s, follow_on_only: %s, legal_expert: %s, "
        "input_schema_id: %s, output_schema_id: %s)",
        name,
        id,
        output_mode,
        follow_on_only,
        legal_expert_prompt_id,
        input_schema_id,
        output_schema_id,
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
                        input_schema_id=input_schema_id,
                        output_schema_id=output_schema_id,
                    )
                    s.add(p)
                else:
                    logger.debug(f"Updating prompt: {existing.name} -> {name}")
                    existing.name = name
                    existing.template_text = template_text
                    existing.output_mode = output_mode
                    existing.verifier_id = verifier_id
                    existing.follow_on_only = follow_on_only
                    existing.legal_expert_prompt_id = legal_expert_prompt_id
                    existing.input_schema_id = input_schema_id
                    existing.output_schema_id = output_schema_id
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
                    input_schema_id=input_schema_id,
                    output_schema_id=output_schema_id,
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
) -> JsonSchema:
    """
    Insert or update a JSON Schema. If id is given, update; else insert.
    The schema_json string should already be valid JSON (we do not enforce it here).
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
                    )
                    s.add(schema)
                else:
                    existing.name = name
                    existing.description = description
                    existing.schema_json = schema_json
                    schema = existing
            else:
                schema = JsonSchema(
                    name=name,
                    description=description,
                    schema_json=schema_json,
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
# Database Import/Export
# -----------------------------------------------------------------------------


def export_database(export_path: str | Path) -> str:
    """
    Export the database to a file by copying the database file.
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
    Import a database from a file by replacing the current database.
    
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