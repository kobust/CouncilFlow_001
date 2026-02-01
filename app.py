"""
Attleboro Council Agent: auth, Drive-backed knowledge base, prompt tasks, Gemini agent.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # type: ignore[misc, assignment]

# Token usage optimization: configurable delays to spread API calls across time
PIPELINE_STEP_DELAY_SECONDS = int(os.environ.get("PIPELINE_STEP_DELAY_SECONDS", "10") or "10")
LEGAL_EXPERT_DELAY_SECONDS = int(os.environ.get("LEGAL_EXPERT_DELAY_SECONDS", "0") or "0")

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import streamlit_authenticator as stauth
import yaml
from docx import Document
from pypdf import PdfReader
from yaml.loader import SafeLoader

# HTML to Markdown conversion (optional)
HTML2MD_AVAILABLE = False
try:
    import html2text
    HTML2MD_AVAILABLE = True
except ImportError:
    HTML2MD_AVAILABLE = False

# OCR imports (optional - will fail gracefully if not available)
OCR_AVAILABLE = False
OCR_ERROR_MSG = None
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    # Check if Tesseract binary is actually available
    try:
        pytesseract.get_tesseract_version()
        OCR_AVAILABLE = True
    except Exception as tess_err:
        OCR_AVAILABLE = False
        OCR_ERROR_MSG = f"Tesseract OCR binary not found: {tess_err}"
except ImportError as import_err:
    OCR_AVAILABLE = False
    OCR_ERROR_MSG = f"OCR Python libraries not installed: {import_err}"

import brain
import workflow as workflow_module
import workflow_graph
from brain import (
    CacheExpiredError,
    get_effective_model,
    get_planner_model,
    list_available_models,
    GEMINI_PACE_DELAY_SECONDS,
    chars_to_tokens,
    expand_queries,
    extract_legal_questions,
    format_context_usage,
    format_reading_equivalent,
    model_max_context,
)
import db
import runs_db
from librarian import get_cached_file_list, get_cached_folder_info
from rag_cache import clear_disk_cache_for_folder
from rag_loader import (
    get_cached_rag_state,
    get_default_plan,
    get_fallback_phrases,
    plan_retrieval,
    retrieve_and_build_context_multi,
    USE_QUERY_EXPANSION,
    USE_RETRIEVAL_PLANNER,
)
from paths import data_path, repo_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Log OCR availability after logger is initialized
if not OCR_AVAILABLE:
    if OCR_ERROR_MSG:
        logger.warning(f"OCR not available: {OCR_ERROR_MSG}")
    else:
        logger.warning("OCR libraries (pytesseract, pdf2image) not available. OCR will be disabled for scanned PDFs.")

# Log HTML2MD availability
if not HTML2MD_AVAILABLE:
    logger.warning("html2text library not available. HTML to Markdown conversion will be disabled.")

APP_NAME = "Attleboro Council Agent"
COUNCILFLOW_VERSION = "1.0.0"


def get_git_version() -> str:
    """Get version string from git commit hash and build number."""
    # Read build number
    build_number = None
    try:
        build_file = repo_path("BUILD_NUMBER")
        if os.path.exists(build_file):
            with open(build_file, "r", encoding="utf-8") as f:
                build_number = f.read().strip()
    except Exception as e:
        logger.debug(f"Could not read build number: {e}")
    
    try:
        # Try to get short commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo_path(),
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            # Try to get tag if available
            tag_result = subprocess.run(
                ["git", "describe", "--tags", "--exact-match", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=repo_path(),
            )
            if tag_result.returncode == 0:
                tag = tag_result.stdout.strip()
                if build_number:
                    return f"{tag}.{build_number}"
                return tag
            # Try to get branch name
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                cwd=repo_path(),
            )
            branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
            version_str = f"{COUNCILFLOW_VERSION}-{commit_hash[:7]}"
            if build_number:
                version_str += f".{build_number}"
            return f"{version_str} ({branch})"
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logger.debug(f"Could not get git version: {e}")
    
    # Fallback: return base version with build number if available
    if build_number:
        return f"{COUNCILFLOW_VERSION}.{build_number}"
    return COUNCILFLOW_VERSION

# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------

logger.info("Starting %s", APP_NAME)

# Page config must be first Streamlit command
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    logger.debug("Loading config.yaml")
    config_path = repo_path("config.yaml")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=SafeLoader)
    logger.debug("Config loaded successfully")
except FileNotFoundError:
    logger.error("config.yaml not found")
    st.error("Configuration file (config.yaml) not found. Place it in the app data folder.")
    st.stop()
except Exception as e:
    logger.error(f"Error loading config.yaml: {e}", exc_info=True)
    st.error(f"Error loading configuration: {e}")
    st.stop()

try:
    logger.debug("Initializing authenticator")
    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )
    logger.debug("Authenticator initialized")
except KeyError as e:
    logger.error(f"Missing config key: {e}")
    st.error(f"Configuration error: missing {e}")
    st.stop()
except Exception as e:
    logger.error(f"Error initializing authenticator: {e}", exc_info=True)
    st.error(f"Authentication setup failed: {e}")
    st.stop()

logger.debug("Rendering login widget")
authenticator.login(location="main")

if not st.session_state.get("authentication_status"):
    if st.session_state.get("authentication_status") is False:
        logger.warning("Login failed: incorrect credentials")
        st.error("Username/password is incorrect")
    else:
        logger.debug("User not authenticated, stopping")
    st.stop()

_username = st.session_state.get("username", "")
logger.info(f"User authenticated: {_username} ({st.session_state.get('name', 'unknown')})")
is_admin = _username == "admin"

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------

logger.debug("Initializing database")
try:
    db.init_db()
    logger.debug("Database initialized")
except Exception as e:
    logger.error(f"Database initialization failed: {e}", exc_info=True)
    st.error(f"Database error: {e}")
    st.stop()

logger.debug("Initializing runs database")
try:
    runs_db.init_runs_db()
    logger.debug("Runs database initialized")
except Exception as e:
    logger.error(f"Runs database initialization failed: {e}", exc_info=True)
    st.error(f"Runs database error: {e}")
    st.stop()

# Pre-load disk caches on startup for faster first use
if "caches_preloaded" not in st.session_state:
    try:
        from brain import _load_all_disk_caches
        _load_all_disk_caches()
        st.session_state["caches_preloaded"] = True
        logger.info("Pre-loaded all disk caches on startup")
    except Exception as e:
        logger.debug(f"Could not pre-load caches (non-fatal): {e}")
        st.session_state["caches_preloaded"] = True  # Mark as attempted

# Load session state cache from disk (persist across sessions)
if "query_embedding_cache" not in st.session_state:
    try:
        from rag_cache import load_query_embedding_cache
        from brain import get_query_hash
        disk_cache = load_query_embedding_cache()
        # Initialize session cache with disk cache (limited to 1000 entries for memory efficiency)
        session_cache = dict(list(disk_cache.items())[-1000:])
        st.session_state["query_embedding_cache"] = session_cache
        if session_cache:
            logger.debug(f"Loaded {len(session_cache)} query embeddings into session cache from disk")
    except Exception as e:
        logger.debug(f"Could not load session cache from disk (non-fatal): {e}")
        st.session_state["query_embedding_cache"] = {}

# Initialize session state
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_mode" not in st.session_state:
    st.session_state["last_mode"] = None
if "last_task_name" not in st.session_state:
    st.session_state["last_task_name"] = None
if "gemini_cache_name" not in st.session_state:
    st.session_state["gemini_cache_name"] = None
if "gemini_cache_model" not in st.session_state:
    st.session_state["gemini_cache_model"] = None
if "gemini_cache_folder_id" not in st.session_state:
    st.session_state["gemini_cache_folder_id"] = None
if "kb_loading_started" not in st.session_state:
    st.session_state["kb_loading_started"] = False
if "kb_loaded" not in st.session_state:
    st.session_state["kb_loaded"] = False
if "kb_load_error" not in st.session_state:
    st.session_state["kb_load_error"] = None
if "transient_items" not in st.session_state:
    st.session_state["transient_items"] = []  # [{id, name, content, type: 'file'|'paste'}, ...]
if "transient_deleted_file_names" not in st.session_state:
    st.session_state["transient_deleted_file_names"] = []  # file names user deleted
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "runner"  # "runner" | "edit_prompts"
if "analysis_session_id" not in st.session_state:
    st.session_state["analysis_session_id"] = 0  # incremented on New Analysis
if "last_rag_retrieval_report" not in st.session_state:
    st.session_state["last_rag_retrieval_report"] = None
if "last_chain" not in st.session_state:
    st.session_state["last_chain"] = None  # [(name, output), ...]
if "last_chain_error" not in st.session_state:
    st.session_state["last_chain_error"] = None
logger.debug("Session state initialized")

# -----------------------------------------------------------------------------
# Load Knowledge Base at Boot (moved to after sidebar setup)
# -----------------------------------------------------------------------------

DEFAULT_FOLDER_ID = "1DBKa-Ol0TU-TVUkl73HomoMdUK8RjE-0"
folder_id = DEFAULT_FOLDER_ID


def _extract_text_via_poppler_html(pdf_bytes: bytes, name: str) -> str | None:
    """
    Extract text from PDF using Poppler's pdftohtml, then convert HTML to Markdown.
    This preserves structure like tables, lists, headings, etc.
    
    Returns markdown text or None if conversion fails.
    """
    if not HTML2MD_AVAILABLE:
        logger.debug("html2text not available, skipping Poppler HTML extraction")
        return None
    
    try:
        # Check if pdftohtml is available
        pdftohtml_cmd = "pdftohtml"
        try:
            subprocess.run([pdftohtml_cmd, "-v"], capture_output=True, check=True, timeout=5)
        except FileNotFoundError:
            # Try common Windows installation paths
            import os
            possible_paths = [
                r"C:\Program Files\poppler\bin\pdftohtml.exe",
                r"C:\poppler\bin\pdftohtml.exe",
                r"C:\Program Files (x86)\poppler\bin\pdftohtml.exe",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pdftohtml_cmd = path
                    logger.debug(f"Found pdftohtml at: {path}")
                    break
            else:
                logger.debug("pdftohtml not found in PATH or common locations, skipping Poppler HTML extraction")
                return None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.debug("pdftohtml found but failed version check, skipping Poppler HTML extraction")
            return None
        
        logger.info(f"Extracting PDF {name} via Poppler HTML (preserves structure)")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as pdf_temp:
            pdf_path = pdf_temp.name
            pdf_temp.write(pdf_bytes)
        
        try:
            html_path = pdf_path.replace(".pdf", ".html")
            
            # Convert PDF to HTML using pdftohtml
            # -i: ignore images
            # -s: single HTML file
            # -noframes: no frames
            result = subprocess.run(
                [pdftohtml_cmd, "-i", "-s", "-noframes", pdf_path, html_path],
                capture_output=True,
                text=True,
                timeout=60,
                check=True
            )
            
            # Read the HTML file
            try:
                with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
                    html_content = f.read()
            except FileNotFoundError:
                # Sometimes pdftohtml creates files with different names
                base_name = html_path.replace(".html", "")
                possible_names = [f"{base_name}.html", f"{base_name}-1.html", f"{base_name}-s.html"]
                html_content = None
                for possible_name in possible_names:
                    try:
                        with open(possible_name, "r", encoding="utf-8", errors="ignore") as f:
                            html_content = f.read()
                            logger.debug(f"Found HTML file: {possible_name}")
                            break
                    except FileNotFoundError:
                        continue
                
                if html_content is None:
                    logger.warning(f"Could not find HTML output file for {name}")
                    return None
            
            if not html_content or len(html_content.strip()) < 50:
                logger.warning(f"HTML output for {name} is too small or empty")
                return None
            
            # Convert HTML to Markdown
            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = True
            h.body_width = 0  # Don't wrap lines
            h.unicode_snob = True  # Use unicode
            markdown_text = h.handle(html_content)
            
            # Clean up temporary files
            try:
                os.unlink(pdf_path)
                if os.path.exists(html_path):
                    os.unlink(html_path)
                # Clean up any numbered HTML files
                base_name = html_path.replace(".html", "")
                for possible_name in [f"{base_name}-1.html", f"{base_name}-s.html"]:
                    if os.path.exists(possible_name):
                        os.unlink(possible_name)
            except Exception as cleanup_err:
                logger.debug(f"Error cleaning up temp files: {cleanup_err}")
            
            if markdown_text and len(markdown_text.strip()) > 50:
                logger.info(f"Successfully extracted {len(markdown_text)} chars from {name} via Poppler HTML")
                return markdown_text
            else:
                logger.warning(f"Poppler HTML extraction for {name} produced too little text")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.warning(f"pdftohtml failed for {name}: {e.stderr or e.stdout}")
            return None
        except subprocess.TimeoutExpired:
            logger.warning(f"pdftohtml timed out for {name}")
            return None
        except Exception as e:
            logger.warning(f"Error in Poppler HTML extraction for {name}: {e}", exc_info=True)
            return None
        finally:
            # Ensure cleanup
            import os
            try:
                if os.path.exists(pdf_path):
                    os.unlink(pdf_path)
            except Exception:
                pass
    
    except Exception as e:
        logger.error(f"Poppler HTML extraction failed for {name}: {e}", exc_info=True)
        return None


def _extract_text_with_ocr(pdf_bytes: bytes, name: str) -> str:
    """Extract text from PDF using OCR (for scanned/image-based PDFs)."""
    if not OCR_AVAILABLE:
        logger.warning(f"OCR not available for {name}: {OCR_ERROR_MSG or 'OCR libraries not installed'}")
        return None
    
    try:
        logger.info(f"Attempting OCR on {name}")
        # Convert PDF pages to images
        try:
            images = convert_from_bytes(pdf_bytes, dpi=300)
            logger.info(f"Converted {len(images)} pages to images for OCR")
        except Exception as pdf2img_err:
            logger.error(f"Failed to convert PDF to images for {name}: {pdf2img_err}", exc_info=True)
            # Check if it's a Poppler error
            error_str = str(pdf2img_err).lower()
            if "poppler" in error_str or "pdftoppm" in error_str:
                logger.error("Poppler (pdf2image dependency) may not be installed. On Windows, download from: https://github.com/oschwartz10612/poppler-windows/releases")
            return None
        
        ocr_parts = []
        successful_ocr_pages = 0
        for page_num, image in enumerate(images, 1):
            try:
                # Perform OCR on the image
                text = pytesseract.image_to_string(image, lang='eng')
                if text and text.strip():
                    ocr_parts.append(text)
                    successful_ocr_pages += 1
                    logger.debug(f"OCR page {page_num}: Extracted {len(text)} chars")
                else:
                    logger.debug(f"OCR page {page_num}: No text found")
            except pytesseract.TesseractNotFoundError as tess_err:
                logger.error(f"Tesseract OCR binary not found for page {page_num} of {name}: {tess_err}")
                logger.error("Please install Tesseract OCR. On Windows: https://github.com/UB-Mannheim/tesseract/wiki")
                return None
            except Exception as ocr_error:
                logger.warning(f"OCR failed on page {page_num} of {name}: {ocr_error}", exc_info=True)
                ocr_parts.append(f"[Page {page_num}: OCR error - {type(ocr_error).__name__}]")
        
        result = "\n\n".join(ocr_parts)
        logger.info(f"OCR completed: {len(result)} chars extracted from {name} ({successful_ocr_pages}/{len(images)} pages successful)")
        return result if result.strip() else None
    except Exception as e:
        logger.error(f"OCR failed for {name}: {e}", exc_info=True)
        # Provide helpful error messages
        error_str = str(e).lower()
        if "tesseract" in error_str or "tesseractnotfound" in error_str:
            logger.error("Tesseract OCR binary not found. Please install Tesseract OCR.")
        elif "poppler" in error_str or "pdftoppm" in error_str:
            logger.error("Poppler not found. Required for pdf2image. On Windows: https://github.com/oschwartz10612/poppler-windows/releases")
        return None


def _extract_text_from_upload(file) -> str:
    """Extract text from uploaded PDF or DOCX."""
    name = (getattr(file, "name", None) or "file").lower()
    logger.info(f"Extracting text from uploaded file: {name} (type: {type(file).__name__})")
    try:
        # Reset file pointer in case it was already read
        if hasattr(file, 'seek'):
            file.seek(0)
        
        # Read file content
        raw = file.read()
        logger.info(f"Read {len(raw)} bytes from {name}")
        
        if len(raw) == 0:
            logger.warning(f"File {name} appears to be empty (0 bytes)")
            return f"[Error: File {name} is empty]"
        
        # Verify it looks like a PDF (starts with %PDF)
        if name.endswith(".pdf"):
            if not raw.startswith(b'%PDF'):
                logger.warning(f"File {name} has .pdf extension but doesn't start with %PDF signature. First bytes: {raw[:20]}")
                # Still try to process it
            
            logger.debug(f"Extracting from PDF: {name}")
            
            # First, try Poppler HTML extraction (preserves structure like tables, lists)
            poppler_result = _extract_text_via_poppler_html(raw, name)
            if poppler_result and len(poppler_result.strip()) > 100:
                logger.info(f"Successfully extracted {len(poppler_result)} chars from {name} via Poppler HTML (with structure)")
                return poppler_result
            
            # Fallback to standard pypdf extraction
            try:
                bio = io.BytesIO(raw)
                reader = PdfReader(bio)
                logger.debug(f"PDF reader created, {len(reader.pages)} pages found")
                
                parts = []
                total_pages = len(reader.pages)
                successful_pages = 0
                failed_pages = []
                
                for page_num, p in enumerate(reader.pages, 1):
                    try:
                        t = p.extract_text()
                        if t and t.strip():
                            parts.append(t)
                            successful_pages += 1
                            logger.debug(f"Page {page_num}: Extracted {len(t)} chars")
                        else:
                            logger.debug(f"Page {page_num}: No text extracted (empty or whitespace only)")
                            parts.append(f"[Page {page_num}: No text found]")
                    except (KeyError, AttributeError, ValueError) as page_error:
                        # Common PDF parsing errors (malformed fonts, missing bbox, etc.)
                        failed_pages.append(page_num)
                        error_type = type(page_error).__name__
                        error_msg = str(page_error)[:100]
                        logger.warning(f"Error extracting text from page {page_num}/{total_pages} of {name}: {error_type} - {error_msg}")
                        parts.append(f"[Page {page_num}: Error extracting text - {error_type}]")
                    except Exception as page_error:
                        # Other unexpected errors
                        failed_pages.append(page_num)
                        logger.warning(f"Unexpected error extracting text from page {page_num}/{total_pages} of {name}: {page_error}", exc_info=True)
                        parts.append(f"[Page {page_num}: Error extracting text - {type(page_error).__name__}]")
                
                result = "\n".join(parts)
                if failed_pages:
                    logger.info(f"Extracted text from {successful_pages}/{total_pages} pages of {name} (failed pages: {failed_pages})")
                logger.info(f"Extracted {len(result)} chars from PDF {name} ({successful_pages}/{total_pages} pages successful)")
                
                # Calculate actual text content (excluding error messages)
                # Filter out error placeholders to get real text length
                actual_text_parts = [p for p in parts if not (p.startswith("[Page") or p.startswith("[Error"))]
                actual_text = "\n".join(actual_text_parts)
                actual_text_len = len(actual_text.strip())
                
                # If no pages succeeded OR very little actual text was extracted, try OCR (for scanned PDFs)
                should_try_ocr = successful_pages == 0 or actual_text_len < 50
                
                if should_try_ocr:
                    logger.info(f"PDF {name} has no successful pages or very little text ({actual_text_len} chars from {successful_pages} pages). Attempting OCR...")
                    ocr_result = _extract_text_with_ocr(raw, name)
                    if ocr_result and len(ocr_result.strip()) > 50:
                        logger.info(f"OCR successfully extracted {len(ocr_result)} chars from {name}")
                        return ocr_result
                    elif ocr_result:
                        logger.warning(f"OCR extracted only {len(ocr_result)} chars from {name}")
                        return ocr_result
                    else:
                        logger.warning(f"OCR failed or unavailable for {name}. Returning minimal text extraction.")
                        if not OCR_AVAILABLE:
                            error_detail = OCR_ERROR_MSG or "OCR libraries not installed"
                            install_msg = ""
                            if "tesseract" in error_detail.lower():
                                install_msg = "\n\nTo install Tesseract OCR on Windows:\n1. Download from: https://github.com/UB-Mannheim/tesseract/wiki\n2. Install to default location or add to PATH\n3. Also install Poppler: https://github.com/oschwartz10612/poppler-windows/releases"
                            return f"[ERROR: PDF {name} appears to be a scanned/image-based document. Text extraction failed on all pages.\n\nOCR is required but not available: {error_detail}{install_msg}\n\nPlease install Tesseract OCR and Poppler, then restart Streamlit and try again.]\n\n{result}"
                        return f"[Warning: PDF {name} appears to be image-based (scanned). OCR was attempted but failed. Check logs for details.]\n\n{result}"
                
                return result
            except Exception as pdf_error:
                logger.error(f"Failed to read PDF {name}: {pdf_error}", exc_info=True)
                return f"[Error reading PDF {name}: {pdf_error}]"
        if name.endswith(".docx") or name.endswith(".doc"):
            logger.debug(f"Extracting from Word doc: {name}")
            try:
                doc = Document(io.BytesIO(raw))
                result = "\n".join(p.text for p in doc.paragraphs if p.text)
                logger.debug(f"Extracted {len(result)} chars from Word doc {name}")
                if len(result) == 0:
                    logger.warning(f"Word document {name} appears to have no extractable text")
                    return f"[Warning: Document {name} contains no extractable text]"
                return result
            except Exception as doc_error:
                logger.error(f"Failed to read Word doc {name}: {doc_error}", exc_info=True)
                return f"[Error reading Word document {name}: {doc_error}]"
        logger.debug(f"Decoding as plain text: {name}")
        result = raw.decode("utf-8", errors="replace")
        logger.debug(f"Decoded {len(result)} chars")
        if len(result) == 0:
            logger.warning(f"Text file {name} appears to be empty")
            return f"[Warning: Text file {name} is empty]"
        return result
    except Exception as e:
        logger.error(f"Error extracting text from {name}: {e}", exc_info=True)
        return f"[Error extracting text from {name}: {e}]"


def _dict_to_dataframe(obj) -> pd.DataFrame | None:
    """Convert JSON result to DataFrame (list of dicts or headers/rows)."""
    logger.debug(f"Converting to DataFrame: type={type(obj).__name__}")
    try:
        if isinstance(obj, list):
            if obj and isinstance(obj[0], dict):
                logger.debug(f"Converting list of {len(obj)} dicts to DataFrame")
                return pd.DataFrame(obj)
            else:
                logger.warning("List is empty or first element is not a dict")
                return None
        if isinstance(obj, dict):
            rows = obj.get("rows")
            headers = obj.get("headers")
            if headers and rows:
                logger.debug(f"Converting headers/rows format: {len(rows)} rows, {len(headers)} cols")
                return pd.DataFrame(rows, columns=headers)
            if "data" in obj and isinstance(obj["data"], list):
                logger.debug(f"Converting data array: {len(obj['data'])} items")
                return pd.DataFrame(obj["data"])
            # Try to convert dict directly (if it's a single row)
            logger.debug("Attempting to convert dict as single-row DataFrame")
            return pd.DataFrame([obj])
        logger.warning(f"Cannot convert {type(obj).__name__} to DataFrame")
        return None
    except Exception as e:
        logger.error(f"Error converting to DataFrame: {e}", exc_info=True)
        return None


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Turn DataFrame into Markdown table string."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def _markdown_with_copy(md: str, key_suffix: str) -> None:
    """Render markdown and show a Copy block so the user can copy the markdown source."""
    st.markdown(md)
    with st.expander(f"📋 Copy markdown ({key_suffix})", expanded=False):
        st.caption("Use the copy icon in the code block below to copy the markdown.")
        st.code(md, language="markdown", line_numbers=False)


def _format_run_datetime(
    dt: datetime | None,
    stored_tz: str = "UTC",
) -> str:
    """Format a run datetime (stored in stored_tz, typically UTC) for display in local time."""
    if dt is None:
        return "?"
    try:
        # Assume naive datetimes from DB are in stored_tz (UTC)
        dt_aware = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        if dt_aware.tzinfo is None:
            dt_aware = dt_aware.replace(tzinfo=timezone.utc)
        # Convert to server's local timezone (user's local when app runs locally)
        local = dt_aware.astimezone()
        return local.strftime("%Y-%m-%d %H:%M %Z")
    except Exception:
        return dt.strftime("%Y-%m-%d %H:%M") + " (UTC)"


def _get_json_view_options(schema_key: str | None = None) -> list[tuple[str, str]]:
    """Return [(value, label), ...] for JSON output view selector: Saved, Raw JSON, then transformer views."""
    options: list[tuple[str, str]] = [("saved", "Saved"), ("raw", "Raw JSON")]
    if not schema_key:
        return options
    try:
        import output_schemas
        output_schemas.ensure_registry_loaded()
        for display_name, _ in output_schemas.get_transformers(schema_key):
            composite = f"{schema_key}::{display_name}"
            options.append((composite, f"{schema_key} :: {display_name}"))
    except Exception as e:
        logger.debug("Could not get transformer options (schema_key=%s): %s", schema_key, e)
    return options


def _render_json_view(
    output_json_str: str | None,
    output_text_saved: str,
    view_choice: str,
) -> str:
    """Return the string to display for the chosen view (Saved, Raw JSON, or transformer result)."""
    if view_choice == "saved":
        return output_text_saved or "(no output)"
    if view_choice == "raw":
        if not output_json_str:
            return "(raw JSON not stored for this run)"
        try:
            parsed = json.loads(output_json_str)
            return json.dumps(parsed, indent=2)
        except Exception:
            return output_json_str
    # Transformer key
    if not output_json_str:
        return "(raw JSON not stored; cannot re-run transformer)"
    try:
        import output_schemas
        output_schemas.ensure_registry_loaded()
        data = json.loads(output_json_str)
        return output_schemas.run_transformer(view_choice, data)
    except Exception as e:
        return f"Transformer failed: {e}"


def _build_prompt_variables(username: str = "", user_name: str = "") -> str:
    """
    Build a variables section to inject into all prompts.
    Includes date/time, user info, and other contextual information.
    """
    now = datetime.now()
    current_date = now.strftime("%B %d, %Y")  # e.g., "January 25, 2026"
    current_time = now.strftime("%I:%M %p")  # e.g., "02:30 PM"
    current_datetime = now.strftime("%B %d, %Y at %I:%M %p")  # e.g., "January 25, 2026 at 02:30 PM"
    current_year = now.year
    current_month = now.strftime("%B")  # e.g., "January"
    current_day = now.day
    
    # Determine if Eastern Daylight Time (EDT) or Eastern Standard Time (EST)
    # Use time.tzname to get the actual timezone abbreviation
    # This works on Windows and Unix systems
    try:
        # Get local timezone info
        if time.daylight:
            # Check if DST is currently active
            local_time = time.localtime()
            is_dst = local_time.tm_isdst
            timezone_str = "Eastern Daylight Time (EDT)" if is_dst else "Eastern Standard Time (EST)"
        else:
            timezone_str = "Eastern Standard Time (EST)"
    except Exception:
        # Fallback: use simple month-based detection
        # DST typically: second Sunday in March to first Sunday in November
        is_dst = 3 <= now.month <= 10 or (now.month == 11 and now.day <= 7) or (now.month == 3 and now.day >= 8)
        timezone_str = "Eastern Daylight Time (EDT)" if is_dst else "Eastern Standard Time (EST)"
    
    # Calculate fiscal year (assuming July 1 - June 30, common for municipalities)
    fiscal_year_start = current_year if now.month >= 7 else current_year - 1
    fiscal_year_end = fiscal_year_start + 1
    fiscal_year = f"FY{fiscal_year_start}-{fiscal_year_end}"
    
    # Build variables section
    variables = f"""
---

**Context Variables**

- **Municipality Name**: City of Attleboro
- **Time Zone**: {timezone_str}
- **Current Date**: {current_date}
- **Current Time**: {current_time}
- **Current Date & Time**: {current_datetime}
- **Current Year**: {current_year}
- **Current Month**: {current_month}
- **Fiscal Year**: {fiscal_year} (July 1, {fiscal_year_start} - June 30, {fiscal_year_end})
"""
    
    if user_name and user_name != "unknown":
        variables += f"- **Analysis Performed By**: {user_name}"
        if username:
            variables += f" ({username})"
        variables += "\n"
    
    variables += "\nUse these variables as needed in your analysis. When referencing dates, use the current date/time provided above.\n"
    
    return variables


def _wrap_transient_content(items: list[dict]) -> str:
    """
    Wrap transient input items (uploads + pastes) as subject-of-analysis, distinct from the knowledge base.
    Each item: {id, name, content, type: 'file'|'paste'}.
    """
    if not items:
        return ""
    parts = [
        "<subject_of_analysis>",
        "The following are the subject of analysis (transient input). They are distinct from the knowledge base.",
        "",
    ]
    for it in items:
        name = (it.get("name") or "Untitled").replace('"', "&quot;").replace("\n", " ")
        content = (it.get("content") or "").strip()
        typ = it.get("type", "file")
        parts.append(f'<document title="{name}" type="{typ}">')
        parts.append(content)
        parts.append("</document>")
        parts.append("")
    parts.append("</subject_of_analysis>")
    return "\n".join(parts)


# -----------------------------------------------------------------------------
# Banner (full-width across entire screen; component iframe injects into parent)
# -----------------------------------------------------------------------------

_BANNER_H = 48
_BANNER_HTML = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body>
<script>
(function() {{
  var doc = parent.document;
  if (doc.getElementById("app-banner")) {{
    var f = window.frameElement;
    if (f) {{ f.style.setProperty("display", "none", "important"); f.style.setProperty("height", "0", "important"); }}
    return;
  }}
  var style = doc.createElement("style");
  style.textContent = "[data-testid=\\"stAppViewContainer\\"] {{ padding-top: {_BANNER_H}px !important; }} " +
    "main .block-container {{ padding-top: 0 !important; margin-top: 0 !important; }} " +
    "[data-testid=\\"stAppViewContainer\\"] > div:first-child {{ margin-top: 0 !important; padding-top: 0 !important; }} " +
    "[data-testid=\\"stHeader\\"] {{ margin-top: 0 !important; padding-top: 0 !important; height: auto !important; min-height: 0 !important; }} " +
    "[data-testid=\\"stHeader\\"] [data-testid=\\"stToolbar\\"] {{ display: flex !important; }} " +
    "[data-testid=\\"stHeader\\"] button[kind=\\"header\\"]:not(#app-banner-right button):not([data-testid=\\"stSidebarCollapseButton\\"]) {{ opacity: 0 !important; pointer-events: none !important; }} " +
    "[data-testid=\\"stSidebarCollapseButton\\"] {{ display: block !important; visibility: visible !important; position: relative !important; z-index: 1000004 !important; pointer-events: auto !important; cursor: pointer !important; }} " +
    "[data-testid=\\"stSidebarCollapseButton\\"]:hover {{ opacity: 0.8 !important; }} " +
    "@media (max-width: 768px) {{ " +
    "[data-testid=\\"stAppViewContainer\\"] .block-container h2 {{ font-size: 1.4rem !important; }} " +
    "}} " +
    "@media (min-width: 769px) {{ #app-menu-btn {{ display: none !important; }} }}";
  doc.head.appendChild(style);
  var bar = doc.createElement("div");
  bar.id = "app-banner";
  bar.style.cssText = "position:fixed;top:0;left:0;right:0;width:100%;height:{_BANNER_H}px;background:#1e3a5f;color:#fff;display:flex;align-items:center;justify-content:space-between;padding:0 1rem;gap:0.75rem;box-shadow:0 1px 4px rgba(0,0,0,0.15);font-family:inherit;z-index:1000002;";
  var leftSection = '<div style="display:flex;align-items:center;gap:0.75rem;">' +
    '<button id="app-menu-btn" aria-label="Menu" style="width:32px;height:32px;border-radius:8px;border:1px solid rgba(255,255,255,0.35);background:transparent;color:#fff;font-size:18px;line-height:1;cursor:pointer;">☰</button>' +
    '<span style="font-size:1.5rem;font-weight:600;white-space:nowrap;">🏛️ {APP_NAME}</span>' +
    '<span class="app-banner-subtitle" style="font-size:0.85rem;opacity:0.9;margin-left:0.5rem;">AI-assisted analysis for municipal council workflows</span>' +
    '</div>';
  var rightSection = '<div id="app-banner-right" style="display:flex;align-items:center;"></div>';
  bar.innerHTML = leftSection + rightSection;
  doc.body.insertBefore(bar, doc.body.firstChild);
  
  // Move Streamlit menu button to banner (NEVER move sidebar collapse button)
  setTimeout(function() {{
    // CRITICAL: Protect sidebar collapse button - ensure it stays in place and works
    var sidebarCollapseBtn = doc.querySelector('[data-testid="stSidebarCollapseButton"]');
    if (sidebarCollapseBtn) {{
      // Keep it in its original parent - don't move it!
      sidebarCollapseBtn.style.cssText = "display: block !important; visibility: visible !important; position: relative !important;";
      // Store original parent to prevent accidental moves
      if (!sidebarCollapseBtn.dataset.originalParent) {{
        sidebarCollapseBtn.dataset.originalParent = sidebarCollapseBtn.parentElement ? sidebarCollapseBtn.parentElement.tagName : 'unknown';
        sidebarCollapseBtn.dataset.protected = 'true';
      }}
    }}
    
    // Find Streamlit menu button - be VERY specific to avoid the collapse button
    var streamlitMenu = null;
    
    // Method 1: Look for the three-dot menu button (most common Streamlit menu)
    var menuButtons = doc.querySelectorAll('[data-testid="stHeader"] button, button[kind="header"]');
    for (var i = 0; i < menuButtons.length; i++) {{
      var btn = menuButtons[i];
      var testId = btn.getAttribute('data-testid') || '';
      var ariaLabel = btn.getAttribute('aria-label') || '';
      var btnText = btn.textContent || '';
      
      // ABSOLUTELY SKIP sidebar collapse button - check multiple ways
      if (testId === 'stSidebarCollapseButton' || 
          testId.includes('Sidebar') ||
          testId.includes('Collapse') ||
          ariaLabel.includes('sidebar') ||
          ariaLabel.includes('Sidebar') ||
          ariaLabel.includes('Collapse') ||
          btnText.trim() === '>>' ||
          btnText.trim() === '<<' ||
          btn.closest('[data-testid="stSidebar"]') ||
          btn.dataset.protected === 'true') {{
        continue;
      }}
      
      // Look for Streamlit menu indicators
      // The menu button usually has aria-label like "Main menu" or contains SVG icons
      if (ariaLabel.toLowerCase().includes('menu') ||
          ariaLabel.toLowerCase().includes('settings') ||
          ariaLabel.toLowerCase().includes('options') ||
          btn.querySelector('svg') ||  // Menu buttons often have SVG icons
          btn.getAttribute('kind') === 'header') {{
        streamlitMenu = btn;
        break;
      }}
    }}
    
    // Only move if we found a menu button AND it's definitely not the collapse button
    if (streamlitMenu && 
        streamlitMenu.getAttribute('data-testid') !== 'stSidebarCollapseButton' &&
        streamlitMenu.dataset.protected !== 'true' &&
        !streamlitMenu.closest('[data-testid="stSidebar"]')) {{
      var bannerRight = doc.getElementById("app-banner-right");
      if (bannerRight) {{
        streamlitMenu.style.cssText = "background:transparent !important;border:1px solid rgba(255,255,255,0.35) !important;color:#fff !important;padding:0.25rem 0.5rem !important;";
        bannerRight.appendChild(streamlitMenu);
      }}
    }}
  }}, 300);
  var subtitle = bar.querySelector(".app-banner-subtitle");
  var bannerStyle = doc.createElement("style");
  bannerStyle.textContent = "@media (max-width: 768px) {{ .app-banner-subtitle {{ display: none; }} }}";
  doc.head.appendChild(bannerStyle);
  var f = window.frameElement;
  if (f) {{
    f.style.setProperty("display", "none", "important");
    f.style.setProperty("height", "0", "important");
    var p = f.parentElement;
    if (p) {{ p.style.setProperty("margin", "0", "important"); p.style.setProperty("padding", "0", "important"); p.style.setProperty("min-height", "0", "important"); }}
  }}
}})();
</script>
</body>
</html>
"""
components.html(_BANNER_HTML, height=0)

_SIDEBAR_HTML = f"""
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body>
<script>
(function() {{
  var doc = parent.document;
  var styleId = "mobile-sidebar-style";
  if (!doc.getElementById(styleId)) {{
    var style = doc.createElement("style");
    style.id = styleId;
    style.textContent = `
@media (max-width: 768px) {{
  [data-testid="stSidebar"] {{
    position: fixed;
    top: 0;
    left: 0;
    height: 100vh;
    width: 80vw;
    max-width: 320px;
    transform: translateX(-100%);
    transition: transform 0.2s ease;
    z-index: 1000001;
    background: var(--background-color, #ffffff);
  }}
  body[data-sidebar-open="true"] [data-testid="stSidebar"] {{
    transform: translateX(0);
    box-shadow: 2px 0 12px rgba(0,0,0,0.25);
  }}
  .mobile-sidebar-backdrop {{
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.35);
    z-index: 1000000;
  }}
  body[data-sidebar-open="true"] .mobile-sidebar-backdrop {{
    display: block;
  }}
  [data-testid="stSidebarCollapseButton"] {{
    display: none !important;
  }}
  [data-testid="stSidebar"] button[aria-label="Close sidebar"] {{
    display: none !important;
  }}
}}
/* Remove horizontal bar above Navigation - only hide the very first hr if it's before Navigation */
[data-testid="stSidebar"] > div:first-child [data-testid="stMarkdownContainer"]:first-child hr:first-of-type {{
  display: none !important;
}}
/* Ensure all other hr elements in sidebar are visible - be very specific */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] hr {{
  display: block !important;
  visibility: visible !important;
  opacity: 1 !important;
  margin: 0.5rem 0 !important;
  border: none !important;
  border-top: 1px solid rgba(250, 250, 250, 0.2) !important;
  height: 1px !important;
}}
/* Make caption text darker */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
  color: #505050 !important;
}}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
  color: #505050 !important;
}}
/* Style section headers consistently - bold and darker */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {{
  color: #262626 !important;
  font-weight: 600 !important;
}}
/* Make H4 headings in sidebar larger */
[data-testid="stSidebar"] h4 {{
  font-size: 1.1rem !important;
  font-weight: 600 !important;
  color: #262626 !important;
  margin-top: 0.5rem !important;
  margin-bottom: 0.5rem !important;
}}
/* Vertically center button text */
[data-testid="stSidebar"] button {{
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  vertical-align: middle !important;
}}
[data-testid="stSidebar"] button > span,
[data-testid="stSidebar"] button > div,
[data-testid="stSidebar"] button > p,
[data-testid="stSidebar"] button > * {{
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  vertical-align: middle !important;
}}
`;
    doc.head.appendChild(style);
  }}

  if (!doc.body.hasAttribute("data-sidebar-open")) {{
    doc.body.setAttribute("data-sidebar-open", "false");
  }}

  var backdrop = doc.getElementById("mobile-sidebar-backdrop");
  if (!backdrop) {{
    backdrop = doc.createElement("div");
    backdrop.id = "mobile-sidebar-backdrop";
    backdrop.className = "mobile-sidebar-backdrop";
    doc.body.appendChild(backdrop);
  }}
  backdrop.onclick = function() {{ doc.body.setAttribute("data-sidebar-open", "false"); }};

  var menuBtn = doc.getElementById("app-menu-btn");
  if (menuBtn) {{
    menuBtn.onclick = function() {{
      var isOpen = doc.body.getAttribute("data-sidebar-open") === "true";
      doc.body.setAttribute("data-sidebar-open", isOpen ? "false" : "true");
    }};
  }}

  var sidebar = doc.querySelector('[data-testid="stSidebar"]');
  if (sidebar && !sidebar.hasAttribute("data-mobile-close-bound")) {{
    sidebar.setAttribute("data-mobile-close-bound", "true");
    sidebar.addEventListener("click", function(event) {{
      var isMobile = window.matchMedia && window.matchMedia("(max-width: 768px)").matches;
      if (!isMobile) return;
      var btn = event.target && event.target.closest("button");
      if (btn) {{
        doc.body.setAttribute("data-sidebar-open", "false");
      }}
    }});
  }}

  var f = window.frameElement;
  if (f) {{
    f.style.setProperty("display", "none", "important");
    f.style.setProperty("height", "0", "important");
  }}
}})();
</script>
</body>
</html>
"""
components.html(_SIDEBAR_HTML, height=0)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

# Navigation (no divider above)
st.sidebar.markdown("#### **Navigation**")
run_analysis_clicked = st.sidebar.button("▶️ Run Analysis", key="nav_run", use_container_width=True)
if run_analysis_clicked:
    st.session_state["current_page"] = "runner"
    st.rerun()

if is_admin:
    open_editor_clicked = st.sidebar.button("✏️ Prompt Editor", key="nav_editor", use_container_width=True)
    if open_editor_clicked:
        st.session_state["current_page"] = "edit_prompts"
        st.rerun()
    run_history_clicked = st.sidebar.button("📋 Run history", key="nav_run_history", use_container_width=True)
    if run_history_clicked:
        st.session_state["current_page"] = "run_history"
        st.rerun()
    transformer_playground_clicked = st.sidebar.button("🔧 Transformer playground", key="nav_playground", use_container_width=True)
    if transformer_playground_clicked:
        st.session_state["current_page"] = "transformer_playground"
        st.rerun()

current_page = st.session_state.get("current_page", "runner")
if current_page == "edit_prompts" and not is_admin:
    st.session_state["current_page"] = "runner"
    st.rerun()
if current_page == "run_history" and not is_admin:
    st.session_state["current_page"] = "runner"
    st.rerun()
if current_page == "transformer_playground" and not is_admin:
    st.session_state["current_page"] = "runner"
    st.rerun()

# About
st.sidebar.divider()
st.sidebar.markdown("#### **About**")
st.sidebar.markdown(
    f"**{APP_NAME}**  \n"
    "RAG-powered LLM analysis tooling with hybrid retrieval (BM25 + semantic) and Gemini context caching to empower municipal decision making."
)
app_version = get_git_version()
st.sidebar.markdown(f"App version: `{app_version}`")
st.sidebar.markdown(
    'Developed by **[Todd Kobus](https://www.facebook.com/kobusforattleboro)**'
)

# Knowledge base section
st.sidebar.divider()
st.sidebar.markdown("#### **Knowledge base**")

# Show folder info first
folder_info = get_cached_folder_info(folder_id) if folder_id else None
if folder_info:
    name = folder_info.get("name", "Drive folder")
    link = folder_info.get("link", f"https://drive.google.com/drive/folders/{folder_id}")
    st.sidebar.markdown(f"Context for the knowledge base dynamically stored here: [**{name}**]({link})")
else:
    st.sidebar.caption(f"Context for the knowledge base dynamically stored here: Folder `{folder_id[:20]}...`")

# RAG knowledge base loading (runs once at boot)
if not st.session_state.get("kb_loading_started") and folder_id:
    st.session_state["kb_loading_started"] = True
    logger.info("Starting RAG knowledge base load at boot")
    kb_status_placeholder = st.sidebar.empty()
    kb_status_placeholder.info("⏳ Loading knowledge base…")
    try:
        with st.spinner("Loading knowledge base…"):
            def _rag_progress(phase: str, *args):
                logger.debug(f"RAG progress: {phase} {args}")
            rag_state = get_cached_rag_state(folder_id, _progress_callback=_rag_progress)
            st.session_state["rag_state"] = rag_state
            n_libs = len(rag_state.get("libraries", []))
            logger.info(f"RAG loaded: Core + {n_libs} libraries")
            st.session_state["kb_loaded"] = True
            st.session_state["kb_load_error"] = None
        kb_status_placeholder.success("✓ Knowledge base loaded")
    except Exception as e:
        logger.error(f"Error loading RAG knowledge base at boot: {e}", exc_info=True)
        st.session_state["kb_load_error"] = str(e)
        st.session_state["kb_loaded"] = False
        st.session_state["rag_state"] = None
        kb_status_placeholder.error(f"⚠ Load error: {str(e)[:60]}…")

# KB status (loaded / error / loading) - show in knowledge base area
# Note: Success/error messages are shown via kb_status_placeholder above, so we only show status if still loading
if st.session_state.get("kb_load_error"):
    st.sidebar.caption(f"⚠ {st.session_state['kb_load_error'][:50]}…")
elif st.session_state.get("kb_loading_started") and not st.session_state.get("kb_loaded"):
    st.sidebar.caption("⏳ Loading knowledge base…")

if is_admin and st.sidebar.button("🔄 Refresh knowledge base", key="refresh_kb", use_container_width=True):
    logger.info("User clicked Refresh Knowledge Base")
    try:
        get_cached_rag_state.clear()
        get_cached_folder_info.clear()
        get_cached_file_list.clear()
        n = clear_disk_cache_for_folder(folder_id)
        if n:
            logger.info(f"Cleared {n} RAG index cache file(s)")
        # Save all caches to disk when clearing RAG state
        try:
            from brain import save_all_caches_to_disk
            save_all_caches_to_disk()
        except Exception:
            pass
        st.session_state["gemini_cache_name"] = None
        st.session_state["gemini_cache_model"] = None
        st.session_state["gemini_cache_folder_id"] = None
        st.session_state["run_cache_key"] = None
        st.session_state["kb_loading_started"] = False
        st.session_state["kb_loaded"] = False
        st.session_state["kb_load_error"] = None
        st.session_state["rag_state"] = None
        logger.info("RAG knowledge base cache cleared")
        msg = "Knowledge base and index cache cleared." if n else "Cache cleared. Reloading…"
        st.sidebar.success(msg)
        st.rerun()
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        st.sidebar.error(f"Error clearing cache: {e}")

# Model & pipeline
st.sidebar.divider()
st.sidebar.markdown("#### **Model & pipeline**")
current_model = get_effective_model()
current_planner_model = get_planner_model()
st.sidebar.caption(f"Model: `{current_model}`")
if current_planner_model and current_planner_model != current_model:
    st.sidebar.caption(f"Planner: `{current_planner_model}`")
max_ctx = model_max_context(current_model)
st.sidebar.caption(f"Context window: {max_ctx:,} tokens")

# Admin model selection
if is_admin:
    with st.sidebar.expander("⚙️ Model Selection (Admin)", expanded=False):
        try:
            available_models = list_available_models()
            if available_models:
                config = db.get_app_config()
                current_selected = config.selected_model if config else current_model
                
                # Find current model index, or default to 0
                try:
                    current_index = available_models.index(current_selected)
                except ValueError:
                    current_index = 0
                    if current_selected not in available_models:
                        available_models.insert(0, current_selected)
                        current_index = 0
                
                selected_model = st.selectbox(
                    "Select Gemini Model",
                    available_models,
                    index=current_index,
                    key="model_selector",
                    help="This model will be used for all users. Changing the model helps manage rate limits by switching to different API quotas."
                )
                
                if selected_model != current_selected:
                    if st.button("💾 Save Model Selection", key="save_model", use_container_width=True):
                        try:
                            db.update_selected_model(selected_model)
                            # Clear Gemini CachedContent cache when model changes (cache is model-specific)
                            if "gemini_cache_name" in st.session_state:
                                old_cache = st.session_state.get("gemini_cache_name")
                                st.session_state["gemini_cache_name"] = None
                                st.session_state["gemini_cache_model"] = None
                                logger.info(f"Cleared Gemini cache (was: {old_cache}) due to model change")
                            st.success(f"✅ Model updated to `{selected_model}`")
                            logger.info(f"Admin updated model to: {selected_model}")
                            # Refresh to apply changes immediately
                            time.sleep(0.5)  # Brief delay to show success message
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error saving model: {e}")
                            logger.error(f"Error saving model: {e}", exc_info=True)
                
                # Planner model selection (optional override)
                st.markdown("---")
                st.caption("**Planner Model** (optional override)")
                planner_options = ["— Use same as main model —"] + available_models
                current_planner_selected = config.planner_model if config and config.planner_model else None
                
                if current_planner_selected:
                    try:
                        planner_index = planner_options.index(current_planner_selected)
                    except ValueError:
                        planner_index = 0
                else:
                    planner_index = 0
                
                selected_planner = st.selectbox(
                    "Planner Model",
                    planner_options,
                    index=planner_index,
                    key="planner_model_selector",
                    help="Optional: Use a different model for retrieval planning. Helps spread quota across models."
                )
                
                planner_to_save = None if selected_planner == "— Use same as main model —" else selected_planner
                if planner_to_save != current_planner_selected:
                    if st.button("💾 Save Planner Model", key="save_planner_model", use_container_width=True):
                        try:
                            db.update_planner_model(planner_to_save)
                            # Note: Planner model change doesn't require cache clearing (only affects retrieval planner, not main agent)
                            st.success(f"✅ Planner model updated to `{planner_to_save or 'same as main'}`")
                            logger.info(f"Admin updated planner model to: {planner_to_save}")
                            # Refresh to apply changes immediately
                            time.sleep(0.5)  # Brief delay to show success message
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error saving planner model: {e}")
                            logger.error(f"Error saving planner model: {e}", exc_info=True)
            else:
                st.warning("⚠️ Could not fetch available models. Check API key configuration.")
        except Exception as e:
            st.error(f"❌ Error loading model selection: {e}")
            logger.error(f"Error in model selection UI: {e}", exc_info=True)

# Token optimization settings
st.sidebar.divider()
st.sidebar.markdown("#### **Token Optimization**")
current_pace = GEMINI_PACE_DELAY_SECONDS
st.sidebar.caption(f"Pace delay: `{current_pace}s` (env: `GEMINI_PACE_DELAY_SECONDS`)")
current_pipeline_delay = PIPELINE_STEP_DELAY_SECONDS
st.sidebar.caption(f"Pipeline step delay: `{current_pipeline_delay}s` (env: `PIPELINE_STEP_DELAY_SECONDS`)")
current_legal_delay = LEGAL_EXPERT_DELAY_SECONDS
st.sidebar.caption(f"Legal expert delay: `{current_legal_delay}s` (env: `LEGAL_EXPERT_DELAY_SECONDS`)")
if USE_QUERY_EXPANSION:
    st.sidebar.caption("⚠️ Query expansion: **Enabled** (uses extra LLM calls)")
    st.sidebar.caption("💡 Disable in `rag_loader.py` to reduce token usage")
else:
    st.sidebar.caption("✅ Query expansion: **Disabled** (saves tokens)")

# Debug info section
st.sidebar.divider()
st.sidebar.markdown("#### **System info**")
python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
st.sidebar.caption(f"Python: `{python_version}`")
st.sidebar.caption(f"Platform: `{platform.system()} {platform.release()}`")
st.sidebar.caption(f"Streamlit: `{st.__version__}`")
try:
    try:
        from google import genai
        sdk_version = getattr(genai, "__version__", "unknown")
    except ImportError:
        import google.generativeai as genai
        sdk_version = getattr(genai, "__version__", "unknown")
    if sdk_version != "unknown":
        st.sidebar.caption(f"Gemini SDK: `{sdk_version}`")
except (ImportError, AttributeError):
    pass
if hasattr(st.session_state, "rag_state") and st.session_state.get("rag_state"):
    rag_state = st.session_state["rag_state"]
    n_libs = len(rag_state.get("libraries", []))
    st.sidebar.caption(f"Libraries loaded: `{n_libs}`")
if folder_id:
    st.sidebar.caption(f"KB folder: `{folder_id[:12]}...`")
st.sidebar.markdown("---")
authenticator.logout(location="sidebar")

# -----------------------------------------------------------------------------
# Prompt Editor page
# -----------------------------------------------------------------------------

if current_page == "edit_prompts":
    st.markdown("### ✏️ Prompt Editor")
    if st.button("← Back to Run Analysis", key="back_to_runner"):
        st.session_state["current_page"] = "runner"
        st.rerun()
    st.caption("Manage prompt templates. Admin only.")
    
    # Database import/export section
    with st.expander("💾 Database Import/Export", expanded=False):
        db_info = db.get_database_info()
        if db_info.get("exists"):
            size_mb = db_info.get("size_bytes", 0) / (1024 * 1024)
            st.caption(f"Database: {db_info.get('path', 'unknown')} ({size_mb:.2f} MB, {db_info.get('prompt_count', 0)} prompts)")
        else:
            st.warning("Database file not found")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Export Database")
            st.caption("Download a backup of the current database file.")
            if st.button("📥 Export Database", key="export_db", use_container_width=True):
                try:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    export_filename = f"councilflow_db_{timestamp}.db"
                    export_path = os.path.join(tempfile.gettempdir(), export_filename)
                    exported_path = db.export_database(export_path)
                    
                    # Read the file and provide download
                    with open(exported_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Database Backup",
                            data=f.read(),
                            file_name=export_filename,
                            mime="application/x-sqlite3",
                            key="download_db"
                        )
                    st.success(f"✅ Database exported successfully: {export_filename}")
                    logger.info(f"Admin exported database to: {exported_path}")
                except Exception as e:
                    st.error(f"❌ Error exporting database: {e}")
                    logger.error(f"Error exporting database: {e}", exc_info=True)
        
        with col2:
            st.markdown("#### Import Database")
            st.caption("⚠️ **Warning**: This will replace the current database. A backup will be created automatically.")
            uploaded_file = st.file_uploader(
                "Choose database file",
                type=["db", "sqlite", "sqlite3"],
                key="import_db_file",
                help="Select a previously exported CouncilFlow database file (.db)"
            )
            
            if uploaded_file is not None:
                st.info(f"📄 File selected: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
                
                if st.button("📤 Import Database", key="import_db", use_container_width=True, type="primary"):
                    try:
                        # Save uploaded file to temp location
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Import the database
                        db.import_database(tmp_path, backup_existing=True)
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                        st.success("✅ Database imported successfully! The page will refresh.")
                        st.info("🔄 Please refresh the page to see the imported data.")
                        logger.info(f"Admin imported database from: {uploaded_file.name}")
                        
                        # Small delay then refresh
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error importing database: {e}")
                        logger.error(f"Error importing database: {e}", exc_info=True)
                        # Clean up temp file if it exists
                        try:
                            if 'tmp_path' in locals():
                                os.unlink(tmp_path)
                        except Exception:
                            pass
    
    st.markdown("---")
    # Load prompts. JSON schemas are code-defined only (output_schemas registry); no DB schema table.
    crud_prompts = sorted(db.get_all_prompts(), key=lambda p: p.name.casefold())

    crud_options = ["+ Add new"] + [p.name for p in crud_prompts]
    crud_select = st.selectbox("Select prompt", crud_options, key="crud_select")
    existing = next((p for p in crud_prompts if p.name == crud_select), None)

    # Show version clearly (default 1 for new or legacy prompts)
    if existing:
        version = getattr(existing, "current_version", None) or 1
        st.markdown(f"**Current version:** {version}")
    else:
        st.markdown("**Current version:** 1 *(new prompt)*")

    # Restored-version banner
    if existing and st.session_state.get("crud_restored_version"):
        st.info(f"Restored to version **{st.session_state['crud_restored_version']}**. You can edit and **Save** to create a new version.")

    # Version history (existing prompt only)
    if existing:
        with st.expander("Version history", expanded=False):
            versions = db.list_prompt_versions(existing.id)
            if not versions:
                st.caption("No version history yet (save this prompt to create the first version).")
            else:
                for pv in versions:
                    saved_str = pv.saved_at.strftime("%Y-%m-%d %H:%M") if getattr(pv, "saved_at", None) else "—"
                    col1, col2, col3 = st.columns([1, 2, 2])
                    with col1:
                        st.markdown(f"**v{pv.version}**")
                    with col2:
                        st.caption(f"Saved: {saved_str}")
                    with col3:
                        if st.button("Restore", key=f"restore_v{pv.version}_{existing.id}"):
                            st.session_state["crud_name"] = pv.name
                            st.session_state["crud_template"] = pv.template_text
                            st.session_state["crud_verifier_id"] = pv.verifier_id
                            st.session_state["crud_follow_on_only"] = getattr(pv, "follow_on_only", False)
                            st.session_state["crud_legal_expert_prompt_id"] = getattr(pv, "legal_expert_prompt_id", None)
                            st.session_state["crud_input_schema_key"] = getattr(pv, "input_schema_key", None)
                            st.session_state["crud_output_schema_key"] = getattr(pv, "output_schema_key", None)
                            st.session_state["crud_use_qa_agent"] = getattr(pv, "use_qa_agent", False)
                            st.session_state["crud_workflow_id"] = getattr(pv, "workflow_id", None)
                            st.session_state["crud_output_transformer_key"] = getattr(pv, "output_transformer_key", None)
                            st.session_state["crud_restored_version"] = pv.version
                            # Sync widget keys so selectboxes show restored selection
                            _fk = f"followon_select_{existing.id}"
                            if pv.verifier_id:
                                _fp = db.get_prompt_by_id(pv.verifier_id)
                                st.session_state[_fk] = f"{_fp.id}: {_fp.name}" if _fp else "— None —"
                            else:
                                st.session_state[_fk] = "— None —"
                            try:
                                crud_workflows = db.get_all_workflows()
                                _wk = f"workflow_select_{existing.id}"
                                if pv.workflow_id and crud_workflows:
                                    _wf = next((w for w in crud_workflows if w.id == pv.workflow_id), None)
                                    st.session_state[_wk] = f"{_wf.id}: {_wf.name}" if _wf else "— Default —"
                                else:
                                    st.session_state[_wk] = "— Default —"
                            except Exception:
                                pass
                            _lk = f"legal_expert_select_{existing.id}"
                            if getattr(pv, "legal_expert_prompt_id", None):
                                _lp = db.get_prompt_by_id(pv.legal_expert_prompt_id)
                                st.session_state[_lk] = f"{_lp.id}: {_lp.name}" if _lp else "— None —"
                            else:
                                st.session_state[_lk] = "— None —"
                            _ik = f"input_schema_select_{existing.id}"
                            _ok = f"output_schema_select_{existing.id}"
                            st.session_state[_ik] = f"code::{pv.input_schema_key}" if getattr(pv, "input_schema_key", None) else "— None —"
                            st.session_state[_ok] = f"code::{pv.output_schema_key}" if getattr(pv, "output_schema_key", None) else "— None —"
                            st.rerun()
    
    # Sync form values only when selection changes (avoid clobbering submitted values)
    selected_id = existing.id if existing else None
    if st.session_state.get("crud_last_selected") != selected_id:
        st.session_state["crud_last_selected"] = selected_id
        if existing:
            st.session_state["crud_name"] = existing.name
            st.session_state["crud_template"] = existing.template_text
            st.session_state["crud_verifier_id"] = existing.verifier_id
            st.session_state["crud_follow_on_only"] = getattr(existing, "follow_on_only", False)
            st.session_state["crud_legal_expert_prompt_id"] = getattr(existing, "legal_expert_prompt_id", None)
            st.session_state["crud_input_schema_key"] = getattr(existing, "input_schema_key", None)
            st.session_state["crud_output_schema_key"] = getattr(existing, "output_schema_key", None)
            st.session_state["crud_use_qa_agent"] = getattr(existing, "use_qa_agent", False)
            st.session_state["crud_workflow_id"] = getattr(existing, "workflow_id", None)
            st.session_state["crud_output_transformer_key"] = getattr(existing, "output_transformer_key", None)
        else:
            st.session_state["crud_name"] = ""
            st.session_state["crud_template"] = ""
            st.session_state["crud_verifier_id"] = None
            st.session_state["crud_follow_on_only"] = False
            st.session_state["crud_legal_expert_prompt_id"] = None
            st.session_state["crud_input_schema_key"] = None
            st.session_state["crud_output_schema_key"] = None
            st.session_state["crud_use_qa_agent"] = False
            st.session_state["crud_workflow_id"] = None
            st.session_state["crud_output_transformer_key"] = None

    with st.form("prompt_form", clear_on_submit=False):
        name_value = st.session_state.get("crud_name", "")
        template_value = st.session_state.get("crud_template", "")
        verifier_id_value = st.session_state.get("crud_verifier_id", None)
        follow_on_only_value = st.session_state.get("crud_follow_on_only", False)
        legal_expert_prompt_id_value = st.session_state.get("crud_legal_expert_prompt_id", None)
        use_qa_agent_value = st.session_state.get("crud_use_qa_agent", False)
        workflow_id_value = st.session_state.get("crud_workflow_id", None)
        
        name = st.text_input("Name", value=name_value, placeholder="e.g. MC Analysis, Constituent Reply")
        template_text = st.text_area("Template", value=template_value, height=200, placeholder="Instructions for the AI. Use {{ content }} for the user's input.")
        st.caption("Output is always Markdown.")
        with st.expander("💡 Tip: Available Variables", expanded=False):
            st.markdown("""
            **Automatic Variables:**
            
            The following context variables are automatically injected into all prompts:
            - **Municipality Name**: City of Attleboro
            - **Time Zone**: Eastern Standard Time (EST) or Eastern Daylight Time (EDT), depending on the season
            - **Current Date**: Full date (e.g., "January 25, 2026")
            - **Current Time**: Time of day (e.g., "02:30 PM")
            - **Current Date & Time**: Combined (e.g., "January 25, 2026 at 02:30 PM")
            - **Current Year**: Year (e.g., 2026)
            - **Current Month**: Month name (e.g., "January")
            - **Fiscal Year**: Calculated fiscal year (e.g., "FY2025-2026" for July 1, 2025 - June 30, 2026)
            - **Analysis Performed By**: Name of the user running the analysis
            
            These are provided in a "Context Variables" section at the start of every prompt. You can reference them in your analysis.
            """)
        
        # Follow-on prompt selector (chainable)
        st.markdown("**Follow-on prompt (optional)**")
        st.caption("Run another prompt after this one. It receives this prompt's output as {{ previous_output }}. Can be chained.")
        _followon_candidates = [p for p in crud_prompts if p.id != (existing.id if existing else None)]
        followon_options = ["— None —"] + [f"{p.id}: {p.name}" for p in _followon_candidates]
        current_followon_str = None
        if verifier_id_value:
            followon_p = db.get_prompt_by_id(verifier_id_value)
            if followon_p:
                current_followon_str = f"{followon_p.id}: {followon_p.name}"
        # Initialize or sync widget only when prompt selection changes
        followon_key = f"followon_select_{selected_id}"
        if followon_key not in st.session_state or st.session_state.get("crud_last_selected") != selected_id:
            st.session_state[followon_key] = current_followon_str or "— None —"
        followon_index = 0
        current_selection = st.session_state.get(followon_key, current_followon_str or "— None —")
        if current_selection and current_selection in followon_options:
            followon_index = followon_options.index(current_selection)
        selected_followon_str = st.selectbox("Follow-on prompt", followon_options, index=followon_index, key=followon_key)
        verifier_id = None
        if selected_followon_str and selected_followon_str != "— None —":
            try:
                verifier_id = int(selected_followon_str.split(":")[0])
                logger.debug(f"Selected follow-on prompt ID: {verifier_id}")
            except (ValueError, IndexError):
                logger.warning(f"Could not parse follow-on ID from: {selected_followon_str}")
                verifier_id = None
        
        follow_on_only = st.checkbox(
            "Follow-on only (exclude from Run Analysis; can only be used as a follow-on)",
            value=follow_on_only_value,
            key="crud_follow_on_only",
        )
        st.caption("If checked, this prompt will not appear in the Analysis type dropdown.")
        
        # Phase 5: Workflow selector (admin only; not exposed on Run Analysis). N/A for follow-on only.
        st.markdown("**Workflow**")
        if follow_on_only:
            st.caption("N/A — Follow-on only prompts are not used at the start of a workflow.")
            workflow_id = None
        else:
            try:
                crud_workflows = db.get_all_workflows()
            except Exception as e:
                logger.error(f"Error loading workflows: {e}", exc_info=True)
                crud_workflows = []
            workflow_options = ["— Default —"] + [f"{w.id}: {w.name}" for w in crud_workflows]
            workflow_key = f"workflow_select_{selected_id}"
            current_workflow_str = "— Default —"
            if workflow_id_value and crud_workflows:
                wf = next((w for w in crud_workflows if w.id == workflow_id_value), None)
                if wf:
                    current_workflow_str = f"{wf.id}: {wf.name}"
            if workflow_key not in st.session_state or st.session_state.get("crud_last_selected") != selected_id:
                st.session_state[workflow_key] = current_workflow_str
            idx = 0
            if st.session_state.get(workflow_key) in workflow_options:
                idx = workflow_options.index(st.session_state[workflow_key])
            selected_workflow_str = st.selectbox(
                "Workflow (which graph runs when this prompt is selected at run start)",
                options=workflow_options,
                index=idx,
                key=workflow_key,
                label_visibility="collapsed",
            )
            if selected_workflow_str and selected_workflow_str != "— Default —":
                try:
                    workflow_id = int(selected_workflow_str.split(":")[0])
                except (ValueError, IndexError):
                    workflow_id = None
            else:
                workflow_id = crud_workflows[0].id if crud_workflows else None  # Default = first workflow
            st.caption("Not shown to users on Run Analysis; determines which workflow graph is used.")
        
        # Phase 4: QA agent (runs after follow-on chain to review and polish final output)
        use_qa_agent = st.checkbox(
            "Use QA agent for this task",
            value=use_qa_agent_value,
            key="crud_use_qa_agent",
            help="After the main analysis, legal review, and follow-on chain, run a QA step to review and polish the final output.",
        )
        st.caption("When enabled, a QA agent reviews the full chain output and produces a polished final document.")
        
        # Legal expert prompt selector
        st.markdown("**Legal expert prompt (optional)**")
        st.caption("If this prompt detects legal questions in its output, it will automatically consult the selected legal expert prompt. The legal expert will perform a separate knowledge base search and integrate its findings into the original output.")
        _legal_expert_candidates = [p for p in crud_prompts if p.id != (existing.id if existing else None)]
        legal_expert_options = ["— None —"] + [f"{p.id}: {p.name}" for p in _legal_expert_candidates]
        current_legal_expert_str = None
        if legal_expert_prompt_id_value:
            legal_expert_p = db.get_prompt_by_id(legal_expert_prompt_id_value)
            if legal_expert_p:
                current_legal_expert_str = f"{legal_expert_p.id}: {legal_expert_p.name}"
        legal_expert_key = f"legal_expert_select_{selected_id}"
        if legal_expert_key not in st.session_state or st.session_state.get("crud_last_selected") != selected_id:
            st.session_state[legal_expert_key] = current_legal_expert_str or "— None —"
        legal_expert_index = 0
        current_legal_expert_selection = st.session_state.get(legal_expert_key, current_legal_expert_str or "— None —")
        if current_legal_expert_selection and current_legal_expert_selection in legal_expert_options:
            legal_expert_index = legal_expert_options.index(current_legal_expert_selection)
        selected_legal_expert_str = st.selectbox("Legal expert prompt", legal_expert_options, index=legal_expert_index, key=legal_expert_key)
        legal_expert_prompt_id = None
        if selected_legal_expert_str and selected_legal_expert_str != "— None —":
            try:
                legal_expert_prompt_id = int(selected_legal_expert_str.split(":")[0])
                logger.debug(f"Selected legal expert prompt ID: {legal_expert_prompt_id}")
            except (ValueError, IndexError):
                logger.warning(f"Could not parse legal expert ID from: {selected_legal_expert_str}")
                legal_expert_prompt_id = None

        # JSON Schema associations (sidecars)
        st.markdown("**JSON Schemas (optional)**")
        st.caption(
            "Attach reusable JSON Schemas as sidecars. The schemas are sent to Gemini "
            "along with the prompt and can also be referenced in the template via "
            "`{{ input_schema_json }}` and `{{ output_schema_json }}`."
        )

        # Code-defined schemas only (from output_schemas registry); no DB schema table
        code_schema_options = ["— None —"]
        try:
            import output_schemas as _osc
            _osc.ensure_registry_loaded()
            for _rk in _osc.get_registry_schema_keys():
                code_schema_options.append(f"code::{_rk}")
        except Exception:
            pass

        # Input schema selector (code-defined only)
        input_schema_key_value = st.session_state.get("crud_input_schema_key") or None
        current_input_schema_str = f"code::{input_schema_key_value}" if input_schema_key_value else "— None —"
        in_schema_widget_key = f"input_schema_select_{selected_id}"
        if in_schema_widget_key not in st.session_state or st.session_state.get("crud_last_selected") != selected_id:
            st.session_state[in_schema_widget_key] = current_input_schema_str
        input_schema_index = 0
        if current_input_schema_str in code_schema_options:
            input_schema_index = code_schema_options.index(current_input_schema_str)
        selected_input_schema_str = st.selectbox(
            "Input JSON schema", code_schema_options, index=input_schema_index, key=in_schema_widget_key,
            help="Code-defined schemas from output_schemas registry (e.g. mayors_communication).",
        )
        input_schema_key = (selected_input_schema_str[6:] if selected_input_schema_str.startswith("code::") else None) if selected_input_schema_str and selected_input_schema_str != "— None —" else None

        # Output schema selector (code-defined only)
        output_schema_key_value = st.session_state.get("crud_output_schema_key") or None
        current_output_schema_str = f"code::{output_schema_key_value}" if output_schema_key_value else "— None —"
        out_schema_widget_key = f"output_schema_select_{selected_id}"
        if out_schema_widget_key not in st.session_state or st.session_state.get("crud_last_selected") != selected_id:
            st.session_state[out_schema_widget_key] = current_output_schema_str
        output_schema_index = 0
        if current_output_schema_str in code_schema_options:
            output_schema_index = code_schema_options.index(current_output_schema_str)
        selected_output_schema_str = st.selectbox(
            "Output JSON schema", code_schema_options, index=output_schema_index, key=out_schema_widget_key,
            help="Code-defined schemas from output_schemas registry (e.g. mayors_communication).",
        )
        output_schema_key = (selected_output_schema_str[6:] if selected_output_schema_str.startswith("code::") else None) if selected_output_schema_str and selected_output_schema_str != "— None —" else None

        # Default transformer (when output schema is selected): run-time transformer for JSON -> Markdown
        output_transformer_key = None
        if output_schema_key:
            transformer_opts = _get_json_view_options(schema_key=output_schema_key)
            # Only transformer keys, not Saved/Raw
            transformer_choices = [(val, lbl) for val, lbl in transformer_opts if val not in ("saved", "raw")]
            if transformer_choices:
                ot_key = f"output_transformer_select_{selected_id}"
                current_ot = st.session_state.get("crud_output_transformer_key")
                labels = ["— None —"] + [lbl for _, lbl in transformer_choices]
                keys = [None] + [val for val, _ in transformer_choices]
                idx = next((i for i, k in enumerate(keys) if k == current_ot), 0)
                st.caption("Optional: default transformer to apply at run time (JSON → Markdown).")
                sel_label = st.selectbox("Default transformer for this prompt", options=labels, index=idx, key=ot_key)
                output_transformer_key = keys[labels.index(sel_label)] if sel_label in labels else None
        
        submitted = st.form_submit_button("Save")
        if submitted and name and template_text:
            logger.info(
                "Saving prompt: %s (id: %s, verifier_id: %s, follow_on_only: %s, "
                "legal_expert: %s, input_schema_key: %s, output_schema_key: %s)",
                name,
                existing.id if existing else "new",
                verifier_id,
                follow_on_only,
                legal_expert_prompt_id,
                input_schema_key,
                output_schema_key,
            )
            try:
                db.save_prompt(
                    name,
                    template_text,
                    "markdown",
                    verifier_id=verifier_id,
                    follow_on_only=follow_on_only,
                    legal_expert_prompt_id=legal_expert_prompt_id,
                    input_schema_key=input_schema_key,
                    output_schema_key=output_schema_key,
                    use_qa_agent=use_qa_agent,
                    workflow_id=workflow_id if not follow_on_only else None,
                    output_transformer_key=output_transformer_key,
                    id=existing.id if existing else None,
                )
                logger.info("Prompt saved successfully")
                st.session_state["crud_restored_version"] = None
                st.success("Saved.")
                st.rerun()
            except Exception as e:
                logger.error(f"Error saving prompt: {e}", exc_info=True)
                st.error(f"Error saving: {e}")

    # Delete prompt (existing only)
    if existing:
        st.markdown("---")
        with st.expander("🗑️ Delete this prompt", expanded=False):
            st.caption("This cannot be undone. Any prompt that used this as a follow-on will have that link cleared.")
            confirm_delete = st.checkbox("I want to delete this prompt", key="confirm_delete_prompt")
            if st.button("Delete prompt", key="delete_prompt_btn", disabled=not confirm_delete, type="secondary"):
                try:
                    db.delete_prompt(existing.id)
                    if "crud_last_selected" in st.session_state:
                        del st.session_state["crud_last_selected"]
                    st.session_state["crud_select"] = "+ Add new"
                    st.success(f"Deleted \"{existing.name}\".")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error deleting prompt: {e}", exc_info=True)
                    st.error(f"Error deleting: {e}")

    st.markdown("---")

# -----------------------------------------------------------------------------
# Run history page (admin only)
# -----------------------------------------------------------------------------

elif current_page == "run_history":
    st.markdown("### 📋 Run history")
    if st.button("← Back to Run Analysis", key="back_from_history"):
        st.session_state["current_page"] = "runner"
        if "run_history_view_id" in st.session_state:
            del st.session_state["run_history_view_id"]
        st.rerun()
    st.caption("Recent analysis runs. Run data is stored locally and is not exported or imported. Times are shown in your local time.")

    view_id = st.session_state.get("run_history_view_id")
    if view_id is not None:
        run_detail = runs_db.get_analysis_run_by_id(view_id)
        if run_detail:
            stored_tz = getattr(run_detail, "stored_timezone", None) or "UTC"
            started_str = _format_run_datetime(run_detail.started_at, stored_tz)
            completed_str = _format_run_datetime(run_detail.completed_at, stored_tz)
            st.subheader(f"Run #{run_detail.id}: {run_detail.task_name}")
            st.caption(
                f"By **{run_detail.username}** · **{run_detail.status}**"
                + (f" · Prompt version: **{run_detail.prompt_version}**" if getattr(run_detail, "prompt_version", None) is not None else "")
            )
            st.caption(f"**Started:** {started_str} · **Completed:** {completed_str}")
            st.caption(f"Stored in **{stored_tz}**, shown in local time.")
            st.markdown("---")
            # Output / Pre-QA / Post-QA when QA was used
            pre_qa = getattr(run_detail, "pre_qa_output", None)
            qa_out = getattr(run_detail, "qa_output", None)
            if pre_qa or qa_out:
                tab_labels = ["Output"]
                if pre_qa:
                    tab_labels.append("Pre-QA")
                if qa_out:
                    tab_labels.append("Post-QA")
                tabs = st.tabs(tab_labels)
                with tabs[0]:
                    _markdown_with_copy(run_detail.output_text or "(no output)", f"run_{run_detail.id}_output")
                idx = 1
                if pre_qa:
                    with tabs[idx]:
                        _markdown_with_copy(pre_qa, f"run_{run_detail.id}_pre_qa")
                    idx += 1
                if qa_out:
                    with tabs[idx]:
                        _markdown_with_copy(qa_out, f"run_{run_detail.id}_post_qa")
            else:
                st.markdown("#### Output")
                out_json = getattr(run_detail, "output_json", None)
                if out_json and run_detail.prompt_template_id:
                    try:
                        prompt = db.get_prompt_by_id(run_detail.prompt_template_id)
                        schema_key = getattr(prompt, "output_schema_key", None) if prompt else None
                    except Exception:
                        schema_key = None
                    if schema_key:
                        view_opts = _get_json_view_options(schema_key=schema_key)
                        view_labels = [lbl for _, lbl in view_opts]
                        view_keys = [val for val, _ in view_opts]
                        sh_key = f"run_detail_view_{run_detail.id}"
                        idx = view_keys.index(st.session_state.get(sh_key, "saved")) if st.session_state.get(sh_key, "saved") in view_keys else 0
                        view_choice = st.selectbox("View as", options=view_labels, index=idx, key=sh_key + "_select")
                        st.session_state[sh_key] = view_keys[view_labels.index(view_choice)]
                        display_str = _render_json_view(out_json, run_detail.output_text or "", st.session_state[sh_key])
                        _markdown_with_copy(display_str, f"run_{run_detail.id}_output")
                    else:
                        _markdown_with_copy(run_detail.output_text or "(no output)", f"run_{run_detail.id}_output")
                else:
                    _markdown_with_copy(run_detail.output_text or "(no output)", f"run_{run_detail.id}_output")
            if run_detail.chain_steps:
                try:
                    steps = json.loads(run_detail.chain_steps) if isinstance(run_detail.chain_steps, str) else run_detail.chain_steps
                    if steps:
                        with st.expander("Chain steps", expanded=False):
                            for s in steps:
                                name = s.get("step_name", "?") if isinstance(s, dict) else "?"
                                out = s.get("output", "") if isinstance(s, dict) else str(s)
                                st.markdown(f"**{name}**")
                                st.text(out[:3000] + ("…" if len(out) > 3000 else ""))
                except Exception:
                    st.text(run_detail.chain_steps[:2000])
            if run_detail.error_message:
                st.markdown("---")
                st.markdown("#### Error")
                st.error(run_detail.error_message)
            events = runs_db.list_run_events(run_detail.id)
            if events:
                with st.expander("Event log", expanded=False):
                    for ev in events:
                        ts = ev.created_at
                        ts_str = _format_run_datetime(ts, getattr(run_detail, "stored_timezone", None) or "UTC") if ts else "—"
                        st.caption(f"{ts_str} · **{ev.step_name}** · {ev.event_type}" + (f" · {ev.payload}" if ev.payload else ""))
            if st.button("← Back to list", key="back_to_list"):
                del st.session_state["run_history_view_id"]
                st.rerun()
        else:
            st.warning(f"Run #{view_id} not found.")
            if "run_history_view_id" in st.session_state:
                del st.session_state["run_history_view_id"]
            st.rerun()
    else:
        # Run history filters (persist in session_state)
        if "run_history_task_filter" not in st.session_state:
            st.session_state["run_history_task_filter"] = "All"
        if "run_history_status_filter" not in st.session_state:
            st.session_state["run_history_status_filter"] = "All"
        if "run_history_version_filter" not in st.session_state:
            st.session_state["run_history_version_filter"] = 0
        if "run_history_filter_by_date" not in st.session_state:
            st.session_state["run_history_filter_by_date"] = False
        if "run_history_date_from" not in st.session_state:
            st.session_state["run_history_date_from"] = datetime.now().date() - timedelta(days=30)
        if "run_history_date_to" not in st.session_state:
            st.session_state["run_history_date_to"] = datetime.now().date()

        try:
            runnable_prompts = [p for p in db.get_all_prompts() if not getattr(p, "follow_on_only", False)]
            task_filter_options = ["All"] + sorted([p.name for p in runnable_prompts], key=str.casefold)
        except Exception:
            task_filter_options = ["All"]
        status_filter_options = ["All", "Completed", "Failed", "Running"]

        with st.expander("Filters", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                task_filter = st.selectbox(
                    "Task",
                    options=task_filter_options,
                    index=task_filter_options.index(st.session_state["run_history_task_filter"])
                    if st.session_state["run_history_task_filter"] in task_filter_options else 0,
                    key="run_history_task_filter_select",
                )
                st.session_state["run_history_task_filter"] = task_filter
            with c2:
                status_filter = st.selectbox(
                    "Status",
                    options=status_filter_options,
                    index=status_filter_options.index(st.session_state["run_history_status_filter"])
                    if st.session_state["run_history_status_filter"] in status_filter_options else 0,
                    key="run_history_status_filter_select",
                )
                st.session_state["run_history_status_filter"] = status_filter
            with c3:
                version_filter = st.number_input(
                    "Prompt version (0 = any)",
                    min_value=0,
                    value=st.session_state["run_history_version_filter"],
                    key="run_history_version_filter_input",
                )
                st.session_state["run_history_version_filter"] = int(version_filter) if version_filter is not None else 0
            filter_by_date = st.checkbox(
                "Filter by date range",
                value=st.session_state["run_history_filter_by_date"],
                key="run_history_filter_by_date_cb",
            )
            st.session_state["run_history_filter_by_date"] = filter_by_date
            if filter_by_date:
                d1, d2 = st.columns(2)
                with d1:
                    date_from = st.date_input(
                        "From",
                        value=st.session_state["run_history_date_from"],
                        key="run_history_date_from_input",
                    )
                    st.session_state["run_history_date_from"] = date_from
                with d2:
                    date_to = st.date_input(
                        "To",
                        value=st.session_state["run_history_date_to"],
                        key="run_history_date_to_input",
                    )
                    st.session_state["run_history_date_to"] = date_to

        task_name_filter = None if st.session_state["run_history_task_filter"] == "All" else st.session_state["run_history_task_filter"]
        status_filter_val = None if st.session_state["run_history_status_filter"] == "All" else st.session_state["run_history_status_filter"]
        prompt_version_filter = None if st.session_state["run_history_version_filter"] == 0 else st.session_state["run_history_version_filter"]
        date_from_dt = None
        date_to_dt = None
        if st.session_state["run_history_filter_by_date"]:
            date_from_dt = datetime.combine(st.session_state["run_history_date_from"], datetime.min.time())
            date_to_dt = datetime.combine(st.session_state["run_history_date_to"], datetime.max.time())

        runs_list = runs_db.list_analysis_runs(
            20,
            task_name_filter=task_name_filter,
            status_filter=status_filter_val,
            prompt_version_filter=prompt_version_filter,
            date_from=date_from_dt,
            date_to=date_to_dt,
        )
        if not runs_list:
            st.info("No runs yet. Run an analysis from the Run Analysis page.")
        else:
            for r in runs_list:
                stored_tz = getattr(r, "stored_timezone", None) or "UTC"
                started_str = _format_run_datetime(r.started_at, stored_tz)
                with st.container():
                    c1, c2, c3 = st.columns([3, 2, 1])
                    with c1:
                        st.markdown(f"**{r.task_name}**")
                    with c2:
                        pv = getattr(r, "prompt_version", None)
                        pv_str = f" · v{pv}" if pv is not None else ""
                        st.caption(f"{r.username} · {started_str} · {r.status}{pv_str}")
                    with c3:
                        if st.button("View", key=f"view_run_{r.id}"):
                            st.session_state["run_history_view_id"] = r.id
                            st.rerun()
                    st.divider()

# -----------------------------------------------------------------------------
# Transformer playground page (admin only)
# -----------------------------------------------------------------------------

elif current_page == "transformer_playground":
    st.markdown("### 🔧 Transformer playground")
    if st.button("← Back to Run Analysis", key="back_from_playground"):
        st.session_state["current_page"] = "runner"
        st.rerun()
    st.caption(
        "Debug transformers without re-running a prompt: pick a past run (with stored raw JSON) or paste JSON, "
        "choose a transformer, and see the Markdown. Edit transformer code and refresh to re-apply."
    )

    import output_schemas
    output_schemas.ensure_registry_loaded()
    all_transformers = output_schemas.get_all_transformer_keys()
    if not all_transformers:
        st.info("No transformers registered. Add schema bundles in `output_schemas/` and register them.")
    else:
        if "playground_source" not in st.session_state:
            st.session_state["playground_source"] = "run"
        if "playground_run_id" not in st.session_state:
            st.session_state["playground_run_id"] = None
        if "playground_transformer_key" not in st.session_state:
            st.session_state["playground_transformer_key"] = all_transformers[0][0] if all_transformers else ""

        source = st.radio(
            "JSON source",
            options=["run", "paste"],
            format_func=lambda x: "From a past run" if x == "run" else "Paste JSON",
            key="playground_source_radio",
            horizontal=True,
        )
        st.session_state["playground_source"] = source

        json_str: str | None = None
        json_source_label = ""

        if source == "run":
            runs_list = runs_db.list_analysis_runs(50)
            runs_with_json = []
            for r in runs_list:
                out_json = getattr(r, "output_json", None)
                if out_json:
                    runs_with_json.append(r)
                else:
                    try:
                        if (r.output_text or "").strip():
                            json.loads(r.output_text)
                            runs_with_json.append(r)
                    except (json.JSONDecodeError, TypeError):
                        pass
            if not runs_with_json:
                st.info("No past runs with stored JSON. Run an analysis that uses an output JSON schema, or use **Paste JSON** below.")
            else:
                run_options = []
                for r in runs_with_json:
                    stored_tz = getattr(r, "stored_timezone", None) or "UTC"
                    started_str = _format_run_datetime(r.started_at, stored_tz)
                    run_options.append((r.id, f"Run #{r.id} — {r.task_name} — {started_str}"))
                default_idx = 0
                if st.session_state.get("playground_run_id") is not None:
                    ids = [ro[0] for ro in run_options]
                    if st.session_state["playground_run_id"] in ids:
                        default_idx = ids.index(st.session_state["playground_run_id"])
                chosen = st.selectbox(
                    "Select run",
                    options=[ro[0] for ro in run_options],
                    format_func=lambda rid: next((ro[1] for ro in run_options if ro[0] == rid), f"Run #{rid}"),
                    index=default_idx,
                    key="playground_run_select",
                )
                st.session_state["playground_run_id"] = chosen
                run_detail = runs_db.get_analysis_run_by_id(chosen)
                if run_detail:
                    out_json = getattr(run_detail, "output_json", None)
                    if out_json:
                        json_str = out_json
                        json_source_label = f"Run #{chosen} (output_json)"
                    else:
                        try:
                            json_str = run_detail.output_text or ""
                            if json_str.strip():
                                json.loads(json_str)
                                json_source_label = f"Run #{chosen} (output_text as JSON)"
                            else:
                                json_str = None
                        except (json.JSONDecodeError, TypeError):
                            st.warning(f"Run #{chosen} has no stored raw JSON and output_text is not valid JSON. Use a run that had an output schema.")
        else:
            pasted = st.text_area(
                "Paste raw JSON",
                value=st.session_state.get("playground_pasted_json", ""),
                height=200,
                key="playground_paste_ta",
                placeholder='{"summary": "...", "points": [], ...}',
            )
            st.session_state["playground_pasted_json"] = pasted
            if pasted.strip():
                try:
                    json.loads(pasted)
                    json_str = pasted.strip()
                    json_source_label = "Pasted JSON"
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")

        transformer_options = [(composite, label) for composite, label in all_transformers]
        transformer_labels = [label for _, label in transformer_options]
        current_key = st.session_state.get("playground_transformer_key", "")
        t_idx = 0
        if current_key:
            for i, (comp, _) in enumerate(transformer_options):
                if comp == current_key:
                    t_idx = i
                    break
        chosen_transformer = st.selectbox(
            "Transformer",
            options=transformer_labels,
            index=t_idx,
            key="playground_transformer_select",
        )
        st.session_state["playground_transformer_key"] = next(c for c, l in transformer_options if l == chosen_transformer)

        st.markdown("---")
        st.markdown("#### Output (Markdown)")

        if json_str and st.session_state["playground_transformer_key"]:
            try:
                data = json.loads(json_str)
                composite_key = st.session_state["playground_transformer_key"]
                md = output_schemas.run_transformer(composite_key, data)
                _markdown_with_copy(md, "playground_output_md")
                st.caption(f"Source: {json_source_label} · Transformer: {chosen_transformer}")
            except json.JSONDecodeError as e:
                st.error(f"JSON parse error: {e}")
            except Exception as e:
                st.error(f"Transformer error: {e}")
        else:
            if not json_str:
                st.caption("Select a run or paste JSON above.")
            else:
                st.caption("Select a transformer above.")

# -----------------------------------------------------------------------------
# Run Analysis page
# -----------------------------------------------------------------------------

if current_page == "runner":
    try:
        prompts = sorted(db.get_all_prompts(), key=lambda p: p.name.casefold())
        runnable = [p for p in prompts if not getattr(p, "follow_on_only", False)]
        task_options = [p.name for p in runnable]
    except Exception as e:
        logger.error(f"Error loading prompts: {e}", exc_info=True)
        prompts = []
        task_options = []

    st.markdown("#### ▶️ Run Analysis")
    st.caption("Select a task, add documents or paste text, then run. Results use the council knowledge base.")

    # New Analysis: reset runner state
    if st.button("🔄 New analysis", key="new_analysis_btn"):
        st.session_state["transient_items"] = []
        st.session_state["transient_deleted_file_names"] = []
        st.session_state["last_result"] = None
        st.session_state["last_task_name"] = None
        st.session_state["last_mode"] = None
        st.session_state["last_chain"] = None
        st.session_state["last_chain_error"] = None
        st.session_state["last_rag_retrieval_report"] = None
        st.session_state["last_run_context_stats"] = None
        st.session_state["last_result_output_json"] = None
        st.session_state["last_result_json_view"] = None
        st.session_state["analysis_session_id"] = st.session_state.get("analysis_session_id", 0) + 1
        logger.info("New Analysis: reset runner state")
        st.rerun()

    sid = st.session_state.get("analysis_session_id", 0)
    st.markdown("---")

    st.markdown("#### Step 1: Select analysis type")
    task_select = st.selectbox("Analysis type", options=task_options or ["—"], key="task_select", label_visibility="collapsed")
    selected = next((p for p in prompts if p.name == task_select), None)

    if selected:
        template_text = selected.template_text
        # Load any JSON Schemas attached to this prompt so they can be passed
        # both into the template (for references) and to Gemini as sidecars.
        input_schema_json = None
        output_schema_json = None
        try:
            import output_schemas as _os
            _os.ensure_registry_loaded()
            input_sk = getattr(selected, "input_schema_key", None)
            if input_sk:
                input_schema_json = _os.get_schema_json(input_sk)
            output_sk = getattr(selected, "output_schema_key", None)
            if output_sk:
                output_schema_json = _os.get_schema_json(output_sk)
        except Exception as e:
            logger.error(f"Error loading JSON Schemas for prompt {selected.id}: {e}", exc_info=True)
            input_schema_json = None
            output_schema_json = None
        transient_items = list(st.session_state.get("transient_items", []))
        deleted_names = set(st.session_state.get("transient_deleted_file_names", []))

        st.markdown("#### Step 2: Add input")
        st.caption("Upload PDF/DOCX files or add named pastes. All become the subject of analysis.")
        files = st.file_uploader("Upload files (PDF, Word)", type=["pdf", "docx", "doc"], accept_multiple_files=True, key=f"file_upload_{sid}", label_visibility="collapsed")
        # Sync uploaded files -> transient_items
        current_file_names = [f.name for f in files] if files else []
        file_names_in_list = {it["name"] for it in transient_items if it.get("type") == "file"}
        for fn in current_file_names:
            if fn in deleted_names:
                continue
            if fn in file_names_in_list:
                continue
            logger.info(f"Adding new file to transient: {fn}")
            with st.spinner(f"Extracting {fn}…"):
                raw = _extract_text_from_upload(next(f for f in files if f.name == fn))
            transient_items.append({
                "id": str(uuid.uuid4()),
                "name": fn,
                "content": raw,
                "type": "file",
            })
            file_names_in_list.add(fn)
        # Remove files no longer in uploader from transient_items and from deleted
        still_uploaded = set(current_file_names)
        new_items = []
        for it in transient_items:
            if it.get("type") == "file":
                n = it.get("name", "")
                if n not in still_uploaded:
                    deleted_names.discard(n)
                    continue
            new_items.append(it)
        transient_items = new_items
        st.session_state["transient_items"] = transient_items
        st.session_state["transient_deleted_file_names"] = list(deleted_names)

        # Add named paste
        with st.expander("➕ Paste text for analysis", expanded=False):
            with st.form("add_paste_form", clear_on_submit=True):
                paste_name = st.text_input("Desciption/Title", placeholder="e.g. Priorities, notes")
                paste_content = st.text_area("Content", height=120, placeholder="Paste text here…")
                if st.form_submit_button("Save"):
                    if paste_name and paste_content:
                        transient_items = st.session_state.get("transient_items", [])
                        transient_items.append({
                            "id": str(uuid.uuid4()),
                            "name": paste_name.strip(),
                            "content": paste_content.strip(),
                            "type": "paste",
                        })
                        st.session_state["transient_items"] = transient_items
                        st.rerun()

        # List transient items with delete
        if not transient_items:
            st.info("Add files and/or named pastes above, then go to **Step 3** to run.")
        else:
            for it in transient_items:
                tid = it["id"]
                name = it.get("name", "Untitled")
                typ = it.get("type", "file")
                content = it.get("content", "")
                n = len(content)
                icon = "📄" if typ == "file" else "📝"
                col1, col2 = st.columns([5, 1])
                with col1:
                    with st.expander(f"{icon} **{name}** ({n:,} chars) — {typ}", expanded=False):
                        preview = (content[:5000] + "…") if len(content) > 5000 else (content or "[No content]")
                        st.markdown(preview)
                with col2:
                    if st.button("🗑️ Delete", key=f"del_{tid}"):
                        st.session_state["transient_items"] = [x for x in st.session_state["transient_items"] if x["id"] != tid]
                        if typ == "file":
                            s = set(st.session_state.get("transient_deleted_file_names", [])) | {name}
                            st.session_state["transient_deleted_file_names"] = list(s)
                        st.rerun()

        st.markdown("#### Step 3: Run")
        run = st.button("Run analysis", key="run_btn", type="primary")

        if run:
            logger.info(f"Run button clicked for task: {selected.name}")
            user_content = _wrap_transient_content(transient_items)
            can_proceed = True
            if not folder_id:
                logger.warning("Run attempted without Drive folder ID")
                st.error("Drive folder ID is missing.")
                can_proceed = False
            elif not transient_items:
                logger.warning("Run attempted without any input")
                st.warning("Add at least one file or named paste, then run.")
                can_proceed = False
            elif not user_content.strip():
                logger.warning("Run attempted but all transient items are empty")
                st.warning("Add content: ensure at least one file or paste has extractable text.")
                can_proceed = False

            if can_proceed:
                run_started_at = datetime.now(timezone.utc)
                st.session_state["_run_started_at"] = run_started_at
                _rs = st.session_state.get("rag_state")
                if _rs is None and folder_id:
                    _rs = get_cached_rag_state(folder_id, _progress_callback=None)
                    st.session_state["rag_state"] = _rs
                if _rs is None:
                    if is_admin:
                        st.error("RAG knowledge base not loaded. Use **Refresh Knowledge Base** and try again.")
                    else:
                        st.error("RAG knowledge base not loaded. Ask an administrator to refresh it, then try again.")
                    st.stop()
                logger.info(f"Starting run: folder_id={folder_id}, content_len={len(user_content)}")
                prompt_variables = _build_prompt_variables(
                    username=_username,
                    user_name=st.session_state.get("name", "unknown"),
                )
                legal_tracking = (
                    workflow_module.LEGAL_TRACKING_MAIN
                    if getattr(selected, "legal_expert_prompt_id", None)
                    else ""
                )
                main_full_template = (
                    prompt_variables + template_text + legal_tracking
                    + "\n\n---\n\nSubject of analysis (transient input):\n{{ content }}"
                )
                # Phase 5.2: state is serializable (no rag_state, selected_prompt, _callbacks, build_prompt_variables)
                state = {
                    "task_name": selected.name,
                    "template_text": template_text,
                    "user_content": user_content,
                    "folder_id": folder_id,
                    "prompt_template_id": selected.id,
                    "username": _username,
                    "user_name": st.session_state.get("name", "unknown"),
                    "prompt_variables": prompt_variables,
                    "main_full_template": main_full_template,
                    "input_schema_json": input_schema_json,
                    "output_schema_json": output_schema_json,
                    "session_cache_name": st.session_state.get("gemini_cache_name"),
                    "session_cache_model": st.session_state.get("gemini_cache_model"),
                    "session_cache_folder_id": st.session_state.get("gemini_cache_folder_id"),
                    "session_run_cache_key": st.session_state.get("run_cache_key"),
                    "gemini_call_count": 0,
                }

                _prompt_version = getattr(selected, "current_version", None) or 1
                # Phase 5: resolve workflow key from prompt's workflow (not exposed to user on Run Analysis)
                _wf_id = getattr(selected, "workflow_id", None)
                _wf = db.get_workflow_by_id(_wf_id) if _wf_id else None
                _workflow_key = _wf.graph_key if _wf else workflow_graph.DEFAULT_WORKFLOW_KEY
                _thread_id = uuid.uuid4().hex
                try:
                    with st.status("Running analysis…", expanded=True) as status:
                        run_config = {
                            "configurable": {
                                "callbacks": {
                                    "write": status.write,
                                    "update_label": lambda label, _s: status.update(label=label),
                                },
                                "build_prompt_variables": _build_prompt_variables,
                                "thread_id": _thread_id,
                                "log_run_event": lambda step_name, event_type, payload=None: runs_db.insert_run_event(
                                    step_name, event_type, thread_id=_thread_id, payload=payload
                                ),
                            }
                        }
                        status.write("Planning retrieval → context → main agent → legal (if needed) → follow-ons…")
                        state = workflow_graph.run_analysis_graph(
                            state, workflow_key=_workflow_key, config=run_config, thread_id=_thread_id
                        )
                except workflow_module.WorkflowError as e:
                    st.error(e.message)
                    if e.details:
                        st.caption(e.details)
                    logger.error(f"Workflow failed: {e}", exc_info=True)
                    try:
                        run_row = runs_db.insert_analysis_run(
                            username=_username,
                            task_name=selected.name,
                            status="failed",
                            prompt_template_id=selected.id,
                            prompt_version=_prompt_version,
                            folder_id=folder_id,
                            started_at=st.session_state.get("_run_started_at") or datetime.now(timezone.utc),
                            completed_at=datetime.now(timezone.utc),
                            input_summary=f"{len(user_content)} chars",
                            error_message=e.message,
                        )
                        if run_row and run_row.id:
                            try:
                                runs_db.update_run_events_run_id(_thread_id, run_row.id)
                            except Exception:
                                pass
                    except Exception as persist_err:
                        logger.warning(f"Could not persist failed run: {persist_err}")
                    st.stop()

                # Sync workflow state to session (Phase 2)
                st.session_state["last_result"] = state["final_output"]
                st.session_state["last_mode"] = "markdown"
                st.session_state["last_task_name"] = state["task_name"]
                st.session_state["last_rag_retrieval_report"] = state.get("retrieval_report")
                st.session_state["pipeline_step_results"] = state.get("pipeline_step_results", [])
                st.session_state["last_legal_questions"] = state.get("legal_questions")
                st.session_state["last_legal_expert_output"] = state.get("legal_expert_output")
                st.session_state["last_legal_expert_report"] = state.get("legal_expert_report")
                st.session_state["legal_questions_by_step"] = state.get("legal_questions_by_step", [])
                st.session_state["last_run_context_stats"] = state.get("last_run_context_stats", {})
                st.session_state["last_chain"] = state.get("chain")
                st.session_state["last_chain_timings"] = state.get("chain_timings", [])
                st.session_state["last_chain_error"] = state.get("last_chain_error")
                st.session_state["last_prompt_version"] = _prompt_version
                if state.get("cache_name"):
                    st.session_state["gemini_cache_name"] = state["cache_name"]
                    st.session_state["gemini_cache_model"] = get_effective_model()
                    st.session_state["gemini_cache_folder_id"] = folder_id
                if state.get("run_cache_key") is not None:
                    st.session_state["run_cache_key"] = state["run_cache_key"]

                # Persist successful run (Phase 1)
                _started = st.session_state.get("_run_started_at") or run_started_at
                _retrieval_summary = ""
                if state.get("retrieval_report"):
                    parts = [
                        f"{r.get('library_name', '?')}: {r.get('chunks_retrieved', 0)} chunks"
                        for r in state["retrieval_report"]
                    ]
                    _retrieval_summary = "; ".join(parts) if parts else ""
                # JSON output: save both raw JSON and (optionally) transformer-generated Markdown.
                # Use last pipeline step's output when it's valid JSON with motions (e.g. analysis
                # injection step adds analysis_block); else use main agent output.
                _output_json = None
                _output_text = state["final_output"]
                if getattr(selected, "output_schema_key", None) and (state.get("main_output_dict") is not None or state.get("main_output")):
                    _parsed = state.get("main_output_dict")
                    _raw_json_str = state.get("main_output")
                    if _parsed is not None:
                        _raw_json_str = json.dumps(_parsed, indent=2) if _raw_json_str is None else _raw_json_str
                    elif _raw_json_str:
                        try:
                            _parsed = json.loads(_raw_json_str)
                        except json.JSONDecodeError:
                            _parsed = None
                    # Prefer last pipeline step's output when it has motions (analysis injection adds analysis_block)
                    _steps = state.get("pipeline_step_results") or []
                    if len(_steps) > 1:
                        _last_out = _steps[-1].get("output") or _steps[-1].get("full_output") or ""
                        if isinstance(_last_out, str) and _last_out.strip():
                            try:
                                _last_parsed = json.loads(_last_out.strip())
                                if isinstance(_last_parsed, dict) and isinstance(_last_parsed.get("motions"), list):
                                    _parsed = _last_parsed
                                    workflow_module._collapse_repeated_newlines(_parsed)
                                    _raw_json_str = json.dumps(_parsed, indent=2)
                                    logger.debug("Using last pipeline step output for transformer (has motions)")
                            except json.JSONDecodeError:
                                pass
                    _output_json = _raw_json_str
                    _transformer_key = getattr(selected, "output_transformer_key", None)
                    if _transformer_key and _parsed is not None:
                        try:
                            import output_schemas
                            output_schemas.ensure_registry_loaded()
                            _output_text = output_schemas.run_transformer(_transformer_key, _parsed)
                        except Exception as te:
                            logger.warning(f"Transformer {_transformer_key} failed: {te}; using pretty JSON for output_text")
                            _output_text = _raw_json_str or state["final_output"]
                    elif _raw_json_str:
                        _output_text = _raw_json_str or state["final_output"]
                    st.session_state["last_result"] = _output_text
                    st.session_state["last_result_output_json"] = _output_json
                else:
                    st.session_state["last_result_output_json"] = None
                try:
                    run_row = runs_db.insert_analysis_run(
                        username=_username,
                        task_name=selected.name,
                        status="completed",
                        prompt_template_id=selected.id,
                        prompt_version=_prompt_version,
                        folder_id=folder_id,
                        started_at=_started,
                        completed_at=datetime.now(timezone.utc),
                        input_summary=f"{len(user_content)} chars",
                        output_text=_output_text,
                        output_mode="markdown",
                        has_legal_review=bool(state.get("legal_questions")),
                        legal_questions=json.dumps(state["legal_questions"]) if state.get("legal_questions") else None,
                        legal_expert_output=state.get("legal_expert_output"),
                        chain_steps=(
                            json.dumps([{"step_name": n, "output": o} for n, o in state["chain"]])
                            if state.get("chain") else None
                        ),
                        retrieval_report_summary=_retrieval_summary or None,
                        model_used=get_effective_model(),
                        pre_qa_output=state.get("pre_qa_output"),
                        qa_output=state.get("qa_output"),
                        output_json=_output_json,
                    )
                    if run_row and run_row.id:
                        runs_db.update_run_events_run_id(_thread_id, run_row.id)
                except Exception as persist_err:
                    logger.warning(f"Could not persist run to runs DB: {persist_err}")

        # Success path continues to Output section below.

        # -------------------------------------------------------------------------
        # 4. Review & 5. Export / copy
        # -------------------------------------------------------------------------

        res = st.session_state.get("last_result")
        res_task = st.session_state.get("last_task_name")
        chain_list = st.session_state.get("last_chain")
        chain_error = st.session_state.get("last_chain_error")
        rag_report = st.session_state.get("last_rag_retrieval_report")

        if res is not None and res_task == selected.name:
            st.markdown("#### Step 4: Review results")
            _pv = st.session_state.get("last_prompt_version")
            if _pv is not None:
                st.caption(f"Prompt version: **{_pv}**")
            logger.debug(f"Displaying output: type={type(res).__name__}")
            
            # Display progressive results for each step in the pipeline
            pipeline_step_results = st.session_state.get("pipeline_step_results", [])
            if pipeline_step_results and len(pipeline_step_results) > 0:
                st.markdown("**Pipeline Results (by step):**")
                st.caption("Each step's results are shown below. You can review them while subsequent steps are processing.")

                # Expand only the final step by default; earlier steps stay collapsed
                last_step_number = pipeline_step_results[-1].get(
                    "step_number", len(pipeline_step_results)
                )

                for step_result in pipeline_step_results:
                    step_num = step_result.get("step_number", 0)
                    step_name = step_result.get("step_name", "Unknown")
                    # Main analysis output for this step (without cumulative chaining)
                    main_output = step_result.get("output", "")
                    # full_output already includes legal expert consultation and any chaining
                    full_output = step_result.get("full_output", main_output)
                    has_legal = step_result.get("has_legal_expert", False)
                    legal_output = step_result.get("legal_expert_output")

                    is_last_step = step_num == last_step_number
                    with st.expander(f"Step {step_num}: {step_name}", expanded=is_last_step):
                        # Show the step's primary analysis output
                        st.markdown("**Step output**")
                        _markdown_with_copy(
                            main_output or full_output, f"step_{step_num}_{step_name}_main"
                        )

                        # If there was a separate legal expert consultation, surface it explicitly
                        if has_legal and legal_output:
                            st.markdown("---")
                            st.markdown("**Legal expert consultation (raw)**")
                            _markdown_with_copy(
                                str(legal_output),
                                f"step_{step_num}_{step_name}_legal",
                            )

                        # If full_output differs (e.g., cumulative with prior steps), show it as well
                        if full_output and full_output != main_output:
                            st.markdown("---")
                            st.markdown("**Combined output (including prior steps / legal)**")
                            _markdown_with_copy(
                                full_output, f"step_{step_num}_{step_name}_full"
                            )
            else:
                # Fallback: if no step results stored, show final result
                md = res if isinstance(res, str) else str(res)
                # JSON view selector when run had JSON output
                json_out = st.session_state.get("last_result_output_json")
                if json_out and (getattr(selected, "output_schema_id", None) or getattr(selected, "output_schema_key", None)):
                    view_opts = _get_json_view_options(schema_id=selected.output_schema_id, schema_key=getattr(selected, "output_schema_key", None))
                    view_labels = [lbl for _, lbl in view_opts]
                    view_keys = [val for val, _ in view_opts]
                    idx = view_keys.index(st.session_state.get("last_result_json_view", "saved")) if st.session_state.get("last_result_json_view", "saved") in view_keys else 0
                    view_choice = st.selectbox("View as", options=view_labels, index=idx, key="last_result_json_view_select")
                    st.session_state["last_result_json_view"] = view_keys[view_labels.index(view_choice)]
                    display_str = _render_json_view(json_out, md, st.session_state["last_result_json_view"])
                    _markdown_with_copy(display_str, "result")
                else:
                    _markdown_with_copy(md, "result")

            ctx_stats = st.session_state.get("last_run_context_stats")
            if ctx_stats:
                with st.expander("📐 Stats for nerds — context & tokens", expanded=False):
                    tot = ctx_stats.get("total_input_tokens", 0)
                    kb = ctx_stats.get("kb_tokens", 0)
                    tr = ctx_stats.get("transient_tokens", 0)
                    pr = ctx_stats.get("prompt_tokens", 0)
                    mx = ctx_stats.get("max_context", 0)
                    out = ctx_stats.get("output_tokens", 0)
                    model = ctx_stats.get("model", "")
                    timings = ctx_stats.get("timings", {})
                    pct = (tot / mx * 100) if mx else 0
                    st.caption("Token estimates (~4 chars/token). Real-world equivalents use Harry Potter, pages, and reading hours.")
                    gemini_calls = ctx_stats.get("gemini_calls", 0)
                    st.markdown(f"**Model**: `{model}` · **Max context**: {mx:,} tokens · **Gemini calls**: **{gemini_calls}**")
                    st.markdown(f"**Input**")
                    st.markdown(f"- Knowledge base (cached): **{kb:,}** tokens")
                    st.markdown(f"- User data (analyzed): **{tr:,}** tokens")
                    st.markdown(f"- Prompt wrapper: **{pr:,}** tokens")
                    st.markdown(f"- **Total input**: **{tot:,}** tokens → {format_context_usage(tot, mx, model)}")
                    st.markdown(f"**Output**: **{out:,}** tokens")
                    st.markdown(f"**Real-world**: Total input ≈ **{format_reading_equivalent(tot)}** · Output ≈ **{format_reading_equivalent(out)}**")
                    if timings:
                        st.markdown("**Timings**")
                        st.markdown(f"- Retrieval planning: **{timings.get('plan_retrieval_s', 0):.2f}s**")
                        st.markdown(f"- Context build: **{timings.get('build_context_s', 0):.2f}s**")
                        st.markdown(f"- Cache create: **{timings.get('cache_create_s', 0):.2f}s**")
                        st.markdown(f"- Model run: **{timings.get('model_run_s', 0):.2f}s**")
                        total_t = sum(
                            float(timings.get(k, 0))
                            for k in ("plan_retrieval_s", "build_context_s", "cache_create_s", "model_run_s")
                        )
                        if total_t > 0:
                            st.markdown(f"- **Total (timed steps)**: **{total_t:.2f}s**")

            chain_timings = st.session_state.get("last_chain_timings")
            if chain_timings:
                with st.expander("⏱️ Timing by follow‑on step", expanded=False):
                    for i, t in enumerate(chain_timings):
                        name = t.get("name", f"Step {i + 1}")
                        st.markdown(f"**{name}**")
                        st.markdown(f"- Planning: **{t.get('plan_retrieval_s', 0):.2f}s**")
                        st.markdown(f"- Context: **{t.get('build_context_s', 0):.2f}s**")
                        st.markdown(f"- Cache: **{t.get('cache_create_s', 0):.2f}s**")
                        st.markdown(f"- Model: **{t.get('model_run_s', 0):.2f}s**")
                        st.markdown(f"- **Total**: **{t.get('total_s', 0):.2f}s**")

            if rag_report:
                with st.expander("📚 Sources used", expanded=False):
                    st.caption("Knowledge-base libraries and files retrieved for this run.")
                    for i, rec in enumerate(rag_report):
                        if i > 0:
                            st.divider()
                        lib_name = rec.get("library_name", "?")
                        n = rec.get("chunks_retrieved", 0)
                        k = rec.get("top_k", 0)
                        srcs = rec.get("sources", [])
                        st.markdown(f"**{lib_name}** — {n} chunks (top_k={k})")
                        if srcs:
                            for s in srcs:
                                fn = s.get("file_name", "?")
                                link = s.get("link", "")
                                cnt = s.get("chunk_count", 0)
                                if link:
                                    st.markdown(f"  - [{fn}]({link}) — {cnt} chunk(s)")
                                else:
                                    st.markdown(f"  - {fn} — {cnt} chunk(s)")
                        else:
                            st.caption("  _(no chunks retrieved)_")

            # Show legal questions status for each step
            legal_questions_by_step = st.session_state.get("legal_questions_by_step", [])
            
            if legal_questions_by_step:
                st.markdown("---")
                with st.expander("⚖️ Legal Review Status by Step", expanded=True):
                    for step_info in legal_questions_by_step:
                        step_name = step_info.get("step_name", "Unknown")
                        questions = step_info.get("questions")
                        expert_output = step_info.get("expert_output")
                        expert_report = step_info.get("expert_report")
                        has_legal_expert = step_info.get("has_legal_expert", False)
                        
                        if has_legal_expert:
                            if questions:
                                st.markdown(f"**{step_name}**: ✅ Legal questions detected ({len(questions)})")
                                for i, q in enumerate(questions, 1):
                                    st.markdown(f"  - Q{i}: {q}")
                                if expert_output:
                                    st.caption("✅ Legal expert consultation completed")
                                    if expert_report:
                                        st.caption("**Legal expert sources:**")
                                        for rec in expert_report:
                                            lib_name = rec.get("library_name", "?")
                                            n = rec.get("chunks_retrieved", 0)
                                            srcs = rec.get("sources", [])
                                            if n > 0:
                                                file_summary = ", ".join(f"{s['file_name']} ({s['chunk_count']})" for s in srcs[:3])
                                                if len(srcs) > 3:
                                                    file_summary += f" +{len(srcs) - 3} more"
                                                st.caption(f"  • **{lib_name}**: {n} chunks from {len(srcs)} file(s) — {file_summary}")
                                else:
                                    st.caption("⚠️ Legal expert consultation was attempted but did not complete")
                            else:
                                st.markdown(f"**{step_name}**: ℹ️ No legal questions identified (legal review not required)")
                        else:
                            st.markdown(f"**{step_name}**: — Legal expert not configured")
                        if step_info != legal_questions_by_step[-1]:
                            st.divider()
            
            if chain_list and len(chain_list) > 1:
                with st.expander("📋 Pipeline steps", expanded=False):
                    st.caption("Cumulative output after each step. Legal expert consultation (if any) appears before follow-on prompts.")
                    for i, (name, step_out) in enumerate(chain_list):
                        if i > 0:
                            st.divider()
                        # Special styling for legal expert step
                        if name == "Legal Expert Consultation":
                            st.markdown(f"**Step {i + 1}: ⚖️ {name}**")
                        else:
                            st.markdown(f"**Step {i + 1}: {name}**")
                        _markdown_with_copy(step_out, f"chain_{i}")
            if chain_error:
                st.markdown("---")
                st.error(f"⚠️ {chain_error}")
            elif selected.verifier_id and (not chain_list or len(chain_list) == 1):
                st.markdown("---")
                st.info("A follow‑on prompt is configured; run again to see chained results.")

            st.markdown("#### Step 5: Export")
            st.caption("Use the **Copy Markdown** expander above each result to copy the text.")
    else:
        if not task_options:
            if is_admin:
                st.info("No analysis types yet. Open **Prompt Editor** in the sidebar to add prompts.")
            else:
                st.info("No analysis types available. An administrator can add them in the Prompt Editor.")
        else:
            st.info("Select an analysis type above to continue.")
