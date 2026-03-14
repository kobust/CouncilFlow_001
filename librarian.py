"""
Google Drive librarian: fetch files, extract text, build XML context.
Uses Streamlit secrets for Service Account auth and @st.cache_resource for per-session caching.
"""

from __future__ import annotations

import html
import io
import logging
import os
from typing import Any

import streamlit as st
from docx import Document
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# OCR (optional): used when building KB from scanned/image PDFs
_OCR_AVAILABLE = False
_OCR_ERROR_MSG: str | None = None
try:
    import pytesseract
    from pdf2image import convert_from_bytes
    try:
        pytesseract.get_tesseract_version()
        _OCR_AVAILABLE = True
    except Exception as e:
        _OCR_AVAILABLE = False
        _OCR_ERROR_MSG = f"Tesseract OCR binary not found: {e}"
except ImportError as e:
    _OCR_AVAILABLE = False
    _OCR_ERROR_MSG = f"OCR libraries not installed: {e}"
if not _OCR_AVAILABLE and _OCR_ERROR_MSG:
    logger.debug("OCR not available for KB extraction: %s", _OCR_ERROR_MSG)

# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
DRIVE_API_SERVICE = "drive"
DRIVE_API_VERSION = "v3"


def _get_drive_service():
    """Build Drive API service using Service Account credentials from st.secrets."""
    try:
        logger.info("Building Drive API service from secrets")
        env_sa = os.environ.get("GCP_SERVICE_ACCOUNT_JSON", "").strip()
        raw = env_sa if env_sa else st.secrets["gcp_service_account"]
        if isinstance(raw, str):
            import json
            logger.debug("Parsing service account credentials from JSON string")
            creds_dict = json.loads(raw)
        else:
            logger.debug("Using service account credentials from dict")
            creds_dict = dict(raw)
        # Normalize private_key: TOML multiline may add leading/trailing newlines
        pk = creds_dict.get("private_key") or ""
        if isinstance(pk, str):
            pk = pk.strip()
            if not pk.startswith("-----BEGIN"):
                logger.error("Invalid private_key format in gcp_service_account")
                raise ValueError("gcp_service_account.private_key missing or invalid")
        creds_dict = {**creds_dict, "private_key": pk}
        creds = service_account.Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
        logger.info("Drive API service created successfully")
        return build(DRIVE_API_SERVICE, DRIVE_API_VERSION, credentials=creds, cache_discovery=False)
    except KeyError as e:
        logger.error(f"Missing secret: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to build Drive service: {e}", exc_info=True)
        raise


def _drive_list_kw() -> dict[str, Any]:
    return {"supportsAllDrives": True, "includeItemsFromAllDrives": True}


def _drive_get_export_kw() -> dict[str, Any]:
    """Params for get_media / export_media (shared-drive support)."""
    # NOTE: Some Drive API client versions do NOT accept supportsAllDrives
    # on export_media, only on get/list. To avoid
    #   TypeError: unexpected keyword argument supportsAllDrives
    # we no longer pass any extra kwargs here and rely on default behaviour.
    return {}


# -----------------------------------------------------------------------------
# Fetch files
# -----------------------------------------------------------------------------


def get_folder_info(folder_id: str) -> dict[str, str] | None:
    """
    Return folder name and Drive link for the given folder ID.
    Returns None if the folder cannot be fetched (e.g. not found, no access).
    """
    try:
        drive = _get_drive_service()
        f = drive.files().get(
            fileId=folder_id,
            fields="name,webViewLink",
            supportsAllDrives=True,
        ).execute()
        name = f.get("name") or "Unknown folder"
        link = f.get("webViewLink") or f"https://drive.google.com/drive/folders/{folder_id}"
        return {"name": name, "link": link}
    except Exception as e:
        logger.warning(f"Could not fetch folder info for {folder_id}: {e}")
        return None


@st.cache_data(ttl=300)
def get_cached_folder_info(folder_id: str) -> dict[str, str] | None:
    """Cached folder name and link for sidebar display."""
    return get_folder_info(folder_id)


def _list_files_in_folder(drive, folder_id: str) -> list[dict[str, Any]]:
    """List all files and folders directly in a folder (non-recursive)."""
    q = f"'{folder_id}' in parents and trashed = false"
    kw = _drive_list_kw()
    kw["q"] = q
    kw["fields"] = "nextPageToken, files(id, name, mimeType, webViewLink)"
    kw["pageSize"] = 1000

    out: list[dict[str, Any]] = []
    page_token: str | None = None

    while True:
        if page_token:
            kw["pageToken"] = page_token
        resp = drive.files().list(**kw).execute()
        items = resp.get("files") or []
        for item in items:
            link = item.get("webViewLink") or f"https://drive.google.com/file/d/{item['id']}/view"
            item["link"] = link
            out.append(item)
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return out


def list_core_and_libraries(root_folder_id: str) -> dict[str, Any]:
    """
    List root-level files as Core and each subfolder as a library.
    Returns:
        core_files: list of file dicts (root only)
        libraries: list of {id, name, folder_id, files: [...]} for each subfolder
    """
    logger.info(f"Listing core + libraries from root {root_folder_id}")
    drive = _get_drive_service()
    items = _list_files_in_folder(drive, root_folder_id)
    core_files: list[dict[str, Any]] = []
    libraries: list[dict[str, Any]] = []
    for item in items:
        mime = (item.get("mimeType") or "").lower()
        if mime == "application/vnd.google-apps.folder":
            fid = item["id"]
            name = item.get("name") or "Unnamed"
            sub_files = fetch_drive_files(fid, recursive=True)
            libraries.append({"id": fid, "name": name, "folder_id": fid, "files": sub_files})
            logger.info(f"Library '{name}': {len(sub_files)} files")
        else:
            core_files.append(item)
    logger.info(f"Core: {len(core_files)} files, {len(libraries)} libraries")
    return {"core_files": core_files, "libraries": libraries}


def fetch_drive_files(folder_id: str, recursive: bool = True) -> list[dict[str, Any]]:
    """
    List all files in the given Drive folder, optionally including subdirectories.
    Returns a list of dicts with at least: id, name, mimeType, webViewLink (or link).
    """
    logger.info(f"Fetching files from Drive folder: {folder_id} (recursive: {recursive})")
    try:
        drive = _get_drive_service()
        all_files: list[dict[str, Any]] = []
        folders_to_process = [folder_id]
        processed_folders = set()

        while folders_to_process:
            current_folder = folders_to_process.pop(0)
            if current_folder in processed_folders:
                continue
            processed_folders.add(current_folder)
            logger.debug(f"Processing folder: {current_folder}")

            items = _list_files_in_folder(drive, current_folder)
            logger.debug(f"Found {len(items)} items in folder {current_folder}")

            for item in items:
                mime = (item.get("mimeType") or "").lower()
                # Google Drive folders have this mime type
                if recursive and mime == "application/vnd.google-apps.folder":
                    folder_id_to_add = item["id"]
                    if folder_id_to_add not in processed_folders:
                        logger.debug(f"Adding subfolder to process: {item.get('name', folder_id_to_add)}")
                        folders_to_process.append(folder_id_to_add)
                # Add files (not folders) to results
                if mime != "application/vnd.google-apps.folder":
                    all_files.append(item)

        logger.info(f"Total files fetched (recursive): {len(all_files)} from {len(processed_folders)} folders")
        return all_files
    except Exception as e:
        logger.error(f"Error fetching Drive files from folder {folder_id}: {e}", exc_info=True)
        raise


# -----------------------------------------------------------------------------
# Extract text & build XML
# -----------------------------------------------------------------------------

def _escape_xml(s: str) -> str:
    return html.escape(s, quote=True)


def _extract_text_with_ocr(pdf_bytes: bytes, name: str) -> str | None:
    """Extract text from PDF using OCR (for scanned/image-based PDFs). Returns None if OCR unavailable or fails."""
    if not _OCR_AVAILABLE:
        logger.debug("OCR not available for %s: %s", name, _OCR_ERROR_MSG or "unavailable")
        return None
    try:
        logger.info("Attempting OCR on %s (KB build)", name)
        images = convert_from_bytes(pdf_bytes, dpi=300)
        logger.info("Converted %d pages to images for OCR", len(images))
        ocr_parts = []
        for page_num, image in enumerate(images, 1):
            try:
                text = pytesseract.image_to_string(image, lang="eng")
                if text and text.strip():
                    ocr_parts.append(text)
                    logger.debug("OCR page %d: %d chars", page_num, len(text))
            except Exception as e:
                logger.warning("OCR failed on page %d of %s: %s", page_num, name, e)
                ocr_parts.append(f"[Page {page_num}: OCR error - {type(e).__name__}]")
        result = "\n\n".join(ocr_parts)
        if result.strip():
            logger.info("OCR extracted %d chars from %s", len(result), name)
            return result
        return None
    except Exception as e:
        logger.warning("OCR failed for %s: %s", name, e)
        return None


def _extract_text_from_bytes(data: bytes, mime: str, name: str) -> str:
    """Extract plain text from file bytes. Supports PDF, DOCX, and Google Docs (text/plain)."""
    mime = (mime or "").lower()
    logger.debug(f"Extracting text from {name} (mime: {mime}, size: {len(data)} bytes)")
    try:
        if "pdf" in mime:
            logger.debug(f"Extracting from PDF: {name}")
            bio = io.BytesIO(data)
            try:
                reader = PdfReader(bio)
            except Exception as e:
                logger.error(f"Failed to read PDF {name}: {e}")
                return f"[Error reading PDF {name}: {e}]"
            
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
                    else:
                        parts.append(f"[Page {page_num}: No text found]")
                except (KeyError, AttributeError, ValueError) as page_error:
                    failed_pages.append(page_num)
                    error_type = type(page_error).__name__
                    logger.warning(f"Error extracting text from page {page_num}/{total_pages} of {name}: {error_type} - {str(page_error)[:100]}")
                    parts.append(f"[Page {page_num}: Error extracting text - {error_type}]")
                except Exception as page_error:
                    failed_pages.append(page_num)
                    logger.warning(f"Unexpected error extracting text from page {page_num}/{total_pages} of {name}: {page_error}")
                    parts.append(f"[Page {page_num}: Error extracting text - {type(page_error).__name__}]")
            
            result = "\n".join(parts)
            if failed_pages:
                logger.info(f"Extracted text from {successful_pages}/{total_pages} pages of {name} (failed pages: {failed_pages})")
            logger.debug(f"Extracted {len(result)} chars from PDF {name} ({successful_pages}/{total_pages} pages successful)")
            
            actual_text_parts = [x for x in parts if not (x.startswith("[Page") or x.startswith("[Error"))]
            actual_text = "\n".join(actual_text_parts)
            actual_text_len = len(actual_text.strip())
            if successful_pages == 0 or actual_text_len < 50:
                logger.info(f"PDF {name} has little/no pypdf text ({actual_text_len} chars). Trying OCR for KB build.")
                ocr_result = _extract_text_with_ocr(data, name)
                if ocr_result and len(ocr_result.strip()) > 50:
                    return ocr_result
                if ocr_result:
                    logger.warning(f"OCR for {name} yielded only {len(ocr_result)} chars; using it anyway.")
                    return ocr_result
                if not _OCR_AVAILABLE:
                    logger.warning(f"OCR not available for scanned PDF {name}. KB will use minimal pypdf output.")
            return result
        if "wordprocessingml" in mime or "msword" in mime or mime.endswith("/document"):
            logger.debug(f"Extracting from Word doc: {name}")
            bio = io.BytesIO(data)
            doc = Document(bio)
            result = "\n".join(p.text for p in doc.paragraphs if p.text)
            logger.debug(f"Extracted {len(result)} chars from Word doc {name}")
            return result
        if "text/plain" in mime:
            logger.debug(f"Decoding plain text: {name}")
            result = data.decode("utf-8", errors="replace")
            logger.debug(f"Decoded {len(result)} chars from text {name}")
            return result
        logger.warning(f"Unsupported mime type for {name}: {mime}")
        return ""
    except Exception as e:
        # Only log as ERROR if it's a critical failure (not a per-page issue which is already handled)
        # Per-page errors are already logged as warnings above
        logger.error(f"Critical error extracting text from {name}: {e}", exc_info=True)
        return f"[Error extracting text from {name}: {e}]"


def _download_file(drive, file_info: dict[str, Any]) -> bytes:
    """Download or export a Drive file to bytes."""
    fid = file_info["id"]
    name = file_info.get("name", "unknown")
    mime = (file_info.get("mimeType") or "").lower()
    logger.debug(f"Downloading file {name} (id: {fid}, mime: {mime})")
    kw = _drive_get_export_kw()

    try:
        if "vnd.google-apps." in mime:
            logger.debug(f"Exporting Google Workspace file {name} as text/plain")
            # export_media does not accept supportsAllDrives on some client versions
            req = drive.files().export_media(fileId=fid, mimeType="text/plain")
            buf = io.BytesIO()
            download = MediaIoBaseDownload(buf, req)
            done = False
            while not done:
                _, done = download.next_chunk()
            result = buf.getvalue()
            logger.debug(f"Exported {name}: {len(result)} bytes")
            return result

        logger.debug(f"Downloading binary file {name}")
        # get_media may accept supportsAllDrives on newer clients, but it's optional;
        # we keep kw for forward compatibility, and it's empty on older ones.
        req = drive.files().get_media(fileId=fid, **kw)
        buf = io.BytesIO()
        download = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = download.next_chunk()
        result = buf.getvalue()
        logger.debug(f"Downloaded {name}: {len(result)} bytes")
        return result
    except Exception as e:
        logger.error(f"Error downloading file {name} (id: {fid}): {e}", exc_info=True)
        raise


def fetch_and_extract_files(
    files: list[dict[str, Any]],
    progress_callback: callable | None = None,
) -> list[dict[str, Any]]:
    """
    Download each file, extract text. Returns list of dicts with name, link, id, text.
    Used for RAG indexing (chunking, embedding).
    """
    logger.info(f"Fetch and extract {len(files)} files")
    drive = _get_drive_service()
    out: list[dict[str, Any]] = []
    for idx, f in enumerate(files, 1):
        name = f.get("name") or "Untitled"
        if progress_callback:
            try:
                progress_callback(name, idx, len(files))
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
        mime = (f.get("mimeType") or "").lower()
        export_mime = "text/plain" if "vnd.google-apps.document" in mime else mime
        try:
            raw = _download_file(drive, f)
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")
            text = f"[Error downloading {name}: {e}]"
        else:
            text = _extract_text_from_bytes(raw, export_mime, name)
        out.append({
            "name": name,
            "link": f.get("link") or f"https://drive.google.com/file/d/{f['id']}/view",
            "id": f["id"],
            "text": text,
        })
    return out


def build_context_xml(files: list[dict[str, Any]], progress_callback: callable | None = None) -> str:
    """
    Download each file, extract text (pypdf / python-docx / export), and wrap in
    <document title='X' link='Y'>text</document>.
    
    Args:
        files: List of file dicts from Drive API
        progress_callback: Optional callback function(name, idx, total) called for each file processed
    """
    logger.info(f"Building context XML from {len(files)} files")
    try:
        drive = _get_drive_service()
        docs: list[str] = []

        for idx, f in enumerate(files, 1):
            name = f.get("name") or "Untitled"
            logger.info(f"Processing file {idx}/{len(files)}: {name}")
            # Call progress callback if provided
            if progress_callback:
                try:
                    progress_callback(name, idx, len(files))
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
            title = _escape_xml(name)
            link = _escape_xml(f.get("link") or f"https://drive.google.com/file/d/{f['id']}/view")
            mime = (f.get("mimeType") or "").lower()

            export_mime = mime
            if "vnd.google-apps.document" in mime:
                export_mime = "text/plain"

            try:
                raw = _download_file(drive, f)
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
                raw = f"[Error downloading {name}: {e}]".encode("utf-8")
                export_mime = "text/plain"

            text = _extract_text_from_bytes(raw, export_mime, name)
            text = _escape_xml(text)
            logger.debug(f"Extracted {len(text)} chars from {name}")

            docs.append(f"<document title='{title}' link='{link}'>{text}</document>")

        result = "\n".join(docs)
        logger.info(f"Built context XML: {len(result)} total chars from {len(docs)} documents")
        return result
    except Exception as e:
        logger.error(f"Error building context XML: {e}", exc_info=True)
        raise


# -----------------------------------------------------------------------------
# Cached context
# -----------------------------------------------------------------------------


@st.cache_resource
def get_cached_file_list(folder_id: str) -> list[dict[str, Any]]:
    """
    Fetch and cache the list of files from Drive folder (without downloading content).
    Returns list of file dicts with name, id, mimeType, link.
    """
    logger.info(f"Getting cached file list for folder: {folder_id}")
    try:
        files = fetch_drive_files(folder_id, recursive=True)
        logger.info(f"Found {len(files)} files in folder {folder_id}")
        return files
    except Exception as e:
        logger.error(f"Error fetching file list for folder {folder_id}: {e}", exc_info=True)
        raise


@st.cache_resource
def get_cached_context(folder_id: str, _progress_callback: callable | None = None) -> str:
    """
    Fetch Drive files in folder_id, download and extract text, and return
    concatenated <document ...> XML. Cached per folder_id for the session.
    
    Args:
        folder_id: Google Drive folder ID
        _progress_callback: Optional callback function(name, idx, total) called for each file processed
                           Note: Underscore prefix tells Streamlit to ignore this for cache key purposes
    """
    logger.info(f"Getting cached context for folder: {folder_id}")
    try:
        files = fetch_drive_files(folder_id, recursive=True)
        if not files:
            logger.warning(f"No files found in folder {folder_id} (recursive)")
            return "<document title='No files found' link=''>No files found in Drive folder.</document>"
        logger.info(f"Found {len(files)} files, building context XML")
        result = build_context_xml(files, progress_callback=_progress_callback)
        if len(result) < 100:
            logger.warning(f"Context XML is very small ({len(result)} chars) - may indicate extraction issues")
        logger.info(f"Cached context ready for folder {folder_id}: {len(result)} chars, {len(files)} documents")
        return result
    except Exception as e:
        logger.error(f"Error getting cached context for folder {folder_id}: {e}", exc_info=True)
        raise
