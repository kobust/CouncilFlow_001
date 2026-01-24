"""
Shared filesystem paths for local read/write storage.
On non-Windows, use /app/data (or COUNCILFLOW_DATA_DIR).
"""

from __future__ import annotations

import os
from pathlib import Path

_UNIX_DATA_DIR = Path("/app/data")


def is_windows() -> bool:
    return os.name == "nt"


def data_dir() -> Path:
    if is_windows():
        return Path(__file__).resolve().parent
    return Path(os.environ.get("COUNCILFLOW_DATA_DIR", str(_UNIX_DATA_DIR)))


def ensure_data_dir() -> Path:
    path = data_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_path(*parts: str) -> Path:
    return ensure_data_dir().joinpath(*parts)


def repo_path(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath(*parts)
