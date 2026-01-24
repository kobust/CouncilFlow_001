"""
Generate .streamlit/secrets.toml from env vars and run Streamlit.
Use in Docker/Cloud Run: set GEMINI_API_KEY and GCP_SERVICE_ACCOUNT_JSON (full JSON string).
"""
from __future__ import annotations

import json
import os
import sys

SECRETS_DIR = os.path.join(os.path.dirname(__file__), ".streamlit")
SECRETS_PATH = os.path.join(SECRETS_DIR, "secrets.toml")


def _toml_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def main() -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    sa_json = os.environ.get("GCP_SERVICE_ACCOUNT_JSON", "").strip()
    if not api_key:
        print("GEMINI_API_KEY env var is required", file=sys.stderr)
        sys.exit(1)
    if not sa_json:
        print("GCP_SERVICE_ACCOUNT_JSON env var is required", file=sys.stderr)
        sys.exit(1)

    try:
        sa = json.loads(sa_json)
    except json.JSONDecodeError as e:
        print(f"GCP_SERVICE_ACCOUNT_JSON invalid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(SECRETS_DIR, exist_ok=True)
    pk = (sa.get("private_key") or "").strip()
    lines = [
        'GEMINI_API_KEY = "%s"' % _toml_escape(api_key),
        "",
        "[gcp_service_account]",
        'type = "%s"' % _toml_escape(sa.get("type") or "service_account"),
        'project_id = "%s"' % _toml_escape(sa.get("project_id") or ""),
        'private_key_id = "%s"' % _toml_escape(sa.get("private_key_id") or ""),
        'private_key = """',
        pk,
        '"""',
        'client_email = "%s"' % _toml_escape(sa.get("client_email") or ""),
        'client_id = "%s"' % _toml_escape(str(sa.get("client_id") or "")),
        'auth_uri = "%s"' % _toml_escape(sa.get("auth_uri") or "https://accounts.google.com/o/oauth2/auth"),
        'token_uri = "%s"' % _toml_escape(sa.get("token_uri") or "https://oauth2.googleapis.com/token"),
        'auth_provider_x509_cert_url = "%s"' % _toml_escape(
            sa.get("auth_provider_x509_cert_url") or "https://www.googleapis.com/oauth2/v1/certs"
        ),
        'client_x509_cert_url = "%s"' % _toml_escape(sa.get("client_x509_cert_url") or ""),
        'universe_domain = "%s"' % _toml_escape(sa.get("universe_domain") or "googleapis.com"),
    ]
    with open(SECRETS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    port = os.environ.get("PORT", "8080")
    addr = os.environ.get("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    app = os.path.join(os.path.dirname(__file__), "app.py")
    os.execvp(
        "streamlit",
        [
            "streamlit", "run", app,
            "--server.port", port,
            "--server.address", addr,
            "--server.headless", "true",
        ],
    )


if __name__ == "__main__":
    main()
