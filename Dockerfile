# Attleboro Council Agent – Cloud Run
FROM python:3.12-slim

# System deps for PDF/OCR (tesseract, poppler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY docker_secrets.py ./
COPY config.yaml ./
RUN mkdir -p .streamlit
COPY .streamlit/config.toml .streamlit/

# Secrets are provided at runtime via env (docker_secrets.py) or mount.
# config.yaml must exist; override via mount or custom build if needed.

ENV PORT=8080
EXPOSE 8080

CMD ["python", "docker_secrets.py"]
