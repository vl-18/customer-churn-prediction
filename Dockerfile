# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps (needed for some ML wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder (keeps image lean)
COPY --from=builder /install /usr/local

# Copy application code
COPY configs/     configs/
COPY src/         src/
COPY api/         api/
COPY models/artifacts/  models/artifacts/

# Ensure log directories exist
RUN mkdir -p logs data/raw data/processed

# Non-root user for security
RUN useradd -m -u 1000 mluser && chown -R mluser:mluser /app
USER mluser

# Health check — matches our /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

# Use gunicorn + uvicorn workers for production (single worker for dev)
CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
