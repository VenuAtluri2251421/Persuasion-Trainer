# HuggingFace Docker Space — Persuasion Trainer Environment
# Builds the FastAPI OpenEnv server directly from the repo root.

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies (copy manifests first for layer caching) ─────
COPY pyproject.toml ./
COPY server/requirements.txt ./server/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "openenv-core[core]>=0.2.1" \
        "torch>=2.0.0" \
        "groq>=0.11.0" \
        "openai>=1.12.0" \
        "pydantic>=2.0.0" \
        "fastapi>=0.115.0" \
        "uvicorn[standard]>=0.24.0"

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Expose port (matches app_port in README frontmatter) ─────────────────────
EXPOSE 8000

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
