# HuggingFace Docker Space — Persuasion Trainer Environment

FROM python:3.11-slim

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
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

# ── Copy source & install package (enables relative imports) ──────────────────
COPY . .
RUN pip install --no-cache-dir -e .

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# ── Run via fully-qualified package path so relative imports resolve ──────────
CMD ["uvicorn", "persuasion_trainer.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
