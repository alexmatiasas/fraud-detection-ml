# ── Builder stage: install only the API dependency set ────────────────────────
# The container ships ONLY the serving API. Training/runtime deps (xgboost,
# lightgbm, optuna, dvc, ...) are deliberately excluded — see the `api`
# optional dependency group in pyproject.toml.
#
# uv.lock is copied and --frozen used so the container installs exactly the
# versions resolved locally; uv never re-resolves the dependency graph.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN UV_COMPILE_BYTECODE=1 uv sync --extra api --no-dev --no-install-project --frozen

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Run as non-root: the API only reads files, so it needs no write access.
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home --home-dir /app appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Model-loading config, demo sample, and application code.
COPY configs/ configs/
COPY data/samples/ data/samples/
COPY src/ src/

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health/')"]

CMD ["uvicorn", "fdml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
