# ── Builder stage: install only the API dependency set ────────────────────────
# The container ships ONLY the serving API. Training/runtime deps (xgboost,
# lightgbm, optuna, dvc, ...) are deliberately excluded — see the `api`
# optional dependency group in pyproject.toml.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra api --no-dev --no-install-project

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Model-loading config, demo sample, and application code.
COPY configs/ configs/
COPY data/samples/ data/samples/
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "fdml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
