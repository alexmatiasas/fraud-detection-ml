# ── Builder stage: install only the API dependency set ────────────────────────
# The container ships ONLY the serving API. Training-only deps (xgboost, dvc,
# optuna, shap, ...) live in the `train` group and are deliberately excluded —
# see pyproject.toml. scikit-learn + lightgbm ARE installed: the served artifact
# is a sklearn Pipeline wrapping an LGBMClassifier.
#
# uv.lock is copied and --frozen used so the container installs exactly the
# versions resolved locally; uv never re-resolves the dependency graph.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=8

WORKDIR /app

# Layer 1: dependencies only. The cache mount persists downloaded wheels across
# builds; the bind mounts mean pyproject.toml/uv.lock changes do not bake the
# project source into this layer, so it is reused on every rebuild.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --extra api --no-dev --no-install-project --frozen

# Layer 2: project source + install the package itself into .venv.
# Only the files uv needs are copied, so unrelated edits (Makefile, CI, ...)
# do not invalidate this layer's cache. .dockerignore keeps .venv and heavy
# assets out of the context, so this COPY does not clobber layer 1's .venv.
COPY pyproject.toml uv.lock ./
COPY src/ src/
COPY configs/ configs/
COPY data/samples/ data/samples/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra api --no-dev --frozen

# Layer 3: strip the venv. Compiled bytecode, tests, and type stubs are not
# needed at runtime; removing them keeps the image lean without touching code.
RUN find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /app/.venv -name "*.pyc" -delete 2>/dev/null; \
    find /app/.venv -name "*.pyo" -delete 2>/dev/null; \
    find /app/.venv -name "*.pyi" -delete 2>/dev/null; \
    find /app/.venv -type d \( -name "tests" -o -name "test" \) -exec rm -rf {} + 2>/dev/null; \
    true

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Run as non-root: the API only reads files, so it needs no write access.
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home --home-dir /app appuser

# libgomp1 (OpenMP) is required by lightgbm at runtime; python:3.11-slim does
# not ship it. scikit-learn bundles its own copy, but the loader searches the
# system paths for libgomp.so.1 when importing lightgbm.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Single copy: .venv, configs, src, and the demo sample, all chowned once.
COPY --from=builder --chown=appuser:appuser /app /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health/')"]

# Cloud Run injects $PORT (default 8080) and routes all traffic to it, so
# uvicorn must bind there — the shell form expands the env var. Locally
# ($PORT unset) it falls back to 8000, matching `make docker-run`.
CMD ["sh", "-c", "uvicorn fdml.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
