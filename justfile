# Fraud-detection MLOps toolkit. Run `just` to list every recipe by group.
set default-list := true

# ── Setup ─────────────────────────────────────────────────────────────────────

# Full installation: environment and git hooks
[group('setup')]
setup: install hooks

# Install dependencies (dev group + train extra)
[group('setup')]
install:
    uv sync --extra train

# Install prek hooks (pre-commit + commit-msg)
[group('setup')]
hooks:
    uv run prek install --hook-type commit-msg --hook-type pre-commit

# ── Code quality ──────────────────────────────────────────────────────────────

# Lint with ruff
[group('quality')]
lint:
    uv run ruff check src/ tests/

# Format code with ruff (writes)
[group('quality')]
format:
    uv run ruff format src/ tests/

# Verify formatting without writing
[group('quality')]
fmt-check:
    uv run ruff format --check src/ tests/

# Type-check with basedpyright
[group('quality')]
typecheck:
    uv run basedpyright src/

# Gate after every build step: lint + format check + type check
[group('quality')]
check: lint fmt-check typecheck

# ── Tests ─────────────────────────────────────────────────────────────────────

# Fast tests, no coverage
[group('test')]
test:
    uv run python -m pytest --no-cov

# Tests with coverage gate (CI parity, fails under 80%)
[group('test')]
test-cov:
    uv run python -m pytest --cov=src/fdml --cov-report=term-missing --cov-fail-under=80

# ── Data ──────────────────────────────────────────────────────────────────────

# Download the dataset from Kaggle (needs a Kaggle API key)
[group('data')]
data-download:
    @echo "Downloading dataset IEEE-CIS Fraud Detection..."
    kaggle competitions download -c ieee-fraud-detection -p data/raw
    unzip -o data/raw/ieee-fraud-detection.zip -d data/raw
    rm data/raw/ieee-fraud-detection.zip
    @echo "Ready dataset in data/raw"

# Transform CSV files to optimized Parquet
[group('data')]
data-convert:
    uv run python scripts/prepare_data.py

# Generate JSON Schema from the Pydantic models
[group('data')]
schemas:
    uv run python -m fdml.schemas.generate

# ── DVC pipeline ──────────────────────────────────────────────────────────────

# Run the full DVC pipeline
[group('pipeline')]
dvc-repro:
    uv run dvc repro

# Run an experiment (compare with `just dvc-exp-show`)
[group('pipeline')]
dvc-exp-run:
    uv run dvc exp run

# Show the comparison table between experiments
[group('pipeline')]
dvc-exp-show:
    uv run dvc exp show

# Compare metrics between experiments
[group('pipeline')]
dvc-metrics-diff:
    uv run dvc metrics diff

# Compare plots between experiments (opens HTML)
[group('pipeline')]
dvc-plots-diff:
    uv run dvc plots diff

# ── Pipeline (direct, without DVC) ────────────────────────────────────────────

# Train a model, e.g. `just train model.name=xgboost optuna.enabled=true`
[group('pipeline')]
train *ARGS:
    uv run python -m fdml.models.train {{ARGS}}

# Feature ablation study (leave-one-group-out)
[group('pipeline')]
ablation *ARGS:
    uv run python -m fdml.models.train.ablation {{ARGS}}

# Full evaluation (metrics, plots, model card)
[group('pipeline')]
evaluate:
    uv run python -m fdml.models.evaluate

# ── API ───────────────────────────────────────────────────────────────────────

# Generate a stratified sample from the validation fold, e.g. `just sample --size 5000`
[group('api')]
sample *ARGS:
    uv run python scripts/generate_sample.py {{ARGS}}

# Run the API locally with reload
[group('api')]
serve-dev:
    uv run fastapi dev

# Run the API in production mode
[group('api')]
serve-prod:
    uv run uvicorn fdml.api.main:app --host 0.0.0.0 --port 8000 --workers 2

# ── MLflow ────────────────────────────────────────────────────────────────────

# Open the MLflow UI
[group('mlflow')]
mlflow-ui:
    uv run mlflow ui --port 5000

# ── Docker ────────────────────────────────────────────────────────────────────

# Build the API-only image (fdml-api)
[group('docker')]
docker-build:
    docker build -t fdml-api:latest .

# Run the API container locally (needs DAGSHUB_*/FDML_API_KEY env)
[group('docker')]
docker-run:
    docker run --env-file .env -p 8000:8000 fdml-api:latest

# ── Cleaning ──────────────────────────────────────────────────────────────────

# Remove caches and temporary files
[group('clean')]
clean:
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type d -name .pytest_cache -exec rm -rf {} +
    find . -type d -name .ruff_cache -exec rm -rf {} +
    find . -name "*.pyc" -delete
    @echo "Finished cleaning"

# Remove processed data (keeps raw)
[group('clean')]
clean-data:
    rm -rf data/processed/*
    touch data/processed/.gitkeep
