.PHONY: help setup install lint test train ablation serve-dev docker-build clean schemas

# Variables
PYTHON := uv run python
DATA_DIR := data/raw
RUFF_RUN := uv run ruff
DVC_RUN := uv run dvc

## ── Help ────────────────────────────────────────────────────────────────────
help:  ## Shows this message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

## ── Setup ────────────────────────────────────────────────────────────────────
setup: install hooks  ## Full installation: environment and hooks

install:  ## Install dependences with uv (dev + train + api extras)
	uv sync --extra dev --extra train --extra api

hooks:  ## Installs pre-commit hooks
	uv run prek install --hook-type commit-msg --hook-type pre-commit

## ── Code Quality ────────────────────────────────────────────────────────
lint:  ## Linting with ruff
	$(RUFF_RUN) check src/ tests/

format:  ## Formats code with ruff
	$(RUFF_RUN) format src/ tests/

## ── Tests ────────────────────────────────────────────────────────────────────
test:  ## Run all tests in coverage
	$(PYTHON) -m pytest

test-fast:  ## Tests without test converge (faster)
	$(PYTHON) -m pytest --no-cov

## ── Data ────────────────────────────────────────────────────────────────────
data-download:  ## Downloads dataset from kaggle (needs a Kaggle API key)
	@echo "Downloading dataset IEEE-CIS Fraud Detection..."
	kaggle competitions download -c ieee-fraud-detection -p $(DATA_DIR)
	unzip -o $(DATA_DIR)/ieee-fraud-detection.zip -d $(DATA_DIR)
	rm $(DATA_DIR)/ieee-fraud-detection.zip
	@echo "Ready dataset in $(DATA_DIR)"

data-convert:  ## Transforms CSV file to Parquet file with optimized types.
	$(PYTHON) scripts/prepare_data.py

## ── Config schemas ────────────────────────────────────────────────────────────
schemas:  ## Generates JSON Schema from Pydantic models
	$(PYTHON) -m fdml.schemas.generate

## ── DVC Pipeline ────────────────────────────────────────────────────────────
dvc-repro:  ## Runs full pipeline with DVC
	$(DVC_RUN) repro

dvc-exp-run:  ## Runs expetiment in DVC (comparable with `dvc exp show`)
	$(DVC_RUN) exp run

dvc-exp-show:  ## Shows comparative table between experiments
	$(DVC_RUN) exp show

dvc-metrics-diff:  ## Compare metrics between experiments
	$(DVC_RUN) metrics diff

dvc-plots-diff:  ## Compare plots between experiments (open HTML)
	$(DVC_RUN) plots diff

## ── Pipeline (direct, without DVC) ─────────────────────────────────────────────
train:  ## Trains model (pass args through ARGS: make train ARGS="model.name=xgboost")
	$(PYTHON) -m fdml.models.train $(ARGS)

ablation:  ## Feature ablation study (leave-one-group-out, see make train ablation.enabled=true)
	$(PYTHON) -m fdml.models.train.ablation $(ARGS)

evaluate:  ## Full evaluation (metrics, plots, model card)
	$(PYTHON) -m fdml.models.evaluate

## ── API ──────────────────────────────────────────────────────────────────────
sample:  ## Generates a stratified sample from the validation fold for demos (ARGS="--size 5000")
	$(PYTHON) scripts/generate_sample.py $(ARGS)

serve-dev:  ## Sets up the API locally
	uv run fastapi dev

serve-prod:  ## Sets up the API in production mode
	uv run uvicorn fdml.api.main:app --host 0.0.0.0 --port 8000 --workers 2

## ── MLflow ───────────────────────────────────────────────────────────────────
mlflow-ui:  ## Opens MLflow UI
	uv run mlflow ui --port 5000

## ── Docker ───────────────────────────────────────────────────────────────────
docker-build:  ## Builds the API-only Docker image (fdml-api)
	docker build -t fdml-api:latest .

docker-run:  ## Runs the API container locally (needs DAGSHUB_*/FDML_API_KEY env)
	docker run --env-file .env -p 8000:8000 fdml-api:latest

## ── Cleaning ─────────────────────────────────────────────────────────────────
clean:  ## Cleans caches and temporal files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "Finished cleaning"

clean-data:  ## Removes processed data (preserves raw)
	rm -rf data/processed/*
	touch data/processed/.gitkeep
