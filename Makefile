.PHONY: help setup install lint test train ablation serve docker-build clean

# Variables
PYTHON := uv run python
DATA_DIR := data/raw

## ── Ayuda ────────────────────────────────────────────────────────────────────
help:  ## Muestra este mensaje
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

## ── Setup ────────────────────────────────────────────────────────────────────
setup: install hooks  ## Instalación completa: entorno + hooks

install:  ## Instala dependencias con uv
	uv sync --extra dev

hooks:  ## Instala pre-commit hooks
	uv run prek install --hook-type commit-msg --hook-type pre-commit

## ── Calidad de código ────────────────────────────────────────────────────────
lint:  ## Linting con ruff
	uv run ruff check src/ tests/

format:  ## Formatea código con ruff
	uv run ruff format src/ tests/

## ── Tests ────────────────────────────────────────────────────────────────────
test:  ## Corre todos los tests con cobertura
	uv run python -m pytest

test-fast:  ## Tests sin reporte de cobertura (más rápido)
	uv run python -m pytest --no-cov

## ── Datos ────────────────────────────────────────────────────────────────────
data-download:  ## Descarga el dataset de Kaggle (requiere kaggle CLI configurado)
	@echo "Descargando dataset IEEE-CIS Fraud Detection..."
	kaggle competitions download -c ieee-fraud-detection -p $(DATA_DIR)
	unzip -o $(DATA_DIR)/ieee-fraud-detection.zip -d $(DATA_DIR)
	rm $(DATA_DIR)/ieee-fraud-detection.zip
	@echo "Dataset listo en $(DATA_DIR)"

data-convert:  ## Convierte CSVs a Parquet con tipos optimizados
	$(PYTHON) scripts/prepare_data.py

## ── Config schemas ────────────────────────────────────────────────────────────
schemas:  ## Genera JSON Schema desde modelos Pydantic
	$(PYTHON) -m fdml.schemas.generate

.PHONY: schemas

## ── DVC Pipeline ────────────────────────────────────────────────────────────
dvc-repro:  ## Corre el pipeline completo con DVC
	uv run dvc repro

dvc-exp-run:  ## Corre experimento DVC (comparable con dvc exp show)
	uv run dvc exp run

dvc-exp-show:  ## Muestra tabla comparativa de experimentos
	uv run dvc exp show

dvc-metrics-diff:  ## Compara métricas entre experimentos
	uv run dvc metrics diff

dvc-plots-diff:  ## Compara plots entre experimentos (abrir HTML)
	uv run dvc plots diff

## ── Pipeline (directo, sin DVC) ─────────────────────────────────────────────
train:  ## Entrena modelo (pasa args via ARGS: make train ARGS="model.name=xgboost")
	$(PYTHON) -m fdml.models.train $(ARGS)

ablation:  ## Feature ablation study (leave-one-group-out, ver make train ablation.enabled=true)
	$(PYTHON) -m fdml.models.train.ablation $(ARGS)

evaluate:  ## Evaluación completa (métricas, plots, model card)
	$(PYTHON) -m fdml.models.evaluate

## ── API ──────────────────────────────────────────────────────────────────────
serve:  ## Levanta la API localmente
	uv run uvicorn fdml.api.main:app --reload --port 8000

serve-prod:  ## Levanta la API en modo producción
	uv run uvicorn fdml.api.main:app --host 0.0.0.0 --port 8000 --workers 2

## ── MLflow ───────────────────────────────────────────────────────────────────
mlflow-ui:  ## Abre MLflow UI
	uv run mlflow ui --port 5000

## ── Docker ───────────────────────────────────────────────────────────────────
docker-build:  ## Construye imagen Docker (sin datos)
	docker build -t fdml:latest .

docker-run:  ## Corre el contenedor de la API
	docker run -p 8000:8000 fdml:latest

## ── Limpieza ─────────────────────────────────────────────────────────────────
clean:  ## Limpia caches y archivos temporales
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "Limpieza completada"

clean-data:  ## Elimina datos procesados (conserva raw)
	rm -rf data/processed/*
	touch data/processed/.gitkeep
