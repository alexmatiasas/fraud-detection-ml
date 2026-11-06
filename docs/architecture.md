# Architecture

How the pieces fit. Code is the source of truth; this document explains the
contracts that are hard to see from a single file.

## Package layout

Python package is **`fdml`** (src layout, hatchling). Everything runs as a module
(`python -m fdml.models.train`), never as a bare script path.

```
src/fdml/
  features/      feature transformers + pipeline factory (sklearn Pipeline)
  models/
    train/       model builders, temporal split, Optuna HPO, MLflow registry
    evaluate/    metrics, plots, drift, stability, segments, model card
  api/           FastAPI app
    internal/    model loader, SHAP explain, registry helpers
    routers/     health, ready, metrics, transactions, features, models, predict
  schemas/       Pydantic config models + JSON Schema generator
  utils/         logging, project-root discovery
```

## The serving contract

The artifact logged to MLflow is a single sklearn estimator:

```python
Pipeline([("features", <feature pipeline>), ("model", model)])
```

It is served on **raw transactions**, not on featurized rows. Two consequences:

- Every new transform must live *inside* the sklearn pipeline. A transform that
  runs outside it (in a script, in a notebook, in the API layer) cannot be
  reproduced at serving time.
- `CategoryEncoder` must stay the **last** feature step: the model is fitted on
  ordinal codes, so the encoding has to run at scoring time too.

This is guarded by
`tests/integration/test_train.py::test_serving_pipeline_is_self_contained`, so a
regression fails CI rather than production.

## Model resolution

`src/fdml/api/internal/loader.py` resolves the served model lazily, in order:

1. MLflow registry alias **`champion`**.
2. Fallback: newest version with `READY` status.
3. Fallback: local `models/pipeline.joblib`.

If none loads, `/v1/predict` and `/v1/ready` return **503** until
`PUT /v1/model/reload` succeeds. `/v1/model/switch` hot-swaps to a specific
registry version and requires the `X-API-Key` header.

All algorithms are registered under a single registry name
(`fraud-detection-lgbm`, see `configs/mlflow.yaml`); the version number
distinguishes runs.

## Config flow

```
configs/*.yaml  →  OmegaConf (interpolation, dot-access)  →  Pydantic model  →  code
        │                                                        │
        └── header `# $schema: ../schemas/*.schema.json`          └── src/fdml/schemas/
```

- Each YAML carries a `$schema` header pointing at a generated JSON Schema.
- The Pydantic models in `src/fdml/schemas/` are the validation contract; run
  `just schemas` (`python -m fdml.schemas.generate`) after changing one.
- Config files are layered: `data.yaml`, `features.yaml`, `train.yaml`,
  `evaluate.yaml`, `mlflow.yaml`.

## MLflow and DagsHub

`resolve_mlflow_tracking` picks the backend at runtime:

- `DAGSHUB_*` env vars (or a `.env` with them) → DagsHub remote tracking + registry.
- otherwise → local SQLite (`sqlite:///mlruns.db`), which also supports the Model
  Registry.

The access token is deliberately never embedded in the tracking URI.

Registry flow (see the diagram in the README): a run is first checked against a
quality gate (`min_auc`, `min_average_precision`); rejected versions are tagged
`validation_status=rejected` and never registered. Accepted versions are compared
against the current `champion` by AUC — better wins `champion`, otherwise the new
version becomes `challenger`.

## API cross-cutting concerns

- **Rate limiting** — slowapi, configured in `api/limiter.py`.
- **Auth** — `X-API-Key` on `/v1/model/*` mutating routes (`api/dependencies.py`).
- **CORS** — restricted to the frontend origin in production (`CORS_ORIGINS`).
- **Logging** — structured JSON logs (`utils/logging.py`).
- **Errors** — Sentry when `SENTRY_DSN` is set.
- **Explanations** — per-prediction SHAP values via `api/internal/explain.py`.

## Data and reproducibility

- `data/raw` and `data/processed` are DVC-managed and gitignored.
- `data/samples/val_sample.parquet` (the API demo sample) **is** committed — the
  API needs it to score raw rows without a `dvc pull`.
- The project root is discovered by walking up to `pyproject.toml`/`dvc.yaml`/`.git`;
  override with `PROJECT_ROOT` (`src/fdml/utils/paths.py`).
- The dataset is static: there is no Airflow / incremental pipeline.

## Tooling

- `justfile` groups recipes (`just --list`). `just check` = lint + format check +
  basedpyright; `just test` = fast tests; `just test-cov` = 80% coverage gate.
- `pytest` collection is scoped to `tests/` (`testpaths` in `pyproject.toml`).
- Ruff lint/format scope is `src/ tests/` only — the same scope CI uses.
