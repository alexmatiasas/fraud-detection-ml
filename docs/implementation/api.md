# Implementation — serving API

The FastAPI service that scores raw transactions with the promoted pipeline.

Source: `src/fdml/api/`, `src/fdml/api/metadata.py`, `Dockerfile`,
`.github/workflows/cd.yml`.

## App layout

```
src/fdml/api/
  main.py             FastAPI app, lifespan, CORS, Sentry, request logging
  dependencies.py     singletons for ModelLoader / MultiModelLoader, API-key guard
  limiter.py          shared slowapi limiter + 429 handler
  metadata.py         title/version/tags, API_PREFIX = /v1
  routers/            health, ready, metrics, transactions, features, model,
                      models, predict
  schemas/            Pydantic request/response models
  internal/
    loader.py         single-model resolution, demo sample, prediction
    registry.py       multi-model discovery from the MLflow registry
    explain.py        raw-feature overrides + SHAP explanations
    metrics.py        metric summary for /v1/model/info
```

At startup (`lifespan`) the app builds the loaders, loads the default model
(MLflow → local), loads the demo sample, and loads the multi-model registry.
Failures are logged and leave the service up; prediction/ready return 503 until
a model is available.

## Model resolution

Single model (`ModelLoader`, default name `fraud-detection-lgbm`):

1. MLflow registry alias `champion` (`models:/<name>@champion`).
2. Newest non-deleted `READY` version.
3. Local `models/pipeline.joblib`.

The evaluation report is read from `models/report.json` when available. If no
source loads, `is_loaded` stays false and guarded endpoints return 503 until
`PUT /v1/model/reload` succeeds.

Multi-model (`MultiModelLoader`) resolves each registered algorithm
(`fraud-detection-lgbm`, `-xgboost`, `-rf`) for the comparison endpoints. It
uses each version's logged report for its threshold and metrics.

## Endpoints

All under `/v1`. Limits are per client (slowapi, 1-minute window).

| Method | Path | Limit | Auth | Purpose |
|---|---|---|---:|---|
| GET | `/v1/health/` | — | — | Liveness |
| GET | `/v1/ready/` | — | — | Readiness; 503 until a model is loaded |
| GET | `/v1/metrics` | 120/min | — | Offline evaluation metrics |
| GET | `/v1/transactions` | 60/min | — | Demo-sample transactions |
| GET | `/v1/features` | 30/min | — | Feature metadata for overrides |
| GET | `/v1/models` | 120/min | — | Registered-model leaderboard |
| GET | `/v1/models/{name}` | 120/min | — | One registered model |
| GET | `/v1/model/info` | 120/min | — | Served model, version, source, metrics |
| POST | `/v1/predict/` | 60/min | — | Score one transaction (overrides + SHAP) |
| POST | `/v1/predict/batch` | 20/min | — | Score up to 50 (no overrides/SHAP) |
| POST | `/v1/predict/compare` | 30/min | — | Score across registered models |
| POST | `/v1/predict/{name}` | 60/min | — | Score with a specific model |
| PUT | `/v1/model/reload` | 10/min | `X-API-Key` | Reload from configured sources |
| PUT | `/v1/model/switch` | 10/min | `X-API-Key` | Hot-swap to a registry version |

`/v1/model/info` sets `Cache-Control: public, max-age=120`.

## Scoring flow

`POST /v1/predict/` (`routers/predict.py`):

1. Look up `transaction_id` in the demo sample; 404 if absent.
2. Apply optional raw-feature overrides; unknown columns or bad casts → 422.
3. Drop `TransactionID`/`isFraud` and score the **raw** row through the full
   pipeline.
4. Read `best_threshold` from the evaluation report (0.5 fallback).
5. Compute `is_fraud = probability >= threshold`,
   `confidence = |probability - threshold|`, and `risk_level`.
6. Compute SHAP values (top 20) when the model supports them.
7. Audit-log the decision and return the raw row, overrides and explanation.

`risk_level` is decision **uncertainty**, not fraud severity: `critical` when
the probability is within 0.15 of the threshold, then `high` (<0.30), `medium`
(<0.45), `low` otherwise.

### Overrides

`overrides` is a map of raw feature name → value (e.g.
`{"TransactionAmt": 500.0}`). Values are cast to the column dtype; unknown
features raise `OverrideError` → 422. This is what powers the "what if" demo.
Batch predictions reject overrides to keep the response cheap.

### Explanations

`compute_explanation` runs SHAP on the featurized row and maps the top features
back to human-readable names. The response includes `base_value`, `n_features`
and `top_features` (`feature`, `value`, `shap`). If the model or explainer is
unavailable, the field is null rather than failing the prediction.

## Errors

| Status | When |
|---|---|
| 404 | Unknown transaction ID or model name |
| 422 | Invalid override column/value |
| 429 | Rate limit exceeded; `Retry-After: 60` |
| 503 | No model loaded, registry unavailable, or `FDML_API_KEY` not configured |

`ModelLoadError` has a dedicated handler returning 503 with the reason.

## Security and configuration

| Control | Behavior |
|---|---|
| Mutating endpoints | `X-API-Key` compared with `hmac.compare_digest` against `FDML_API_KEY` |
| Unset API key | Mutating endpoints return 503 (fail closed) |
| Swagger/Redoc/OpenAPI | Disabled when `ENVIRONMENT=production` |
| CORS | Explicit allowlist; production defaults to the Vercel frontend |
| Rate limiting | Per-endpoint, in-memory slowapi storage |
| Request logging | JSON line with `request_id`, method, path, status, latency; echoed as `X-Request-ID` |
| Audit log | Every scored prediction (ID, probability, decision, model version, override/SHAP flags) |
| Sentry | Enabled when `SENTRY_DSN` is set (traces 10%, profiling 100%) |

Environment: `DAGSHUB_USERNAME`/`DAGSHUB_TOKEN`/`DAGSHUB_REPO`,
`FDML_API_KEY`, `SENTRY_DSN`, `CORS_ORIGINS`, `ENVIRONMENT`, `PORT` (Cloud Run).

## Deployment

- Multi-stage `Dockerfile`: builder installs only the serving dependency set
  (`uv sync --no-dev --frozen`), runtime is `python:3.11-slim` with `libgomp1`,
  runs as non-root, exposes 8000, binds `${PORT:-8000}`, and healthchecks
  `/v1/health/`.
- Training-only packages (xgboost, dvc, optuna) are excluded; scikit-learn,
  LightGBM and SHAP are installed because the artifact and explanations need
  them.
- CD (`.github/workflows/cd.yml`) builds/pushes the image to GHCR on tags
  `v*`, deploys to Cloud Run, then runs `scripts/verify_deploy.sh`.
- The smoke test checks health, readiness, a sample prediction and model info.
  Note: it sends `include_shap`, which is not in `PredictionRequest` and is
  silently ignored (Pydantic default is to ignore extra fields); explanations
  are returned regardless.

## Known gaps

- **Rate limiting behind a proxy**: `get_remote_address` sees the proxy IP in
  Cloud Run, so the limit is effectively global, not per user. In-memory storage
  is also per replica, so limits are not shared across instances.
- **Hot-swap is not locked**: `ModelLoader` mutates cached fields without a
  lock. A concurrent predict during `reload`/`switch` can observe mixed state,
  and a failed MLflow load can leave `source`/`version` newer than the cached
  pipeline.
- **Threshold fallback**: if `models/report.json` is missing or from another
  model, predictions fall back to 0.5, silently changing decisions.
- **Demo-only predictions**: the endpoint scores IDs present in
  `data/samples/val_sample.parquet`; it is not a general transaction-scoring
  endpoint.
- **Default model name is hard-coded** to `fraud-detection-lgbm`; serving a
  different algorithm's champion requires the per-model endpoints or a registry
  alias change.
