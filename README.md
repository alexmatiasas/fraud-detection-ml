# Fraud Detection ML

End-to-end MLOps pipeline for [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Kaggle 2019): data versioning, feature engineering, experiment tracking, model training and evaluation, and a containerized serving API.

[![CI](https://github.com/alexmatiasas/fraud-detection-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/alexmatiasas/fraud-detection-ml/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)

> **Deployment status**: the serving API is live on [Cloud Run](https://fdml-api-bxpgqqidaq-uc.a.run.app). The frontend demo is at [alexmatias.vercel.app](https://alexmatias.vercel.app).

---

## About

Fraud detection on transactional data is a classic imbalanced-classification problem: only **3.4%** of transactions are fraudulent, so accuracy is meaningless. The project focuses on:

- **Reproducible data science** — DVC for data/artifact versioning, MLflow + DagsHub for experiment tracking and model registry.
- **EDA-driven feature engineering** — a 341-feature space built from 17 purpose-built transformers, each documented and tested.
- **Production-ready serving** — a versioned FastAPI service with rate limiting, API-key auth for model management, structured JSON logs, and a multi-stage Docker image running as a non-root user.

The project is written to portfolio standards: every decision is documented (`configs/*.yaml`, notebooks, model card), and the suite of **499 tests** covers transformers, training, evaluation, and API behavior.

## Current model

The served model is an **XGBoost** classifier evaluated on a temporal hold-out (test set is the last 20% of the transaction timeline, no shuffle).

| Metric | Value |
|---|---|
| ROC AUC | **0.9145** |
| Average Precision | **0.5484** (95% CI [0.5318, 0.5650]) |
| F1 @ optimal threshold (0.65) | 0.5406 |
| Precision / Recall | 0.4693 / 0.5689 |
| Brier score | 0.0322 |
| Expected cost / transaction | 0.1619 @ thr 0.32 |
| Recall@top 1% / 5% | 0.2571 / 0.6021 |

The full model card with top features and evaluation settings is at `models/model_card.md`.

## Architecture

```mermaid
flowchart LR
    subgraph Data
        A[Kaggle IEEE-CIS] --> B[DVC · data/raw]
        B --> C[prepare_data.py]
        C --> D[DVC · data/processed]
    end

    subgraph Training
        D --> E[Feature pipeline · 17 transformers]
        E --> F[LightGBM / XGBoost / RF]
        F --> G[Optuna HPO]
        G --> H[models/pipeline.joblib]
    end

    subgraph Tracking
        H --> I[(MLflow + DagsHub)]
        D --> I
    end

    subgraph Serving
        H --> J[Docker image]
        J --> K[FastAPI · /v1]
        L[(Demo sample)] --> K
        K --> M[Cloud Run]
    end
```

## Feature engineering

All feature decisions come from the [EDA in R](notebooks/01_eda_r/01_EDA_in_R.Rmd). Highlights:

- **Time features** — cyclic encoding of hour, day of week; D1–D15 timedelta columns kept only where they carry signal (D6–D9, D12–D14 dropped at 87–94% missing).
- **Identity signals** — `has_identity` binary flag; the ~76% structurally-missing identity block is treated explicitly.
- **Categorical encoding** — product/card/email-domain categories with ordinal and frequency encodings, plus a transactional `card1` frequency feature.
- **V-feature curation** — 0.01 variance filter + 0.95 correlation filter (see `configs/features.yaml`).
- **Missing-value convention** — numerical missing → `-999` (LightGBM sentinel), categorical missing → `"missing"`, M flags → `{T:1, F:0, NaN:-1}`.
- **Class imbalance** — `scale_pos_weight = 27.6`; the decision threshold is tuned on F1, and evaluation uses **Average Precision**, never accuracy.

## Tech stack

| Layer | Technology |
|---|---|
| Data versioning | DVC + DagsHub storage |
| Experiment tracking / registry | MLflow + DagsHub |
| Feature pipeline | scikit-learn Pipeline + pandas |
| Training | LightGBM / XGBoost / RandomForest |
| HPO | Optuna |
| Serving API | FastAPI + uvicorn + slowapi |
| Container | Docker (multi-stage, non-root) |
| Deploy | Cloud Run (scale-to-zero) |
| Frontend | Astro + Tailwind CSS (Vercel) |
| Monitoring | Sentry |
| CI/CD | GitHub Actions (ruff, basedpyright, pytest) |
| EDA | R 4.4 · tidyverse · arrow |
| Testing | pytest + hypothesis |

## Repository structure

```
configs/                  # data/features/train/evaluate/mlflow YAML + schemas
data/                     # raw + processed parquet (DVC, not in git)
models/                   # pipeline.joblib, report.json, model card, plots
notebooks/                # R EDA + Quarto FE/training verification
scripts/                  # download_data, prepare_data, generate_sample
src/fdml/features/        # 17 feature transformers + pipeline factory
src/fdml/models/          # train (LGBM/XGB/RF), evaluate (metrics/plots/...)
src/fdml/api/             # FastAPI app: routers, schemas, model loader, limiter
tests/                    # 499 unit + integration tests
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
make install              # creates the environment (dev + train + api extras)
```

### Data

```bash
make data-download        # needs a Kaggle API key (~1.1M transactions)
make data-convert         # CSV -> optimized Parquet
```

### Train and evaluate

```bash
make train ARGS="model.name=xgboost"   # or lightgbm / random_forest
make evaluate                          # metrics, plots, report.json, model card
make sample                            # builds data/samples for the API demo
```

Run the full DVC pipeline instead:

```bash
dvc repro
```

### Run the API locally

```bash
make serve-dev                        # uvicorn with reload on :8000
```

Interactive docs at http://localhost:8000/docs (disabled in production).

## API

All routes are under `/v1`. Predictions are scored by `TransactionID` against a labeled demo sample so the raw row is available and the full feature pipeline runs server-side.

| Method | Path | Description | Auth |
|---|---|---|---|
| `GET` | `/v1/health` | Liveness probe | — |
| `GET` | `/v1/ready` | Readiness probe (model loaded) | — |
| `GET` | `/v1/metrics` | Offline evaluation metrics | — |
| `GET` | `/v1/transactions` | List demo-sample transactions | — |
| `GET` | `/v1/features` | Feature metadata for overrides | — |
| `GET` | `/v1/models` | All registered models (leaderboard) | — |
| `GET` | `/v1/models/{name}` | Single model detail | — |
| `GET` | `/v1/model/info` | Current served model + metrics | — |
| `POST` | `/v1/predict` | Score one transaction | — |
| `POST` | `/v1/predict/batch` | Score up to 50 transactions | — |
| `POST` | `/v1/predict/compare` | Score across all models | — |
| `POST` | `/v1/predict/{name}` | Score with a specific model | — |
| `PUT` | `/v1/model/reload` | Reload from source | `X-API-Key` |
| `PUT` | `/v1/model/switch` | Hot-swap registry version | `X-API-Key` |

```bash
curl -X POST https://fdml-api-bxpgqqidaq-uc.a.run.app/v1/predict/ \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": 3538759}'
```

```json
{
  "transaction_id": 3538759,
  "is_fraud": false,
  "probability": 0.2874,
  "threshold": 0.6534,
  "risk_level": "low",
  "confidence": 0.3660,
  "is_above_threshold": false,
  "model_version": "fraud-detection-lgbm:49",
  "raw": { "TransactionAmt": 24.90, "...": "432 fields" },
  "overrides_applied": null,
  "explanation": {
    "base_value": -2.13,
    "n_features": 341,
    "top_features": [
      { "feature": "TransactionAmt", "value": 24.90, "shap": 0.85 }
    ]
  }
}
```

### Docker

```bash
make docker-build         # multi-stage image, API-only dependency set
make docker-run           # needs DAGSHUB_* / FDML_API_KEY via .env
```

The image ships only the serving API: training-only packages (xgboost, dvc, optuna, ...) are excluded from the build. The `uv` layer uses cache mounts so dependency downloads persist across rebuilds.

### Environment variables

| Variable | Purpose |
|---|---|
| `DAGSHUB_USERNAME` / `DAGSHUB_TOKEN` / `DAGSHUB_REPO` | MLflow remote tracking/registry (or `.env`) |
| `FDML_API_KEY` | Guards `/v1/model/*` mutating endpoints |
| `SENTRY_DSN` | Error tracking (optional) |
| `CORS_ORIGINS` | Override allowed origins (default: Vercel only in prod) |
| `ENVIRONMENT` | `production` or `development` (default: development) |

## Notebooks

- [EDA in R](notebooks/01_eda_r/01_EDA_in_R.Rmd) — full dataset exploration that drives every feature decision.
- [Feature engineering verification](notebooks/02_feature_engineering/02_feature_engineering.qmd) — proves each transformer output against `configs/features.yaml`.
- [Training verification](notebooks/03_training/03_training_verification.qmd) — temporal split, MLflow tracking, registry, benchmarks.

## Roadmap

- [x] Model registry — register all trained models with `champion`/`challenger` aliases.
- [x] Deployment — build and push the image from CI to Cloud Run.
- [x] API hardening — rate limiting, API-key auth, CORS, Swagger disabled in production.
- [x] Frontend demo — interactive predict UI with what-if explorer and SHAP explanations.
- **Ensemble models** — weighted-average and stacking of LightGBM + XGBoost + RandomForest, with blend weights logged to MLflow.
- **Fairness analysis** — segment-level metrics across card brand, email domain, and device type.

## Contact

- Repository: [github.com/alexmatiasas/fraud-detection-ml](https://github.com/alexmatiasas/fraud-detection-ml)
- Personal site: [alexmatias.vercel.app](https://alexmatias.vercel.app)
