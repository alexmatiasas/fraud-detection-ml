# Project map — start here

One-page mental model of the repo. Read this before any other document.
If this file and the code disagree, the code wins; fix this file.

## What this is

Binary classifier for IEEE-CIS fraud detection (~590k transactions, ~3.5%
fraud). Training compares LightGBM (default), XGBoost and RandomForest on a
**temporal** split with embargo. The promoted model is a single sklearn
`Pipeline([("features", …), ("model", …)])` that scores **raw transactions**,
served by FastAPI and resolved through the MLflow registry alias `champion`.

## The whole flow

```
Kaggle CSVs
  │ scripts/download_data.py            DVC stage "download" (frozen)
  ▼
data/raw/*.csv
  │ scripts/prepare_data.py             DVC stage "prepare": dtype downcast, tx ⋈ identity
  ▼
data/processed/*.parquet
  │ fdml.features.factory.load_data + merge_tables
  ▼
one dataframe (transaction + identity columns)
  │ fdml.models.split.TemporalSplitter  time-ordered split + embargo
  ▼
train fold ──────────────► val fold
  │ create_pipeline()        │ transform() only (never refit on val)
  ▼                          ▼
featurized X (~341 cols) ──► featurized X
  │ fdml.models.train.model_builder (LGBM / XGB / RF)
  ▼
fitted estimator
  │ fdml.models.evaluate.runner.evaluate
  ├─► models/report.json, model_card.md, plots
  │ fdml.models.train.registry
  └─► MLflow alias "champion"
  │
  ▼
Pipeline([features, model]) ──► models/pipeline.joblib
  │ fdml.api.internal.loader  (registry champion → local joblib fallback)
  ▼
FastAPI ──► POST /v1/predict  (raw transaction JSON in, probability out)
```

## Modules: what each one does

### Configuration and contracts

| Path | Responsibility |
|---|---|
| `configs/*.yaml` | The real knobs: data files, feature toggles, model params, split, HPO, evaluation, MLflow. **Source of truth.** |
| `src/fdml/schemas/*.py` | Pydantic models that validate each YAML. Change schema → `just schemas`. |
| `src/fdml/config.py`, `src/fdml/models/config.py` | Load YAML with OmegaConf; resolve MLflow tracking URI (DagsHub vs local SQLite). |
| `src/fdml/utils/paths.py` | Project-root discovery (`pyproject.toml` / `dvc.yaml` / `.git`), config paths. |
| `src/fdml/utils/logging.py` | Logging setup and the `step(...)` phase banner used by training. |

### Data and features

| Path | Responsibility |
|---|---|
| `scripts/download_data.py` | Kaggle download (DVC `download`, frozen). |
| `scripts/prepare_data.py` | CSV → Parquet, join transaction + identity, dtype reduction. |
| `src/fdml/features/factory.py` | **The FE contract.** `create_pipeline()` defines step order; `load_data`/`merge_tables` read parquet. |
| `src/fdml/features/base.py` | Shared transformer base/interface. |
| `src/fdml/features/{identity,email,freq,device,flags,time,id_encoder,card,amount,vfilter,imputer,dtypes,selector,category_encoder}.py` | One transform each, composed in `create_pipeline()`. |

Pipeline order (simplified): identity → email → freq → device → M-flags →
time → ID codes → card aggregates → amount → V filter → C filter → imputer →
dtypes → selector → **CategoryEncoder (always last)**.

Why the order matters: `freq` must run before `device` (it encodes
`DeviceInfo`, which `device` drops); `CategoryEncoder` must be last so the
logged pipeline can consume raw strings at serving time.

### Training

| Path | Responsibility |
|---|---|
| `src/fdml/models/split.py` | `TemporalSplitter` (default), `StratifiedSplitter`, `StratifiedKFoldSplitter`. |
| `src/fdml/models/train/data_prep.py` | Load → merge → split → fit FE on train only → transform val; optional train-fold cap. |
| `src/fdml/models/train/runner.py` | **Orchestrator.** `train()` runs phases 1–6: prepare, build, fit, evaluate, track, save. |
| `src/fdml/models/train/model_builder.py` | Per-algorithm builders and their parameter spaces. |
| `src/fdml/models/train/fitting.py` | Fit loop, early stopping, metric history. |
| `src/fdml/models/train/callbacks.py` | Per-iteration validation curves / drift callbacks. |
| `src/fdml/models/train/hpo.py` | Optuna search strategies (opt-in via `optuna.enabled=true`). |
| `src/fdml/models/train/ablation.py` | Leave-one-group-out feature study, one MLflow run per variant × seed. |
| `src/fdml/models/train/registry.py` | Quality gate and MLflow alias promotion. |
| `src/fdml/models/train/tracking.py`, `artifacts.py` | MLflow tags/params/metrics; native MLflow evaluation. |

### Evaluation

| Path | Responsibility |
|---|---|
| `src/fdml/models/evaluate/runner.py` | **`evaluate()`**: metrics, thresholds, segments, plots, drift, stability, reporters; also the standalone `just evaluate` entry point. |
| `src/fdml/models/evaluate/metrics.py` | ROC AUC, AP, F-beta, Brier, expected cost, recall@k, bootstrap CI, threshold tuning. |
| `src/fdml/models/evaluate/segments.py` | Per-segment metrics (`ProductCD`, `card4`, …). |
| `src/fdml/models/evaluate/drift.py` | Adversarial validation train-vs-val. |
| `src/fdml/models/evaluate/stability.py` | Learning curves and feature-importance stability. |
| `src/fdml/models/evaluate/plots.py` | ROC, PR, calibration, error-analysis plots. |
| `src/fdml/models/evaluate/reporter.py` | Console / JSON / logging / DVCLive / MLflow reporters. |
| `src/fdml/models/evaluate/model_card.py` | Generates `models/model_card.md`. |

### API and serving

| Path | Responsibility |
|---|---|
| `src/fdml/api/main.py` | App assembly: lifespan, middleware, routers, root endpoint. |
| `src/fdml/api/internal/loader.py` | **Serving core.** Model resolution (`champion` → newest READY → local joblib), prediction, report cache. |
| `src/fdml/api/internal/registry.py` | MLflow registry client and report download. |
| `src/fdml/api/internal/explain.py` | SHAP-based explanations. |
| `src/fdml/api/internal/metrics.py` | In-process request/latency metrics. |
| `src/fdml/api/routers/*.py` | Endpoints: health, ready, predict, model, models, metrics, features. |
| `src/fdml/api/schemas/*.py` | Request/response models (Pydantic). |
| `src/fdml/api/dependencies.py`, `limiter.py`, `context.py` | Auth dependency, rate limiting, request context. |

### Scripts, tests, docs

| Path | Responsibility |
|---|---|
| `scripts/generate_sample.py` | Stratified validation sample committed at `data/samples/val_sample.parquet` (API demo). |
| `scripts/verify_deploy.sh` | Post-deploy smoke test used by CD. |
| `scripts/analyze_runs.py`, `compare_runs.py` | Paired analysis of MLflow runs / ablation. |
| `tests/unit/` | Fast tests, mirroring `src/fdml/`. |
| `tests/integration/` | Train/API/FE integration; self-skip when processed data is absent. |
| `tests/load/locustfile.py` | Load test. |
| `docs/index.md` | Documentation map. |
| `docs/decisions.md`, `docs/results.md` | Why things are the way they are; measured numbers with provenance. |
| `docs/implementation/*.md` | Deep dives per subsystem. |

## Read in this order (~60 min)

| # | File | Question it answers |
|---|---|---|
| 1 | `README.md` | What is this and what does it claim? |
| 2 | `configs/train.yaml` + `configs/features.yaml` | Which choices are actually active? |
| 3 | `src/fdml/features/factory.py::create_pipeline` | How is a raw row turned into 341 features? |
| 4 | `src/fdml/models/train/data_prep.py` | How is the split made, and where is leakage prevented? |
| 5 | `src/fdml/models/train/runner.py::train` | What happens end to end, in which phases? |
| 6 | `src/fdml/models/evaluate/runner.py::evaluate` | What is measured and what artifacts come out? |
| 7 | `src/fdml/api/internal/loader.py` | How does a raw transaction become a probability? |
| 8 | `tests/integration/test_train.py` | Which contract must never break? |

After that, only open a file when a question sends you there. Do not read the
repo top to bottom.

## Run it yourself (cheap → expensive)

```bash
just                      # list recipes
just test                 # fast tests (integration self-skips without data)
just sample --size 5000   # rebuild the API demo sample
just evaluate             # needs models/pipeline.joblib
just serve-dev            # http://localhost:8000/docs
just train                # full training (expensive)
just train model.name=xgboost optuna.enabled=true
```

`just check` = lint + format check + type check. CI also runs `pip-audit`,
coverage ≥ 80%, and an OmegaConf config-load check.

## Efficiency levers (why it is not slower than it is)

- Parquet + dtype downcast in `prepare_data.py`.
- FE is fitted once on the train fold, then only `transform` on validation.
- Optional train-fold cap (`_cap_train_fold`) for quick iterations.
- Early stopping and per-iteration callbacks instead of full retrains.
- One estimator per algorithm, parameters centralized in builders/config.
- API loads the pipeline and report once at startup; prediction is in-process.
- Batch prediction endpoint and request rate limiting.

## Current state (verified 2026-11-02)

| Area | State |
|---|---|
| Data | `data/raw` and `data/processed` present locally; parquet generated. |
| Local model | `models/pipeline.joblib` + `models/report.json` present; last local report is **random_forest**, AUC 0.8622, AP 0.4276, 341 features. |
| Training | Works; LGBM is the default and the strongest of the three. |
| Evaluation | Works, but learning curves / importance stability are broken in the train path (see known gaps). |
| API | Code complete; local serving works; the deployed URL could not be verified (timeout). |
| Docs | Complete: index, decisions, results, four implementation deep dives. |
| CI/CD | Strict CI on every push; CD on `v*` tags to Cloud Run. |

## Known gaps (honest list)

1. **Evaluation pipeline mismatch.** Training passes already-featurized frames
   plus the FE-only pipeline to `evaluate()`, while `learning_curves` and
   `feature_importance_stability` expect a full pipeline. They re-apply the FE
   step, so the two paths disagree. Fix pending.
2. **Single-class segments crash.** `per_segment_analysis` calls ROC AUC on
   segments with one class; it should skip them (and return an empty, typed
   DataFrame when nothing survives).
3. **Adversarial validation is in-sample.** `adversarial_validation` reports
   training-fold AUC, not held-out AUC, so the number is optimistic.
4. **Loader mutation/race.** Model reload mutates shared state without a lock;
   concurrent predict/reload can observe a half-updated model.
5. **HPO persistence.** Optuna does not use a shared `storage=`, and the
   objective ignores the configured metric/direction.
6. **Champion promotion compares AUC only**, while the project's primary metric
   is Average Precision.

## What to ignore

- `notes/`, `reports/`, `dvclive/`, `models/*` — scratch and generated output.
- `PLANS.md`, `.github/instructions/project-status.instructions.md`,
  `.github/instructions/dataset.instructions.md` — stale; local-only, excluded
  from git. The dataset one still says `scale_pos_weight = 10`; the truth is
  `27.6` in `configs/train.yaml`.
- `docs/tasks/*` — historical task logs, not current state.
- `src/fdml/models/dvc_plots/index.html` — generated DVC artifact.

## Suggested cleanup (not executed)

1. Revert or intentionally commit `uv.lock` (currently a large unrelated diff).
2. Delete the stale local-only docs listed above.
3. Branches are linear, not tangled: `main` is an ancestor of `docs/project`
   (34 commits ahead). Fast-forward `develop` and `main`, delete
   `feat/model-training` after the PR.
4. Do **not** restructure `src/` until the map above is understood; the current
   layout matches the pipeline stages, and moving modules breaks imports,
   tests and the serving contract.
