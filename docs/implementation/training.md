# Implementation — training

How a training run moves from processed Parquet to a logged, registered model.

Source: `src/fdml/models/train/`, `configs/train.yaml`, `configs/mlflow.yaml`.

## Entry points

| Entry | Use |
|---|---|
| `python -m fdml.models.train` / `just train *ARGS` | Full run: data → model → evaluation → MLflow → registry |
| `python -m fdml.models.train.ablation` / `just train ablation.enabled=true` | Feature ablation study (returns before HPO) |
| `from fdml.models.train.runner import train` | Programmatic training; returns `TrainResult` |

`just train` forwards OmegaConf dot-overrides as a variadic argument, e.g.
`just train model.name=xgboost optuna.enabled=true ablation.enabled=true`.
The YAML config is merged with CLI overrides and validated by
`TrainConfig` before anything runs.

## Phases

`runner.main()` opens one MLflow run, then:

| Phase | Function | What happens |
|---|---|---|
| 1 | `load_split` | Read `data/processed/train_{transaction,identity}.parquet`, left-merge on `TransactionID` |
| 2 | `load_split` | Temporal (or stratified) split + optional `max_train_rows` cap |
| 3 | `featurize` | Fit the feature pipeline on train, transform validation; return feature names |
| 4 | `train` | Build the estimator, optional HPO, fit with early stopping |
| 5 | `_evaluate_and_track` | Predict validation, compute metrics, run reporters, log params/metrics |
| 6 | `_log_and_save_artifacts` | Save `pipeline.joblib`, log model + native evaluation, register version |

Phase timings and the per-step feature cost are logged and, in phase 5,
recorded as `phase_N_seconds` metrics.

## Split

Default (`configs/train.yaml`):

```yaml
split:
  strategy: temporal
  time_col: TransactionDT
  test_size: 0.2
  embargo_seconds: 172800
```

`TemporalSplitter` sorts by time, holds out the last 20%, and drops two days of
rows immediately before the boundary. `strategy: random` selects
`StratifiedSplitter` and is intended only for debugging — it leaks future
information for this dataset.

`data.max_train_rows` caps the training fold to the earliest N rows when set
(0 = all). Validation is never capped, so experiments remain comparable.

## HPO

`OptunaHPO` (`src/fdml/models/train/hpo.py`) is constructed only when
`optuna.enabled: true`. It searches over `ModelBuilder.get_search_space()` and
evaluates each trial on the featurized training/validation frames with early
stopping.

Observed behavior and limits:

- The objective is hard-coded to ROC AUC; `optuna.metric` and
  `optuna.direction` are not read by the objective.
- The study is created **without `storage=`/`load_if_exists`**, so trials are
  not resumable. `save_best_params` persists the winning params, not the study.
- The validation fold used by HPO is the same fold used for early stopping and
  the final report. Treat the promoted score as mildly optimistic until a
  separate selection fold exists.

## Early stopping and fitting

`_fit_model` (`fitting.py`) decides the eval sets, then delegates to
`ModelBuilder.fit`:

| Model | Early stopping |
|---|---|
| LightGBM | native `lgb.early_stopping(rounds, first_metric_only=True)` callback |
| XGBoost ≥ 3.x | `eval_metric`, `early_stopping_rounds` and callbacks set via `set_params`, then `fit(eval_set=...)`; validation must be the **last** eval set |
| RandomForest | none; fits all trees (`class_weight="balanced_subsample"`) |

Policy:

- Early stopping evaluates on a fixed, stratified 20k-row subsample of the
  validation fold (`random_state=0`), so all seeds stop on the same rows.
- Final metrics are computed on the full validation fold.
- When `train_eval_max_rows > 0`, a fixed random sample of training rows is
  added as a second eval set to show the per-iteration train curve. LightGBM
  appends it after validation; XGBoost inserts it before validation and renames
  the datasets through `xgb_name_map`.
- Per-iteration logging (`IterationCallback`) writes train/validation metrics
  to MLflow and optionally DVCLive every `training_callbacks.mlflow_every`
  iterations (default 10) to avoid remote-call overhead.

## Artifacts and tracking

Saved to `models/` and the MLflow run:

| Artifact | Contents |
|---|---|
| `pipeline.joblib` | `Pipeline([("features", FE), ("model", model)])`, served on raw rows |
| `val_proba.npy`, `y_val.npy` | Validation predictions for cross-run comparison/seed bagging |
| `report.json` | Full evaluation report (see [evaluation](evaluation.md)) |
| `model_card.md` | Human-readable summary |
| plots, `dvclive/` | Calibration/error plots and DVC metrics/plots |
| `train_<run-name>.log` | Run log |
| MLflow model | Logged with a signature/input example inferred from **raw** validation rows |
| Native MLflow evaluation | Standard metrics + ROC/PR/matrix + SHAP explainer, optional 20k-row sample |

The signature uses raw rows (categorical dtypes cast to `object`) because the
served pipeline consumes raw transactions. The dataset hash is read from
`dvc.lock` when present.

Tracking resolution (`src/fdml/config.py`):

1. If `DAGSHUB_TOKEN`/`DAGSHUB_USER_TOKEN`, `DAGSHUB_USERNAME` and
   `DAGSHUB_REPO` are set (env, or `.env` locally), use the DagsHub URI.
2. Otherwise use `sqlite:///mlruns.db`.

If remote experiment setup fails, the run falls back to local SQLite with a
warning. The token is supplied via `MLFLOW_TRACKING_USERNAME/_PASSWORD`, never
embedded in the URI.

## Registry

After a successful run:

1. **Quality gate** — reject unless `AUC ≥ 0.83` and `AP ≥ 0.35`
   (`configs/mlflow.yaml`). Rejected runs are tagged `validation_status=rejected`
   and not registered.
2. **Create version** under the per-algorithm name (`fraud-detection-lgbm`,
   `fraud-detection-xgboost`, `fraud-detection-rf`) using the logged-model URI.
3. **Alias** — if the new AUC is strictly greater than the current champion's
   AUC, set `champion`; otherwise set `challenger`. The first approved version
   becomes `champion`.

The API resolves `champion`, falling back to the newest READY version, then to
the local joblib (see [api](api.md)).

## Ablation

`python -m fdml.models.train.ablation` (or `ablation.enabled: true`) runs the
leave-one-group-out study **before** HPO, with fixed parameters. It logs one
MLflow run per variant × seed and writes `models/feature_ablation.csv`.

Groups: `vesta_features`, `count_corr_filter`, `cyclical`, `has_identity`,
`log_amount`, `is_round_amount`, `email_domain`, `device_os`, `device_brand`,
`card_aggregations`, `frequency_encoding`, plus `id_codes_encoding` (toggled
on). See [results](../results.md#3-feature-ablation--lightgbm-fixed-parameters)
for the current snapshot and its caveats.

## Outputs

| Path | Produced by |
|---|---|
| `models/pipeline.joblib` | Every successful training run |
| `models/report.json`, `models/model_card.md` | Phase 5 (and standalone evaluate) |
| `models/val_proba.npy`, `models/y_val.npy` | Phase 6 |
| `dvclive/metrics.json`, `dvclive/plots/` | DVCLive reporter / callbacks |
| `mlruns.db` (or DagsHub) | MLflow tracking + registry |
| `models/feature_ablation.csv` | Ablation study |
| `train_*.log`, `evaluate.log` | Logging |

## Known gaps

- `report.json` and `model_card.md` are overwritten by whichever model ran last;
  they are not per-model snapshots. Cross-check `model_name` before quoting.
- Training-time learning curves and feature-importance stability receive
  `X_train`/`X_val` that are **already featurized** but a feature-only pipeline;
  these opt-in steps fail into a warning. See [evaluation](evaluation.md).
- HPO study persistence and validation reuse (above).
- Champion promotion uses AUC only; AP is the primary metric.
- `models/` and `dvc.lock` are referenced as relative paths, so runs are only
  reproducible when launched from the project root.
