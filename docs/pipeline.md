# Pipeline

The data → model → report path, stage by stage. For the module layout and the
serving contract, see [architecture](architecture.md).

## DVC stages

`dvc.yaml` currently defines three stages (`just dvc-repro` runs them):

| Stage | Command | Inputs | Outputs |
|---|---|---|---|
| `download` (frozen) | `scripts/download_data.py` | — | `data/raw/` (persisted) |
| `prepare` | `scripts/prepare_data.py` | `configs/data.yaml`, `data/raw` | `data/processed/` (persisted) |
| `train` | `python -m fdml.models.train` | `configs/{train,features,data}.yaml`, `src/fdml/{models,features,schemas}/`, `data/processed` | `models/pipeline.joblib` + DVCLive metrics/plots |

**Known gap:** `evaluate` is not a DVC stage, so `just dvc-repro` stops after
training. Run `just evaluate` separately (tracked as a roadmap item).

## Split — temporal with embargo

`configs/train.yaml` fixes `strategy: temporal` on `TransactionDT`:

- Validation is the **last 20%** of the transaction timeline. Never a random split:
  fraud models leak badly across time through aggregates (card frequencies,
  device history).
- An **embargo** of 172,800 s (2 days) is dropped between train and validation,
  so aggregate features cannot see the boundary.
- The competition's real train→test gap is ~30 days; 2 days is conservative.

## Features

Built by a **14-step** sklearn pipeline (`src/fdml/features/factory.py`); the
decision of *what* to keep lives in `configs/features.yaml`:

1. `identity` — `has_identity` flag (~76% of rows have no identity block).
2. `email` — domain-level email features.
3. `freq` — frequency encoding (must run before `device`, which drops the raw column).
4. `device` — OS / browser / brand parses.
5. `mflags` — M-columns encoded `{T: 1, F: 0, NaN: -1}`.
6. `time` — cyclical hour / day-of-week encodings.
7. `card` — aggregations over `card1..card6` groups.
8. `amount` — log amount, round-amount flag.
9. `vfilter` / 10. `vfilter_c` — variance (0.01) + correlation (0.95) filters over V and C blocks.
11. `imputer` — numerical → `-999`, categorical → `"missing"`.
12. `dtypes` — dtype optimization (memory).
13. `selector` — fixed expected column list.
14. `categories` — ordinal encoding (last step, required for serving).

Feature decisions already made (do not re-derive): C3 dropped, D6–D9 and
D12–D14 dropped (87–94% missing), V variance/correlation filters as above. The
full audit is in the R EDA notebook.

## Class imbalance

- `scale_pos_weight = 27.6` — the exact ratio `(1 - 0.0344) / 0.0344`. Never SMOTE.
- Fraud is ~3.4% of rows, so the accuracy of an all-negative classifier is ~96.5%.
  **Accuracy is never used.**

## Metrics

Priority: **Average Precision (PR AUC) > ROC AUC > F1**. Accuracy is meaningless here.

`just evaluate` (`src/fdml/models/evaluate/`) produces `models/report.json` and
`models/model_card.md` with:

- ROC AUC, Average Precision with a 1000-iteration bootstrap 95% CI.
- F1 at the optimal threshold, plus precision/recall at the default threshold.
- **Expected cost per transaction** (FN costs 10× an FP; the threshold minimising
  cost is reported — not the F1-optimal one).
- Recall@top 1% / 5% — fraud captured if only the highest-risk fraction is reviewed.
- Brier score (probability calibration), F-beta (β=2, recall-weighted).
- Segment metrics (`ProductCD`, `card4`), drift, feature-importance stability,
  error analysis, calibration curve.

Optional (opt-in in `configs/evaluate.yaml`, they retrain the model N times):
learning curves, importance stability, adversarial validation.

## Model comparison and HPO

- `just train model.name=lightgbm|xgboost|random_forest` (default `lightgbm`).
- `just train optuna.enabled=true` runs Optuna HPO (metric `roc_auc`, 50 trials,
  1 h timeout by default).
- `just ablation ablation.enabled=true` runs a leave-one-group-out feature
  ablation: one variant per feature group toggled off (plus an
  `id_codes_encoding` variant toggled on), each variant × seed logged as its own
  MLflow run tagged `D_ablation`, summarised to `models/feature_ablation.csv`.
- Early stopping uses a fixed stratified 20k-row slice of the validation fold;
  final metrics are still computed on the full validation fold.
- Per-iteration `val/auc` is logged to MLflow every 10 iterations (each remote
  log costs ~1.5 s to DagsHub).

`scripts/compare_runs.py` and `scripts/analyze_runs.py` read those runs back
(paired-by-seed statistics) — they are dev tools outside the lint scope.
