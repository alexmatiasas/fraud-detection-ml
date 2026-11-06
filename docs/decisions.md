# Decision log

Why the project does what it does. Each entry records the context, the choice,
the reason, and the consequences. Sources point at the executable form of the
decision; if this file and the source disagree, the source wins.

Format is ADR-lite: one decision per entry, numbered for reference from code
review and onboarding.

---

## D-001 — Temporal split with an embargo, never a random split

**Decision.** Validation is the last 20% of the transaction timeline
(`TransactionDT`), with a 2-day (`172800 s`) embargo gap removed immediately
before the boundary.

**Why.** Fraud patterns drift and aggregate features (card statistics,
frequency encodings) are learned from the training window. A random split lets
rows from the future leak into training and inflates every metric.

**Consequences.** Validation scores are pessimistic compared with a random
split, which is the point. The embargo discards the rows closest to the
boundary; the real competition train→test gap is ~30 days, so 2 days is
conservative.

**Source.** `configs/train.yaml` (`split`), `src/fdml/models/split.py`.

---

## D-002 — `scale_pos_weight = 27.6`, no SMOTE

**Decision.** Tree models receive `scale_pos_weight` equal to the exact
negative/positive ratio (≈27.6 at a 3.5% fraud rate). No resampling, no SMOTE.

**Why.** The imbalance is real, not an artifact of sampling. Reweighting keeps
all rows and their temporal order, does not synthesize fraudulent transactions,
and is natively supported by LightGBM/XGBoost.

**Consequences.** Predicted probabilities are shifted by the class weight and
must be calibrated/tuned before being read as risk. RandomForest uses
`class_weight="balanced_subsample"` instead, because it has no
`scale_pos_weight`.

**Source.** `configs/train.yaml` (`model.params.scale_pos_weight`),
`src/fdml/models/train/model_builder.py`.

---

## D-003 — Average Precision is the primary metric; accuracy is banned

**Decision.** Report ROC AUC and Average Precision (PR AUC). Use F1/F2,
precision/recall, cost and recall@top-k as operating-point metrics. Never use
accuracy.

**Why.** At a ~3.4% fraud rate, predicting "never fraud" scores ~96.6%
accuracy while catching nothing. AP concentrates on the positive class and is
the honest ranking metric for rare events.

**Consequences.** Model selection and the registry gate use AP alongside AUC.
Thresholds are reported explicitly because a single probability threshold is a
business decision, not a property of the model.

**Source.** `configs/evaluate.yaml`, `src/fdml/models/evaluate/metrics.py`,
`src/fdml/models/train/registry.py`.

---

## D-004 — Feature set is fixed in config, with documented exclusions

**Decision.** Drop `C3` (99.59% zeros, inverse signal), `D2` (r ≈ 0.98 with
`D1`), and `D6–D9`/`D12–D14` (87–94% missing). Keep V features behind a
variance (`0.01`) + correlation (`0.95`) filter, and C count features behind a
`0.99` correlation filter.

**Why.** The EDA showed these columns add noise, duplicate signal, or are
mostly missing. The thresholds are explicit so the choice is reproducible
rather than folklore.

**Consequences.** The feature space is 341 columns under the default config.
Changing any list or threshold changes the model's input contract and should
trigger a re-run of the ablation study.

**Source.** `configs/features.yaml` (comments cite the EDA sections),
`src/fdml/features/factory.py`.

---

## D-005 — Missing-value conventions are semantic, not numeric

**Decision.** Numerical missing → `-999`; categorical missing → the literal
`"missing"`; `M` match flags map `T → 1`, `F → 0`, `NaN → -1`.

**Why.** `-999` is a LightGBM-native sentinel that cannot collide with real
amounts/distances; `"missing"` preserves missingness as a category; the
three-state M mapping keeps "no match recorded" distinct from "did not match".

**Consequences.** Every transformer and the serving pipeline must apply the
same convention. A new transform that introduces `NaN` after imputation breaks
the contract.

**Source.** `configs/features.yaml` (`imputation`,
`encoding.match_flags_encoding`), `src/fdml/features/imputer.py`,
`src/fdml/features/flags.py`.

---

## D-006 — Serve one self-contained pipeline on raw transactions

**Decision.** The logged artifact is
`Pipeline([("features", <feature pipeline>), ("model", model)])`.
`CategoryEncoder` is the last feature step, and the MLflow signature is inferred
from the raw validation frame.

**Why.** The API receives raw transactions, not pre-featurized rows. If any
transform lives outside the pipeline, the served artifact silently diverges
from training. Keeping encoding inside the pipeline also makes the artifact
portable and testable.

**Consequences.** Every new feature transform must be added inside
`create_pipeline` (and to the column contract), not in a script. The invariant
is guarded by
`tests/integration/test_train.py::test_serving_pipeline_is_self_contained`.

**Source.** `src/fdml/features/factory.py`,
`src/fdml/models/train/runner.py` (`_log_and_save_artifacts`).

---

## D-007 — Frequency-encode only true high-cardinality categoricals

**Decision.** Frequency-encode `P_emaildomain`, `R_emaildomain`, `DeviceInfo`,
`card1`, `id_30`, `id_31`, `id_33`. Leave integer-coded columns
(`card2/3/5`, `addr1/2`) as raw numerics.

**Why.** Those integer codes have centralized distributions where relative
frequency is near-constant (e.g. `addr2` top-1 share 99.2%) and is
anti-predictive; their raw value carries more signal. `id_02` is effectively a
row identifier and is excluded.

**Consequences.** Adding a column to `frequency_encoding` changes the feature
count. The ablation `frequency_encoding` variant measures the whole group.

**Source.** `configs/features.yaml` (`engineered.frequency_encoding` comments),
`src/fdml/features/freq.py`.

---

## D-008 — Ablation runs before HPO, with fixed hyperparameters

**Decision.** The leave-one-group-out study re-fits a fixed model configuration
on the same temporal split for every variant (plus seeds 40/41/42 in the
configured study), and runs before Optuna.

**Why.** If HPO ran first, each variant would get different hyperparameters and
the comparison would mix feature effect with tuning effect. Fixing the feature
set first makes the ablation interpretable.

**Consequences.** Ablation results are about feature groups, not about the best
achievable score. The committed CSV is a single-row-per-variant snapshot; the
current `summarize()` writes multi-seed means, so regenerate before treating it
as seed-averaged evidence.

**Source.** `src/fdml/models/train/ablation.py`,
`configs/train.yaml` (`ablation`).

---

## D-009 — Early stopping on a fixed stratified subsample; final metrics on full validation

**Decision.** Early stopping and the per-iteration validation curve use a
fixed, stratified 20k-row subsample of the validation fold (`random_state=0`).
Final metrics are computed on the full validation fold.

**Why.** AUC evaluation on 118k rows costs seconds per iteration; 20k makes
per-iteration curves cheap and comparable across seeds. A constant sampling
seed keeps every run early-stopping on the same rows.

**Consequences.** The stopping point is chosen on a subset; the reported
metrics are not. The train curve optionally uses a 50k-row sample of the
training fold, so it is diagnostic, not the training loss.

**Source.** `configs/train.yaml` (`early_stopping`),
`src/fdml/models/train/data_prep.py` (`_sample_eval_set`),
`src/fdml/models/train/fitting.py`.

---

## D-010 — Registry quality gate, then `champion` by AUC

**Decision.** A run is registered only if `AUC ≥ 0.83` and
`AP ≥ 0.35`. An approved version becomes `champion` when its AUC is strictly
greater than the current champion's AUC; otherwise it becomes `challenger`.

**Why.** The gate prevents a broken or degenerate run from becoming
servable. The alias keeps the API pointing at a named, stable version instead
of a mutable path.

**Consequences.** Promotion currently ignores AP even though AP is the primary
metric (see [evaluation](implementation/evaluation.md#known-gaps)). A
challenger with better AP but slightly lower AUC is not promoted.

**Source.** `configs/mlflow.yaml` (`registry`),
`src/fdml/models/train/registry.py`.

---

## D-011 — One registry name per algorithm

**Decision.** `lightgbm` → `fraud-detection-lgbm`, `xgboost` →
`fraud-detection-xgboost`, `random_forest` → `fraud-detection-rf`.

**Why.** Names are stable identifiers for the multi-model API
(`GET /v1/models`, `/v1/predict/{name}`, `/v1/predict/compare`). Version
numbers identify a run within an algorithm.

**Consequences.** The default API loader resolves the `champion` alias of
`fraud-detection-lgbm`; the other algorithms are served through the multi-model
loader. The README's earlier claim that all algorithms share one name is
incorrect.

**Source.** `src/fdml/models/train/registry.py` (`_REGISTRY_NAMES`),
`src/fdml/api/internal/loader.py`, `src/fdml/api/internal/registry.py`.

---

## D-012 — DagsHub when credentials exist, local SQLite otherwise

**Decision.** If `DAGSHUB_TOKEN`/`DAGSHUB_USER_TOKEN`, `DAGSHUB_USERNAME` and
`DAGSHUB_REPO` are present (env or `.env`), use the DagsHub MLflow URI;
otherwise use `sqlite:///mlruns.db`. The token is never embedded in the URI.

**Why.** The same code must run locally without secrets and in Cloud Run with
injected env vars. MLflow prints the tracking URI in run URLs and logs, so a
token in the URI would leak.

**Consequences.** Auth is supplied via `MLFLOW_TRACKING_USERNAME` /
`MLFLOW_TRACKING_PASSWORD` env vars. A failed remote setup falls back to local
SQLite, which can mask a misconfigured deployment (logged as a warning).

**Source.** `src/fdml/config.py` (`resolve_mlflow_tracking`),
`configs/mlflow.yaml`.

---

## D-013 — Two thresholds, reported separately

**Decision.** Tune an F1-optimal threshold over a 100-point grid for reporting,
and a cost-optimal threshold separately with `FN = 10 × FP`. The API classifies
with the report's `best_threshold` (falling back to `0.5` if absent).

**Why.** "Fraud or not" is a business decision. F1 balances precision/recall
symmetrically; the cost curve expresses the actual asymmetry (missing fraud is
10× worse than a manual review). Reporting both avoids hiding the trade-off.

**Consequences.** The served threshold can differ from the threshold that
maximizes F1. It is tuned on the same validation fold used to report metrics,
so it is mildly optimistic; a held-out threshold set would be stricter.

**Source.** `configs/evaluate.yaml` (`threshold`, `costs`),
`src/fdml/models/evaluate/metrics.py`,
`src/fdml/api/routers/predict.py`.

---

## D-014 — Score by `TransactionID` against a committed demo sample

**Decision.** `POST /v1/predict/` accepts a `TransactionID`, looks up the full
raw row in `data/samples/val_sample.parquet`, optionally applies raw-feature
overrides, and always returns a SHAP explanation when the model supports it.

**Why.** The service needs the full 434-column raw row to run the feature
pipeline, and a browser demo cannot ship the dataset. A committed sample keeps
the demo self-contained; overrides enable "what if the amount were higher?"
without a custom request schema.

**Consequences.** Predictions are limited to transactions present in the
sample. Unknown IDs return 404; invalid override columns return 422. The sample
is committed to git while `data/raw` and `data/processed` are DVC-managed.

**Source.** `src/fdml/api/routers/predict.py`,
`src/fdml/api/internal/loader.py`, `src/fdml/api/internal/explain.py`,
`scripts/generate_sample.py`.

---

## D-015 — OmegaConf + Pydantic, with generated JSON Schemas

**Decision.** YAML configs are loaded with OmegaConf and validated by Pydantic
models in `src/fdml/schemas/`. Each YAML carries a
`# $schema: ../schemas/*.schema.json` header; `just schemas` regenerates them.

**Why.** One validation layer catches typos and type errors before a long
training run, and the JSON Schema gives editors autocomplete.

**Consequences.** Adding or renaming a config field requires updating the
Pydantic model and running `just schemas`. CI loads every config to catch
validation failures.

**Source.** `src/fdml/schemas/`, `src/fdml/schemas/generate.py`,
`configs/*.yaml`, `justfile`.

---

## D-016 — DVC owns data and the training stage; evaluation stays standalone

**Decision.** `dvc.yaml` defines `download` (frozen), `prepare`, and `train`.
`evaluate` is not a DVC stage; run it with `just evaluate`.

**Why.** Data preparation and training are the expensive, cacheable steps.
Evaluation reads the trained artifact and is cheap to run separately.

**Consequences.** `just dvc-repro` reproduces data and training but stops before
evaluation. This is a known roadmap item.

**Source.** `dvc.yaml`, `justfile`, [pipeline](pipeline.md).

---

## D-017 — Mutating endpoints require an API key; docs are off in production

**Decision.** `PUT /v1/model/reload` and `PUT /v1/model/switch` require
`X-API-Key`, compared with `hmac.compare_digest` against `FDML_API_KEY`.
Swagger/Redoc/OpenAPI are disabled when `ENVIRONMENT=production`. CORS is an
explicit allowlist.

**Why.** Model management must not be publicly writable; interactive docs
expand the attack surface; an origin allowlist keeps browser calls to the
frontend.

**Consequences.** If `FDML_API_KEY` is unset, mutating endpoints return 503
rather than allowing unauthenticated access. Prediction endpoints stay public
(rate-limited) so the portfolio demo works without a key.

**Source.** `src/fdml/api/dependencies.py`, `src/fdml/api/main.py`,
`src/fdml/api/routers/model.py`.

---

## D-018 — CI is stricter than the local `just check`

**Decision.** CI runs, in order: `ruff check` → `ruff format --check` →
`pip-audit --strict` → `basedpyright` → `pytest --cov --cov-fail-under=80` →
OmegaConf config load. `just check` covers lint, format and types only.

**Why.** The local gate must be fast; CI must also catch vulnerable
dependencies, coverage regressions and broken configs.

**Consequences.** A green `just check` is not a green CI. Run
`just test-cov` and the config check before pushing. `basedpyright` needs the
`train` extra installed, otherwise optional imports are reported unresolved.

**Source.** `.github/workflows/ci.yml`, `justfile`, [AGENTS.md](../AGENTS.md).

---

## Open decision debt

These are known, verified gaps in decisions already made. They are recorded
here so they are fixed deliberately, not rediscovered.

- HPO does not persist its Optuna study (`storage`/`load_if_exists` are
  absent), and its objective ignores `optuna.metric`/`direction`.
- Champion promotion compares AUC only, while AP is the primary metric.
- Adversarial validation scores the drift classifier in-sample.
- Training-time learning curves / feature-importance stability receive the
  featurized frame and a feature-only pipeline; both are opt-in and currently
  fail into a warning.
- `per_segment_analysis` can raise on a single-class segment and abort the
  evaluation phase.

See the implementation docs for file/line detail.
