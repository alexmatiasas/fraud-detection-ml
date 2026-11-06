# Results

Measured results and where each number comes from. **Read the provenance
column before quoting a number.** Local artifacts are not committed to git
(`models/` is ignored except `.gitkeep`); this page is the committed snapshot.

Snapshot captured **2026-10-19** at `HEAD 63d27a6`.

---

## 1. Last local evaluation — RandomForest artifact

Source: `models/report.json` (produced by `just evaluate` or the last training
run in this working copy). The artifact currently on disk is a RandomForest
model, **not necessarily the deployed champion**.

Dataset: IEEE-CIS, merged transaction + identity, temporal split with a 2-day
embargo. 467,054 train rows / 118,108 validation rows, 341 features, 3.44%
fraud in validation.

| Metric | Value |
|---|---|
| ROC AUC | 0.8622 |
| Average Precision (primary) | 0.4276 |
| AP 95% bootstrap CI | [0.4107, 0.4443] |
| F1 @ 0.5 | 0.2398 |
| Best F1 (tuned threshold 0.8613) | 0.4470 |
| Precision / Recall @ 0.5 | 0.1444 / 0.7077 |
| F2 @ 0.5 (β = 2) | 0.3975 |
| Brier score | 0.1203 |
| Expected cost / transaction (FN = 10×FP) | 0.2087 @ threshold 0.7227 |
| Recall @ top 1% / top 5% | 0.2311 / 0.4884 |

Segments: 10 rows reported across `ProductCD` and `card4` (minimum 50 rows per
segment). Top split-gain features: `V258` (0.233), `V317` (0.133), `C14`
(0.096), `C8` (0.049), `V90` (0.040).

Generated alongside: `models/model_card.md`, calibration/ROC/PR plots,
`dvclive/`.

---

## 2. Reported deployed champion

Source: the table in [README](../README.md#current-model), described there as
reported by `GET /v1/model/info`. **Not verified in this snapshot**: the live
endpoint timed out from the development machine on 2026-10-19.

| Metric | Value |
|---|---|
| ROC AUC | 0.9145 |
| Average Precision | 0.5484 (95% CI [0.5318, 0.5650]) |
| F1 @ threshold 0.65 | 0.5406 |
| Precision / Recall | 0.4693 / 0.5689 |
| Brier score | 0.0322 |
| Expected cost / transaction | 0.1619 @ threshold 0.32 |
| Recall @ top 1% / top 5% | 0.2571 / 0.6021 |

To refresh and confirm:

```bash
curl -sS https://fdml-api-bxpgqqidaq-uc.a.run.app/v1/model/info | python -m json.tool
```

The README table and the API example disagree on the algorithm:
the example shows `fraud-detection-lgbm:49`, while the metrics table does not
name an algorithm. The registry actually keeps one name per algorithm
(`fraud-detection-lgbm`, `-xgboost`, `-rf`), and the default loader resolves the
`champion` alias of `fraud-detection-lgbm`. Treat the algorithm name as
unconfirmed until the endpoint responds.

> Do not compare this row with the RandomForest row above as if they were the
> same experiment. They are different models and different evaluation runs.

---

## 3. Feature ablation — LightGBM, fixed parameters

Source: `models/feature_ablation.csv` (produced by
`just train ablation.enabled=true`). Fixed LightGBM parameters; same temporal
split for every variant. Delta = variant − baseline, so a **positive AUC delta**
means removing that group *increased* AUC.

| Variant (group toggled) | Features | Δ AUC | Δ AP |
|---|---:|---:|---:|
| `vesta_features` (all V off) | 105 | +0.00556 | −0.00532 |
| `has_identity` off | 340 | +0.00367 | −0.00418 |
| `device_brand` off | 340 | +0.00318 | **+0.01091** |
| `email_domain` off | 340 | +0.00226 | +0.00676 |
| `is_round_amount` off | 340 | +0.00195 | −0.00115 |
| `cyclical` off | 335 | +0.00186 | −0.00612 |
| `device_os` off | 340 | +0.00174 | +0.01084 |
| **baseline** | **341** | **0.91248** | **0.54636** |
| `id_codes_encoding` on | 341 | 0.00000 | −0.00758 |
| `log_amount` off | 340 | −0.00062 | −0.00478 |
| `frequency_encoding` off | 334 | −0.00077 | +0.00211 |
| `card_aggregations` off | 337 | −0.00095 | −0.00534 |
| `count_corr_filter` off | 345 | −0.00202 | +0.00127 |

Baseline run: 341 features, best iteration 617, ~30.6 s.

Reading it:

- Removing `device_brand`, `device_os` or `email_domain` **raises both AUC and
  AP** in this snapshot (e.g. `device_brand` off: Δ AUC +0.00318, Δ AP
  +0.01091). On this evidence those groups are not earning their place, but the
  deltas are small enough that they need the configured 3-seed study before a
  removal decision.
- `id_codes_encoding` (toggled on), `log_amount` and `card_aggregations` all
  reduce AP when toggled, so they are kept.
- Removing all V features raises AUC (+0.0056) but lowers AP (−0.0053): the V
  block buys ranking quality at the top of the score distribution, which is
  what AP rewards.

Data-quality caveats:

- The CSV has one row per variant and **no seed column**, although
  `configs/train.yaml` configures seeds `[40, 41, 42]`. The current
  `summarize()` writes multi-seed means (`auc_mean`, `ap_mean`), so this file
  predates (or was post-processed outside) that format. Do not cite the deltas
  as seed-averaged until regenerated.
- The `count_corr_filter` row is internally inconsistent: 345 features with
  `n_removed = 0`, although disabling the filter adds columns relative to the
  341-feature baseline. Treat that row as suspect.

---

## 4. Model comparison

| Model | Best local evidence | AUC | AP |
|---|---|---:|---:|
| LightGBM (fixed params, ablation baseline) | `models/feature_ablation.csv` | 0.91248 | 0.54636 |
| RandomForest (last local evaluate) | `models/report.json` | 0.8622 | 0.4276 |
| XGBoost | no local artifact in this snapshot | — | — |
| Deployed champion | README / `GET /v1/model/info` | 0.9145 | 0.5484 |

There is no committed, reproducible head-to-head table for all three
algorithms. To build one, train each with the same split and seeds and compare
the logged `val/roc_auc` and `val/average_precision` metrics, or use
`scripts/compare_runs.py`.

---

## 5. How to regenerate

```bash
just evaluate                              # report.json, model card, plots
just train model.name=lightgbm             # new MLflow run + pipeline.joblib
just train model.name=xgboost
just train model.name=random_forest
just train ablation.enabled=true           # feature_ablation.csv + MLflow runs
```

Requirements: `data/processed/train_transaction.parquet` and
`train_identity.parquet` must exist (`just data-download` then
`just data-convert`, or `dvc pull`). Evaluation re-fits the feature pipeline on
the training fold, so it needs the full processed data even though it loads an
already-fitted model.

Runtime notes:

- Early stopping evaluates on a fixed 20k-row stratified validation subsample;
  final metrics use all 118,108 validation rows.
- `just evaluate` evaluates whatever `models/pipeline.joblib` contains, which
  may be any algorithm. The model card records the class name.

---

## 6. Methodology caveats

- **Threshold optimism.** The F1-optimal and cost-optimal thresholds are tuned
  on the same validation fold used to report metrics. A held-out threshold set
  would give a less optimistic operating point.
- **AP confidence interval** is a 1,000-iteration bootstrap over validation
  rows (seed 42), not over training runs.
- **Brier score** is reported for calibration but the probabilities are
  class-weighted (LightGBM/XGBoost) or balanced-subsample (RF), so they are not
  calibrated by construction.
- **Adversarial validation** (opt-in) currently trains and scores on the same
  rows, so its AUC is an in-sample drift estimate.
- **Drift** is not the same as performance degradation; monitor AP on labeled
  future data for that.
