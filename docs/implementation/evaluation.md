# Implementation — evaluation

How metrics, thresholds, segments, plots and the model card are produced.

Source: `src/fdml/models/evaluate/`, `configs/evaluate.yaml`.

## Two entry points

| Entry | Pipeline passed | Data passed |
|---|---|---|
| In-training (`runner._evaluate_and_track`) | feature-only `Pipeline` | already-featurized train/val frames |
| Standalone (`python -m fdml.models.evaluate` / `just evaluate`) | full `Pipeline([features, model])` loaded from `models/pipeline.joblib` | featurized frames rebuilt by re-fitting the feature pipeline |

Both call the same `evaluate()` function. They differ in what they hand it,
which matters for the opt-in learning-curve/stability steps (see
[known gaps](#known-gaps)).

`evaluate()` computes metrics, optionally segments/plots/drift, writes the model
card, and fans the report out to reporters. It returns an
`EvaluationReport` Pydantic model.

## Metrics

`compute_metrics(y_true, y_proba, threshold=0.5)`:

| Metric | Definition | Role |
|---|---|---|
| `roc_auc` | Area under the ROC curve | Secondary ranking metric |
| `average_precision` | Area under the precision-recall curve | **Primary metric** |
| `f1` | F1 at the configured threshold (0.5 by default) | Operating point |
| `precision`, `recall` | At the same threshold | Operating point |

Additional metrics (`metrics.py`):

- **Best F1** — the maximum F1 over a 100-point threshold grid
  (`np.linspace(0.01, 0.99, 100)`), with the threshold that achieved it. This is
  `best_f1`/`best_threshold`, not the `f1` at 0.5.
- **F-beta** — β = 2 by default, weighting recall twice precision, at the
  configured threshold.
- **Brier score** — mean squared error of the probabilities; lower is better.
- **Expected cost** — mean cost per transaction over the same grid, with
  `FP = 1.0`, `FN = 10.0` (configurable). Returns the cost-optimal threshold and
  the full curve.
- **Recall@top-k** — fraction of frauds captured when reviewing the top 1% and
  top 5% of scores, mirroring a manual-review queue.
- **Bootstrap CI** — 1,000 resamples (seed 42) of the validation rows, default
  metric Average Precision, 2.5/97.5 percentiles. Resamples that raise
  `ValueError` (single-class) are skipped.

Accuracy is deliberately absent. See
[decisions](decisions.md#d-003--average-precision-is-the-primary-metric-accuracy-is-banned).

## Threshold policy

Two different thresholds are reported:

- `best_threshold` / `best_f1`: maximizes F1 on validation.
- `cost_best_threshold` / `expected_cost`: minimizes expected cost
  (`FN = 10×FP`).

The API classifies with the report's `best_threshold` (default 0.5 if the
report is missing). Both thresholds are tuned on the same validation fold used
for reporting, so they are mildly optimistic.

## Segments

Enabled by default for `ProductCD` and `card4`, with `min_samples: 50`.
`per_segment_analysis` reports count, fraud rate, ROC AUC and AP per value.
Segments smaller than 50 rows are skipped.

Known failure: a segment whose remaining rows contain a single class raises
`ValueError` from `roc_auc_score`/`average_precision_score`. This call is not
wrapped, so it aborts the whole evaluation phase rather than skipping the
segment. With a 3.4% fraud rate, a 50-row all-negative segment is plausible
(~18% for an exactly-50-row segment).

## Plots

Controlled by `configs/evaluate.yaml` → `plots.plots`:

- `calibration` (default) — reliability curve.
- `error_analysis` (enabled separately) — error distribution by configured
  features (`TransactionAmt` by default).
- `roc_curve` / `pr_curve` — optional custom plots; their curve tables are only
  populated when enabled. In-training, MLflow's native evaluation also logs
  interactive ROC/PR curves and a confusion matrix.

Output format defaults to `webp` (set `png` for print), DPI 150.

## Opt-in analyses

| Analysis | Default | What it does |
|---|---|---|
| Learning curves | off | Re-fits the model on increasing train fractions |
| Feature-importance stability | off | Re-fits across seeds, compares importance rankings |
| Adversarial validation | off | Trains a classifier to distinguish train from validation (drift) |

These are expensive (extra fits), which is why they are off.

## Reporters and outputs

| Reporter | Destination |
|---|---|
| `ConsoleReporter` | Formatted tables to stdout/log |
| `JSONFileReporter` | `models/report.json` |
| `LoggingReporter` | Structured log lines |
| `DVCLiveReporter` | `dvclive/` metrics and plots for DVC |
| `MLflowReporter` | MLflow metrics/artifacts (in-training only) |
| Model card | `models/model_card.md` |

`CompositeReporter` fans the same `EvaluationReport` out to all configured
reporters. `report.json` is the canonical local artifact: model name, split,
feature count, row counts, fraud rate, metrics, CI, thresholds, cost/PR/ROC/
calibration curves, segments, feature importances, drift AUC, plot paths and
model-card path.

## Model card

`generate_model_card()` renders the headline metrics, AP confidence interval,
top-10 features by split gain, cost/threshold notes, and the split strategy.
It is enabled by default (`model_card.enabled: true`).

## How to run

```bash
just evaluate        # standalone; needs models/pipeline.joblib + processed data
```

The standalone path:

1. Loads `models/pipeline.joblib` and splits it into `model` and `features`.
2. Rebuilds the temporal split from `configs/train.yaml`.
3. Re-fits the feature pipeline on the training fold and transforms validation.
4. Predicts with the loaded model and calls `evaluate()`.

Phase-5 (in-training) evaluation is automatic and writes the same report/model
card so the two paths stay comparable at the metric level.

## Known gaps

- **Segments can abort evaluation** — no single-class guard (above).
- **Adversarial validation is in-sample** — `drift.py` fits the classifier on
  `concat(train, val)` and scores the same rows, so `auc_adv` is inflated.
- **Learning curves / stability pipeline mismatch** — in-training, the
  feature-only pipeline has no `"features"` step (`KeyError`); in both paths the
  frames are already featurized, so applying the feature pipeline again would
  double-transform. Both calls are wrapped in `try/except` and only emit a
  warning, so the plots silently never appear.
- **Threshold optimism** — tuned and reported on the same fold.
- **No held-out test set** — all reported numbers are validation numbers.
- **Drift ≠ degradation** — `auc_adv` measures covariate shift only; track AP
  on labeled future data for performance.
