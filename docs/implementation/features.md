# Implementation — feature pipeline

How raw transactions become the model's 341-column feature frame, and the
invariants that keep training and serving identical.

Source: `src/fdml/features/`, `configs/features.yaml`,
`src/fdml/models/train/data_prep.py`.

## Contract

```
raw merged DataFrame (transaction ⟕ identity, 434 columns)
        │
        ▼  Pipeline.fit_transform(train) / .transform(val)
341-column DataFrame (numeric + ordinal codes, no NaN)
```

- Input is the **raw** merged frame. `TransactionID` and `isFraud` are present
  during training; the target is dropped by the selector.
- Output is a flat frame of model-ready columns. Categorical columns are
  ordinal codes by the last step.
- The same fitted pipeline is stored inside the served artifact, so the API
  calls `pipeline.predict(raw_row)` — never a separate featurization script.

## Step order (default config)

`create_pipeline()` in `src/fdml/features/factory.py` returns 14 steps with the
default config:

| # | Step name | Class | Produces / does |
|---:|---|---|---|
| 1 | `identity` | `IdentityFlagExtractor` | `has_identity`; identity-block signals |
| 2 | `email` | `EmailFeatureExtractor` | payer/recipient domain signals; contract keeps `p_r_domain_match` |
| 3 | `freq` | `FrequencyEncoder` | `<column>_freq` for the configured categoricals |
| 4 | `device` | `DeviceFeatureExtractor` | `device_os`, `device_brand`; drops `DeviceInfo` after frequency encoding |
| 5 | `mflags` | `MFlagEncoder` | `M1–M9`: `T → 1`, `F → 0`, `NaN → -1` |
| 6 | `time` | `TimeFeatureExtractor` | `hour_of_day`, `day_of_week`, `day_of_month`, optional sin/cos |
| 7 | `card` | `CardAggregator` | `card_{mean,std,max,count}_transactionamt` |
| 8 | `amount` | `AmountFeatureExtractor` | `TransactionAmt_log`, `is_round_amount` |
| 9 | `vfilter` | `VFeatureFilter` | variance + correlation filter over `V*` |
| 10 | `vfilter_c` | `VFeatureFilter` | correlation filter over `C*` |
| 11 | `imputer` | `MissingImputer` | numerical `-999`, categorical `"missing"` |
| 12 | `dtypes` | `DtypeOptimizer` | compact dtypes |
| 13 | `selector` | `FeatureSelector` | enforces the expected column contract, drops target |
| 14 | `categories` | `CategoryEncoder` | ordinal codes; **must stay last** |

`idcodes` (`IdCodeEncoder`) is inserted between `time` and `card` only when
`engineered.id_codes_encoding: true` (off by default). That makes 15 steps, so
"14-step pipeline" describes the default config, not every config.

## Ordering constraints

These are load-bearing; reordering breaks either correctness or the serving
contract.

- **`freq` before `device`.** `DeviceInfo` is frequency-encoded and then
  dropped by the device step. Encoding it later would fail.
- **`vfilter`/`vfilter_c` before `imputer`.** The filters operate on the raw
  V/C distributions; imputation would change variance/correlation.
- **`selector` before `categories`.** The selector matches names against the
  expected contract; `CategoryEncoder` then converts the surviving categorical
  columns to codes.
- **`categories` last.** The model must receive the same encoded frame at
  training and serving. `tests/integration/test_train.py` asserts the served
  pipeline is self-contained.

## Column contract

`_build_feature_columns(cfg)` derives the expected output names from config, in
order: numerical, categorical, high-cardinality, count (C), delta (D), identity
categoricals and `id_*`, datetime, cyclical sin/cos, `has_identity`, amount,
email match, device, frequency, card aggregations.

`FeatureSelector(feature_columns=..., target=..., vesta_include=...)` keeps
exactly those columns. V features are handled specially because the filter is
data-dependent: the selector keeps the surviving `V*` columns when
`vesta_features.include` is true.

Consequence: adding a transform without adding its outputs to
`_build_feature_columns` makes the selector silently drop them. The feature
count in `report.json` is the observable check.

## Fit/transform discipline

`src/fdml/models/train/data_prep.py`:

- `fit_transform_steps(pipeline, X_train, y_train)` — fits every step on the
  training fold only, logging per-step shape and time.
- `transform_steps(pipeline, X_val)` — applies the fitted steps to validation.
- `featurize(...)` returns `(X_train_fe, X_val_fe, pipeline, feature_names)`.

No step may be fitted on validation data. Aggregate features (card statistics,
frequency counts, V/C filters) are learned from the training window; this is
why the temporal split has an embargo (see
[decisions](decisions.md#d-001--temporal-split-with-an-embargo-never-a-random-split)).

## Missing values

| Kind | Convention | Applied by |
|---|---|---|
| Numerical | `-999` | `MissingImputer` |
| Categorical | literal `"missing"` | `MissingImputer` |
| `M` flags | `T → 1`, `F → 0`, `NaN → -1` | `MFlagEncoder` |
| `M4` | treated as categorical (`M0/M1/M2`) | config + `MFlagEncoder` |

`-999` is a sentinel, not an imputed mean. A model that treats `-999` as a
real magnitude will learn a spurious split; tree models handle it natively.

## Adding a feature — checklist

1. Add the config knob/list in `configs/features.yaml` and the matching
   Pydantic model in `src/fdml/schemas/features.py`; run `just schemas`.
2. Implement the transformer in `src/fdml/features/` as a scikit-learn
   transformer (`fit` stores state, `transform` is pure).
3. Insert it in `create_pipeline()` at the right position (see ordering above).
4. Add its output names to `_build_feature_columns()`.
5. Add unit tests under `tests/unit/features/` and, if it changes the serving
   surface, extend the integration pipeline test.
6. Retrain, run the ablation, and update [results](../results.md). If the
   choice is non-obvious, add a decision entry.

## Known limits

- The V/C filters learn which columns survive from the training fold. A
  different split or data version can change the feature count.
- The pipeline keeps raw `TransactionDT` alongside the derived time features;
  it is a valid split key but a risky model input outside the training period.
- `data/raw` and `data/processed` are DVC-managed; only the demo sample
  (`data/samples/val_sample.parquet`) is committed.
