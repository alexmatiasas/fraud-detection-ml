"""Leave-one-group-out feature ablation study.

Estimates the contribution of each engineered / selection feature group by
re-fitting a *fixed* model configuration (no Optuna) on the same temporal
train/val split, once with the full feature set (baseline) and once per
variant with one group toggled OFF (or, for ``id_codes_encoding``, toggled
ON — an experiment that adds a transformation).  Reports validation AUC /
average precision and the delta versus the baseline, so groups can be kept,
cut, or flagged as noisy.

Run via ``make train ablation.enabled=true`` (uses the ``ablation`` section
of ``configs/train.yaml``) or ``python -m fdml.models.train.ablation``.

The ablation runs *before* Optuna HPO so the feature set is fixed first;
HPO then optimises the model hyperparameters on that feature set.
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from fdml.models.config import load_train_config
from fdml.models.train.model_builder import model_builder_registry
from fdml.models.train.runner import featurize, load_split
from fdml.schemas.features import FeaturesConfig
from fdml.utils.logging import setup_logging
from fdml.utils.paths import features_config

logger = logging.getLogger(__name__)


def _off_vesta_features(d: dict) -> None:
    d["transaction"]["vesta_features"]["include"] = False


def _off_count_corr_filter(d: dict) -> None:
    d["transaction"]["count_corr_filter"]["include"] = False


def _off_cyclical(d: dict) -> None:
    d["engineered"]["cyclical"] = False


def _off_has_identity(d: dict) -> None:
    d["engineered"]["has_identity"] = False


def _off_log_amount(d: dict) -> None:
    d["engineered"]["log_amount"] = False


def _off_is_round_amount(d: dict) -> None:
    d["engineered"]["is_round_amount"] = False


def _off_email_domain(d: dict) -> None:
    d["engineered"]["email_domain"] = False


def _off_device_os(d: dict) -> None:
    d["engineered"]["device_os"] = False


def _off_device_brand(d: dict) -> None:
    d["engineered"]["device_brand"] = False


def _off_card_aggregations(d: dict) -> None:
    d["engineered"]["card_aggregations"] = None


def _off_frequency_encoding(d: dict) -> None:
    d["engineered"]["frequency_encoding"] = []


def _on_id_codes_encoding(d: dict) -> None:
    d["engineered"]["id_codes_encoding"] = True


FEATURE_GROUPS: "OrderedDict[str, Callable[[dict], None]]" = OrderedDict(
    vesta_features=_off_vesta_features,
    count_corr_filter=_off_count_corr_filter,
    cyclical=_off_cyclical,
    has_identity=_off_has_identity,
    log_amount=_off_log_amount,
    is_round_amount=_off_is_round_amount,
    email_domain=_off_email_domain,
    device_os=_off_device_os,
    device_brand=_off_device_brand,
    card_aggregations=_off_card_aggregations,
    frequency_encoding=_off_frequency_encoding,
    id_codes_encoding=_on_id_codes_encoding,
)


def load_base_features() -> FeaturesConfig:
    """The full feature configuration (baseline), resolved from yaml."""
    return FeaturesConfig.model_validate(features_config)


def variant_features(
    mutate: Callable[[dict], None] | None, base: FeaturesConfig | None = None
) -> FeaturesConfig:
    """Return the features config for a variant.

    ``mutate`` is applied to a deep-copy of the base config to toggle one
    feature group OFF.  ``None`` returns the (baseline) config unchanged.
    """
    if mutate is None:
        return base or load_base_features()
    d = copy.deepcopy(dict(features_config))
    mutate(d)
    return FeaturesConfig.model_validate(d)


def _fit_and_score(
    builder: Any,
    params: dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    es_rounds: int,
    es_metric: str,
) -> tuple[Any, float, float, int | None, float]:
    """Fit on the variant's features and return (model, auc, ap, iters, seconds)."""

    model = builder.build(params)
    start = time.perf_counter()

    if isinstance(model, lgb.LGBMClassifier):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_names=["validation"],
            eval_metric=es_metric,
            callbacks=[lgb.early_stopping(es_rounds, first_metric_only=True)],
        )
    elif isinstance(model, xgb.XGBClassifier):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    elapsed = time.perf_counter() - start
    best_iter = getattr(model, "best_iteration_", None)

    y_proba = np.asarray(model.predict_proba(X_val))[:, 1]
    auc = float(roc_auc_score(y_val, y_proba))
    ap = float(average_precision_score(y_val, y_proba))
    return model, auc, ap, best_iter, elapsed


def run_ablation(cfg: Any, max_train_rows: int = 0) -> pd.DataFrame:
    """Run the study and return a per-variant results DataFrame.

    The results table is returned (and, in ``main()``, saved to
    ``cfg.ablation.report_path``).
    """

    X_train, X_val, y_train, y_val = load_split(cfg)

    if max_train_rows and len(X_train) > max_train_rows:
        X_train = X_train.iloc[:max_train_rows]
        y_train = y_train.iloc[:max_train_rows]
        logger.info(
            "  Ablation: capping train set to %s rows (temporal order kept)",
            f"{max_train_rows:,}",
        )

    builder = model_builder_registry.get(cfg.model.name)
    params = cfg.model.params.model_dump()
    es_rounds = cfg.early_stopping.rounds
    es_metric = cfg.early_stopping.eval_metric
    logger.info("  Model: %s (fixed params, no Optuna)", cfg.model.name)
    logger.info("  params: %s", builder.format_params(params))

    base_features = load_base_features()
    variants: list[tuple[str, Callable[[dict], None] | None]] = [
        ("baseline", None),
        *FEATURE_GROUPS.items(),
    ]

    baseline_cols: list[str] = []
    rows: list[dict[str, Any]] = []

    for name, mutate in variants:
        features = variant_features(mutate, base=base_features)
        logger.info("─ Variant: %s", name)
        X_tr, X_va, _, feature_names = featurize(X_train, X_val, y_train, features)

        removed = (
            sorted(set(baseline_cols) - set(feature_names)) if baseline_cols else []
        )

        model, auc, ap, best_iter, elapsed = _fit_and_score(
            builder,
            params,
            X_tr,
            y_train,
            X_va,
            y_val,
            es_rounds,
            es_metric,
        )
        del model  # not retained — each variant is a single fit

        logger.info(
            "  → AUC=%.5f  AP=%.5f  (%d features, %s removed, %d iters, %.1fs)",
            auc,
            ap,
            len(feature_names),
            len(removed),
            best_iter or 0,
            elapsed,
        )

        rows.append(
            {
                "variant": name,
                "n_features": len(feature_names),
                "n_removed": len(removed),
                "removed_columns": " ".join(removed),
                "best_iteration": best_iter,
                "train_seconds": round(elapsed, 1),
                "val_auc": round(auc, 5),
                "val_ap": round(ap, 5),
            }
        )

        if not baseline_cols:
            baseline_cols = feature_names.copy()

    results = pd.DataFrame(rows)
    base_auc = float(results.loc[results.variant == "baseline", "val_auc"].iloc[0])
    base_ap = float(results.loc[results.variant == "baseline", "val_ap"].iloc[0])
    results["delta_auc"] = results["val_auc"] - base_auc
    results["delta_ap"] = results["val_ap"] - base_ap
    results = results.sort_values("val_auc", ascending=False).reset_index(drop=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature ablation study")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Cap train rows per variant (overrides ablation.max_train_rows)",
    )
    args = parser.parse_args()

    cfg = load_train_config()
    setup_logging(log_path="train_ablation.log")

    max_rows = args.max_train_rows or cfg.ablation.max_train_rows
    results = run_ablation(cfg, max_train_rows=max_rows)

    report_path = Path(cfg.ablation.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(report_path, index=False)
    logger.info("Ablation report saved to %s", report_path)
    logger.info("\n%s", results.to_string(index=False))


if __name__ == "__main__":
    main()
