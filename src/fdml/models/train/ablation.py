"""Leave-one-group-out feature ablation study.

Estimates the contribution of each engineered / selection feature group by
re-fitting a *fixed* model configuration (no Optuna) on the same temporal
train/val split, once with the full feature set (baseline) and once per
variant with one group toggled OFF (or, for ``id_codes_encoding``, toggled
ON — an experiment that adds a transformation).

Each (variant × seed) fit is logged as its own MLflow run tagged
``D_ablation`` with the same logging standard as the training runner
(``experiment``/``variant``/``features_hash`` tags, per-iteration val/auc
curve, ``val_proba.npy`` + ``y_val.npy`` artifacts for ``compare_runs`` /
``analyze_runs``), so variants can be compared with the paired-by-seed
statistics in ``scripts/analyze_runs.py``.

Run via ``make train ablation.enabled=true`` (uses the ``ablation`` section
of ``configs/train.yaml``) or ``python -m fdml.models.train.ablation``.

The ablation runs *before* Optuna HPO so the feature set is fixed first;
HPO then optimises the model hyperparameters on that feature set.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fdml.models.config import (
    load_mlflow_config,
    load_train_config,
    resolve_mlflow_tracking,
)
from fdml.models.train.model_builder import model_builder_registry
from fdml.models.train.runner import (
    _build_callbacks,
    _fit_model,
    _log_configs_and_env,
    _log_git_tags,
    _register_abort_handler,
    features_fingerprint,
    featurize,
    load_split,
)
from fdml.schemas.features import FeaturesConfig
from fdml.utils.logging import setup_logging
from fdml.utils.paths import features_config

logger = logging.getLogger(__name__)

ABLATION_TAG = "D_ablation"


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


def _setup_mlflow() -> None:
    """Point MLflow at the tracking store (DagsHub via .env) with fallback."""
    mlflow_cfg = resolve_mlflow_tracking(load_mlflow_config())
    mlflow.set_tracking_uri(mlflow_cfg.tracking.tracking_uri)
    try:
        mlflow.set_experiment(mlflow_cfg.tracking.experiment_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "  MLflow: experiment setup failed (%s), falling back to local SQLite",
            exc,
        )
        mlflow_cfg.tracking.tracking_uri = "sqlite:///mlruns.db"
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
        mlflow.set_experiment(mlflow_cfg.tracking.experiment_name)


def _log_variant_features(features: FeaturesConfig) -> None:
    """Snapshot the variant's resolved feature config as an artifact."""
    payload = json.dumps(features.model_dump(mode="json"), indent=2, default=str)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".json", encoding="utf-8", delete=False
    ) as f:
        f.write(payload)
        tmp_path = f.name
    mlflow.log_artifact(tmp_path, "configs")
    Path(tmp_path).unlink(missing_ok=True)


def _log_predictions(model: Any, X_val: pd.DataFrame, y_val: pd.Series) -> None:
    """Persist val predictions so ``compare_runs`` / ``analyze_runs`` can
    seed-bag this variant without re-fitting."""
    y_proba = np.asarray(model.predict_proba(X_val))[:, 1]
    with tempfile.TemporaryDirectory() as tmp:
        proba_path = Path(tmp) / "val_proba.npy"
        y_path = Path(tmp) / "y_val.npy"
        np.save(proba_path, y_proba)
        np.save(y_path, np.asarray(y_val))
        mlflow.log_artifact(str(proba_path))
        mlflow.log_artifact(str(y_path))


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per-variant mean ± std across seeds, sorted by mean AUC, with the
    delta versus the baseline variant's mean AUC / AP."""
    base_auc = float(results.loc[results.variant == "baseline", "val_auc"].mean())
    base_ap = float(results.loc[results.variant == "baseline", "val_ap"].mean())
    grouped = (
        results.groupby("variant", sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            auc_mean=("val_auc", "mean"),
            auc_std=("val_auc", "std"),
            ap_mean=("val_ap", "mean"),
            ap_std=("val_ap", "std"),
            best_iteration=("best_iteration", "mean"),
        )
        .reset_index()
    )
    grouped["delta_auc"] = grouped["auc_mean"] - base_auc
    grouped["delta_ap"] = grouped["ap_mean"] - base_ap
    return grouped.sort_values("auc_mean", ascending=False).reset_index(drop=True)


def run_ablation(cfg: Any, max_train_rows: int = 0) -> pd.DataFrame:
    """Run the study and return a per-(variant × seed) results DataFrame.

    Every fit is logged to MLflow under the ``D_ablation`` experiment tag.
    The long results table is returned (one row per variant × seed); use
    :func:`summarize` for the per-variant mean ± std report.
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
    base_params = cfg.model.params.model_dump()
    seeds = cfg.ablation.seeds or [cfg.seed]
    logger.info("  Model: %s (fixed params, no Optuna)", cfg.model.name)
    logger.info("  params: %s", builder.format_params(base_params))
    logger.info("  seeds: %s", seeds)
    logger.info("  MLflow tag: %s — one run per variant × seed", ABLATION_TAG)

    _setup_mlflow()
    callbacks = _build_callbacks(cfg)

    base_features = load_base_features()
    variants: list[tuple[str, Callable[[dict], None] | None]] = [
        ("baseline", None),
        *FEATURE_GROUPS.items(),
    ]

    baseline_cols: list[str] = []
    rows: list[dict[str, Any]] = []

    for name, mutate in variants:
        features = variant_features(mutate, base=base_features)
        X_tr, X_va, _, feature_names = featurize(X_train, X_val, y_train, features)

        removed = (
            sorted(set(baseline_cols) - set(feature_names)) if baseline_cols else []
        )
        fhash = features_fingerprint(features)
        logger.info(
            "─ Variant: %s (%d features, %s removed, hash %s)",
            name,
            len(feature_names),
            len(removed),
            fhash[:8],
        )

        for seed in seeds:
            params = {**base_params, "random_state": seed}
            run_name = f"{cfg.model.name}_s{seed}_{ABLATION_TAG}_{name}"
            with mlflow.start_run(run_name=run_name):
                _register_abort_handler()
                _log_git_tags()
                mlflow.set_tag("experiment", ABLATION_TAG)
                mlflow.log_param("experiment", ABLATION_TAG)
                mlflow.set_tag("variant", name)
                mlflow.log_param("variant", name)
                mlflow.set_tag("features_hash", fhash)
                mlflow.log_param("features_hash", fhash)
                mlflow.log_params({"seed": seed})
                mlflow.log_params(
                    {"n_features": len(feature_names), "n_removed": len(removed)}
                )
                mlflow.log_params(params)
                _log_configs_and_env(cfg)
                _log_variant_features(features)

                model = builder.build(params)
                start = time.perf_counter()
                model = _fit_model(model, X_tr, y_train, X_va, y_val, cfg, callbacks)
                elapsed = time.perf_counter() - start
                best_iter = getattr(model, "best_iteration_", None)
                if best_iter is None:
                    best_iter = getattr(model, "best_iteration", None)

                y_proba = np.asarray(model.predict_proba(X_va))[:, 1]
                auc = float(roc_auc_score(y_val, y_proba))
                ap = float(average_precision_score(y_val, y_proba))

                mlflow.log_metrics(
                    {
                        "val/roc_auc": auc,
                        "val/average_precision": ap,
                        "val/best_iteration": float(best_iter or 0),
                        "training_elapsed_s": round(elapsed, 1),
                    }
                )
                _log_predictions(model, X_va, y_val)

                logger.info(
                    "  s%s → AUC=%.5f AP=%.5f (%d iters, %.1fs)",
                    seed,
                    auc,
                    ap,
                    best_iter or 0,
                    elapsed,
                )

                rows.append(
                    {
                        "variant": name,
                        "seed": seed,
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

    return pd.DataFrame(rows)


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
    summarize(results).to_csv(report_path, index=False)
    logger.info(
        "Ablation runs logged to MLflow (tag=%s); report saved to %s",
        ABLATION_TAG,
        report_path,
    )
    logger.info("\n%s", summarize(results).to_string(index=False))


if __name__ == "__main__":
    main()
