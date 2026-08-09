from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fdml.models.config import load_evaluation_config, load_train_config
from fdml.models.evaluate.drift import adversarial_validation
from fdml.models.evaluate.metrics import (
    brier_score,
    bootstrap_ci,
    compute_metrics,
    expected_cost,
    f_beta_score,
    recall_at_top_k,
    threshold_tuning,
)
from fdml.models.evaluate.model_card import generate_model_card
from fdml.models.evaluate.plots import (
    CalibrationPlotter,
    ErrorAnalysisPlotter,
    PRCurvePlotter,
    ROCCurvePlotter,
)
from fdml.models.evaluate.reporter import (
    CompositeReporter,
    ConsoleReporter,
    CostPoint,
    DVCLiveReporter,
    EvaluationReport,
    JSONFileReporter,
    LoggingReporter,
    Reporter,
    SegmentResult,
    ThresholdPoint,
)
from fdml.models.evaluate.segments import per_segment_analysis
from fdml.models.evaluate.stability import (
    feature_importance_stability,
    learning_curves,
)
from fdml.schemas.evaluate import EvaluateConfig

logger = logging.getLogger(__name__)


def _has_plot(cfg: EvaluateConfig, name: str) -> bool:
    return name in cfg.plots.plots


def evaluate(
    model_name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model: Any,
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None,
    feature_names: list[str],
    split_strategy: str,
    eval_cfg: EvaluateConfig,
    reporters: list[Reporter] | None = None,
) -> EvaluationReport:
    default_thr = eval_cfg.threshold.threshold
    metrics = compute_metrics(y_true, y_proba, threshold=default_thr)

    brier_val: float | None = None
    if eval_cfg.brier:
        try:
            brier_val = brier_score(y_true, y_proba)
        except ValueError:
            logger.warning("  Brier score failed (single-class sample)")

    f_beta_val: float | None = None
    if eval_cfg.f_beta.enabled:
        f_beta_val = f_beta_score(
            y_true, y_proba, beta=eval_cfg.f_beta.beta, threshold=default_thr
        )

    cost_thr: float | None = None
    cost_val: float | None = None
    cost_curve_list: list[dict[str, float]] = []
    if eval_cfg.costs.enabled:
        cost_thr, cost_val, cost_curve_list = expected_cost(
            y_true,
            y_proba,
            fp_cost=eval_cfg.costs.false_positive_cost,
            fn_cost=eval_cfg.costs.false_negative_cost,
            n_thresholds=eval_cfg.costs.n_thresholds,
        )

    recall_at_k: dict[str, float] = {}
    if eval_cfg.recall_at_k.enabled:
        for fraction in eval_cfg.recall_at_k.fractions:
            recall_at_k[f"{fraction:.4f}"] = recall_at_top_k(
                y_true, y_proba, k_fraction=fraction
            )

    ci: tuple[float, float] | None = None
    if eval_cfg.threshold.bootstrap:
        ci = bootstrap_ci(
            y_true,
            y_proba,
            metric=eval_cfg.threshold.bootstrap_metric,
            n_iterations=eval_cfg.threshold.bootstrap_iterations,
            seed=eval_cfg.threshold.bootstrap_seed,
        )

    best_thr, best_f1, thr_curve = threshold_tuning(
        y_true,
        y_proba,
        n_thresholds=eval_cfg.threshold.n_thresholds,
    )

    seg_df = None
    if X_val is not None and eval_cfg.segments.enabled and eval_cfg.segments.columns:
        seg_df = per_segment_analysis(
            y_true,
            y_proba,
            X_val[eval_cfg.segments.columns],
            min_samples=eval_cfg.segments.min_samples,
        )

    top_features: list[dict[str, Any]] = []
    feature_importance = getattr(model, "feature_importances_", None)
    if feature_importance is not None and len(feature_importance) == len(feature_names):
        imp_sorted = sorted(
            zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True
        )
        top_features = [
            {"feature": col, "importance": float(val)} for col, val in imp_sorted
        ]

    plot_paths: list[str] = []
    output_dir = eval_cfg.plots.output_dir

    if _has_plot(eval_cfg, "roc_curve"):
        path = ROCCurvePlotter().plot(y_true, y_proba, output_dir)
        plot_paths.append(str(path))
    if _has_plot(eval_cfg, "pr_curve"):
        path = PRCurvePlotter().plot(y_true, y_proba, output_dir)
        plot_paths.append(str(path))
    if _has_plot(eval_cfg, "calibration"):
        path = CalibrationPlotter().plot(y_true, y_proba, output_dir)
        plot_paths.append(str(path))

    if _has_plot(eval_cfg, "error_analysis") and X_val is not None:
        error_features = (
            eval_cfg.plots.error_features or eval_cfg.error_analysis.features
        )
        paths = ErrorAnalysisPlotter(features_to_plot=error_features).plot(
            y_true, y_proba, X_val, output_dir
        )
        plot_paths.extend(str(p) for p in paths)

    y_val_series = pd.Series(y_true, name="isFraud")
    y_train_series = y_train if isinstance(y_train, pd.Series) else pd.Series(y_train)

    if eval_cfg.learning_curves.enabled and X_val is not None:
        try:
            model_cls = model.__class__
            path = learning_curves(
                X_train,
                y_train_series,
                X_val,
                y_val_series,
                pipeline,
                lambda: model_cls(**model.get_params()),
                train_sizes=eval_cfg.learning_curves.train_sizes,
                output_dir=output_dir,
            )
            plot_paths.append(str(path))
        except Exception as exc:
            logger.warning("Learning curves failed: %s", exc)

    if eval_cfg.stability.feature_importance and X_val is not None:
        try:
            path = feature_importance_stability(
                model,
                feature_names,
                X_train,
                y_train_series,
                X_val,
                y_val_series,
                pipeline,
                n_iterations=eval_cfg.stability.n_iterations,
                seed=eval_cfg.stability.seed,
                output_dir=output_dir,
            )
            plot_paths.append(str(path))
        except Exception as exc:
            logger.warning("Feature importance stability failed: %s", exc)

    auc_adv = None
    if eval_cfg.adversarial_validation.enabled and X_val is not None:
        try:
            auc_adv, _ = adversarial_validation(X_train, X_val, output_dir)
        except Exception as exc:
            logger.warning("Adversarial validation failed: %s", exc)

    n_train = len(X_train)
    n_val = len(y_true) if X_val is None else len(y_true)
    fraud_rate = float(y_true.mean())

    segments_list: list[SegmentResult] = []
    if seg_df is not None and not seg_df.empty:
        for _, row in seg_df.iterrows():
            segments_list.append(
                SegmentResult(
                    segment_col=str(row["segment_col"]),
                    segment_value=str(row["segment_value"]),
                    count=int(row["count"]),
                    fraud_rate=float(row["fraud_rate"]),
                    roc_auc=float(row["roc_auc"]),
                    average_precision=float(row["average_precision"]),
                )
            )

    report = EvaluationReport(
        model_name=model_name,
        split_strategy=split_strategy,
        n_features=len(feature_names),
        n_train=n_train,
        n_val=n_val,
        fraud_rate=fraud_rate,
        roc_auc=metrics["roc_auc"],
        average_precision=metrics["average_precision"],
        f1=metrics["f1"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        ci_lower=float(ci[0]) if ci else None,
        ci_upper=float(ci[1]) if ci else None,
        best_threshold=best_thr,
        best_f1=best_f1,
        brier=brier_val,
        f_beta=f_beta_val,
        cost_best_threshold=cost_thr,
        expected_cost=cost_val,
        recall_at_k=recall_at_k,
        cost_curve=[
            CostPoint(
                threshold=float(p["threshold"]),
                expected_cost=float(p["expected_cost"]),
                precision=float(p["precision"]),
                recall=float(p["recall"]),
            )
            for p in cost_curve_list
        ],
        threshold_curve=[
            ThresholdPoint(
                threshold=float(p["threshold"]),
                f1=float(p["f1"]),
                precision=float(p["precision"]),
                recall=float(p["recall"]),
            )
            for p in thr_curve
        ],
        segments=segments_list,
        top_features=top_features,
        auc_adv=auc_adv,
        plot_paths=plot_paths,
    )

    if eval_cfg.model_card.enabled:
        card_text = generate_model_card(
            model_name=model_name,
            metrics=metrics,
            ci=ci,
            best_threshold=best_thr,
            best_f1=best_f1,
            n_features=len(feature_names),
            n_train=n_train,
            n_val=n_val,
            fraud_rate=fraud_rate,
            split_strategy=split_strategy,
            feature_importance=[
                (ft["feature"], ft["importance"]) for ft in top_features
            ],
            auc_adv=auc_adv,
            brier=brier_val,
            f_beta=f_beta_val,
            cost_best_threshold=cost_thr,
            expected_cost=cost_val,
            recall_at_k=recall_at_k,
        )
        card_path = Path(output_dir) / eval_cfg.model_card.filename
        card_path.parent.mkdir(parents=True, exist_ok=True)
        card_path.write_text(card_text)
        report.model_card_path = str(card_path)
        logger.info("  Model card saved to %s", card_path)

    if reporters:
        composite = CompositeReporter(reporters)
        composite.report(report, eval_cfg)

    return report


def main() -> None:
    from fdml.utils.logging import setup_logging

    setup_logging(log_path="evaluate.log")

    import joblib

    from fdml.features.factory import (
        load_data,
        load_data_config,
        merge_tables,
    )
    from fdml.models.categoricals import (
        encode_categoricals as _encode_categoricals,
    )
    from fdml.models.split import TemporalSplitter

    train_cfg = load_train_config()
    eval_cfg = load_evaluation_config()
    data_cfg = load_data_config()

    logger.info("=== Evaluation (standalone) ===")

    pipeline_path = "models/pipeline.joblib"
    logger.info("Loading pipeline from %s ...", pipeline_path)
    full_pipeline = joblib.load(pipeline_path)
    model = full_pipeline.named_steps["model"]
    fe_pipeline = full_pipeline.named_steps["features"]
    logger.info("  Loaded: model=%s", model.__class__.__name__)

    logger.info("Loading data ...")
    train_df, identity_df = load_data(data_cfg)
    df = merge_tables(train_df, identity_df)

    splitter = TemporalSplitter(
        time_col=train_cfg.split.time_col, test_size=train_cfg.split.test_size
    )
    train_idx, val_idx = next(splitter.split(df, df["isFraud"]))

    X_train, X_val = (
        df.drop(columns=["isFraud"]).iloc[train_idx],
        df.drop(columns=["isFraud"]).iloc[val_idx],
    )
    y_train, y_val = df["isFraud"].iloc[train_idx], df["isFraud"].iloc[val_idx]
    logger.info(
        "  Train: %s rows / Val: %s rows", f"{len(X_train):,}", f"{len(X_val):,}"
    )

    X_train_fe = fe_pipeline.fit_transform(X_train, y_train)
    X_val_fe = fe_pipeline.transform(X_val)
    X_train_fe, X_val_fe = _encode_categoricals(X_train_fe, X_val_fe)
    logger.info("  Features: %s", X_train_fe.shape[1])

    y_proba = model.predict_proba(X_val_fe)[:, 1]
    y_true = y_val.values
    feature_names = X_val_fe.columns.tolist()

    reporters: list[Reporter] = [
        ConsoleReporter(),
        JSONFileReporter(eval_cfg.report.path),
        LoggingReporter(),
        DVCLiveReporter(
            eval_cfg=eval_cfg,
            y_true=y_true,
            y_proba=y_proba,
            model_name=model.__class__.__name__,
            split_strategy="temporal",
        ),
    ]

    report = evaluate(
        model_name=model.__class__.__name__,
        y_true=y_true,
        y_proba=y_proba,
        model=model,
        pipeline=full_pipeline,
        X_train=X_train_fe,
        y_train=y_train,
        X_val=X_val_fe,
        feature_names=feature_names,
        split_strategy="temporal",
        eval_cfg=eval_cfg,
        reporters=reporters,
    )

    logger.info("Evaluation complete: AP=%.4f", report.average_precision)


if __name__ == "__main__":
    main()
