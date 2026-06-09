"""Backward-compatible shim: re-exports from the new evaluate/ subpackage."""

import sys
from pathlib import Path

import logging

from src.models.evaluate.runner import evaluate

logger = logging.getLogger(__name__)

# permitir `python src/models/evaluate.py` sin -m
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def run_evaluation(
    model,
    pipeline,
    X_train,
    X_val,
    y_train,
    y_val,
    cfg,
    feature_names=None,
    output_dir="models/",
):
    from src.models.evaluate.reporter import ConsoleReporter
    from src.schemas.evaluate import EvaluateConfig

    y_proba = model.predict_proba(X_val)[:, 1]
    if feature_names is None:
        feature_names = X_val.columns.tolist()

    eval_cfg = EvaluateConfig()

    report = evaluate(
        model_name=model.__class__.__name__,
        y_true=y_val.values,
        y_proba=y_proba,
        model=model,
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        feature_names=feature_names,
        split_strategy=getattr(cfg.split, "strategy", "temporal"),
        eval_cfg=eval_cfg,
        reporters=[ConsoleReporter()],
    )

    return {
        "metrics": {
            "roc_auc": report.roc_auc,
            "average_precision": report.average_precision,
            "f1": report.f1,
            "precision": report.precision,
            "recall": report.recall,
        },
        "ci": (
            (report.ci_lower, report.ci_upper) if report.ci_lower is not None else None
        ),
        "best_threshold": report.best_threshold,
        "best_f1": report.best_f1,
        "model_card": "",
    }


def main() -> None:
    from src.models.evaluate.runner import main as _main

    _main()


if __name__ == "__main__":
    main()
