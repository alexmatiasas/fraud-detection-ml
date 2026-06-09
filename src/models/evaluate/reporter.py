from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.schemas.evaluate import EvaluateConfig

logger = logging.getLogger(__name__)


class ThresholdPoint(BaseModel):
    threshold: float
    f1: float
    precision: float
    recall: float


class SegmentResult(BaseModel):
    segment_col: str
    segment_value: str | float | int
    count: int
    fraud_rate: float
    roc_auc: float
    average_precision: float


class EvaluationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str
    split_strategy: str
    n_features: int
    n_train: int
    n_val: int
    fraud_rate: float
    roc_auc: float
    average_precision: float
    f1: float
    precision: float
    recall: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    best_threshold: float
    best_f1: float
    threshold_curve: list[ThresholdPoint] = []
    segments: list[SegmentResult] = []
    top_features: list[dict[str, Any]] = []
    auc_adv: float | None = None
    plot_paths: list[str] = []
    model_card_path: str | None = None
    report_path: str | None = None


class Reporter(ABC):
    @abstractmethod
    def report(self, report: EvaluationReport, eval_cfg: EvaluateConfig) -> None: ...


class ConsoleReporter(Reporter):
    def report(self, report: EvaluationReport, eval_cfg: EvaluateConfig) -> None:
        ci_str = (
            f"  AP 95% CI: [{report.ci_lower:.4f}, {report.ci_upper:.4f}]"
            if report.ci_lower is not None
            else ""
        )
        print(f"\n{'=' * 50}")
        print(f"  Model: {report.model_name}")
        print(f"  ROC AUC:          {report.roc_auc:.4f}")
        print(f"  Average Precision: {report.average_precision:.4f}{ci_str}")
        print(
            f"  Best threshold:    {report.best_threshold:.2f} (F1={report.best_f1:.4f})"
        )
        print(f"  Precision:         {report.precision:.4f}")
        print(f"  Recall:            {report.recall:.4f}")
        print(f"  Features:          {report.n_features}")
        print(f"  Train/Val:         {report.n_train:,} / {report.n_val:,}")
        print(f"  Fraud rate:        {report.fraud_rate:.2%}")
        if report.auc_adv is not None:
            print(f"  Adv. validation:   AUC={report.auc_adv:.3f}")
        print(f"{'=' * 50}\n")

        if report.segments:
            print("  Segment analysis (worst AP):")
            for seg in report.segments[:5]:
                print(
                    f"    {seg.segment_col}={seg.segment_value}: AP={seg.average_precision:.4f} (n={seg.count})"
                )
            print()

        if report.top_features:
            print("  Top features:")
            for i, ft in enumerate(report.top_features[:10]):
                print(f"    {i + 1}. {ft['feature']}: {ft['importance']:.4f}")
            print()


class JSONFileReporter(Reporter):
    def __init__(self, path: str = "models/report.json"):
        self._path = path

    def report(self, report: EvaluationReport, eval_cfg: EvaluateConfig) -> None:
        path = Path(self._path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2)
        report.report_path = str(path)
        logger.info("  Evaluation report saved to %s", path)


class LoggingReporter(Reporter):
    def report(self, report: EvaluationReport, eval_cfg: EvaluateConfig) -> None:
        logger.info(
            "%s: AUC=%.4f, AP=%.4f, F1=%.4f, thr=%.2f",
            report.model_name,
            report.roc_auc,
            report.average_precision,
            report.best_f1,
            report.best_threshold,
        )


class CompositeReporter(Reporter):
    def __init__(self, reporters: list[Reporter]):
        self._reporters = reporters

    def report(self, report: EvaluationReport, eval_cfg: EvaluateConfig) -> None:
        for r in self._reporters:
            try:
                r.report(report, eval_cfg)
            except Exception as exc:
                logger.warning("Reporter %s failed: %s", r.__class__.__name__, exc)
