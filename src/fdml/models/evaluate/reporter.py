from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from rich.console import Console
from rich.table import Table

from fdml.schemas.evaluate import EvaluateConfig
from fdml.schemas.mlflow import MlflowFullConfig

logger = logging.getLogger(__name__)
_rich_console = Console()


class ThresholdPoint(BaseModel):
    threshold: float
    f1: float
    precision: float
    recall: float


class CostPoint(BaseModel):
    threshold: float
    expected_cost: float
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
    brier: float | None = None
    f_beta: float | None = None
    cost_best_threshold: float | None = None
    expected_cost: float | None = None
    recall_at_k: dict[str, float] = {}
    cost_curve: list[CostPoint] = []
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
            f"[{report.ci_lower:.4f}, {report.ci_upper:.4f}]"
            if report.ci_lower is not None
            else ""
        )

        table = Table(title=f"  {report.model_name}", box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("ROC AUC", f"{report.roc_auc:.4f}")
        table.add_row("Avg Precision", f"{report.average_precision:.4f}")
        if ci_str:
            table.add_row("AP 95% CI", ci_str)
        table.add_row("Best thr", f"{report.best_threshold:.2f}")
        table.add_row("F1 (best)", f"{report.best_f1:.4f}")
        if report.f_beta is not None:
            table.add_row("F2 (default thr)", f"{report.f_beta:.4f}")
        if report.brier is not None:
            table.add_row("Brier", f"{report.brier:.4f}")
        table.add_row("Precision", f"{report.precision:.4f}")
        table.add_row("Recall", f"{report.recall:.4f}")
        if report.expected_cost is not None:
            table.add_row(
                "Cost/thr (FN=10×FP)",
                f"{report.expected_cost:.4f} @ {report.cost_best_threshold:.2f}",
            )
        for k, v in report.recall_at_k.items():
            table.add_row(f"Recall@top {float(k):.0%}", f"{v:.4f}")
        table.add_row("Features", str(report.n_features))
        table.add_row("Train/Val", f"{report.n_train:,} / {report.n_val:,}")
        table.add_row("Fraud rate", f"{report.fraud_rate:.2%}")
        if report.auc_adv is not None:
            table.add_row("Adv. AUC", f"{report.auc_adv:.3f}")

        _rich_console.print()
        _rich_console.print(table)
        _rich_console.print()

        if report.segments:
            seg_table = Table(title="  Segment analysis (worst AP)", box=None)
            seg_table.add_column("Segment", style="cyan")
            seg_table.add_column("Value", style="magenta")
            seg_table.add_column("AP", style="yellow")
            seg_table.add_column("Count", style="green")
            for seg in report.segments[:5]:
                seg_table.add_row(
                    seg.segment_col,
                    str(seg.segment_value),
                    f"{seg.average_precision:.4f}",
                    str(seg.count),
                )
            _rich_console.print(seg_table)
            _rich_console.print()

        if report.top_features:
            ft_table = Table(title="  Top features", box=None)
            ft_table.add_column("#", style="dim")
            ft_table.add_column("Feature", style="cyan")
            ft_table.add_column("Importance", style="yellow")
            for i, ft in enumerate(report.top_features[:10]):
                ft_table.add_row(str(i + 1), ft["feature"], f"{ft['importance']:.4f}")
            _rich_console.print(ft_table)
            _rich_console.print()


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


class MLflowReporter(Reporter):
    def __init__(
        self,
        mlflow_cfg: MlflowFullConfig,
        eval_cfg: EvaluateConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        split_strategy: str,
    ):
        self._mlflow_cfg = mlflow_cfg
        self._eval_cfg = eval_cfg
        self._X_train = X_train
        self._y_train = y_train
        self._model_name = model_name
        self._split_strategy = split_strategy

    def report(self, report: EvaluationReport, eval_cfg: EvaluateConfig) -> None:
        if mlflow.active_run() is None:
            logger.warning("  MLflowReporter: no active run, skipping")
            return

        run = mlflow.active_run()
        run_id = run.info.run_id

        mlflow.log_params(
            {
                "model_name": self._model_name,
                "split_strategy": self._split_strategy,
                "n_features": report.n_features,
                "n_train": report.n_train,
                "n_val": report.n_val,
                "best_threshold": report.best_threshold,
            }
        )

        mlflow.log_metrics(
            {
                "roc_auc": report.roc_auc,
                "average_precision": report.average_precision,
                "f1": report.f1,
                "precision": report.precision,
                "recall": report.recall,
            }
        )

        extra: dict[str, float] = {}
        if report.brier is not None:
            extra["brier"] = report.brier
        if report.f_beta is not None:
            extra["f_beta"] = report.f_beta
        if report.expected_cost is not None:
            extra["expected_cost"] = report.expected_cost
            extra["cost_best_threshold"] = report.cost_best_threshold or 0.0
        for k, v in report.recall_at_k.items():
            extra[f"recall_at_top_{float(k):.2f}"] = v
        if extra:
            mlflow.log_metrics(extra)

        if report.threshold_curve:
            mlflow.log_table(
                pd.DataFrame([t.model_dump() for t in report.threshold_curve]),
                "segments/threshold_curve.json",
            )
        if report.cost_curve:
            mlflow.log_table(
                pd.DataFrame([c.model_dump() for c in report.cost_curve]),
                "segments/cost_curve.json",
            )
        if report.segments:
            mlflow.log_table(
                pd.DataFrame([s.model_dump() for s in report.segments]),
                "segments/segments.json",
            )

        if report.ci_lower is not None:
            mlflow.log_metrics(
                {
                    "ap_ci_lower": report.ci_lower,
                    "ap_ci_upper": report.ci_upper,
                }
            )
        if report.auc_adv is not None:
            mlflow.log_metric("adversarial_auc", report.auc_adv)

        if report.model_card_path:
            p = Path(report.model_card_path)
            if p.exists():
                mlflow.log_artifact(str(p))

        if report.report_path:
            p = Path(report.report_path)
            if p.exists():
                mlflow.log_artifact(str(p))

        if report.top_features:
            mlflow.log_dict(
                {"features": report.top_features}, "features/importance.json"
            )

        if self._mlflow_cfg.datasets.enabled:
            try:
                dataset = mlflow.data.from_pandas(self._X_train)
                mlflow.log_input(dataset, context=self._mlflow_cfg.datasets.context)
                logger.info("  MLflow: dataset lineage logged")
            except Exception as exc:
                logger.warning("  MLflow: log_input failed: %s", exc)

        logger.info(
            "  MLflow: run %s — AP=%.4f, AUC=%.4f",
            run_id[:8],
            report.average_precision,
            report.roc_auc,
        )


class DVCLiveReporter(Reporter):
    def __init__(
        self,
        eval_cfg: EvaluateConfig,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        split_strategy: str,
    ):
        self._eval_cfg = eval_cfg
        self._y_true = y_true
        self._y_proba = y_proba
        self._model_name = model_name
        self._split_strategy = split_strategy

    def report(self, report: EvaluationReport, eval_cfg: EvaluateConfig) -> None:
        if not eval_cfg.dvclive.enabled:
            return

        try:
            from dvclive import Live

            dvclive_dir = eval_cfg.dvclive.dir
            with Live(dir=dvclive_dir, dvcyaml=False, report="notebook") as live:
                live.log_params(
                    {
                        "model_name": self._model_name,
                        "split_strategy": self._split_strategy,
                        "n_features": report.n_features,
                        "n_train": report.n_train,
                        "n_val": report.n_val,
                        "best_threshold": report.best_threshold,
                    }
                )

                live.log_metric("roc_auc", report.roc_auc)
                live.log_metric("average_precision", report.average_precision)
                live.log_metric("f1", report.f1)
                live.log_metric("precision", report.precision)
                live.log_metric("recall", report.recall)

                if report.brier is not None:
                    live.log_metric("brier", report.brier)
                if report.f_beta is not None:
                    live.log_metric("f_beta", report.f_beta)
                if report.expected_cost is not None:
                    live.log_metric("expected_cost", report.expected_cost)
                    live.log_metric(
                        "cost_best_threshold",
                        report.cost_best_threshold or 0.0,
                    )
                for k, v in report.recall_at_k.items():
                    live.log_metric(f"recall_at_top_{float(k):.2f}", v)

                if report.ci_lower is not None:
                    live.log_metric("ap_ci_lower", report.ci_lower)
                    live.log_metric("ap_ci_upper", report.ci_upper)
                if report.auc_adv is not None:
                    live.log_metric("adversarial_auc", report.auc_adv)

                try:
                    live.log_sklearn_plot("roc", self._y_true, self._y_proba)
                except Exception:
                    pass

                try:
                    live.log_sklearn_plot(
                        "precision_recall", self._y_true, self._y_proba
                    )
                except Exception:
                    pass

                try:
                    live.log_sklearn_plot(
                        "calibration_curve", self._y_true, self._y_proba
                    )
                except Exception:
                    pass

                for path_str in report.plot_paths:
                    p = Path(path_str)
                    if p.exists() and p.suffix in (".png", ".jpg", ".jpeg"):
                        try:
                            img = _read_image(p)
                            if img is not None:
                                live.log_image(p.name, img)
                        except Exception:
                            pass

                if report.top_features:
                    imp_path = Path("reports") / "feature_importance.json"
                    imp_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(imp_path, "w") as f:
                        json.dump(report.top_features, f, indent=2)
                    live.log_artifact(str(imp_path))

                live.make_summary()

            logger.info("  DVCLive: metrics/plots logged to %s/", dvclive_dir)
        except Exception as exc:
            logger.warning("  DVCLive: failed with %s", exc)


def _read_image(path: Path) -> np.ndarray | None:
    try:
        from matplotlib.image import imread

        img = imread(str(path))
        if img.dtype == "float32" or img.dtype == "float64":
            img = (img * 255).astype("uint8")
        return img
    except Exception:
        return None


class CompositeReporter(Reporter):
    def __init__(self, reporters: list[Reporter]):
        self._reporters = reporters

    def report(self, report: EvaluationReport, eval_cfg: EvaluateConfig) -> None:
        for r in self._reporters:
            try:
                r.report(report, eval_cfg)
            except Exception as exc:
                logger.warning("Reporter %s failed: %s", r.__class__.__name__, exc)
