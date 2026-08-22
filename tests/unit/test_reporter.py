from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from fdml.models.evaluate.reporter import (
    CompositeReporter,
    ConsoleReporter,
    CostPoint,
    DVCLiveReporter,
    EvaluationReport,
    JSONFileReporter,
    LoggingReporter,
    MLflowReporter,
    SegmentResult,
    ThresholdPoint,
    _read_image,
    log_dataset_lineage,
)
from fdml.schemas.evaluate import EvaluateConfig
from fdml.schemas.mlflow import MlflowFullConfig


def _mlflow_cfg(**overrides) -> MlflowFullConfig:
    return MlflowFullConfig(**overrides)


def _eval_cfg() -> EvaluateConfig:
    return EvaluateConfig()


def _report(**overrides) -> EvaluationReport:
    defaults: dict[str, Any] = dict(
        model_name="lightgbm",
        split_strategy="temporal",
        n_features=100,
        n_train=50_000,
        n_val=10_000,
        fraud_rate=0.035,
        roc_auc=0.91,
        average_precision=0.5,
        f1=0.6,
        precision=0.7,
        recall=0.5,
        best_threshold=0.3,
        best_f1=0.75,
        brier=0.05,
        f_beta=0.65,
        cost_best_threshold=0.25,
        expected_cost=0.08,
        recall_at_k={"0.0100": 0.4, "0.0500": 0.6},
        ci_lower=0.48,
        ci_upper=0.52,
        auc_adv=0.8,
        threshold_curve=[
            ThresholdPoint(threshold=0.1, f1=0.5, precision=0.6, recall=0.4)
        ],
        cost_curve=[
            CostPoint(threshold=0.2, expected_cost=0.1, precision=0.6, recall=0.4)
        ],
        roc_curve=[{"fpr": 0.0, "tpr": 0.0, "threshold": 1.0}],
        pr_curve=[{"precision": 1.0, "recall": 0.0}],
        calibration_curve=[{"prob_pred": 0.1, "prob_true": 0.05}],
        segments=[
            SegmentResult(
                segment_col="ProductCD",
                segment_value="W",
                count=500,
                fraud_rate=0.03,
                roc_auc=0.8,
                average_precision=0.3,
            )
        ],
        top_features=[{"feature": "V1", "importance": 0.5}],
    )
    defaults.update(overrides)
    return EvaluationReport(**defaults)


class TestMLflowReporter:
    def test_metrics_use_val_prefix_and_context_goes_to_tags(self):
        reporter = MLflowReporter(
            mlflow_cfg=_mlflow_cfg(),
            eval_cfg=_eval_cfg(),
            model_name="lightgbm",
            split_strategy="temporal",
        )
        report = _report()

        with (
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("mlflow.set_tags") as mock_tags,
            patch("mlflow.log_metrics") as mock_metrics,
            patch("mlflow.log_table"),
            patch("mlflow.log_artifact"),
        ):
            reporter.report(report, _eval_cfg())

        logged = {
            k: v for d in mock_metrics.call_args_list for k, v in d.args[0].items()
        }
        assert logged["val/roc_auc"] == 0.91
        assert logged["val/average_precision"] == 0.5
        assert logged["val/f1_best"] == 0.75
        assert logged["val/best_threshold"] == 0.3
        assert logged["val/fraud_rate"] == 0.035
        assert "roc_auc" not in logged
        assert "best_threshold" not in logged

        tags = mock_tags.call_args.args[0]
        assert tags["model_name"] == "lightgbm"
        assert tags["n_train"] == "50000"
        assert tags["split_strategy"] == "temporal"

    def test_metrics_never_logged_as_params(self):
        reporter = MLflowReporter(
            mlflow_cfg=_mlflow_cfg(),
            eval_cfg=_eval_cfg(),
            model_name="lightgbm",
            split_strategy="temporal",
        )
        with (
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("mlflow.set_tags"),
            patch("mlflow.log_metrics"),
            patch("mlflow.log_params") as mock_params,
            patch("mlflow.log_table"),
            patch("mlflow.log_artifact"),
        ):
            reporter.report(_report(), _eval_cfg())

        assert mock_params.call_count == 0

    def test_curve_tables_logged_under_curves(self):
        reporter = MLflowReporter(
            mlflow_cfg=_mlflow_cfg(),
            eval_cfg=_eval_cfg(),
            model_name="lightgbm",
            split_strategy="temporal",
        )
        with (
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("mlflow.set_tags"),
            patch("mlflow.log_metrics"),
            patch("mlflow.log_table") as mock_table,
            patch("mlflow.log_artifact"),
        ):
            reporter.report(_report(), _eval_cfg())

        paths = [c.args[1] for c in mock_table.call_args_list]
        assert "curves/roc.json" in paths
        assert "curves/pr.json" in paths
        assert "curves/calibration.json" in paths
        assert "curves/threshold_curve.json" in paths
        assert "curves/cost_curve.json" in paths
        assert "segments/segments.json" in paths
        assert "features/importance.json" in paths

    def test_feature_importance_gated_by_config(self):
        reporter = MLflowReporter(
            mlflow_cfg=_mlflow_cfg(log_feature_importance=False),
            eval_cfg=_eval_cfg(),
            model_name="lightgbm",
            split_strategy="temporal",
        )
        with (
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("mlflow.set_tags"),
            patch("mlflow.log_metrics"),
            patch("mlflow.log_table") as mock_table,
            patch("mlflow.log_artifact"),
        ):
            reporter.report(_report(), _eval_cfg())

        paths = [c.args[1] for c in mock_table.call_args_list]
        assert "features/importance.json" not in paths

    def test_plot_paths_uploaded_as_artifacts(self, tmp_path):
        plot = tmp_path / "roc_curve.webp"
        plot.write_bytes(b"\x00")
        report = _report(plot_paths=[str(plot)])

        reporter = MLflowReporter(
            mlflow_cfg=_mlflow_cfg(),
            eval_cfg=_eval_cfg(),
            model_name="lightgbm",
            split_strategy="temporal",
        )
        with (
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("mlflow.set_tags"),
            patch("mlflow.log_metrics"),
            patch("mlflow.log_table"),
            patch("mlflow.log_artifact") as mock_artifact,
        ):
            reporter.report(report, _eval_cfg())

        mock_artifact.assert_any_call(str(plot), "plots")

    def test_no_active_run_skips(self):
        reporter = MLflowReporter(
            mlflow_cfg=_mlflow_cfg(),
            eval_cfg=_eval_cfg(),
            model_name="lightgbm",
            split_strategy="temporal",
        )
        with (
            patch("mlflow.active_run", return_value=None),
            patch("mlflow.log_metrics") as mock_metrics,
        ):
            reporter.report(_report(), _eval_cfg())

        mock_metrics.assert_not_called()


class TestLogDatasetLineage:
    def test_logs_raw_frames_with_target_and_hash(self):
        X_train = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        y_train = pd.Series([0, 1, 0], name="isFraud")
        X_val = pd.DataFrame({"A": [4, 5], "B": ["a", "b"]})
        y_val = pd.Series([0, 0], name="isFraud")

        with (
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("fdml.models.evaluate.reporter.from_pandas") as mock_from,
            patch("mlflow.log_input") as mock_input,
        ):
            log_dataset_lineage(X_train, y_train, X_val, y_val)

        assert mock_from.call_count == 2
        assert mock_input.call_count == 2

        train_df = mock_from.call_args_list[0].args[0]
        assert "isFraud" in train_df.columns
        assert mock_from.call_args_list[0].kwargs["targets"] == "isFraud"
        assert mock_input.call_args_list[0].kwargs["context"] == "training"
        assert mock_input.call_args_list[1].kwargs["context"] == "validation"

    def test_no_active_run_skips(self):
        X = pd.DataFrame({"A": [1]})
        y = pd.Series([0])
        with (
            patch("mlflow.active_run", return_value=None),
            patch("fdml.models.evaluate.reporter.from_pandas") as mock_from,
        ):
            log_dataset_lineage(X, y, X, y)
        mock_from.assert_not_called()


def test_report_accepts_curve_tables():
    report = _report()
    assert isinstance(report.roc_curve, list)
    assert report.roc_curve[0]["fpr"] == 0.0
    assert report.pr_curve[0]["precision"] == 1.0


class TestConsoleReporter:
    def test_report_runs_without_error(self):
        reporter = ConsoleReporter()
        report = _report()
        reporter.report(report, _eval_cfg())

    def test_report_with_minimal_fields(self):
        reporter = ConsoleReporter()
        report = _report(
            ci_lower=None,
            ci_upper=None,
            f_beta=None,
            brier=None,
            expected_cost=None,
            cost_best_threshold=None,
            recall_at_k={},
            auc_adv=None,
            segments=[],
            top_features=[],
        )
        reporter.report(report, _eval_cfg())

    def test_report_with_segments_and_features(self):
        reporter = ConsoleReporter()
        report = _report()
        reporter.report(report, _eval_cfg())


class TestJSONFileReporter:
    def test_creates_json_file(self, tmp_path):
        path = tmp_path / "report.json"
        reporter = JSONFileReporter(path=str(path))
        report = _report()
        reporter.report(report, _eval_cfg())
        assert path.exists()
        import json

        data = json.loads(path.read_text())
        assert data["model_name"] == "lightgbm"
        assert data["roc_auc"] == 0.91

    def test_sets_report_path(self, tmp_path):
        path = tmp_path / "report.json"
        reporter = JSONFileReporter(path=str(path))
        report = _report()
        reporter.report(report, _eval_cfg())
        assert report.report_path == str(path)


class TestLoggingReporter:
    def test_report_logs_metrics(self, caplog):
        reporter = LoggingReporter()
        report = _report()
        with caplog.at_level("INFO", logger="fdml.models.evaluate.reporter"):
            reporter.report(report, _eval_cfg())
        assert "AUC=0.9100" in caplog.text
        assert "AP=0.5000" in caplog.text


class TestCompositeReporter:
    def test_delegates_to_all_reporters(self):
        r1 = MagicMock()
        r2 = MagicMock()
        composite = CompositeReporter([r1, r2])
        report = _report()
        composite.report(report, _eval_cfg())
        r1.report.assert_called_once()
        r2.report.assert_called_once()

    def test_continues_when_reporter_fails(self):
        r1 = MagicMock()
        r1.report.side_effect = RuntimeError("boom")
        r2 = MagicMock()
        composite = CompositeReporter([r1, r2])
        report = _report()
        composite.report(report, _eval_cfg())
        r2.report.assert_called_once()


class TestReadImage:
    def test_reads_float_image(self, tmp_path):
        from PIL import Image

        img_path = tmp_path / "test.png"
        arr = np.random.rand(10, 10, 3).astype(np.float32)
        Image.fromarray((arr * 255).astype(np.uint8)).save(img_path)
        result = _read_image(img_path)
        assert result is not None
        assert result.dtype == np.uint8

    def test_returns_none_on_missing_file(self):
        result = _read_image(Path("/nonexistent/image.png"))
        assert result is None


class TestDVCLiveReporter:
    def test_skips_when_disabled(self):
        eval_cfg = EvaluateConfig()
        eval_cfg.dvclive.enabled = False
        reporter = DVCLiveReporter(
            eval_cfg=eval_cfg,
            y_true=np.array([0, 1, 0]),
            y_proba=np.array([0.1, 0.8, 0.2]),
            model_name="lightgbm",
            split_strategy="temporal",
        )
        report = _report()
        reporter.report(report, eval_cfg)

    def test_calls_live_methods_when_enabled(self):
        eval_cfg = EvaluateConfig()
        eval_cfg.dvclive.enabled = True
        eval_cfg.dvclive.dir = "/tmp/dvclive_test"
        eval_cfg.dvclive.report = False
        reporter = DVCLiveReporter(
            eval_cfg=eval_cfg,
            y_true=np.array([0, 1, 0]),
            y_proba=np.array([0.1, 0.8, 0.2]),
            model_name="lightgbm",
            split_strategy="temporal",
        )
        report = _report()
        with patch("dvclive.live.Live") as mock_live:
            mock_instance = MagicMock()
            mock_live.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_live.return_value.__exit__ = MagicMock(return_value=False)
            reporter.report(report, eval_cfg)
            mock_instance.log_params.assert_called_once()
            assert mock_instance.log_metric.call_count >= 5


class TestMLflowReporterArtifactBranches:
    def test_logs_model_card_and_report(self, tmp_path):
        card = tmp_path / "model_card.md"
        card.write_text("test")
        report_file = tmp_path / "report.json"
        report_file.write_text("{}")
        report = _report(model_card_path=str(card), report_path=str(report_file))
        reporter = MLflowReporter(
            mlflow_cfg=_mlflow_cfg(),
            eval_cfg=_eval_cfg(),
            model_name="lightgbm",
            split_strategy="temporal",
        )
        with (
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("mlflow.set_tags"),
            patch("mlflow.log_metrics"),
            patch("mlflow.log_table"),
            patch("mlflow.log_artifact") as mock_artifact,
        ):
            reporter.report(report, _eval_cfg())
        assert mock_artifact.call_count >= 2


class TestDVCLiveReporterExtended:
    def test_logs_all_branches(self):
        eval_cfg = EvaluateConfig()
        eval_cfg.dvclive.enabled = True
        eval_cfg.dvclive.dir = "/tmp/dvclive_test2"
        eval_cfg.dvclive.report = False
        reporter = DVCLiveReporter(
            eval_cfg=eval_cfg,
            y_true=np.array([0, 1, 0, 1]),
            y_proba=np.array([0.1, 0.8, 0.2, 0.9]),
            model_name="lightgbm",
            split_strategy="temporal",
        )
        report = _report(
            ci_lower=0.48,
            ci_upper=0.52,
            auc_adv=0.85,
            f_beta=0.65,
            expected_cost=0.08,
            cost_best_threshold=0.25,
            recall_at_k={"0.01": 0.4, "0.05": 0.6},
        )
        with patch("dvclive.live.Live") as mock_live:
            mock_instance = MagicMock()
            mock_live.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_live.return_value.__exit__ = MagicMock(return_value=False)
            with patch.object(Path, "exists", return_value=True):
                with patch("fdml.models.evaluate.reporter._read_image") as mock_img:
                    mock_img.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
                    reporter.report(report, eval_cfg)
            assert mock_instance.log_sklearn_plot.call_count >= 3
            assert mock_instance.log_metric.call_count >= 10
