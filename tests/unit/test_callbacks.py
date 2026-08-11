from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from fdml.models.train.callbacks import IterationCallback


class TestIterationCallback:
    def test_lgbm_format(self):
        callback = IterationCallback(log_mlflow=True, log_dvclive=False)
        env = MagicMock()
        env.iteration = 5
        env.evaluation_result_list = [
            ("validation", "auc", 0.85, True),
            ("validation", "log_loss", 0.35, False),
        ]

        with (
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("mlflow.log_metric") as mock_log,
        ):
            callback(env)

        assert mock_log.call_count == 2
        mock_log.assert_any_call("val/auc", 0.85, step=5)
        mock_log.assert_any_call("val/log_loss", 0.35, step=5)

    def test_xgboost_format(self):
        callback = IterationCallback(log_mlflow=True, log_dvclive=False)
        env = MagicMock()
        env.iteration = 10
        env.evaluation_result_list = [
            ("validation", "auc", 0.82),
            ("validation", "log_loss", 0.40),
        ]

        with (
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("mlflow.log_metric") as mock_log,
        ):
            callback(env)

        assert mock_log.call_count == 2
        mock_log.assert_any_call("val/auc", 0.82, step=10)
        mock_log.assert_any_call("val/log_loss", 0.40, step=10)

    def test_no_active_run_skips_mlflow(self):
        callback = IterationCallback(log_mlflow=True, log_dvclive=False)
        env = MagicMock()
        env.iteration = 1
        env.evaluation_result_list = [("validation", "auc", 0.8, True)]

        with (
            patch("mlflow.active_run", return_value=None),
            patch("mlflow.log_metric") as mock_log,
        ):
            callback(env)

        mock_log.assert_not_called()

    def test_empty_eval_list_returns_early(self):
        callback = IterationCallback(log_mlflow=True, log_dvclive=False)
        env = MagicMock()
        env.iteration = 1
        env.evaluation_result_list = []

        with (
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("mlflow.log_metric") as mock_log,
        ):
            callback(env)

        mock_log.assert_not_called()

    def test_no_eval_list_attr_returns_early(self):
        callback = IterationCallback(log_mlflow=True, log_dvclive=False)
        env = MagicMock(spec=[])

        with (
            patch("mlflow.active_run", return_value=MagicMock()),
            patch("mlflow.log_metric") as mock_log,
        ):
            callback(env)

        mock_log.assert_not_called()

    def test_close_no_live(self):
        callback = IterationCallback(log_mlflow=True, log_dvclive=False)
        callback.close()

    def test_close_with_live(self):
        callback = IterationCallback(log_mlflow=False, log_dvclive=True)
        live_mock = MagicMock()
        callback._live = live_mock
        callback.close()
        live_mock.make_summary.assert_called_once()


class TestIterationCallbackConsole:
    def test_prints_on_console_every_interval(self, caplog):
        callback = IterationCallback(
            log_mlflow=False, log_dvclive=False, log_console=True, console_every=50
        )
        env = MagicMock()
        env.iteration = 50
        env.evaluation_result_list = [("validation", "auc", 0.9, True)]

        with caplog.at_level(logging.INFO, logger="fdml.models.train.callbacks"):
            callback(env)

        assert "iter    50 | auc=0.9000 (best 0.9000 @ 50)" in caplog.text

    def test_prints_only_on_new_best_between_intervals(self, caplog):
        callback = IterationCallback(
            log_mlflow=False, log_dvclive=False, log_console=True, console_every=50
        )
        env1 = MagicMock()
        env1.iteration = 5
        env1.evaluation_result_list = [("validation", "auc", 0.8, True)]
        env2 = MagicMock()
        env2.iteration = 6
        env2.evaluation_result_list = [("validation", "auc", 0.79, True)]

        with caplog.at_level(logging.INFO, logger="fdml.models.train.callbacks"):
            callback(env1)
            callback(env2)

        lines = [r for r in caplog.records if "iter" in r.getMessage()]
        assert len(lines) == 1
        assert "best 0.8000 @ 5" in lines[0].getMessage()

    def test_xgboost_three_tuple_higher_better(self, caplog):
        callback = IterationCallback(
            log_mlflow=False, log_dvclive=False, log_console=True, console_every=1
        )
        env = MagicMock()
        env.iteration = 10
        env.evaluation_result_list = [("validation", "log_loss", 0.4)]

        with caplog.at_level(logging.INFO, logger="fdml.models.train.callbacks"):
            callback(env)

        assert "best 0.4000 @ 10" in caplog.text
