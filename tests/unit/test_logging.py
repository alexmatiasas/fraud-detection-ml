from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from fdml.utils import logging as logging_utils


@pytest.fixture()
def isolated_logging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[Path, list[logging.Handler]]]:
    """Snapshot root handlers and module state; restore them after the test.

    Returns the tmp dir and the list of handlers present at snapshot time
    (pytest's own logging plugin attaches handlers to the root logger).
    """
    root = logging.getLogger()
    before_handlers = list(root.handlers)
    monkeypatch.setattr(logging_utils, "_stdout_handler_attached", False)
    monkeypatch.setattr(logging_utils, "_file_handlers_attached", set())
    monkeypatch.setattr(logging_utils, "_primary_log_path", tmp_path / "fdml.log")
    yield tmp_path, before_handlers
    root.handlers[:] = before_handlers


def _added_handlers(
    before: list[logging.Handler], handler_cls: type | None = None
) -> list[logging.Handler]:
    root = logging.getLogger()
    return [
        h
        for h in root.handlers
        if h not in before and (handler_cls is None or isinstance(h, handler_cls))
    ]


class TestSetupLogging:
    def test_writes_messages_to_file(self, isolated_logging: tuple[Path, list]):
        tmp_path, _ = isolated_logging
        log_path = logging_utils.setup_logging(log_path=tmp_path / "out.log")
        logging.getLogger("test").info("hello observability")

        assert log_path == tmp_path / "out.log"
        assert "hello observability" in log_path.read_text()

    def test_relative_path_resolves_under_logs(
        self, isolated_logging: tuple[Path, list], monkeypatch: pytest.MonkeyPatch
    ):
        tmp_path, _ = isolated_logging
        monkeypatch.setattr(logging_utils, "LOGS_DIR", tmp_path)
        path = logging_utils.setup_logging(log_path="train_lgbm.log")

        assert path == tmp_path / "train_lgbm.log"

    def test_stdout_handler_attached_at_most_once(
        self, isolated_logging: tuple[Path, list]
    ):
        tmp_path, before = isolated_logging
        logging_utils.setup_logging(log_path=tmp_path / "a.log")
        logging_utils.setup_logging(log_path=tmp_path / "b.log")

        stdout = [
            h
            for h in _added_handlers(before, logging.StreamHandler)
            if not isinstance(h, logging.FileHandler)
        ]
        assert len(stdout) == 1

    def test_file_handler_attached_at_most_once_per_path(
        self, isolated_logging: tuple[Path, list]
    ):
        tmp_path, before = isolated_logging
        logging_utils.setup_logging(log_path=tmp_path / "a.log")
        logging_utils.setup_logging(log_path=tmp_path / "a.log")

        files = _added_handlers(before, logging.FileHandler)
        assert len(files) == 1

    def test_returns_primary_path_when_no_path(
        self, isolated_logging: tuple[Path, list]
    ):
        tmp_path, _ = isolated_logging
        path = logging_utils.setup_logging()

        assert path == tmp_path / "fdml.log"


class TestEnsureLogging:
    def test_configures_when_no_handlers(self, isolated_logging: tuple[Path, list]):
        tmp_path, _ = isolated_logging
        path = logging_utils.ensure_logging()

        assert path == tmp_path / "fdml.log"

    def test_returns_same_path_without_new_handlers(
        self, isolated_logging: tuple[Path, list]
    ):
        tmp_path, before = isolated_logging
        first = logging_utils.setup_logging(log_path=tmp_path / "out.log")
        second = logging_utils.ensure_logging()

        assert second == first
        files = _added_handlers(before, logging.FileHandler)
        assert len(files) == 1


class TestStep:
    def test_logs_start_and_elapsed(
        self, isolated_logging: tuple[Path, list], caplog: pytest.LogCaptureFixture
    ):
        tmp_path, _ = isolated_logging
        logging_utils.setup_logging(log_path=tmp_path / "out.log")
        with caplog.at_level(logging.INFO, logger="fdml.pipeline"):
            with logging_utils.step("PHASE 4: Model training"):
                pass

        assert "PHASE 4: Model training" in caplog.text
        assert "done in" in caplog.text
