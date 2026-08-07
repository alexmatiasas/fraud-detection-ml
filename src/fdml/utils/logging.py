from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fdml.utils.paths import LOGS_DIR

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"

_stdout_handler_attached = False
_file_handlers_attached: set[Path] = set()
_primary_log_path: Path = LOGS_DIR / "fdml.log"


def setup_logging(
    level: int = logging.INFO, log_path: Path | str | None = None
) -> Path:
    """Configure the root logger for stdout plus a file handler.

    Idempotent: the stdout handler is attached at most once, and a file
    handler at most once per path.  Safe to call from CLI entry points,
    from ``train()``/``evaluate()`` library functions and from notebooks.

    Args:
        level: Minimum level for both handlers.
        log_path: Destination of the file handler.  Relative paths are
            resolved under the project ``logs/`` directory.  Defaults to
            ``logs/fdml.log``.

    Returns:
        The log file path currently being written to.
    """
    global _stdout_handler_attached, _primary_log_path

    resolved = Path(log_path or _primary_log_path)
    if not resolved.is_absolute():
        resolved = LOGS_DIR / resolved

    root = logging.getLogger()

    if not _stdout_handler_attached:
        stdout = logging.StreamHandler()
        stdout.setFormatter(logging.Formatter(_FMT))
        root.addHandler(stdout)
        _stdout_handler_attached = True

    if resolved not in _file_handlers_attached:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FMT))
        root.addHandler(file_handler)
        _file_handlers_attached.add(resolved)

    root.setLevel(level)
    _primary_log_path = resolved
    return resolved


def ensure_logging(level: int = logging.INFO) -> Path:
    """Idempotent convenience for library callers (e.g. notebooks).

    Only configures logging when the root logger has no handlers yet;
    otherwise returns the path already in use.
    """
    if not logging.getLogger().hasHandlers():
        return setup_logging(level=level)
    return _primary_log_path


@contextmanager
def step(name: str, *, level: int = logging.INFO) -> Iterator[None]:
    """Log the start and end of a named pipeline step with elapsed time.

    Example:
        >>> with step("PHASE 4: Model training"):
        ...     model.fit(X, y)
    """
    logger = logging.getLogger("fdml.pipeline")
    start = time.perf_counter()
    logger.log(level, "=" * 56)
    logger.log(level, "  %s", name)
    logger.log(level, "=" * 56)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(level, "  %s — done in %.1fs", name, elapsed)
        logger.log(level, "")
