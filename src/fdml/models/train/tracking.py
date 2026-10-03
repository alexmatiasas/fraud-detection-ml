from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


def _log_configs_and_env(cfg: Any) -> None:
    for config_path in [
        "configs/train.yaml",
        "configs/features.yaml",
        "configs/evaluate.yaml",
        "configs/mlflow.yaml",
    ]:
        if Path(config_path).exists():
            mlflow.log_artifact(config_path, "configs")

    # Resolved config (after CLI overrides) — makes the run reproducible from
    # MLflow alone, since the raw yaml files don't reflect `key=value` overrides.

    try:
        resolved = json.dumps(cfg.model_dump(mode="json"), indent=2, default=str)
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as f:
            f.write(resolved)
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, "configs")
        Path(tmp_path).unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  Config snapshot failed: %s", exc)
    for dep_path in ["pyproject.toml", "uv.lock"]:
        if Path(dep_path).exists():
            mlflow.log_artifact(dep_path, "env")


def _log_git_tags() -> None:
    for tag_cmd, tag_name in [
        (["git", "rev-parse", "--abbrev-ref", "HEAD"], "git_branch"),
        (["git", "rev-parse", "--short", "HEAD"], "git_commit"),
    ]:
        try:
            result = subprocess.run(
                tag_cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            mlflow.set_tag(tag_name, result.stdout.strip())
        except Exception as exc:  # noqa: BLE001
            logger.debug("  git tag %s unavailable: %s", tag_name, exc)
    try:
        dirty = (
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout
            != ""
        )
        mlflow.set_tag("git_dirty", "yes" if dirty else "no")
    except Exception as exc:  # noqa: BLE001
        logger.debug("  git dirty check failed: %s", exc)


def features_fingerprint(features_cfg: Any) -> str:
    """Canonical sha1 of the resolved feature configuration.

    Distinguishes runs that used different feature sets even though the
    ``features.yaml`` file on disk changed between runs. Two runs share a
    ``features_hash`` iff their feature pipeline is identical.
    """

    payload = json.dumps(features_cfg.model_dump(mode="json"), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()


def _register_abort_handler() -> None:
    """Tag the active MLflow run ``status=aborted`` on Ctrl-C / SIGTERM.

    The tag lets ``compare_runs.py`` exclude interrupted runs instead of
    showing them as metric-less noise. Runs that are killed hard (SIGKILL)
    cannot be tagged — they are simply dropped by the completeness filter.
    """
    import signal

    def _mark_aborted(signum, frame):
        try:
            if mlflow.active_run() is not None:
                mlflow.set_tag("status", "aborted")
                logger.warning("  Interrupted — run tagged status=aborted")
        except Exception as exc:  # noqa: BLE001
            logger.debug("  failed to mark run aborted: %s", exc)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _mark_aborted)
    signal.signal(signal.SIGTERM, _mark_aborted)
