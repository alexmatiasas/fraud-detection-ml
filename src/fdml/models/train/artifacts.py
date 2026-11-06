from __future__ import annotations

import logging
import warnings
from typing import Any

import pandas as pd

from fdml.schemas.mlflow import MlflowFullConfig

logger = logging.getLogger(__name__)


def _run_native_evaluation(
    mlflow_cfg: MlflowFullConfig,
    model_info: Any,
    X_val_raw: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Native ``mlflow.models.evaluate`` on the logged full pipeline.

    Runs over RAW transactions (the pipeline's serving contract), computes the
    standard classifier metric set, interactive ROC/PR/confusion-matrix
    artifacts and optionally a SHAP explainer, and links everything to both
    the run and the LoggedModel. Failures degrade to a warning — the custom
    evaluate runner has already produced the authoritative metrics.
    """
    cfg = mlflow_cfg.native_evaluate
    if not cfg.enabled or model_info is None:
        return

    df = X_val_raw.copy()
    # pandas Categorical breaks the default evaluator ("Cannot interpret
    # CategoricalDtype ... as a data type"); the served pipeline accepts
    # object columns equally well (CategoryEncoder handles both).
    cat_cols = df.select_dtypes(include="category").columns
    if len(cat_cols):
        df[cat_cols] = df[cat_cols].astype(object)
    df["isFraud"] = y_val.to_numpy()
    if cfg.max_rows > 0 and len(df) > cfg.max_rows:
        df = df.sample(n=cfg.max_rows, random_state=0)
    logger.info(
        "  Native evaluation: %s rows (explainer=%s)",
        f"{len(df):,}",
        cfg.log_explainer,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Hint: Inferred schema contains integer column"
            )
            from mlflow.models import evaluate

            evaluate(
                model_info.model_uri,
                df,
                targets="isFraud",
                model_type="classifier",
                evaluator_config={"log_explainer": cfg.log_explainer},
            )
            logger.info("  ✓ Native evaluation logged")
    except Exception as exc:  # noqa: BLE001
        logger.warning("  Native evaluation failed: %s", exc)
