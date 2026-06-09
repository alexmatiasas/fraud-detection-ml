from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def encode_categoricals(
    X_train: pd.DataFrame, X_val: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return X_train, X_val

    logger.info("Encoding %d categorical columns → int32 ordinal codes", len(cat_cols))
    X_train = X_train.copy()
    X_val = X_val.copy()
    for col in cat_cols:
        train_vals = X_train[col].astype(str)
        categories = sorted(train_vals.unique())
        cat_to_code = {c: i for i, c in enumerate(categories)}
        X_train[col] = train_vals.map(cat_to_code).astype("int32")

        val_str = X_val[col].astype(str)
        X_val[col] = val_str.map(cat_to_code).fillna(-1).astype("int32")

    return X_train, X_val
