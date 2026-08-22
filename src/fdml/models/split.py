from collections.abc import Generator
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedKFold,
    train_test_split,
)


class TemporalSplitter(BaseCrossValidator):
    """Split data by time order — preserves temporal ordering.

    Sorts by *time_col*, takes the first ``(1 - test_size)`` rows as the
    training set and the rest as the validation set.  No shuffling, no
    stratification — the only honest way to evaluate a forward-looking model.

    An *embargo* window of ``embargo_seconds`` is dropped immediately before
    the train/validation boundary: aggregate features (card stats, frequency
    encodings) are learned from train, so a temporally adjacent row would
    inherit those values and leak the future into the past.  Rows within
    ``[boundary - embargo_seconds, boundary)`` are discarded.

    Args:
        time_col: Column containing the temporal ordering (e.g. TransactionDT).
        test_size: Fraction of rows to hold out as validation set.
        embargo_seconds: Time gap (in ``time_col`` units) dropped between train
            and validation.  The Kaggle train/test windows are ~30 days apart.
    """

    def __init__(
        self,
        time_col: str = "TransactionDT",
        test_size: float = 0.2,
        embargo_seconds: int = 0,
    ):
        self.time_col = time_col
        self.test_size = test_size
        self.embargo_seconds = embargo_seconds

    def get_n_splits(  # type: ignore[reportIncompatibleMethodOverride]
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> int:
        return 1

    def _time_split(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df = X.sort_values(self.time_col)
        dt = df[self.time_col]
        n = len(dt)
        split_idx = int(n * (1 - self.test_size))
        boundary = dt.iloc[split_idx]
        train_mask = dt < boundary - self.embargo_seconds
        val_mask = dt >= boundary
        return df.index[train_mask].to_numpy(), df.index[val_mask].to_numpy()

    def split(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        if self.time_col not in X.columns:
            msg = f"Column '{self.time_col}' not found in X"
            raise ValueError(msg)
        yield self._time_split(X)

    def _iter_test_indices(  # type: ignore[reportIncompatibleMethodOverride]
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> Generator[np.ndarray, None, None]:
        yield self._time_split(X)[1]


class StratifiedSplitter(BaseCrossValidator):
    """Stratified random train/validation split.

    Wraps :func:`sklearn.model_selection.train_test_split` with
    ``stratify=y`` to preserve class proportions in both folds.

    Args:
        test_size: Fraction of rows for the validation set.
        random_state: PRNG seed for reproducibility.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def get_n_splits(  # type: ignore[reportIncompatibleMethodOverride]
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> int:
        return 1

    def split(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        train_idx, val_idx = train_test_split(
            np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        yield np.asarray(train_idx), np.asarray(val_idx)

    def _iter_test_indices(  # type: ignore[reportIncompatibleMethodOverride]
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> Generator[np.ndarray, None, None]:
        _, val_idx = train_test_split(
            np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        yield np.asarray(val_idx)


class StratifiedKFoldSplitter(BaseCrossValidator):
    """Stratified K-Fold cross-validator.

    Wraps :class:`sklearn.model_selection.StratifiedKFold`.

    Args:
        n_splits: Number of folds.
        shuffle: Whether to shuffle before splitting.
        random_state: PRNG seed.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self._kfold = StratifiedKFold(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def get_n_splits(  # type: ignore[reportIncompatibleMethodOverride]
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> int:
        return self.n_splits

    def split(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        yield from self._kfold.split(X, y)

    def _iter_test_indices(  # type: ignore[reportIncompatibleMethodOverride]
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> Generator[np.ndarray, None, None]:
        yield from self._kfold._iter_test_indices(X, y)
