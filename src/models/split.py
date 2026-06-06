from typing import Iterator, Optional

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

    Args:
        time_col: Column containing the temporal ordering (e.g. TransactionDT).
        test_size: Fraction of rows to hold out as validation set.
    """

    def __init__(self, time_col: str = "TransactionDT", test_size: float = 0.2):
        self.time_col = time_col
        self.test_size = test_size

    def get_n_splits(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> int:
        return 1

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        df = X.copy()
        if self.time_col not in df.columns:
            msg = f"Column '{self.time_col}' not found in X"
            raise ValueError(msg)

        df = df.sort_values(self.time_col)
        n = len(df)
        split_idx = int(n * (1 - self.test_size))
        train_idx = df.index[:split_idx].to_numpy()
        val_idx = df.index[split_idx:].to_numpy()
        yield train_idx, val_idx

    def _iter_test_indices(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> Iterator[np.ndarray]:
        df = X.sort_values(self.time_col)
        n = len(df)
        split_idx = int(n * (1 - self.test_size))
        yield df.index[split_idx:].to_numpy()


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

    def get_n_splits(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> int:
        return 1

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        train_idx, val_idx = train_test_split(
            np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        yield train_idx, val_idx

    def _iter_test_indices(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> Iterator[np.ndarray]:
        _, val_idx = train_test_split(
            np.arange(len(X)),
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        yield val_idx


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

    def get_n_splits(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> int:
        return self.n_splits

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        yield from self._kfold.split(X, y)

    def _iter_test_indices(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> Iterator[np.ndarray]:
        yield from self._kfold._iter_test_indices(X, y)
