import numpy as np
import pandas as pd
import pytest

from fdml.models.split import (
    StratifiedKFoldSplitter,
    StratifiedSplitter,
    TemporalSplitter,
)


@pytest.fixture()
def ts_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 1000
    return pd.DataFrame(
        {
            "TransactionDT": rng.permutation(
                np.arange(0, 86400 * n, 86400, dtype=np.int32)
            ),
            "isFraud": rng.choice([0, 1], n, p=[0.965, 0.035]),
            "feature_a": rng.normal(0, 1, n),
        }
    )


class TestTemporalSplitter:
    def test_split_preserves_time_order(self, ts_data: pd.DataFrame):
        # Arrange
        X = ts_data.drop(columns=["isFraud"])
        y = ts_data["isFraud"].values
        splitter = TemporalSplitter(time_col="TransactionDT", test_size=0.2)

        # Act
        train_idx, val_idx = next(splitter.split(X, y))

        # Assert
        train_dt = X.iloc[train_idx]["TransactionDT"].max()
        val_dt = X.iloc[val_idx]["TransactionDT"].min()
        assert train_dt < val_dt, "Train max time must be before val min time"

    def test_split_raises_on_missing_col(self, ts_data: pd.DataFrame):
        # Arrange
        X = ts_data.drop(columns=["TransactionDT"])
        y = ts_data["isFraud"].values
        splitter = TemporalSplitter(time_col="TransactionDT")

        # Act / Assert
        with pytest.raises(ValueError, match="not found"):
            next(splitter.split(X, y))

    def test_split_respects_test_size(self, ts_data: pd.DataFrame):
        # Arrange
        X = ts_data.drop(columns=["isFraud"])
        y = ts_data["isFraud"].values
        test_size = 0.2

        # Act
        splitter = TemporalSplitter(test_size=test_size)
        train_idx, val_idx = next(splitter.split(X, y))

        # Assert
        n = len(X)
        assert abs(len(val_idx) / n - test_size) < 0.01
        assert len(train_idx) + len(val_idx) == n

    def test_get_n_splits_returns_one(self, ts_data: pd.DataFrame):
        X = ts_data.drop(columns=["isFraud"])
        assert TemporalSplitter().get_n_splits(X) == 1

    def test_split_embargo_zero_keeps_all_rows(self, ts_data: pd.DataFrame):
        # Arrange
        X = ts_data.drop(columns=["isFraud"])
        y = ts_data["isFraud"].values
        splitter = TemporalSplitter(test_size=0.2, embargo_seconds=0)

        # Act
        train_idx, val_idx = next(splitter.split(X, y))

        # Assert
        assert len(train_idx) + len(val_idx) == len(X)

    def test_split_embargo_drops_boundary_rows(self, ts_data: pd.DataFrame):
        # Arrange
        X = ts_data.drop(columns=["isFraud"])
        y = ts_data["isFraud"].values
        embargo = 86400  # one day in the fixture
        splitter = TemporalSplitter(test_size=0.2, embargo_seconds=embargo)

        # Act
        train_idx, val_idx = next(splitter.split(X, y))

        # Assert — the embargo window right before val is dropped
        train_dt_max = X.iloc[train_idx]["TransactionDT"].max()
        val_dt_min = X.iloc[val_idx]["TransactionDT"].min()
        assert val_dt_min - train_dt_max >= embargo
        assert len(train_idx) + len(val_idx) < len(X), "embargo must drop rows"


class TestStratifiedSplitter:
    def test_split_preserves_class_ratio(self, ts_data: pd.DataFrame):
        # Arrange
        X = ts_data.drop(columns=["isFraud"])
        y = ts_data["isFraud"].values
        splitter = StratifiedSplitter(test_size=0.2, random_state=42)

        # Act
        train_idx, val_idx = next(splitter.split(X, y))

        # Assert
        orig_ratio = y.mean()
        val_ratio = y[val_idx].mean()
        assert abs(val_ratio - orig_ratio) < 0.02

    def test_split_respects_test_size(self, ts_data: pd.DataFrame):
        # Arrange
        X = ts_data.drop(columns=["isFraud"])
        y = ts_data["isFraud"].values
        test_size = 0.2
        splitter = StratifiedSplitter(test_size=test_size)

        # Act
        train_idx, val_idx = next(splitter.split(X, y))

        # Assert
        n = len(X)
        assert abs(len(val_idx) / n - test_size) < 0.01

    def test_get_n_splits_returns_one(self, ts_data: pd.DataFrame):
        X = ts_data.drop(columns=["isFraud"])
        assert StratifiedSplitter().get_n_splits(X) == 1

    def test_split_is_deterministic(self, ts_data: pd.DataFrame):
        # Arrange
        X = ts_data.drop(columns=["isFraud"])
        y = ts_data["isFraud"].values

        # Act
        splitter_a = StratifiedSplitter(random_state=42)
        splitter_b = StratifiedSplitter(random_state=42)
        _, val_a = next(splitter_a.split(X, y))
        _, val_b = next(splitter_b.split(X, y))

        # Assert
        np.testing.assert_array_equal(val_a, val_b)


class TestStratifiedKFoldSplitter:
    def test_split_returns_correct_number_of_folds(self, ts_data: pd.DataFrame):
        # Arrange
        X = ts_data.drop(columns=["isFraud"])
        y = ts_data["isFraud"].values
        n_splits = 5
        splitter = StratifiedKFoldSplitter(n_splits=n_splits)

        # Act
        splits = list(splitter.split(X, y))
        n_train = [len(t) for t, _ in splits]

        # Assert
        assert len(splits) == n_splits
        assert all(0.75 < n / len(X) < 0.85 for n in n_train)

    def test_preserves_class_ratio_in_each_fold(self, ts_data: pd.DataFrame):
        # Arrange
        X = ts_data.drop(columns=["isFraud"])
        y = ts_data["isFraud"].values
        splitter = StratifiedKFoldSplitter(n_splits=5)
        orig_ratio = y.mean()

        # Act / Assert
        for _, val_idx in splitter.split(X, y):
            val_ratio = y[val_idx].mean()
            assert abs(val_ratio - orig_ratio) < 0.03

    def test_get_n_splits(self, ts_data: pd.DataFrame):
        X = ts_data.drop(columns=["isFraud"])
        assert StratifiedKFoldSplitter(n_splits=5).get_n_splits(X) == 5
        assert StratifiedKFoldSplitter(n_splits=10).get_n_splits(X) == 10
