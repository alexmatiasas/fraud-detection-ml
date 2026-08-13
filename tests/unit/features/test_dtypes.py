import numpy as np
import pandas as pd

from fdml.features.dtypes import DtypeOptimizer


class TestDtypeOptimizer:
    def test_float64_to_float32(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [1, 2]})
        result = DtypeOptimizer().transform(df)
        assert result["a"].dtype == np.dtype("float32")
        assert result["b"].dtype == np.dtype("int32")

    def test_large_int64_not_downcast(self):
        df = pd.DataFrame({"a": [0, 2**40]})
        result = DtypeOptimizer().transform(df)
        assert result["a"].dtype == np.dtype("int64")

    def test_categorical_preserved(self):
        df = pd.DataFrame({"c": pd.Categorical(["x", "y"])})
        result = DtypeOptimizer().transform(df)
        assert isinstance(result["c"].dtype, pd.CategoricalDtype)

    def test_disabled_unchanged(self):
        df = pd.DataFrame({"a": [1.0]})
        result = DtypeOptimizer(enabled=False).transform(df)
        assert result["a"].dtype == np.dtype("float64")
