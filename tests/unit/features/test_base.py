import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from src.features.base import BaseFeatureTransformer


class TestBaseFeatureTransformer:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseFeatureTransformer()

    def test_get_params_returns_enabled(self):
        class Concrete(BaseFeatureTransformer):
            def transform(self, X):
                return X

        t = Concrete(enabled=False)
        params = t.get_params()
        assert params["enabled"] is False

    def test_set_params_updates_enabled(self):
        class Concrete(BaseFeatureTransformer):
            def transform(self, X):
                return X

        t = Concrete(enabled=True)
        t.set_params(enabled=False)
        assert t.enabled is False

    def test_fit_returns_self(self):
        class Concrete(BaseFeatureTransformer):
            def transform(self, X):
                return X

        t = Concrete()
        result = t.fit(pd.DataFrame())
        assert result is t

    def test_inherits_from_sklearn_base_estimator(self):
        class Concrete(BaseFeatureTransformer):
            def transform(self, X):
                return X

        assert isinstance(Concrete(), BaseEstimator)
