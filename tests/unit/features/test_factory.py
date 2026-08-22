from fdml.features.factory import (
    _build_feature_columns,
    create_pipeline,
    load_fe_config,
)


class TestFactory:
    def test_pipeline_has_expected_steps(self):
        cfg = load_fe_config()
        pipeline, _ = create_pipeline(cfg)
        step_names = [s[0] for s in pipeline.steps]
        assert "device" in step_names
        assert "mflags" in step_names
        assert "time" in step_names
        assert "freq" in step_names
        assert "card" in step_names
        assert "imputer" in step_names
        assert "selector" in step_names

    def test_feature_columns_is_list_of_strings(self):
        cfg = load_fe_config()
        _, feature_cols = create_pipeline(cfg)
        assert isinstance(feature_cols, list)
        assert len(feature_cols) > 0
        assert all(isinstance(c, str) for c in feature_cols)

    def test_pipeline_get_params_has_all_steps(self):
        cfg = load_fe_config()
        pipeline, _ = create_pipeline(cfg)
        params = pipeline.get_params()
        for step_name in ["device", "mflags", "time", "selector"]:
            assert f"{step_name}__enabled" in params


class TestBuildFeatureColumns:
    def test_includes_device_engineered_features(self):
        cfg = load_fe_config()
        cols = _build_feature_columns(cfg)
        assert "device_os" in cols
        assert "device_brand" in cols

    def test_includes_datetime_features(self):
        cfg = load_fe_config()
        cols = _build_feature_columns(cfg)
        assert "hour_of_day" in cols
        assert "day_of_week" in cols

    def test_includes_freq_columns(self):
        cfg = load_fe_config()
        cols = _build_feature_columns(cfg)
        freq_suffixes = [c for c in cols if c.endswith("_freq")]
        assert len(freq_suffixes) > 0
