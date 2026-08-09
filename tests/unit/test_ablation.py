from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fdml.features.factory import create_pipeline
from fdml.features.id_encoder import IdCodeEncoder
from fdml.models.train.ablation import (
    FEATURE_GROUPS,
    load_base_features,
    variant_features,
)


class TestIdCodeEncoder:
    def test_recasts_listed_columns_to_category(self):
        X = pd.DataFrame(
            {
                "id_03": [11.0, 12.0, 11.0, np.nan],
                "id_04": [1.0, 2.0, 3.0, 4.0],
                "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            }
        )
        out = IdCodeEncoder(columns=["id_03", "id_04"]).transform(X)
        assert isinstance(out["id_03"].dtype, pd.CategoricalDtype)
        assert isinstance(out["id_04"].dtype, pd.CategoricalDtype)
        assert out["TransactionAmt"].dtype.name == "float64"
        assert out.columns.tolist() == X.columns.tolist()

    def test_disabled_returns_unchanged(self):
        X = pd.DataFrame({"id_03": [1.0, 2.0]})
        out = IdCodeEncoder(enabled=False, columns=["id_03"]).transform(X)
        pd.testing.assert_frame_equal(out, X)

    def test_skips_columns_not_present(self):
        X = pd.DataFrame({"id_03": [1.0, 2.0]})
        out = IdCodeEncoder(columns=["id_99"]).transform(X)
        pd.testing.assert_frame_equal(out, X)

    def test_does_not_double_recast(self):
        X = pd.DataFrame({"id_03": pd.Series([1.0, 2.0], dtype="category")})
        out = IdCodeEncoder(columns=["id_03"]).transform(X)
        assert isinstance(out["id_03"].dtype, pd.CategoricalDtype)


class TestFeatureGroups:
    def test_expected_groups_registered(self):
        assert list(FEATURE_GROUPS) == [
            "vesta_features",
            "count_corr_filter",
            "cyclical",
            "has_identity",
            "log_amount",
            "is_round_amount",
            "email_domain",
            "device_os",
            "device_brand",
            "card_aggregations",
            "frequency_encoding",
            "id_codes_encoding",
        ]

    def test_baseline_unchanged(self):
        base = load_base_features()
        variant = variant_features(None)
        assert variant == base

    @pytest.mark.parametrize(
        ("group", "flip_field", "expected"),
        [
            ("vesta_features", "transaction.vesta_features.include", False),
            ("count_corr_filter", "transaction.count_corr_filter.include", False),
            ("cyclical", "engineered.cyclical", False),
            ("has_identity", "engineered.has_identity", False),
            ("log_amount", "engineered.log_amount", False),
            ("is_round_amount", "engineered.is_round_amount", False),
            ("email_domain", "engineered.email_domain", False),
            ("device_os", "engineered.device_os", False),
            ("device_brand", "engineered.device_brand", False),
            ("card_aggregations", "engineered.card_aggregations", None),
            ("frequency_encoding", "engineered.frequency_encoding", []),
            ("id_codes_encoding", "engineered.id_codes_encoding", True),
        ],
    )
    def test_toggles_group(self, group: str, flip_field: str, expected: object):
        mutate = FEATURE_GROUPS[group]
        variant = variant_features(mutate)

        def _nested(obj, dotted_path: str):
            for part in dotted_path.split("."):
                obj = getattr(obj, part)
            return obj

        assert _nested(variant, flip_field) == expected

    def test_each_variant_drops_its_own_features(self):
        base = load_base_features()
        _, base_cols = create_pipeline(base)

        # vesta_features and count_corr_filter are covered by their vfilter-step
        # tests: V/C columns are dropped at fit time by the VFeatureFilter step
        # (or appended by the selector), not listed in feature_columns, so
        # nothing changes here. id_codes_encoding is an "add" experiment (dtype
        # change only, same columns) — covered by TestIdCodeEncoder.
        for group, mutate in FEATURE_GROUPS.items():
            if group in ("vesta_features", "count_corr_filter", "id_codes_encoding"):
                continue
            variant = variant_features(mutate)
            _, cols = create_pipeline(variant)
            removed = set(base_cols) - set(cols)
            assert removed, f"group {group} removed nothing"

    @pytest.mark.parametrize(
        ("group", "dropped_column"),
        [
            ("cyclical", "hour_of_day_sin"),
            ("has_identity", "has_identity"),
            ("log_amount", "TransactionAmt_log"),
            ("is_round_amount", "is_round_amount"),
            ("email_domain", "p_r_domain_match"),
            ("device_os", "device_os"),
            ("device_brand", "device_brand"),
            ("card_aggregations", "card_mean_transactionamt"),
            ("frequency_encoding", "P_emaildomain_freq"),
        ],
    )
    def test_specific_column_dropped(self, group: str, dropped_column: str):
        variant = variant_features(FEATURE_GROUPS[group])
        _, cols = create_pipeline(variant)
        assert dropped_column not in cols

    def test_baseline_keeps_all_group_features(self):
        _, cols = create_pipeline(load_base_features())
        for col in [
            "has_identity",
            "TransactionAmt_log",
            "is_round_amount",
            "p_r_domain_match",
            "device_os",
            "device_brand",
            "hour_of_day_sin",
            "card_mean_transactionamt",
            "P_emaildomain_freq",
        ]:
            assert col in cols

    def test_vesta_off_removes_vfilter_step(self):
        variant = variant_features(FEATURE_GROUPS["vesta_features"])
        pipeline, _ = create_pipeline(variant)
        assert "vfilter" not in dict(pipeline.steps)

    def test_count_corr_off_removes_vfilter_c_step(self):
        variant = variant_features(FEATURE_GROUPS["count_corr_filter"])
        pipeline, _ = create_pipeline(variant)
        assert "vfilter_c" not in dict(pipeline.steps)

    def test_id_codes_on_adds_idcodes_step(self):
        variant = variant_features(FEATURE_GROUPS["id_codes_encoding"])
        pipeline, _ = create_pipeline(variant)
        assert "idcodes" in dict(pipeline.steps)

    def test_id_codes_off_by_default_no_step(self):
        pipeline, _ = create_pipeline(load_base_features())
        assert "idcodes" not in dict(pipeline.steps)

    def test_id_codes_variant_keeps_same_columns(self):
        base = load_base_features()
        _, base_cols = create_pipeline(base)
        variant = variant_features(FEATURE_GROUPS["id_codes_encoding"])
        _, cols = create_pipeline(variant)
        assert cols == base_cols
