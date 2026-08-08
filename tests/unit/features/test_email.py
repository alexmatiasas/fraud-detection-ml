import numpy as np
import pandas as pd

from fdml.features.email import EmailFeatureExtractor


class TestEmailFeatureExtractor:
    def test_microsoft_domains_grouped(self):
        df = pd.DataFrame(
            {
                "P_emaildomain": ["Outlook.com", "gmail.com", "hotmail.com", None],
                "R_emaildomain": ["outlook.com", "yahoo.com", "live.com", "gmail.com"],
            }
        )
        result = EmailFeatureExtractor().transform(df)
        assert list(result["P_emaildomain"]) == [
            "microsoft",
            "gmail.com",
            "microsoft",
            None,
        ]
        assert list(result["R_emaildomain"]) == [
            "microsoft",
            "yahoo.com",
            "microsoft",
            "gmail.com",
        ]

    def test_domain_match(self):
        df = pd.DataFrame(
            {
                "P_emaildomain": ["gmail.com", "outlook.com", "gmail.com", None],
                "R_emaildomain": ["gmail.com", "hotmail.com", "yahoo.com", "yahoo.com"],
            }
        )
        result = EmailFeatureExtractor().transform(df)
        values = result["p_r_domain_match"]
        assert list(values.iloc[:3]) == [1.0, 1.0, 0.0]
        assert np.isnan(values.iloc[3])

    def test_use_grouping_only(self):
        df = pd.DataFrame(
            {
                "P_emaildomain": ["outlook.com"],
                "R_emaildomain": ["gmail.com"],
            }
        )
        result = EmailFeatureExtractor(use_match=False).transform(df)
        assert "p_r_domain_match" not in result.columns
        assert result["P_emaildomain"].iloc[0] == "microsoft"

    def test_disabled_unchanged(self):
        df = pd.DataFrame({"P_emaildomain": ["outlook.com"]})
        result = EmailFeatureExtractor(enabled=False).transform(df)
        assert result["P_emaildomain"].iloc[0] == "outlook.com"

    def test_no_email_columns_unchanged(self):
        df = pd.DataFrame({"a": [1]})
        result = EmailFeatureExtractor().transform(df)
        assert list(result.columns) == ["a"]
