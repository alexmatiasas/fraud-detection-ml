from __future__ import annotations

from fdml.models.evaluate.model_card import generate_model_card


class TestGenerateModelCard:
    @staticmethod
    def _base(**overrides):
        defaults = dict(
            model_name="lightgbm",
            metrics={
                "roc_auc": 0.91,
                "average_precision": 0.5,
                "precision": 0.7,
                "recall": 0.5,
            },
            ci=None,
            best_threshold=0.3,
            best_f1=0.75,
            n_features=100,
            n_train=50_000,
            n_val=10_000,
            fraud_rate=0.035,
            split_strategy="temporal",
        )
        defaults.update(overrides)
        return defaults

    def test_basic_card(self):
        card = generate_model_card(**self._base())
        assert "# Model Card" in card
        assert "lightgbm" in card
        assert "ROC AUC" in card
        assert "0.9100" in card

    def test_with_ci(self):
        card = generate_model_card(**self._base(ci=(0.48, 0.52)))
        assert "AP 95% CI" in card
        assert "[0.4800, 0.5200]" in card

    def test_with_brier(self):
        card = generate_model_card(**self._base(brier=0.05))
        assert "Brier" in card

    def test_with_f_beta(self):
        card = generate_model_card(**self._base(f_beta=0.65))
        assert "F2" in card

    def test_with_expected_cost(self):
        card = generate_model_card(
            **self._base(expected_cost=0.08, cost_best_threshold=0.25)
        )
        assert "Expected cost" in card

    def test_with_recall_at_k(self):
        card = generate_model_card(
            **self._base(recall_at_k={"0.0100": 0.4, "0.0500": 0.6})
        )
        assert "Recall@top" in card

    def test_with_auc_adv_drift(self):
        card = generate_model_card(**self._base(auc_adv=0.85))
        assert "possible drift detected" in card

    def test_with_auc_adv_no_drift(self):
        card = generate_model_card(**self._base(auc_adv=0.6))
        assert "no significant drift" in card

    def test_with_feature_importance(self):
        card = generate_model_card(
            **self._base(feature_importance=[("V1", 0.5), ("V2", 0.3)])
        )
        assert "Top-10 features" in card
        assert "V1" in card

    def test_with_all_optionals(self):
        card = generate_model_card(
            **self._base(
                ci=(0.48, 0.52),
                brier=0.05,
                f_beta=0.65,
                expected_cost=0.08,
                cost_best_threshold=0.25,
                recall_at_k={"0.0100": 0.4},
                auc_adv=0.85,
                feature_importance=[("V1", 0.5)],
            )
        )
        assert "# Model Card" in card
        assert "Performance" in card
        assert "Adversarial validation" in card
        assert "Top-10 features" in card
