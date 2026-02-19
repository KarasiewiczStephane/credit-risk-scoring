"""Tests for bias mitigation and fairness reporting."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.fairness.analyzer import FairnessMetrics
from src.fairness.mitigator import BiasMitigator
from src.fairness.report_generator import FairnessReportGenerator


@pytest.fixture
def biased_setup() -> tuple[LogisticRegression, pd.DataFrame, np.ndarray, np.ndarray]:
    """Create a trained model with biased data."""
    rng = np.random.RandomState(42)
    x, y = make_classification(n_samples=300, n_features=5, random_state=42, weights=[0.7, 0.3])
    x_df = pd.DataFrame(x, columns=[f"feat_{i}" for i in range(5)])
    groups = rng.choice(["A", "B"], 300)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_df, y)

    return model, x_df, y, groups


class TestBiasMitigator:
    """Tests for BiasMitigator."""

    def test_reweight_samples_produces_weights(
        self, biased_setup: tuple[LogisticRegression, pd.DataFrame, np.ndarray, np.ndarray]
    ) -> None:
        model, x_df, y, groups = biased_setup
        mitigator = BiasMitigator(model)
        weights = mitigator.reweight_samples(x_df, y, groups)
        assert len(weights) == len(y)
        assert all(w > 0 for w in weights)

    def test_train_with_reweighting(
        self, biased_setup: tuple[LogisticRegression, pd.DataFrame, np.ndarray, np.ndarray]
    ) -> None:
        model, x_df, y, groups = biased_setup
        mitigator = BiasMitigator(model)
        reweighted_model = mitigator.train_with_reweighting(x_df, y, groups)
        preds = reweighted_model.predict(x_df)
        assert len(preds) == len(y)
        assert "reweighted" in mitigator.mitigated_models

    def test_threshold_optimization(
        self, biased_setup: tuple[LogisticRegression, pd.DataFrame, np.ndarray, np.ndarray]
    ) -> None:
        model, x_df, y, groups = biased_setup
        mitigator = BiasMitigator(model)
        optimizer = mitigator.threshold_optimization(x_df, y, groups)
        preds = optimizer.predict(x_df, sensitive_features=groups)
        assert len(preds) == len(y)
        assert "threshold_optimized" in mitigator.mitigated_models

    def test_exponentiated_gradient(
        self, biased_setup: tuple[LogisticRegression, pd.DataFrame, np.ndarray, np.ndarray]
    ) -> None:
        model, x_df, y, groups = biased_setup
        mitigator = BiasMitigator(model)
        eg_model = mitigator.exponentiated_gradient(x_df, y, groups)
        preds = eg_model.predict(x_df)
        assert len(preds) == len(y)
        assert "exponentiated_gradient" in mitigator.mitigated_models


class TestFairnessReportGenerator:
    """Tests for FairnessReportGenerator."""

    @pytest.fixture
    def sample_metrics(self) -> tuple[dict[str, FairnessMetrics], dict[str, FairnessMetrics]]:
        """Create sample before/after metrics."""
        before = {
            "group": FairnessMetrics(
                demographic_parity_diff=0.15,
                demographic_parity_ratio=0.85,
                equalized_odds_diff=0.12,
                tpr_by_group={"A": 0.8, "B": 0.68},
                fpr_by_group={"A": 0.1, "B": 0.2},
                selection_rate_by_group={"A": 0.3, "B": 0.45},
                overall_accuracy=0.75,
            )
        }
        after = {
            "group": FairnessMetrics(
                demographic_parity_diff=0.05,
                demographic_parity_ratio=0.95,
                equalized_odds_diff=0.04,
                tpr_by_group={"A": 0.76, "B": 0.72},
                fpr_by_group={"A": 0.12, "B": 0.14},
                selection_rate_by_group={"A": 0.35, "B": 0.38},
                overall_accuracy=0.73,
            )
        }
        return before, after

    def test_comparison_report_structure(
        self,
        sample_metrics: tuple[dict[str, FairnessMetrics], dict[str, FairnessMetrics]],
    ) -> None:
        before, after = sample_metrics
        gen = FairnessReportGenerator(protected_attributes=["group"])
        report = gen.generate_comparison_report(before, after, "reweighting")
        assert isinstance(report, pd.DataFrame)
        assert "attribute" in report.columns
        assert "before" in report.columns
        assert "after" in report.columns
        assert "improvement" in report.columns
        assert len(report) == 2  # dp_diff and eo_diff

    def test_comparison_shows_improvement(
        self,
        sample_metrics: tuple[dict[str, FairnessMetrics], dict[str, FairnessMetrics]],
    ) -> None:
        before, after = sample_metrics
        gen = FairnessReportGenerator(protected_attributes=["group"])
        report = gen.generate_comparison_report(before, after, "reweighting")
        # after.dp_diff (0.05) < before.dp_diff (0.15), improvement should be positive
        dp_row = report[report["metric"] == "demographic_parity_diff"]
        assert dp_row["improvement"].values[0] > 0

    def test_full_report_structure(
        self,
        sample_metrics: tuple[dict[str, FairnessMetrics], dict[str, FairnessMetrics]],
    ) -> None:
        before, after = sample_metrics
        gen = FairnessReportGenerator(protected_attributes=["group"])
        report = gen.generate_full_report(before, {"reweighting": after})
        assert "summary" in report
        assert "before_mitigation" in report
        assert "after_mitigation" in report
        assert "recommendations" in report
        assert len(report["recommendations"]) > 0

    def test_full_report_selects_best_method(
        self,
        sample_metrics: tuple[dict[str, FairnessMetrics], dict[str, FairnessMetrics]],
    ) -> None:
        before, after = sample_metrics
        gen = FairnessReportGenerator(protected_attributes=["group"])
        report = gen.generate_full_report(before, {"reweighting": after})
        assert any("reweighting" in r for r in report["recommendations"])
