"""Tests for fairness analysis."""

import numpy as np
import pandas as pd
import pytest

from src.fairness.analyzer import FairnessAnalyzer, FairnessMetrics


@pytest.fixture
def synthetic_fair_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Create synthetic data where predictions are perfectly fair."""
    rng = np.random.RandomState(42)
    n = 200
    y_true = rng.choice([0, 1], n, p=[0.7, 0.3])
    y_pred = y_true.copy()  # Perfect predictions = fair
    y_prob = y_pred.astype(float)
    sensitive = pd.DataFrame({"group": rng.choice(["A", "B"], n)})
    return y_true, y_pred, y_prob, sensitive


@pytest.fixture
def synthetic_unfair_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Create synthetic data with biased predictions."""
    rng = np.random.RandomState(42)
    n = 200
    groups = rng.choice(["A", "B"], n)
    y_true = rng.choice([0, 1], n, p=[0.7, 0.3])
    y_pred = y_true.copy()
    # Make predictions biased for group B
    mask_b = groups == "B"
    y_pred[mask_b & (y_true == 0)] = 1  # More false positives for group B
    y_prob = y_pred.astype(float)
    sensitive = pd.DataFrame({"group": groups})
    return y_true, y_pred, y_prob, sensitive


class TestFairnessAnalyzer:
    """Tests for FairnessAnalyzer."""

    def test_analyze_returns_metrics(
        self,
        synthetic_fair_data: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ) -> None:
        y_true, y_pred, y_prob, sensitive = synthetic_fair_data
        analyzer = FairnessAnalyzer(protected_attributes=["group"])
        result = analyzer.analyze(y_true, y_pred, y_prob, sensitive)
        assert "group" in result
        assert isinstance(result["group"], FairnessMetrics)

    def test_fair_predictions_low_disparity(
        self,
        synthetic_fair_data: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ) -> None:
        y_true, y_pred, y_prob, sensitive = synthetic_fair_data
        analyzer = FairnessAnalyzer(protected_attributes=["group"])
        result = analyzer.analyze(y_true, y_pred, y_prob, sensitive)
        metrics = result["group"]
        # Perfect predictions should have very low disparity
        assert abs(metrics.equalized_odds_diff) < 0.15

    def test_unfair_predictions_high_disparity(
        self,
        synthetic_unfair_data: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ) -> None:
        y_true, y_pred, y_prob, sensitive = synthetic_unfair_data
        analyzer = FairnessAnalyzer(protected_attributes=["group"])
        result = analyzer.analyze(y_true, y_pred, y_prob, sensitive)
        metrics = result["group"]
        assert abs(metrics.demographic_parity_diff) > 0.1

    def test_tpr_fpr_by_group(
        self,
        synthetic_fair_data: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ) -> None:
        y_true, y_pred, y_prob, sensitive = synthetic_fair_data
        analyzer = FairnessAnalyzer(protected_attributes=["group"])
        result = analyzer.analyze(y_true, y_pred, y_prob, sensitive)
        metrics = result["group"]
        assert "A" in metrics.tpr_by_group
        assert "B" in metrics.tpr_by_group
        assert "A" in metrics.fpr_by_group
        assert "B" in metrics.fpr_by_group

    def test_check_fairness_constraints(
        self,
        synthetic_fair_data: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ) -> None:
        y_true, y_pred, y_prob, sensitive = synthetic_fair_data
        analyzer = FairnessAnalyzer(protected_attributes=["group"])
        analyzer.analyze(y_true, y_pred, y_prob, sensitive)
        constraints = analyzer.check_fairness_constraints(threshold=0.2)
        assert "group_demographic_parity" in constraints
        assert "group_equalized_odds" in constraints

    def test_disparity_report_structure(
        self,
        synthetic_fair_data: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ) -> None:
        y_true, y_pred, y_prob, sensitive = synthetic_fair_data
        analyzer = FairnessAnalyzer(protected_attributes=["group"])
        analyzer.analyze(y_true, y_pred, y_prob, sensitive)
        report = analyzer.get_disparity_report()
        assert isinstance(report, pd.DataFrame)
        assert "protected_attribute" in report.columns
        assert "demographic_parity_diff" in report.columns
        assert "equalized_odds_diff" in report.columns
        assert len(report) == 1

    def test_to_dict_serializable(
        self,
        synthetic_fair_data: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ) -> None:
        y_true, y_pred, y_prob, sensitive = synthetic_fair_data
        analyzer = FairnessAnalyzer(protected_attributes=["group"])
        analyzer.analyze(y_true, y_pred, y_prob, sensitive)
        d = analyzer.to_dict()
        assert "group" in d
        assert "demographic_parity_difference" in d["group"]
        assert "overall_accuracy" in d["group"]

    def test_missing_attribute_skipped(
        self,
        synthetic_fair_data: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ) -> None:
        y_true, y_pred, y_prob, sensitive = synthetic_fair_data
        analyzer = FairnessAnalyzer(protected_attributes=["nonexistent"])
        result = analyzer.analyze(y_true, y_pred, y_prob, sensitive)
        assert len(result) == 0

    def test_overall_accuracy_correct(
        self,
        synthetic_fair_data: tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame],
    ) -> None:
        y_true, y_pred, y_prob, sensitive = synthetic_fair_data
        analyzer = FairnessAnalyzer(protected_attributes=["group"])
        result = analyzer.analyze(y_true, y_pred, y_prob, sensitive)
        # Perfect predictions should give 100% accuracy
        assert result["group"].overall_accuracy == pytest.approx(1.0)
