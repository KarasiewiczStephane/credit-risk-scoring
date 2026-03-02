"""Tests for the credit risk scoring dashboard data generators."""

import pandas as pd

from src.dashboard.app import (
    generate_fairness_metrics,
    generate_model_metrics,
    generate_pd_by_score,
    generate_score_distribution,
    generate_scorecard_points,
)


class TestScoreDistribution:
    def test_returns_dataframe(self) -> None:
        df = generate_score_distribution()
        assert isinstance(df, pd.DataFrame)

    def test_has_1000_scores(self) -> None:
        df = generate_score_distribution()
        assert len(df) == 1000

    def test_scores_in_valid_range(self) -> None:
        df = generate_score_distribution()
        assert (df["score"] >= 300).all()
        assert (df["score"] <= 850).all()

    def test_reproducible(self) -> None:
        df1 = generate_score_distribution(seed=99)
        df2 = generate_score_distribution(seed=99)
        pd.testing.assert_frame_equal(df1, df2)


class TestModelMetrics:
    def test_returns_dict(self) -> None:
        metrics = generate_model_metrics()
        assert isinstance(metrics, dict)

    def test_has_required_keys(self) -> None:
        metrics = generate_model_metrics()
        for key in ["auc_roc", "gini", "ks_statistic", "accuracy", "precision", "recall"]:
            assert key in metrics

    def test_auc_bounded(self) -> None:
        metrics = generate_model_metrics()
        assert 0.5 <= metrics["auc_roc"] <= 1.0

    def test_gini_derived_from_auc(self) -> None:
        metrics = generate_model_metrics()
        expected_gini = round(2 * metrics["auc_roc"] - 1, 4)
        assert metrics["gini"] == expected_gini


class TestScorecardPoints:
    def test_returns_dataframe(self) -> None:
        df = generate_scorecard_points()
        assert isinstance(df, pd.DataFrame)

    def test_has_correct_feature_count(self) -> None:
        df = generate_scorecard_points()
        assert len(df) == 8

    def test_iv_positive(self) -> None:
        df = generate_scorecard_points()
        assert (df["iv"] > 0).all()


class TestFairnessMetrics:
    def test_returns_dataframe(self) -> None:
        df = generate_fairness_metrics()
        assert isinstance(df, pd.DataFrame)

    def test_has_groups(self) -> None:
        df = generate_fairness_metrics()
        assert len(df) == 5

    def test_approval_rate_bounded(self) -> None:
        df = generate_fairness_metrics()
        assert (df["approval_rate"] >= 0).all()
        assert (df["approval_rate"] <= 1).all()


class TestPdByScore:
    def test_returns_dataframe(self) -> None:
        df = generate_pd_by_score()
        assert isinstance(df, pd.DataFrame)

    def test_has_score_ranges(self) -> None:
        df = generate_pd_by_score()
        assert len(df) == 6

    def test_pd_bounded(self) -> None:
        df = generate_pd_by_score()
        assert (df["probability_of_default"] >= 0).all()
        assert (df["probability_of_default"] <= 1).all()

    def test_count_positive(self) -> None:
        df = generate_pd_by_score()
        assert (df["count"] > 0).all()
