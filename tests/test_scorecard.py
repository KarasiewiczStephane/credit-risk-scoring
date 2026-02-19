"""Tests for traditional credit scorecard."""

import numpy as np
import pandas as pd
import pytest

from src.models.scorecard import CreditScorecard, ScorecardConfig


@pytest.fixture
def sample_credit_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample data for scorecard testing."""
    rng = np.random.RandomState(42)
    n = 200
    x = pd.DataFrame(
        {
            "feat_a": rng.choice([0, 1, 2, 3, 4], n),
            "feat_b": rng.choice([0, 1, 2], n),
            "feat_c": rng.choice([0, 1, 2, 3, 4, 5], n),
        }
    )
    y = pd.Series(rng.choice([0, 1], n, p=[0.7, 0.3]))
    return x, y


@pytest.fixture
def fitted_scorecard(
    sample_credit_data: tuple[pd.DataFrame, pd.Series],
) -> CreditScorecard:
    """Create a fitted scorecard."""
    x, y = sample_credit_data
    scorecard = CreditScorecard()
    scorecard.fit(x, y)
    return scorecard


class TestScorecardConfig:
    """Tests for ScorecardConfig."""

    def test_default_values(self) -> None:
        config = ScorecardConfig()
        assert config.pdo == 20
        assert config.base_score == 600
        assert config.base_odds == 50.0

    def test_custom_values(self) -> None:
        config = ScorecardConfig(pdo=30, base_score=700, base_odds=100.0)
        assert config.pdo == 30
        assert config.base_score == 700


class TestScalingFactors:
    """Tests for PDO scaling formula."""

    def test_factor_formula(self) -> None:
        scorecard = CreditScorecard(ScorecardConfig(pdo=20))
        factor, _ = scorecard._calculate_scaling_factors()
        expected_factor = 20 / np.log(2)
        assert factor == pytest.approx(expected_factor)

    def test_offset_formula(self) -> None:
        config = ScorecardConfig(pdo=20, base_score=600, base_odds=50.0)
        scorecard = CreditScorecard(config)
        _, offset = scorecard._calculate_scaling_factors()
        expected = 600 - (20 / np.log(2)) * np.log(50.0)
        assert offset == pytest.approx(expected)


class TestCreditScorecard:
    """Tests for CreditScorecard."""

    def test_fit_creates_feature_scorecards(self, fitted_scorecard: CreditScorecard) -> None:
        assert len(fitted_scorecard.feature_scorecards) == 3
        assert "feat_a" in fitted_scorecard.feature_scorecards
        assert "feat_b" in fitted_scorecard.feature_scorecards
        assert "feat_c" in fitted_scorecard.feature_scorecards

    def test_score_returns_integers(
        self,
        sample_credit_data: tuple[pd.DataFrame, pd.Series],
        fitted_scorecard: CreditScorecard,
    ) -> None:
        x, _ = sample_credit_data
        scores = fitted_scorecard.score(x)
        assert scores.dtype in [np.int64, np.int32]
        assert len(scores) == len(x)

    def test_score_to_pd_returns_probabilities(self, fitted_scorecard: CreditScorecard) -> None:
        scores = np.array([300, 500, 600, 700, 800])
        pds = fitted_scorecard.score_to_pd(scores)
        assert all(0 <= p <= 1 for p in pds)

    def test_higher_score_lower_pd(self, fitted_scorecard: CreditScorecard) -> None:
        scores = np.array([400, 500, 600, 700, 800])
        pds = fitted_scorecard.score_to_pd(scores)
        for i in range(len(pds) - 1):
            assert pds[i] >= pds[i + 1]

    def test_scorecard_table_structure(self, fitted_scorecard: CreditScorecard) -> None:
        table = fitted_scorecard.get_scorecard_table()
        assert isinstance(table, pd.DataFrame)
        assert "Feature" in table.columns
        assert "Bin" in table.columns
        assert "WoE" in table.columns
        assert "Points" in table.columns
        assert len(table) > 0

    def test_explain_score_structure(
        self,
        sample_credit_data: tuple[pd.DataFrame, pd.Series],
        fitted_scorecard: CreditScorecard,
    ) -> None:
        x, _ = sample_credit_data
        explanation = fitted_scorecard.explain_score(x.iloc[0])
        assert "total_score" in explanation
        assert "probability_of_default" in explanation
        assert "breakdown" in explanation
        assert explanation["breakdown"][0]["component"] == "Base Score"

    def test_explain_score_pd_valid(
        self,
        sample_credit_data: tuple[pd.DataFrame, pd.Series],
        fitted_scorecard: CreditScorecard,
    ) -> None:
        x, _ = sample_credit_data
        explanation = fitted_scorecard.explain_score(x.iloc[0])
        assert 0 <= explanation["probability_of_default"] <= 1

    def test_score_to_pd_at_base_score(self) -> None:
        config = ScorecardConfig(pdo=20, base_score=600, base_odds=50.0)
        scorecard = CreditScorecard(config)
        scorecard._calculate_scaling_factors()
        pd_at_base = scorecard.score_to_pd(np.array([600]))[0]
        expected_pd = 1 / (1 + 50.0)
        assert pd_at_base == pytest.approx(expected_pd, rel=1e-4)
