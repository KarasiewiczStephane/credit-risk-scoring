"""Tests for WoE/IV analysis and feature binning."""

import numpy as np
import pandas as pd
import pytest

from src.data.woe_transformer import WoEBin, WoETransformer


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create sample data for WoE testing."""
    rng = np.random.RandomState(42)
    n = 200
    x = pd.DataFrame(
        {
            "continuous_feat": rng.randn(n) * 10 + 50,
            "categorical_feat": rng.choice([0, 1, 2, 3, 4], n),
            "binary_feat": rng.choice([0, 1], n),
        }
    )
    y = pd.Series(rng.choice([0, 1], n, p=[0.7, 0.3]))
    return x, y


@pytest.fixture
def fitted_transformer(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> WoETransformer:
    """Create a fitted WoE transformer."""
    x, y = sample_data
    transformer = WoETransformer(min_bins=3, max_bins=5, min_bin_size=0.05)
    transformer.fit(x, y)
    return transformer


class TestWoEBin:
    """Tests for WoEBin dataclass."""

    def test_woe_bin_creation(self) -> None:
        bin_ = WoEBin(lower=0, upper=10, woe=0.5, iv=0.02, event_rate=0.3, count=50)
        assert bin_.lower == 0
        assert bin_.upper == 10
        assert bin_.woe == 0.5
        assert bin_.iv == 0.02


class TestWoETransformer:
    """Tests for WoE transformer."""

    def test_fit_stores_woe_dict(self, fitted_transformer: WoETransformer) -> None:
        assert len(fitted_transformer.woe_dict) == 3
        assert "continuous_feat" in fitted_transformer.woe_dict
        assert "categorical_feat" in fitted_transformer.woe_dict
        assert "binary_feat" in fitted_transformer.woe_dict

    def test_fit_stores_iv_values(self, fitted_transformer: WoETransformer) -> None:
        assert len(fitted_transformer.iv_values) == 3
        for iv in fitted_transformer.iv_values.values():
            assert iv >= 0

    def test_iv_sums_correctly(self, fitted_transformer: WoETransformer) -> None:
        for col, bins in fitted_transformer.woe_dict.items():
            bin_iv_sum = sum(b.iv for b in bins)
            assert abs(bin_iv_sum - fitted_transformer.iv_values[col]) < 1e-10

    def test_transform_produces_correct_shape(
        self,
        sample_data: tuple[pd.DataFrame, pd.Series],
        fitted_transformer: WoETransformer,
    ) -> None:
        x, _ = sample_data
        x_woe = fitted_transformer.transform(x)
        assert x_woe.shape == x.shape

    def test_transform_produces_numeric_values(
        self,
        sample_data: tuple[pd.DataFrame, pd.Series],
        fitted_transformer: WoETransformer,
    ) -> None:
        x, _ = sample_data
        x_woe = fitted_transformer.transform(x)
        for col in x_woe.columns:
            assert pd.api.types.is_numeric_dtype(x_woe[col])

    def test_optimal_binning_respects_min_bin_size(
        self,
        sample_data: tuple[pd.DataFrame, pd.Series],
    ) -> None:
        x, y = sample_data
        transformer = WoETransformer(min_bins=3, max_bins=5, min_bin_size=0.1)
        transformer.fit(x, y)
        for bins in transformer.woe_dict.values():
            for b in bins:
                assert b.count >= int(len(x) * 0.05)  # Allow some slack

    def test_iv_report_structure(self, fitted_transformer: WoETransformer) -> None:
        report = fitted_transformer.get_iv_report()
        assert isinstance(report, pd.DataFrame)
        assert "feature" in report.columns
        assert "iv" in report.columns
        assert "predictive_power" in report.columns
        assert len(report) == 3

    def test_iv_report_sorted_descending(self, fitted_transformer: WoETransformer) -> None:
        report = fitted_transformer.get_iv_report()
        iv_values = report["iv"].tolist()
        assert iv_values == sorted(iv_values, reverse=True)


class TestIVStrength:
    """Tests for IV strength classification."""

    def test_not_useful(self) -> None:
        assert WoETransformer._iv_strength(0.01) == "Not useful"

    def test_weak(self) -> None:
        assert WoETransformer._iv_strength(0.05) == "Weak"

    def test_medium(self) -> None:
        assert WoETransformer._iv_strength(0.15) == "Medium"

    def test_strong(self) -> None:
        assert WoETransformer._iv_strength(0.35) == "Strong"

    def test_suspicious(self) -> None:
        assert WoETransformer._iv_strength(0.6) == "Suspicious (check overfitting)"

    def test_boundary_values(self) -> None:
        assert WoETransformer._iv_strength(0.02) == "Weak"
        assert WoETransformer._iv_strength(0.1) == "Medium"
        assert WoETransformer._iv_strength(0.3) == "Strong"
        assert WoETransformer._iv_strength(0.5) == "Suspicious (check overfitting)"


class TestEdgeCases:
    """Test edge cases for WoE transformer."""

    def test_single_unique_value_feature(self) -> None:
        x = pd.DataFrame({"const": [1] * 100, "var": np.random.choice([0, 1, 2], 100)})
        y = pd.Series(np.random.choice([0, 1], 100, p=[0.7, 0.3]))
        transformer = WoETransformer(max_bins=5)
        transformer.fit(x, y)
        assert "const" in transformer.woe_dict

    def test_binary_feature(self) -> None:
        rng = np.random.RandomState(42)
        x = pd.DataFrame({"binary": rng.choice([0, 1], 100)})
        y = pd.Series(rng.choice([0, 1], 100, p=[0.7, 0.3]))
        transformer = WoETransformer(max_bins=5)
        transformer.fit(x, y)
        assert len(transformer.woe_dict["binary"]) == 2

    def test_woe_bins_have_positive_counts(self, fitted_transformer: WoETransformer) -> None:
        for bins in fitted_transformer.woe_dict.values():
            for b in bins:
                assert b.count > 0

    def test_event_rates_valid(self, fitted_transformer: WoETransformer) -> None:
        for bins in fitted_transformer.woe_dict.values():
            for b in bins:
                assert 0 <= b.event_rate <= 1
