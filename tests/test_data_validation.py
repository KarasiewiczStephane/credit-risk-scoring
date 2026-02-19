"""Tests for data validation and loading utilities."""

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_data
from src.data.validator import DataValidationReport, DataValidator
from src.utils.config import Config, DataConfig


@pytest.fixture
def clean_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create clean data that passes validation."""
    rng = np.random.RandomState(42)
    n = 100
    x = pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "feat_b": rng.randint(0, 5, n),
            "feat_c": rng.randn(n) * 10,
        }
    )
    y = pd.Series(rng.choice([0, 1], n, p=[0.7, 0.3]))
    return x, y


@pytest.fixture
def validator() -> DataValidator:
    """Create a default validator."""
    return DataValidator(max_missing_ratio=0.1, min_class_ratio=0.1)


class TestDataValidator:
    """Tests for DataValidator."""

    def test_clean_data_passes(
        self,
        clean_data: tuple[pd.DataFrame, pd.Series],
        validator: DataValidator,
    ) -> None:
        x, y = clean_data
        report = validator.validate(x, y)
        assert report.passed is True
        assert len(report.warnings) == 0

    def test_report_structure(
        self,
        clean_data: tuple[pd.DataFrame, pd.Series],
        validator: DataValidator,
    ) -> None:
        x, y = clean_data
        report = validator.validate(x, y)
        assert report.n_samples == 100
        assert report.n_features == 3
        assert isinstance(report.missing_values, dict)
        assert isinstance(report.target_distribution, dict)
        assert isinstance(report.feature_types, dict)

    def test_catches_missing_values(self, validator: DataValidator) -> None:
        x = pd.DataFrame(
            {
                "a": [1, 2, np.nan, np.nan, np.nan] * 20,
                "b": range(100),
            }
        )
        y = pd.Series([0] * 70 + [1] * 30)
        report = validator.validate(x, y)
        assert report.passed is False
        assert any("missing" in w.lower() for w in report.warnings)

    def test_catches_imbalanced_classes(self, validator: DataValidator) -> None:
        x = pd.DataFrame({"a": range(100)})
        y = pd.Series([0] * 95 + [1] * 5)
        report = validator.validate(x, y)
        assert report.passed is False
        assert any("imbalanced" in w.lower() for w in report.warnings)

    def test_catches_zero_variance(self, validator: DataValidator) -> None:
        x = pd.DataFrame({"constant": [1] * 100, "variable": range(100)})
        y = pd.Series([0] * 70 + [1] * 30)
        report = validator.validate(x, y)
        assert report.passed is False
        assert any("variance" in w.lower() for w in report.warnings)

    def test_catches_duplicates(self, validator: DataValidator) -> None:
        x = pd.DataFrame({"a": [1, 2] * 50, "b": [3, 4] * 50})
        y = pd.Series([0, 1] * 50)
        report = validator.validate(x, y)
        assert report.passed is False
        assert any("duplicate" in w.lower() for w in report.warnings)

    def test_custom_thresholds(self) -> None:
        validator = DataValidator(max_missing_ratio=0.5, min_class_ratio=0.01)
        x = pd.DataFrame(
            {
                "a": [1, 2, 3, np.nan, np.nan] * 20,
                "b": range(100),
            }
        )
        y = pd.Series([0] * 70 + [1] * 30)
        report = validator.validate(x, y)
        # 40% missing is under 50% threshold
        assert not any("missing" in w.lower() for w in report.warnings)


class TestDataValidationReport:
    """Tests for DataValidationReport dataclass."""

    def test_default_values(self) -> None:
        report = DataValidationReport(
            n_samples=100,
            n_features=5,
            missing_values={},
            target_distribution={0: 0.7, 1: 0.3},
            feature_types={"a": "int64"},
        )
        assert report.passed is True
        assert report.warnings == []


class TestLoadData:
    """Tests for the load_data convenience function."""

    def test_load_sample_data(self) -> None:
        config = Config(
            data=DataConfig(source="sample", test_size=0.2, random_state=42),
        )
        x_train, x_test, y_train, y_test = load_data(config)
        assert len(x_train) + len(x_test) == 200
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)

    def test_load_unknown_source_raises(self) -> None:
        config = Config(
            data=DataConfig(source="nonexistent"),
        )
        with pytest.raises(ValueError, match="Unknown data source"):
            load_data(config)

    def test_load_with_sample_size(self) -> None:
        config = Config(
            data=DataConfig(source="sample", sample_size=100, test_size=0.2, random_state=42),
        )
        x_train, x_test, y_train, y_test = load_data(config)
        assert len(x_train) + len(x_test) == 100
