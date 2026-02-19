"""Data validation utilities for the credit risk scoring system.

Provides quality checks for training data including missing values,
class balance, duplicates, and feature variance.
"""

from dataclasses import dataclass, field

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataValidationReport:
    """Report containing data quality check results.

    Attributes:
        n_samples: Total number of samples.
        n_features: Total number of features.
        missing_values: Count of missing values per column.
        target_distribution: Fraction of each target class.
        feature_types: Data type of each feature.
        warnings: List of validation warnings.
        passed: Whether all validation checks passed.
    """

    n_samples: int
    n_features: int
    missing_values: dict[str, int]
    target_distribution: dict[int, float]
    feature_types: dict[str, str]
    warnings: list[str] = field(default_factory=list)
    passed: bool = True


class DataValidator:
    """Validate data quality before model training.

    Args:
        max_missing_ratio: Maximum allowed fraction of missing values per column.
        min_class_ratio: Minimum required fraction for any target class.
    """

    def __init__(self, max_missing_ratio: float = 0.1, min_class_ratio: float = 0.1) -> None:
        self.max_missing_ratio = max_missing_ratio
        self.min_class_ratio = min_class_ratio

    def validate(self, x: pd.DataFrame, y: pd.Series) -> DataValidationReport:
        """Run all validation checks on the dataset.

        Args:
            x: Feature DataFrame.
            y: Target Series.

        Returns:
            DataValidationReport with check results.
        """
        warnings: list[str] = []

        # Check missing values
        missing = x.isnull().sum().to_dict()
        high_missing = {k: v for k, v in missing.items() if v / len(x) > self.max_missing_ratio}
        if high_missing:
            warnings.append(f"High missing ratio in: {list(high_missing.keys())}")

        # Check class balance
        class_dist = y.value_counts(normalize=True).to_dict()
        if min(class_dist.values()) < self.min_class_ratio:
            warnings.append(f"Imbalanced classes: {class_dist}")

        # Check for duplicate rows
        n_duplicates = x.duplicated().sum()
        if n_duplicates > 0:
            warnings.append(f"{n_duplicates} duplicate rows found")

        # Check feature variance
        low_variance = [
            c for c in x.select_dtypes(include=["number"]).columns if x[c].nunique() == 1
        ]
        if low_variance:
            warnings.append(f"Zero variance features: {low_variance}")

        passed = len(warnings) == 0

        if passed:
            logger.info("Data validation passed: %d samples, %d features", len(x), len(x.columns))
        else:
            logger.warning("Data validation found %d warnings", len(warnings))
            for w in warnings:
                logger.warning("  - %s", w)

        return DataValidationReport(
            n_samples=len(x),
            n_features=len(x.columns),
            missing_values=missing,
            target_distribution=class_dist,
            feature_types={c: str(x[c].dtype) for c in x.columns},
            warnings=warnings,
            passed=passed,
        )
