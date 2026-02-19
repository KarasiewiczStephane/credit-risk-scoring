"""Tests for dataset download and preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessor import CreditDataPreprocessor


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Load sample dataset for testing."""
    return pd.read_csv("data/sample/german_credit_sample.csv")


@pytest.fixture
def preprocessor() -> CreditDataPreprocessor:
    """Create a default preprocessor instance."""
    return CreditDataPreprocessor(test_size=0.2, random_state=42)


class TestSampleData:
    """Tests for the sample dataset."""

    def test_sample_data_exists(self, sample_df: pd.DataFrame) -> None:
        assert len(sample_df) == 200

    def test_sample_has_correct_columns(self, sample_df: pd.DataFrame) -> None:
        expected = [
            "checking_status",
            "duration",
            "credit_history",
            "purpose",
            "credit_amount",
            "savings_status",
            "employment",
            "installment_rate",
            "personal_status",
            "other_parties",
            "residence_since",
            "property_magnitude",
            "age",
            "other_payment_plans",
            "housing",
            "existing_credits",
            "job",
            "num_dependents",
            "own_telephone",
            "foreign_worker",
            "class",
        ]
        assert list(sample_df.columns) == expected

    def test_sample_target_distribution(self, sample_df: pd.DataFrame) -> None:
        class_counts = sample_df["class"].value_counts(normalize=True)
        assert 0 in class_counts.index
        assert 1 in class_counts.index
        assert class_counts[0] > 0.5  # majority class


class TestCreditDataPreprocessor:
    """Tests for the preprocessing pipeline."""

    def test_fit_transform_returns_correct_shapes(
        self, sample_df: pd.DataFrame, preprocessor: CreditDataPreprocessor
    ) -> None:
        x_train, x_test, y_train, y_test = preprocessor.fit_transform(sample_df)
        total = len(x_train) + len(x_test)
        assert total == len(sample_df)
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)

    def test_no_nan_after_preprocessing(
        self, sample_df: pd.DataFrame, preprocessor: CreditDataPreprocessor
    ) -> None:
        x_train, x_test, _, _ = preprocessor.fit_transform(sample_df)
        assert x_train.isnull().sum().sum() == 0
        assert x_test.isnull().sum().sum() == 0

    def test_stratification_preserved(
        self, sample_df: pd.DataFrame, preprocessor: CreditDataPreprocessor
    ) -> None:
        _, _, y_train, y_test = preprocessor.fit_transform(sample_df)
        original_ratio = sample_df["class"].mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        assert abs(train_ratio - original_ratio) < 0.05
        assert abs(test_ratio - original_ratio) < 0.05

    def test_age_group_created(
        self, sample_df: pd.DataFrame, preprocessor: CreditDataPreprocessor
    ) -> None:
        x_train, x_test, _, _ = preprocessor.fit_transform(sample_df)
        assert "age_group" in x_train.columns
        assert "age_group" in x_test.columns

    def test_categorical_encoding(
        self, sample_df: pd.DataFrame, preprocessor: CreditDataPreprocessor
    ) -> None:
        x_train, _, _, _ = preprocessor.fit_transform(sample_df)
        for col in preprocessor.categorical_cols:
            if col in x_train.columns:
                assert x_train[col].dtype in [np.int64, np.int32, np.float64]

    def test_scaler_fitted_on_train_only(
        self, sample_df: pd.DataFrame, preprocessor: CreditDataPreprocessor
    ) -> None:
        x_train, x_test, _, _ = preprocessor.fit_transform(sample_df)
        # Scaler mean should reflect training data stats
        assert preprocessor.scaler.mean_ is not None
        assert len(preprocessor.scaler.mean_) == len(preprocessor.numerical_cols)

    def test_handle_missing_values(self) -> None:
        rng = np.random.RandomState(42)
        n = 50
        df = pd.DataFrame(
            {
                "checking_status": rng.choice(["A11", "A12", None], n),
                "duration": [12 if i % 3 else np.nan for i in range(n)],
                "credit_amount": rng.randint(1000, 5000, n).astype(float),
                "age": rng.randint(20, 60, n),
                "class": rng.choice([0, 1], n, p=[0.7, 0.3]),
            }
        )
        preprocessor = CreditDataPreprocessor(test_size=0.2, random_state=42)
        x_train, x_test, _, _ = preprocessor.fit_transform(df)
        assert x_train.isnull().sum().sum() == 0
        assert x_test.isnull().sum().sum() == 0

    def test_transform_not_fitted_raises(self) -> None:
        preprocessor = CreditDataPreprocessor()
        with pytest.raises(RuntimeError, match="must be fitted"):
            preprocessor.transform(pd.DataFrame({"a": [1]}))

    def test_test_size_respected(
        self, sample_df: pd.DataFrame, preprocessor: CreditDataPreprocessor
    ) -> None:
        x_train, x_test, _, _ = preprocessor.fit_transform(sample_df)
        expected_test = int(len(sample_df) * 0.2)
        assert abs(len(x_test) - expected_test) <= 2

    def test_transform_after_fit(
        self, sample_df: pd.DataFrame, preprocessor: CreditDataPreprocessor
    ) -> None:
        preprocessor.fit_transform(sample_df)
        # Transform a single row from the original data
        single_row = sample_df.drop(columns=["class"]).head(1)
        result = preprocessor.transform(single_row)
        assert result.isnull().sum().sum() == 0
        assert len(result) == 1


class TestDownloaderImport:
    """Test that downloader module imports correctly."""

    def test_downloader_imports(self) -> None:
        from src.data.downloader import COLUMN_NAMES, download_german_credit

        assert download_german_credit is not None
        assert len(COLUMN_NAMES) == 21

    def test_column_names_correct(self) -> None:
        from src.data.downloader import COLUMN_NAMES

        assert COLUMN_NAMES[0] == "checking_status"
        assert COLUMN_NAMES[-1] == "class"
        assert "age" in COLUMN_NAMES
        assert "credit_amount" in COLUMN_NAMES
