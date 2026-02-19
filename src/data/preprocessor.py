"""Data preprocessing pipeline for the credit risk scoring system.

Handles categorical encoding, missing value imputation, feature scaling,
and train/test splitting with stratification.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CreditDataPreprocessor:
    """Preprocessor for credit risk data with encoding, scaling, and splitting.

    Attributes:
        test_size: Fraction of data reserved for testing.
        random_state: Random seed for reproducibility.
        label_encoders: Fitted label encoders for categorical columns.
        scaler: Fitted standard scaler for numerical columns.
        categorical_cols: List of categorical column names.
        numerical_cols: List of numerical column names.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.categorical_cols: list[str] = []
        self.numerical_cols: list[str] = []
        self._is_fitted: bool = False

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "class"
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Fit preprocessor on data and return train/test splits.

        Args:
            df: Raw input DataFrame.
            target_col: Name of the target column.

        Returns:
            Tuple of (X_train, X_test, y_train, y_test).
        """
        df = df.copy()
        logger.info("Preprocessing dataset with %d rows and %d columns", len(df), len(df.columns))

        self.categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        self.numerical_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns if c != target_col
        ]

        # Handle missing values
        for col in self.categorical_cols:
            df[col] = df[col].fillna("Unknown")
        for col in self.numerical_cols:
            df[col] = df[col].fillna(df[col].median())

        # Encode categoricals
        for col in self.categorical_cols:
            self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])

        # Create age_group for fairness analysis
        if "age" in df.columns:
            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 25, 35, 45, 60, 100],
                labels=["<25", "25-35", "35-45", "45-60", "60+"],
            )
            age_group_encoder = LabelEncoder()
            df["age_group"] = age_group_encoder.fit_transform(df["age_group"].astype(str))
            self.label_encoders["age_group"] = age_group_encoder

        # Split data
        x = df.drop(columns=[target_col])
        y = df[target_col]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        # Scale numerical features
        x_train[self.numerical_cols] = self.scaler.fit_transform(x_train[self.numerical_cols])
        x_test[self.numerical_cols] = self.scaler.transform(x_test[self.numerical_cols])

        self._is_fitted = True
        logger.info(
            "Preprocessing complete: train=%d, test=%d, features=%d",
            len(x_train),
            len(x_test),
            len(x_train.columns),
        )

        return x_train, x_test, y_train, y_test

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders and scaler.

        Args:
            df: New data to transform.

        Returns:
            Transformed DataFrame.

        Raises:
            RuntimeError: If the preprocessor has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform()")

        df = df.copy()

        for col in self.categorical_cols:
            if col in df.columns and col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col])

        if "age" in df.columns and "age_group" not in df.columns:
            df["age_group"] = pd.cut(
                df["age"],
                bins=[0, 25, 35, 45, 60, 100],
                labels=["<25", "25-35", "35-45", "45-60", "60+"],
            )
            df["age_group"] = self.label_encoders["age_group"].transform(
                df["age_group"].astype(str)
            )

        numerical_in_df = [c for c in self.numerical_cols if c in df.columns]
        if numerical_in_df:
            df[numerical_in_df] = self.scaler.transform(df[numerical_in_df])

        return df
