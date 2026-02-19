"""LIME-based model explanation for credit risk scoring.

Provides local interpretable explanations for individual predictions
using LIME (Local Interpretable Model-agnostic Explanations).
"""

from typing import Any

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from src.utils.logger import get_logger

logger = get_logger(__name__)


class LIMEExplainer:
    """LIME-based explainer for credit risk model predictions.

    Attributes:
        model: Trained classifier with predict_proba method.
        feature_names: List of feature names.
        explainer: LIME tabular explainer instance.
    """

    def __init__(self, model: Any, x_train: pd.DataFrame, mode: str = "classification") -> None:
        self.model = model
        self.feature_names = list(x_train.columns)
        self.explainer = LimeTabularExplainer(
            training_data=x_train.values,
            feature_names=self.feature_names,
            mode=mode,
            discretize_continuous=True,
        )

    def explain_prediction(self, x_row: np.ndarray, num_features: int = 10) -> dict:
        """Generate LIME explanation for a single prediction.

        Args:
            x_row: Feature values for one sample (1D array).
            num_features: Number of top features to include.

        Returns:
            Dictionary with prediction probabilities, local prediction,
            intercept, and feature contributions.
        """
        explanation = self.explainer.explain_instance(
            x_row,
            self.model.predict_proba,
            num_features=num_features,
        )

        return {
            "prediction_proba": explanation.predict_proba.tolist(),
            "local_prediction": float(explanation.local_pred[0]),
            "intercept": float(explanation.intercept[1]),
            "feature_contributions": [
                {"feature": feat, "contribution": float(weight)}
                for feat, weight in explanation.as_list()
            ],
            "r_squared": float(explanation.score) if hasattr(explanation, "score") else None,
        }
