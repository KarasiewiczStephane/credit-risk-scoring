"""Model evaluation metrics for credit risk scoring.

Provides industry-standard metrics including AUC-ROC, Gini coefficient,
KS statistic, and model comparison utilities.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate credit risk models with industry metrics.

    Computes AUC-ROC, Gini, KS statistic, accuracy, precision, and recall.
    """

    @staticmethod
    def calculate_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic.

        Args:
            y_true: True binary labels.
            y_prob: Predicted probabilities for the positive class.

        Returns:
            Maximum separation between TPR and FPR curves.
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        return float(max(tpr - fpr))

    @staticmethod
    def calculate_gini(auc: float) -> float:
        """Calculate Gini coefficient from AUC.

        Args:
            auc: Area Under the ROC Curve.

        Returns:
            Gini coefficient (2 * AUC - 1).
        """
        return 2 * auc - 1

    def evaluate(self, model: Any, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Compute all evaluation metrics for a model.

        Args:
            model: Trained classifier with predict and predict_proba methods.
            x_test: Test feature matrix.
            y_test: True test labels.

        Returns:
            Dictionary of metric name to value.
        """
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)

        metrics = {
            "auc_roc": float(auc),
            "gini": float(self.calculate_gini(auc)),
            "ks_statistic": float(self.calculate_ks_statistic(y_test, y_prob)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        }

        logger.info(
            "Model evaluation: AUC=%.4f, Gini=%.4f, KS=%.4f",
            auc,
            metrics["gini"],
            metrics["ks_statistic"],
        )
        return metrics

    def compare_models(
        self,
        models: dict[str, Any],
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """Compare multiple models side by side.

        Args:
            models: Dictionary of model name to trained model.
            x_test: Test feature matrix.
            y_test: True test labels.

        Returns:
            DataFrame with metrics for each model.
        """
        results = []
        for name, model in models.items():
            metrics = self.evaluate(model, x_test, y_test)
            metrics["model"] = name
            results.append(metrics)

        return pd.DataFrame(results).set_index("model")
