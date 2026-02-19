"""Fairness analysis for credit risk models using Fairlearn.

Computes demographic parity, equalized odds, and TPR/FPR parity
metrics across protected attributes.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FairnessMetrics:
    """Container for fairness metric results.

    Attributes:
        demographic_parity_diff: Difference in selection rates across groups.
        demographic_parity_ratio: Ratio of selection rates across groups.
        equalized_odds_diff: Maximum difference in TPR/FPR across groups.
        tpr_by_group: True positive rate per group.
        fpr_by_group: False positive rate per group.
        selection_rate_by_group: Selection rate per group.
        overall_accuracy: Overall model accuracy.
    """

    demographic_parity_diff: float
    demographic_parity_ratio: float
    equalized_odds_diff: float
    tpr_by_group: dict[str, float]
    fpr_by_group: dict[str, float]
    selection_rate_by_group: dict[str, float]
    overall_accuracy: float


class FairnessAnalyzer:
    """Analyze model fairness across protected attributes.

    Attributes:
        protected_attributes: List of protected attribute column names.
        metrics_by_attribute: Computed fairness metrics per attribute.
    """

    def __init__(self, protected_attributes: list[str]) -> None:
        self.protected_attributes = protected_attributes
        self.metrics_by_attribute: dict[str, FairnessMetrics] = {}

    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        sensitive_features: pd.DataFrame,
    ) -> dict[str, FairnessMetrics]:
        """Compute fairness metrics for each protected attribute.

        Args:
            y_true: True binary labels.
            y_pred: Predicted binary labels.
            y_prob: Predicted probabilities (unused but kept for API consistency).
            sensitive_features: DataFrame with protected attribute columns.

        Returns:
            Dictionary mapping attribute name to FairnessMetrics.
        """
        for attr in self.protected_attributes:
            if attr not in sensitive_features.columns:
                logger.warning("Protected attribute '%s' not found in data, skipping", attr)
                continue

            groups = sensitive_features[attr]

            metric_frame = MetricFrame(
                metrics={
                    "accuracy": accuracy_score,
                    "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
                    "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
                    "selection_rate": selection_rate,
                    "tpr": true_positive_rate,
                    "fpr": false_positive_rate,
                },
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=groups,
            )

            dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=groups)
            dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=groups)
            eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=groups)

            tpr_dict = {str(k): float(v) for k, v in metric_frame.by_group["tpr"].items()}
            fpr_dict = {str(k): float(v) for k, v in metric_frame.by_group["fpr"].items()}
            sr_dict = {str(k): float(v) for k, v in metric_frame.by_group["selection_rate"].items()}

            self.metrics_by_attribute[attr] = FairnessMetrics(
                demographic_parity_diff=float(dp_diff),
                demographic_parity_ratio=float(dp_ratio),
                equalized_odds_diff=float(eo_diff),
                tpr_by_group=tpr_dict,
                fpr_by_group=fpr_dict,
                selection_rate_by_group=sr_dict,
                overall_accuracy=float(accuracy_score(y_true, y_pred)),
            )

            logger.info(
                "Fairness for '%s': DP_diff=%.4f, EO_diff=%.4f",
                attr,
                dp_diff,
                eo_diff,
            )

        return self.metrics_by_attribute

    def check_fairness_constraints(self, threshold: float = 0.1) -> dict[str, bool]:
        """Check if fairness constraints are satisfied.

        Args:
            threshold: Maximum acceptable disparity.

        Returns:
            Dictionary mapping constraint name to pass/fail.
        """
        results = {}
        for attr, metrics in self.metrics_by_attribute.items():
            results[f"{attr}_demographic_parity"] = (
                abs(metrics.demographic_parity_diff) <= threshold
            )
            results[f"{attr}_equalized_odds"] = abs(metrics.equalized_odds_diff) <= threshold
        return results

    def get_disparity_report(self) -> pd.DataFrame:
        """Generate detailed disparity report.

        Returns:
            DataFrame with disparity metrics per protected attribute.
        """
        rows = []
        for attr, metrics in self.metrics_by_attribute.items():
            rows.append(
                {
                    "protected_attribute": attr,
                    "demographic_parity_diff": metrics.demographic_parity_diff,
                    "demographic_parity_ratio": metrics.demographic_parity_ratio,
                    "equalized_odds_diff": metrics.equalized_odds_diff,
                    "tpr_disparity": max(metrics.tpr_by_group.values())
                    - min(metrics.tpr_by_group.values()),
                    "fpr_disparity": max(metrics.fpr_by_group.values())
                    - min(metrics.fpr_by_group.values()),
                }
            )
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        """Convert analysis results to dictionary for JSON serialization.

        Returns:
            Nested dictionary with all fairness metrics.
        """
        return {
            attr: {
                "demographic_parity_difference": m.demographic_parity_diff,
                "demographic_parity_ratio": m.demographic_parity_ratio,
                "equalized_odds_difference": m.equalized_odds_diff,
                "true_positive_rate_by_group": m.tpr_by_group,
                "false_positive_rate_by_group": m.fpr_by_group,
                "selection_rate_by_group": m.selection_rate_by_group,
                "overall_accuracy": m.overall_accuracy,
            }
            for attr, m in self.metrics_by_attribute.items()
        }
