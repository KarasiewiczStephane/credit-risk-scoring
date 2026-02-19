"""Bias mitigation strategies for credit risk models.

Implements reweighting (pre-processing), threshold adjustment (post-processing),
and exponentiated gradient (in-processing) methods.
"""

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
from sklearn.linear_model import LogisticRegression

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BiasMitigator:
    """Bias mitigation strategies for credit risk models.

    Supports reweighting, threshold optimization, and exponentiated gradient.

    Attributes:
        base_model: The original (possibly biased) model.
        mitigated_models: Dictionary of mitigated model variants.
        sample_weights: Computed reweighting weights.
    """

    def __init__(self, base_model: Any) -> None:
        self.base_model = base_model
        self.mitigated_models: dict[str, Any] = {}
        self.sample_weights: np.ndarray | None = None

    def reweight_samples(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> np.ndarray:
        """Compute sample weights to balance group positive rates.

        Args:
            x: Feature DataFrame.
            y: Binary target array.
            sensitive_features: Group membership array.

        Returns:
            Array of sample weights.
        """
        unique_groups = np.unique(sensitive_features)
        weights = np.ones(len(y), dtype=float)
        overall_positive_rate = y.mean()

        for group in unique_groups:
            mask = sensitive_features == group
            group_rate = y[mask].mean()

            if group_rate > 0 and group_rate < 1:
                pos_mask = mask & (y == 1)
                neg_mask = mask & (y == 0)
                weights[pos_mask] = overall_positive_rate / group_rate
                weights[neg_mask] = (1 - overall_positive_rate) / (1 - group_rate)

        self.sample_weights = weights
        logger.info("Computed reweighting for %d groups", len(unique_groups))
        return weights

    def train_with_reweighting(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> Any:
        """Train model with reweighted samples.

        Args:
            x: Feature DataFrame.
            y: Binary target array.
            sensitive_features: Group membership array.

        Returns:
            Trained model with sample reweighting.
        """
        weights = self.reweight_samples(x, y, sensitive_features)

        if isinstance(self.base_model, lgb.LGBMClassifier):
            params = self.base_model.get_params()
            model = lgb.LGBMClassifier(**params)
            model.fit(x, y, sample_weight=weights)
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(x, y, sample_weight=weights)

        self.mitigated_models["reweighted"] = model
        logger.info("Trained reweighted model")
        return model

    def threshold_optimization(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        sensitive_features: np.ndarray,
        constraint: str = "demographic_parity",
    ) -> ThresholdOptimizer:
        """Post-processing: optimize decision thresholds per group.

        Args:
            x: Feature DataFrame.
            y: Binary target array.
            sensitive_features: Group membership array.
            constraint: Fairness constraint type.

        Returns:
            Fitted ThresholdOptimizer.
        """
        optimizer = ThresholdOptimizer(
            estimator=self.base_model,
            constraints=constraint,
            prefit=True,
        )
        optimizer.fit(x, y, sensitive_features=sensitive_features)

        self.mitigated_models["threshold_optimized"] = optimizer
        logger.info("Trained threshold optimizer with constraint='%s'", constraint)
        return optimizer

    def exponentiated_gradient(
        self,
        x: pd.DataFrame,
        y: np.ndarray,
        sensitive_features: np.ndarray,
        constraint: str = "demographic_parity",
    ) -> ExponentiatedGradient:
        """In-processing: train with exponentiated gradient reduction.

        Args:
            x: Feature DataFrame.
            y: Binary target array.
            sensitive_features: Group membership array.
            constraint: Fairness constraint type.

        Returns:
            Fitted ExponentiatedGradient model.
        """
        constraint_obj = (
            DemographicParity() if constraint == "demographic_parity" else EqualizedOdds()
        )

        base_estimator = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)

        mitigator = ExponentiatedGradient(
            estimator=base_estimator,
            constraints=constraint_obj,
            eps=0.01,
        )
        mitigator.fit(x, y, sensitive_features=sensitive_features)

        self.mitigated_models["exponentiated_gradient"] = mitigator
        logger.info("Trained exponentiated gradient with constraint='%s'", constraint)
        return mitigator
