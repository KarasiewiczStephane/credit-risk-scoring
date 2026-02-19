"""Partial Dependence Plot (PDP) generator and explanation serialization.

Provides PDP computation for model interpretation and JSON-serializable
explanation output for API responses.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PDPGenerator:
    """Partial Dependence Plot generator for model interpretation.

    Attributes:
        model: Trained classifier.
        x: Feature DataFrame used for PDP computation.
        feature_names: List of feature names.
    """

    def __init__(self, model: Any, x: pd.DataFrame, feature_names: list[str] | None = None) -> None:
        self.model = model
        self.x = x
        self.feature_names = feature_names or list(x.columns)

    def compute_pdp(self, feature: str, grid_resolution: int = 50) -> dict[str, Any]:
        """Compute partial dependence for a single feature.

        Args:
            feature: Feature name.
            grid_resolution: Number of grid points.

        Returns:
            Dictionary with grid values, PDP values, and feature range.
        """
        feature_idx = self.feature_names.index(feature)

        pd_result = partial_dependence(
            self.model,
            self.x,
            features=[feature_idx],
            grid_resolution=grid_resolution,
            kind="average",
        )

        return {
            "feature": feature,
            "grid_values": pd_result["grid_values"][0].tolist(),
            "pdp_values": pd_result["average"][0].tolist(),
            "feature_range": {
                "min": float(self.x[feature].min()),
                "max": float(self.x[feature].max()),
                "mean": float(self.x[feature].mean()),
            },
        }

    def compute_pdp_top_features(
        self,
        n_features: int = 5,
        importance_scores: dict[str, float] | None = None,
    ) -> list[dict]:
        """Compute PDP for top N important features.

        Args:
            n_features: Number of top features to include.
            importance_scores: Optional feature importance scores.

        Returns:
            List of PDP result dictionaries.
        """
        if importance_scores:
            top_features = sorted(importance_scores.items(), key=lambda x: -x[1])[:n_features]
            top_features = [f[0] for f in top_features]
        elif hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            top_idx = np.argsort(importance)[::-1][:n_features]
            top_features = [self.feature_names[i] for i in top_idx]
        else:
            top_features = self.feature_names[:n_features]

        logger.info("Computing PDP for top %d features: %s", n_features, top_features)
        return [self.compute_pdp(f) for f in top_features]

    def plot_pdp(
        self,
        features: list[str],
        save_path: str | None = None,
        figsize: tuple[int, int] = (12, 8),
    ) -> None:
        """Generate PDP plot for multiple features.

        Args:
            features: List of feature names.
            save_path: Path to save the plot image.
            figsize: Figure size as (width, height).
        """
        feature_indices = [self.feature_names.index(f) for f in features]

        fig, ax = plt.subplots(figsize=figsize)
        PartialDependenceDisplay.from_estimator(
            self.model,
            self.x,
            features=feature_indices,
            feature_names=self.feature_names,
            ax=ax,
        )
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        plt.close()


class ExplanationSerializer:
    """Serialize model explanations to JSON for API responses."""

    @staticmethod
    def serialize_prediction_explanation(
        application_id: str,
        credit_score: int,
        probability_of_default: float,
        decision: str,
        shap_explanation: dict,
        lime_explanation: dict,
        scorecard_breakdown: dict | None = None,
    ) -> dict:
        """Create comprehensive explanation JSON.

        Args:
            application_id: Unique application identifier.
            credit_score: Computed credit score.
            probability_of_default: PD value.
            decision: Approval decision string.
            shap_explanation: SHAP explanation dictionary.
            lime_explanation: LIME explanation dictionary.
            scorecard_breakdown: Optional scorecard breakdown.

        Returns:
            Dictionary containing full explanation.
        """
        return {
            "application_id": application_id,
            "timestamp": pd.Timestamp.now().isoformat(),
            "decision": {
                "credit_score": credit_score,
                "probability_of_default": round(probability_of_default, 4),
                "recommendation": decision,
                "score_band": ExplanationSerializer._get_score_band(credit_score),
            },
            "explanations": {
                "shap": shap_explanation,
                "lime": lime_explanation,
                "scorecard": scorecard_breakdown,
            },
            "top_risk_factors": ExplanationSerializer._extract_top_factors(shap_explanation),
            "top_positive_factors": ExplanationSerializer._extract_positive_factors(
                shap_explanation
            ),
        }

    @staticmethod
    def _get_score_band(score: int) -> str:
        """Classify credit score into risk bands.

        Args:
            score: Credit score.

        Returns:
            Score band label.
        """
        if score >= 750:
            return "Excellent"
        elif score >= 700:
            return "Good"
        elif score >= 650:
            return "Fair"
        elif score >= 600:
            return "Poor"
        else:
            return "Very Poor"

    @staticmethod
    def _extract_top_factors(shap_explanation: dict, n: int = 3) -> list[dict]:
        """Extract top negative (risk-increasing) factors.

        Args:
            shap_explanation: SHAP explanation dictionary.
            n: Number of factors to return.

        Returns:
            List of factor dictionaries.
        """
        if "shap_values" not in shap_explanation:
            return []

        factors = list(shap_explanation["shap_values"].items())
        factors.sort(key=lambda x: x[1])

        return [
            {"feature": f, "impact": round(v, 4), "direction": "increases_risk"}
            for f, v in factors[:n]
            if v < 0
        ]

    @staticmethod
    def _extract_positive_factors(shap_explanation: dict, n: int = 3) -> list[dict]:
        """Extract top positive (risk-decreasing) factors.

        Args:
            shap_explanation: SHAP explanation dictionary.
            n: Number of factors to return.

        Returns:
            List of factor dictionaries.
        """
        if "shap_values" not in shap_explanation:
            return []

        factors = list(shap_explanation["shap_values"].items())
        factors.sort(key=lambda x: -x[1])

        return [
            {"feature": f, "impact": round(v, 4), "direction": "decreases_risk"}
            for f, v in factors[:n]
            if v > 0
        ]

    @staticmethod
    def save_explanation(explanation: dict, path: str) -> None:
        """Save explanation to JSON file.

        Args:
            explanation: Explanation dictionary.
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(explanation, f, indent=2, default=str)
