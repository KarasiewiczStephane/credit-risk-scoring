"""SHAP-based model explanation for credit risk scoring.

Provides global feature importance, summary plots, waterfall plots,
dependence plots, and per-prediction explanations using TreeExplainer.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import shap

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_class1_shap(shap_vals: Any) -> np.ndarray:
    """Extract SHAP values for the positive class (class 1).

    Handles both old-style list output and new-style 3D array output.

    Args:
        shap_vals: Raw SHAP values from the explainer.

    Returns:
        2D array of SHAP values for class 1.
    """
    if isinstance(shap_vals, list):
        return np.array(shap_vals[1])
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        return shap_vals[:, :, 1]
    return np.array(shap_vals)


def _extract_base_value(expected_value: Any) -> float:
    """Extract base value for the positive class.

    Args:
        expected_value: Expected value from the SHAP explainer.

    Returns:
        Float base value for class 1.
    """
    if isinstance(expected_value, list | np.ndarray):
        return float(expected_value[1])
    return float(expected_value)


class SHAPExplainer:
    """SHAP-based model explainer for tree-based credit risk models.

    Attributes:
        model: Trained tree-based model.
        feature_names: List of feature names.
        explainer: SHAP TreeExplainer instance.
        shap_values: Computed SHAP values (cached after first computation).
    """

    def __init__(self, model: Any, x_train: pd.DataFrame) -> None:
        self.model = model
        self.x_train = x_train
        self.feature_names = list(x_train.columns)
        self.explainer = shap.TreeExplainer(model)
        self.shap_values: np.ndarray | None = None

    def compute_shap_values(self, x: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for given data.

        Args:
            x: Feature DataFrame.

        Returns:
            2D array of SHAP values for the positive class.
        """
        raw = self.explainer.shap_values(x)
        self.shap_values = _extract_class1_shap(raw)
        logger.info("Computed SHAP values for %d samples", len(x))
        return self.shap_values

    def global_feature_importance(
        self, x: pd.DataFrame, save_path: str | None = None
    ) -> pd.DataFrame:
        """Calculate and optionally plot global feature importance.

        Args:
            x: Feature DataFrame.
            save_path: Path to save the bar plot image.

        Returns:
            DataFrame with feature names and mean absolute SHAP importance.
        """
        if self.shap_values is None:
            self.compute_shap_values(x)

        importance = np.abs(self.shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values, x, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        return importance_df.reset_index(drop=True)

    def summary_plot(self, x: pd.DataFrame, save_path: str | None = None) -> None:
        """Generate SHAP summary plot.

        Args:
            x: Feature DataFrame.
            save_path: Path to save the plot image.
        """
        if self.shap_values is None:
            self.compute_shap_values(x)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, x, show=False)
        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        plt.close()

    def waterfall_plot(self, x: pd.DataFrame, idx: int, save_path: str | None = None) -> dict:
        """Generate waterfall plot for a single prediction.

        Args:
            x: Feature DataFrame.
            idx: Row index to explain.
            save_path: Path to save the plot image.

        Returns:
            Dictionary with base_value, prediction, and feature contributions.
        """
        if self.shap_values is None:
            self.compute_shap_values(x)

        base = _extract_base_value(self.explainer.expected_value)
        sv_row = self.shap_values[idx]

        shap_explanation = shap.Explanation(
            values=sv_row,
            base_values=base,
            data=x.iloc[idx].values,
            feature_names=self.feature_names,
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_explanation, show=False)
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

        return {
            "base_value": float(base),
            "prediction": float(base + sv_row.sum()),
            "features": [
                {"name": name, "value": float(val), "shap_value": float(sv)}
                for name, val, sv in zip(
                    self.feature_names,
                    x.iloc[idx].values,
                    sv_row,
                    strict=False,
                )
            ],
        }

    def explain_prediction(self, x_row: pd.DataFrame) -> dict:
        """Get SHAP explanation for a single prediction as a dictionary.

        Args:
            x_row: Single-row feature DataFrame.

        Returns:
            Dictionary with base_value, shap_values, and final_prediction.
        """
        raw = self.explainer.shap_values(x_row)
        sv = _extract_class1_shap(raw)
        base = _extract_base_value(self.explainer.expected_value)

        return {
            "base_value": float(base),
            "shap_values": {
                name: float(v) for name, v in zip(self.feature_names, sv[0], strict=False)
            },
            "prediction_contribution": float(sv[0].sum()),
            "final_prediction": float(base + sv[0].sum()),
        }
