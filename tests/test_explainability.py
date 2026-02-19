"""Tests for SHAP and LIME explainability."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.shap_explainer import SHAPExplainer


@pytest.fixture
def trained_model_data() -> tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Create a trained model with data for explanation testing."""
    x, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=42)
    feature_names = [f"feature_{i}" for i in range(5)]
    x_df = pd.DataFrame(x, columns=feature_names)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(x_df, y)

    return model, x_df, x_df.iloc[:50], y


class TestSHAPExplainer:
    """Tests for SHAP explainer."""

    def test_compute_shap_values(
        self,
        trained_model_data: tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, np.ndarray],
    ) -> None:
        model, x_train, x_test, _ = trained_model_data
        explainer = SHAPExplainer(model, x_train)
        shap_values = explainer.compute_shap_values(x_test)
        assert shap_values.shape == x_test.shape

    def test_global_feature_importance(
        self,
        trained_model_data: tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, np.ndarray],
    ) -> None:
        model, x_train, x_test, _ = trained_model_data
        explainer = SHAPExplainer(model, x_train)
        importance = explainer.global_feature_importance(x_test)
        assert isinstance(importance, pd.DataFrame)
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) == 5

    def test_importance_sorted_descending(
        self,
        trained_model_data: tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, np.ndarray],
    ) -> None:
        model, x_train, x_test, _ = trained_model_data
        explainer = SHAPExplainer(model, x_train)
        importance = explainer.global_feature_importance(x_test)
        values = importance["importance"].tolist()
        assert values == sorted(values, reverse=True)

    def test_waterfall_plot_structure(
        self,
        trained_model_data: tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, np.ndarray],
    ) -> None:
        model, x_train, x_test, _ = trained_model_data
        explainer = SHAPExplainer(model, x_train)
        result = explainer.waterfall_plot(x_test, idx=0)
        assert "base_value" in result
        assert "prediction" in result
        assert "features" in result
        assert len(result["features"]) == 5

    def test_explain_prediction(
        self,
        trained_model_data: tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, np.ndarray],
    ) -> None:
        model, x_train, x_test, _ = trained_model_data
        explainer = SHAPExplainer(model, x_train)
        result = explainer.explain_prediction(x_test.iloc[:1])
        assert "base_value" in result
        assert "shap_values" in result
        assert "prediction_contribution" in result
        assert "final_prediction" in result
        assert len(result["shap_values"]) == 5

    def test_shap_values_sum_to_prediction(
        self,
        trained_model_data: tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, np.ndarray],
    ) -> None:
        model, x_train, x_test, _ = trained_model_data
        explainer = SHAPExplainer(model, x_train)
        result = explainer.explain_prediction(x_test.iloc[:1])
        expected = result["base_value"] + result["prediction_contribution"]
        assert result["final_prediction"] == pytest.approx(expected, abs=1e-6)

    def test_summary_plot_no_error(
        self,
        trained_model_data: tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, np.ndarray],
    ) -> None:
        model, x_train, x_test, _ = trained_model_data
        explainer = SHAPExplainer(model, x_train)
        explainer.summary_plot(x_test)  # Should not raise


class TestLIMEExplainer:
    """Tests for LIME explainer."""

    def test_explain_prediction_structure(
        self,
        trained_model_data: tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, np.ndarray],
    ) -> None:
        model, x_train, x_test, _ = trained_model_data
        explainer = LIMEExplainer(model, x_train)
        result = explainer.explain_prediction(x_test.iloc[0].values)
        assert "prediction_proba" in result
        assert "local_prediction" in result
        assert "intercept" in result
        assert "feature_contributions" in result

    def test_explain_prediction_has_contributions(
        self,
        trained_model_data: tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, np.ndarray],
    ) -> None:
        model, x_train, x_test, _ = trained_model_data
        explainer = LIMEExplainer(model, x_train)
        result = explainer.explain_prediction(x_test.iloc[0].values)
        assert len(result["feature_contributions"]) > 0
        for contrib in result["feature_contributions"]:
            assert "feature" in contrib
            assert "contribution" in contrib

    def test_prediction_proba_valid(
        self,
        trained_model_data: tuple[RandomForestClassifier, pd.DataFrame, pd.DataFrame, np.ndarray],
    ) -> None:
        model, x_train, x_test, _ = trained_model_data
        explainer = LIMEExplainer(model, x_train)
        result = explainer.explain_prediction(x_test.iloc[0].values)
        proba = result["prediction_proba"]
        assert len(proba) == 2
        assert all(0 <= p <= 1 for p in proba)
