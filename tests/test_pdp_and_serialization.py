"""Tests for PDP generation and explanation serialization."""

import json
import os
import tempfile

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from src.explainability.pdp_generator import ExplanationSerializer, PDPGenerator


@pytest.fixture
def model_and_data() -> tuple[RandomForestClassifier, pd.DataFrame]:
    """Create a trained model with data for PDP testing."""
    x, y = make_classification(n_samples=200, n_features=5, n_informative=3, random_state=42)
    feature_names = [f"feat_{i}" for i in range(5)]
    x_df = pd.DataFrame(x, columns=feature_names)

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(x_df, y)
    return model, x_df


@pytest.fixture
def pdp_gen(model_and_data: tuple[RandomForestClassifier, pd.DataFrame]) -> PDPGenerator:
    """Create a PDP generator."""
    model, x_df = model_and_data
    return PDPGenerator(model, x_df)


class TestPDPGenerator:
    """Tests for PDPGenerator."""

    def test_compute_pdp_structure(self, pdp_gen: PDPGenerator) -> None:
        result = pdp_gen.compute_pdp("feat_0")
        assert "feature" in result
        assert "grid_values" in result
        assert "pdp_values" in result
        assert "feature_range" in result
        assert result["feature"] == "feat_0"

    def test_pdp_values_are_lists(self, pdp_gen: PDPGenerator) -> None:
        result = pdp_gen.compute_pdp("feat_0")
        assert isinstance(result["grid_values"], list)
        assert isinstance(result["pdp_values"], list)
        assert len(result["grid_values"]) == len(result["pdp_values"])

    def test_feature_range_valid(self, pdp_gen: PDPGenerator) -> None:
        result = pdp_gen.compute_pdp("feat_0")
        r = result["feature_range"]
        assert "min" in r
        assert "max" in r
        assert "mean" in r
        assert r["min"] <= r["mean"] <= r["max"]

    def test_compute_top_features_with_importance(self, pdp_gen: PDPGenerator) -> None:
        scores = {"feat_0": 0.5, "feat_1": 0.3, "feat_2": 0.1, "feat_3": 0.05, "feat_4": 0.05}
        results = pdp_gen.compute_pdp_top_features(n_features=3, importance_scores=scores)
        assert len(results) == 3
        assert results[0]["feature"] == "feat_0"
        assert results[1]["feature"] == "feat_1"

    def test_compute_top_features_from_model(self, pdp_gen: PDPGenerator) -> None:
        results = pdp_gen.compute_pdp_top_features(n_features=3)
        assert len(results) == 3

    def test_plot_pdp_no_error(self, pdp_gen: PDPGenerator) -> None:
        pdp_gen.plot_pdp(["feat_0"])  # Should not raise


class TestExplanationSerializer:
    """Tests for ExplanationSerializer."""

    def test_score_band_excellent(self) -> None:
        assert ExplanationSerializer._get_score_band(750) == "Excellent"
        assert ExplanationSerializer._get_score_band(800) == "Excellent"

    def test_score_band_good(self) -> None:
        assert ExplanationSerializer._get_score_band(700) == "Good"
        assert ExplanationSerializer._get_score_band(749) == "Good"

    def test_score_band_fair(self) -> None:
        assert ExplanationSerializer._get_score_band(650) == "Fair"

    def test_score_band_poor(self) -> None:
        assert ExplanationSerializer._get_score_band(600) == "Poor"

    def test_score_band_very_poor(self) -> None:
        assert ExplanationSerializer._get_score_band(500) == "Very Poor"

    def test_serialize_produces_valid_json(self) -> None:
        shap_expl = {"shap_values": {"feat_a": -0.3, "feat_b": 0.5, "feat_c": -0.1}}
        lime_expl = {"feature_contributions": []}
        result = ExplanationSerializer.serialize_prediction_explanation(
            application_id="APP-001",
            credit_score=700,
            probability_of_default=0.05,
            decision="approved",
            shap_explanation=shap_expl,
            lime_explanation=lime_expl,
        )
        # Should be JSON serializable
        json_str = json.dumps(result, default=str)
        assert json.loads(json_str) is not None

    def test_serialize_has_expected_keys(self) -> None:
        shap_expl = {"shap_values": {"feat_a": 0.1}}
        lime_expl = {}
        result = ExplanationSerializer.serialize_prediction_explanation(
            application_id="APP-002",
            credit_score=650,
            probability_of_default=0.1,
            decision="manual_review",
            shap_explanation=shap_expl,
            lime_explanation=lime_expl,
        )
        assert "application_id" in result
        assert "timestamp" in result
        assert "decision" in result
        assert "explanations" in result
        assert "top_risk_factors" in result
        assert "top_positive_factors" in result

    def test_extract_top_risk_factors(self) -> None:
        shap_expl = {"shap_values": {"a": -0.5, "b": -0.3, "c": 0.2, "d": -0.1}}
        factors = ExplanationSerializer._extract_top_factors(shap_expl, n=2)
        assert len(factors) == 2
        assert factors[0]["feature"] == "a"
        assert factors[0]["direction"] == "increases_risk"

    def test_extract_positive_factors(self) -> None:
        shap_expl = {"shap_values": {"a": 0.5, "b": 0.3, "c": -0.2}}
        factors = ExplanationSerializer._extract_positive_factors(shap_expl, n=2)
        assert len(factors) == 2
        assert factors[0]["feature"] == "a"
        assert factors[0]["direction"] == "decreases_risk"

    def test_extract_factors_empty_shap(self) -> None:
        assert ExplanationSerializer._extract_top_factors({}) == []
        assert ExplanationSerializer._extract_positive_factors({}) == []

    def test_save_explanation(self) -> None:
        explanation = {"test": "data", "value": 42}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        ExplanationSerializer.save_explanation(explanation, path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded == explanation
        os.unlink(path)

    def test_timestamp_is_iso_format(self) -> None:
        shap_expl = {"shap_values": {}}
        result = ExplanationSerializer.serialize_prediction_explanation(
            application_id="APP-003",
            credit_score=700,
            probability_of_default=0.05,
            decision="approved",
            shap_explanation=shap_expl,
            lime_explanation={},
        )
        # ISO 8601 format includes 'T'
        assert "T" in result["timestamp"]
