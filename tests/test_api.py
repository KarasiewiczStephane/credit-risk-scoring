"""Tests for FastAPI credit risk scoring endpoints."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import _get_score_band, app_state, create_app
from src.api.schemas import (
    CreditApplicationRequest,
    DecisionEnum,
    ScoreResponse,
)


@pytest.fixture
def client() -> TestClient:
    """Create a test client with no loaded models."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_request_data() -> dict:
    """Sample credit application request data."""
    return {
        "application_id": "APP-TEST-001",
        "checking_status": "A11",
        "duration": 24,
        "credit_history": "A34",
        "purpose": "A43",
        "credit_amount": 5000,
        "savings_status": "A61",
        "employment": "A73",
        "installment_rate": 2,
        "personal_status": "A93",
        "other_parties": "A101",
        "residence_since": 2,
        "property_magnitude": "A121",
        "age": 35,
        "other_payment_plans": "A143",
        "housing": "A152",
        "existing_credits": 1,
        "job": "A173",
        "num_dependents": 1,
        "own_telephone": "A192",
        "foreign_worker": "A201",
    }


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_healthy(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestScoreBand:
    """Tests for _get_score_band helper."""

    def test_excellent_band(self) -> None:
        assert _get_score_band(750) == "Excellent"
        assert _get_score_band(800) == "Excellent"

    def test_good_band(self) -> None:
        assert _get_score_band(700) == "Good"
        assert _get_score_band(749) == "Good"

    def test_fair_band(self) -> None:
        assert _get_score_band(650) == "Fair"
        assert _get_score_band(699) == "Fair"

    def test_poor_band(self) -> None:
        assert _get_score_band(600) == "Poor"
        assert _get_score_band(649) == "Poor"

    def test_very_poor_band(self) -> None:
        assert _get_score_band(599) == "Very Poor"
        assert _get_score_band(300) == "Very Poor"


class TestScoreEndpoint:
    """Tests for POST /score endpoint."""

    def test_score_returns_503_when_model_not_loaded(
        self, client: TestClient, sample_request_data: dict
    ) -> None:
        app_state.clear()
        response = client.post("/score", json=sample_request_data)
        assert response.status_code == 503

    def test_score_returns_valid_response(
        self, client: TestClient, sample_request_data: dict
    ) -> None:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])

        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = pd.DataFrame(np.random.rand(1, 20))

        mock_scorecard = MagicMock()
        mock_scorecard.score.return_value = np.array([720])

        mock_shap = MagicMock()
        mock_shap.explain_prediction.return_value = {
            "shap_values": {"duration": -0.15, "credit_amount": 0.10, "age": 0.05},
            "base_value": 0.3,
        }

        app_state["model"] = mock_model
        app_state["preprocessor"] = mock_preprocessor
        app_state["scorecard"] = mock_scorecard
        app_state["shap_explainer"] = mock_shap
        app_state["explanations_cache"] = {}

        response = client.post("/score", json=sample_request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["application_id"] == "APP-TEST-001"
        assert data["credit_score"] == 720
        assert data["probability_of_default"] == 0.2
        assert data["decision"] == "approved"
        assert data["score_band"] == "Good"
        assert "top_factors" in data["explanation_summary"]
        assert data["processing_time_ms"] > 0

        app_state.clear()

    def test_score_decision_review(self, client: TestClient, sample_request_data: dict) -> None:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.6, 0.4]])

        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = pd.DataFrame(np.random.rand(1, 20))

        mock_scorecard = MagicMock()
        mock_scorecard.score.return_value = np.array([650])

        mock_shap = MagicMock()
        mock_shap.explain_prediction.return_value = {
            "shap_values": {"duration": -0.1},
            "base_value": 0.3,
        }

        app_state["model"] = mock_model
        app_state["preprocessor"] = mock_preprocessor
        app_state["scorecard"] = mock_scorecard
        app_state["shap_explainer"] = mock_shap
        app_state["explanations_cache"] = {}

        response = client.post("/score", json=sample_request_data)
        assert response.status_code == 200
        assert response.json()["decision"] == "manual_review"

        app_state.clear()

    def test_score_decision_declined(self, client: TestClient, sample_request_data: dict) -> None:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = pd.DataFrame(np.random.rand(1, 20))

        mock_scorecard = MagicMock()
        mock_scorecard.score.return_value = np.array([500])

        mock_shap = MagicMock()
        mock_shap.explain_prediction.return_value = {
            "shap_values": {"duration": -0.3},
            "base_value": 0.3,
        }

        app_state["model"] = mock_model
        app_state["preprocessor"] = mock_preprocessor
        app_state["scorecard"] = mock_scorecard
        app_state["shap_explainer"] = mock_shap
        app_state["explanations_cache"] = {}

        response = client.post("/score", json=sample_request_data)
        assert response.status_code == 200
        assert response.json()["decision"] == "declined"

        app_state.clear()

    def test_score_caches_result(self, client: TestClient, sample_request_data: dict) -> None:
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])

        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = pd.DataFrame(np.random.rand(1, 20))

        mock_scorecard = MagicMock()
        mock_scorecard.score.return_value = np.array([700])

        mock_shap = MagicMock()
        mock_shap.explain_prediction.return_value = {
            "shap_values": {"duration": -0.1},
            "base_value": 0.3,
        }

        app_state["model"] = mock_model
        app_state["preprocessor"] = mock_preprocessor
        app_state["scorecard"] = mock_scorecard
        app_state["shap_explainer"] = mock_shap
        app_state["explanations_cache"] = {}

        client.post("/score", json=sample_request_data)
        assert "APP-TEST-001" in app_state["explanations_cache"]

        app_state.clear()


class TestExplainEndpoint:
    """Tests for GET /explain/{application_id} endpoint."""

    def test_explain_not_found(self, client: TestClient) -> None:
        app_state["explanations_cache"] = {}
        response = client.get("/explain/NONEXISTENT")
        assert response.status_code == 404
        app_state.clear()

    def test_explain_returns_valid_response(self, client: TestClient) -> None:
        cached_x = pd.DataFrame(np.random.rand(1, 5), columns=["a", "b", "c", "d", "e"])

        app_state["explanations_cache"] = {
            "APP-EXPLAIN-001": {
                "X": cached_x,
                "credit_score": 700,
                "prob": 0.25,
            }
        }

        mock_shap = MagicMock()
        mock_shap.explain_prediction.return_value = {
            "shap_values": {"a": -0.2, "b": 0.15, "c": -0.05},
            "base_value": 0.3,
        }

        mock_lime = MagicMock()
        mock_lime.explain_prediction.return_value = {
            "feature_contributions": [("a", -0.18), ("b", 0.12)],
            "prediction_proba": 0.25,
        }

        app_state["shap_explainer"] = mock_shap
        app_state["lime_explainer"] = mock_lime

        response = client.get("/explain/APP-EXPLAIN-001")
        assert response.status_code == 200

        data = response.json()
        assert data["application_id"] == "APP-EXPLAIN-001"
        assert "shap_explanation" in data
        assert "lime_explanation" in data
        assert isinstance(data["top_risk_factors"], list)
        assert isinstance(data["top_positive_factors"], list)

        app_state.clear()


class TestFairnessReportEndpoint:
    """Tests for GET /fairness-report endpoint."""

    def test_fairness_report_not_available(self, client: TestClient) -> None:
        app_state.clear()
        response = client.get("/fairness-report")
        assert response.status_code == 503

    def test_fairness_report_returns_data(self, client: TestClient) -> None:
        app_state["fairness_report"] = {
            "report_date": "2025-01-01",
            "protected_attributes": ["age_group", "personal_status"],
            "metrics": {"demographic_parity_ratio": 0.85},
            "fairness_constraints_satisfied": {"demographic_parity": True},
            "recommendations": ["Continue monitoring"],
        }

        response = client.get("/fairness-report")
        assert response.status_code == 200

        data = response.json()
        assert data["report_date"] == "2025-01-01"
        assert "age_group" in data["protected_attributes"]
        assert data["fairness_constraints_satisfied"]["demographic_parity"] is True

        app_state.clear()


class TestModelCardEndpoint:
    """Tests for GET /model-card endpoint."""

    def test_model_card_not_available(self, client: TestClient) -> None:
        app_state.clear()
        response = client.get("/model-card")
        assert response.status_code == 503

    def test_model_card_returns_data(self, client: TestClient) -> None:
        mock_card = MagicMock()
        mock_card.model_name = "CreditRiskLGBM"
        mock_card.model_version = "1.0.0"
        mock_card.model_type = "LightGBM"
        mock_card.model_date = "2025-01-01"
        mock_card.primary_intended_uses = ["Credit risk assessment"]
        mock_card.quantitative_analysis.metrics = {"auc_roc": 0.78}
        mock_card.fairness_analysis.protected_attributes = ["age_group"]
        mock_card.fairness_analysis.bias_mitigation_applied = True
        mock_card.ethical_considerations = ["Monitor for bias"]
        mock_card.compliance_checks = {"no_prohibited_factors": True}

        app_state["model_card"] = mock_card

        response = client.get("/model-card")
        assert response.status_code == 200

        data = response.json()
        assert data["model_name"] == "CreditRiskLGBM"
        assert data["model_version"] == "1.0.0"
        assert data["evaluation_metrics"]["auc_roc"] == 0.78
        assert data["compliance_status"]["no_prohibited_factors"] is True

        app_state.clear()


class TestSchemas:
    """Tests for Pydantic request/response schemas."""

    def test_credit_application_request_valid(self, sample_request_data: dict) -> None:
        req = CreditApplicationRequest(**sample_request_data)
        assert req.application_id == "APP-TEST-001"
        assert req.duration == 24
        assert req.credit_amount == 5000

    def test_credit_application_request_missing_field(self) -> None:
        with pytest.raises(ValueError):
            CreditApplicationRequest(application_id="APP-001")

    def test_decision_enum_values(self) -> None:
        assert DecisionEnum.APPROVED == "approved"
        assert DecisionEnum.DECLINED == "declined"
        assert DecisionEnum.REVIEW == "manual_review"

    def test_score_response_schema(self) -> None:
        resp = ScoreResponse(
            application_id="APP-001",
            credit_score=700,
            probability_of_default=0.25,
            decision=DecisionEnum.APPROVED,
            score_band="Good",
            explanation_summary={"top_factors": []},
            processing_time_ms=15.5,
        )
        assert resp.credit_score == 700
        assert resp.decision == DecisionEnum.APPROVED


class TestAppFactory:
    """Tests for create_app factory function."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        app = create_app()
        assert app.title == "Credit Risk Scoring API"
        assert app.version == "1.0.0"

    def test_create_app_with_config(self) -> None:
        from src.utils.config import Config

        config = Config(
            data={"source": "sample", "sample_size": None, "test_size": 0.2, "random_state": 42},
            model={"lightgbm": {"n_estimators": 50}},
            scorecard={"pdo": 20, "base_score": 600, "base_odds": 50},
            fairness={"protected_attributes": ["age_group"]},
            api={"host": "0.0.0.0", "port": 8000},
        )
        app = create_app(config=config)
        assert app.title == "Credit Risk Scoring API"
