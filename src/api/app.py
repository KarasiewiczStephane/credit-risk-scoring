"""FastAPI application factory for the credit risk scoring API.

Provides endpoints for credit scoring, explanations, fairness reporting,
and model card access.
"""

import time
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    CreditApplicationRequest,
    DecisionEnum,
    DetailedExplanation,
    FairnessReportResponse,
    ModelCardResponse,
    ScoreResponse,
)
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Global state for loaded models
app_state: dict[str, Any] = {}


def _get_score_band(score: int) -> str:
    """Classify credit score into risk band.

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


def _load_artifacts() -> None:
    """Load pre-trained model artifacts from disk."""
    try:
        app_state["model"] = joblib.load("models/lightgbm_model.pkl")
        app_state["scorecard"] = joblib.load("models/scorecard.pkl")
        app_state["preprocessor"] = joblib.load("models/preprocessor.pkl")
        app_state["shap_explainer"] = joblib.load("models/shap_explainer.pkl")
        app_state["lime_explainer"] = joblib.load("models/lime_explainer.pkl")
        app_state["fairness_report"] = joblib.load("models/fairness_report.pkl")
        app_state["model_card"] = joblib.load("models/model_card.pkl")
        app_state["explanations_cache"] = {}
        logger.info("Model artifacts loaded successfully")
    except FileNotFoundError:
        logger.warning("Model artifacts not found, API running in schema-only mode")
        app_state["explanations_cache"] = {}


def create_app(config: Config | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Application configuration. If None, defaults are used.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Credit Risk Scoring API",
        description="Interpretable ML model for loan approval decisions",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/score", response_model=ScoreResponse)
    async def score_application(request: CreditApplicationRequest) -> ScoreResponse:
        """Score a credit application and return decision with explanation."""
        if "model" not in app_state:
            raise HTTPException(status_code=503, detail="Model not loaded")

        start_time = time.time()

        data = request.model_dump(exclude={"application_id"})
        df = pd.DataFrame([data])

        preprocessor = app_state["preprocessor"]
        x = preprocessor.transform(df)

        model = app_state["model"]
        prob = float(model.predict_proba(x)[0][1])

        scorecard = app_state["scorecard"]
        credit_score = int(scorecard.score(df)[0])

        if prob < 0.3:
            decision = DecisionEnum.APPROVED
        elif prob < 0.5:
            decision = DecisionEnum.REVIEW
        else:
            decision = DecisionEnum.DECLINED

        shap_expl = app_state["shap_explainer"].explain_prediction(x)

        app_state["explanations_cache"][request.application_id] = {
            "X": x,
            "credit_score": credit_score,
            "prob": prob,
        }

        processing_time = (time.time() - start_time) * 1000

        return ScoreResponse(
            application_id=request.application_id,
            credit_score=credit_score,
            probability_of_default=round(prob, 4),
            decision=decision,
            score_band=_get_score_band(credit_score),
            explanation_summary={"top_factors": list(shap_expl["shap_values"].items())[:5]},
            processing_time_ms=round(processing_time, 2),
        )

    @app.get("/explain/{application_id}", response_model=DetailedExplanation)
    async def get_explanation(application_id: str) -> DetailedExplanation:
        """Get detailed SHAP and LIME explanation for a scored application."""
        if application_id not in app_state.get("explanations_cache", {}):
            raise HTTPException(status_code=404, detail="Application not found")

        cached = app_state["explanations_cache"][application_id]
        x = cached["X"]

        shap_expl = app_state["shap_explainer"].explain_prediction(x)
        lime_expl = app_state["lime_explainer"].explain_prediction(x.values[0])

        return DetailedExplanation(
            application_id=application_id,
            shap_explanation=shap_expl,
            lime_explanation=lime_expl,
            scorecard_breakdown=None,
            top_risk_factors=[
                {"feature": k, "impact": v}
                for k, v in list(shap_expl["shap_values"].items())[:3]
                if v < 0
            ],
            top_positive_factors=[
                {"feature": k, "impact": v}
                for k, v in list(shap_expl["shap_values"].items())[:3]
                if v > 0
            ],
        )

    @app.get("/fairness-report", response_model=FairnessReportResponse)
    async def get_fairness_report() -> FairnessReportResponse:
        """Get the fairness analysis report."""
        if "fairness_report" not in app_state:
            raise HTTPException(status_code=503, detail="Fairness report not available")
        return FairnessReportResponse(**app_state["fairness_report"])

    @app.get("/model-card", response_model=ModelCardResponse)
    async def get_model_card() -> ModelCardResponse:
        """Get the model card."""
        if "model_card" not in app_state:
            raise HTTPException(status_code=503, detail="Model card not available")

        card = app_state["model_card"]
        return ModelCardResponse(
            model_name=card.model_name,
            model_version=card.model_version,
            model_type=card.model_type,
            model_date=card.model_date,
            intended_uses=card.primary_intended_uses,
            evaluation_metrics=card.quantitative_analysis.metrics,
            fairness_analysis={
                "protected_attributes": card.fairness_analysis.protected_attributes,
                "mitigation_applied": card.fairness_analysis.bias_mitigation_applied,
            },
            ethical_considerations=card.ethical_considerations,
            compliance_status=card.compliance_checks,
        )

    return app
