"""Pydantic schemas for the credit risk scoring API."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class DecisionEnum(StrEnum):
    """Loan decision categories."""

    APPROVED = "approved"
    DECLINED = "declined"
    REVIEW = "manual_review"


class CreditApplicationRequest(BaseModel):
    """Request schema for credit scoring endpoint."""

    application_id: str = Field(..., description="Unique application identifier")
    checking_status: str
    duration: int
    credit_history: str
    purpose: str
    credit_amount: float
    savings_status: str
    employment: str
    installment_rate: int
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    age: int
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "application_id": "APP-001",
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
            ]
        }
    }


class ScoreResponse(BaseModel):
    """Response schema for credit scoring endpoint."""

    application_id: str
    credit_score: int
    probability_of_default: float
    decision: DecisionEnum
    score_band: str
    explanation_summary: dict[str, Any]
    processing_time_ms: float


class DetailedExplanation(BaseModel):
    """Response schema for detailed explanation endpoint."""

    application_id: str
    shap_explanation: dict[str, Any]
    lime_explanation: dict[str, Any]
    scorecard_breakdown: dict[str, Any] | None
    top_risk_factors: list[dict]
    top_positive_factors: list[dict]


class FairnessReportResponse(BaseModel):
    """Response schema for fairness report endpoint."""

    report_date: str
    protected_attributes: list[str]
    metrics: dict[str, Any]
    fairness_constraints_satisfied: dict[str, bool]
    recommendations: list[str]


class ModelCardResponse(BaseModel):
    """Response schema for model card endpoint."""

    model_name: str
    model_version: str
    model_type: str
    model_date: str
    intended_uses: list[str]
    evaluation_metrics: dict[str, float]
    fairness_analysis: dict[str, Any]
    ethical_considerations: list[str]
    compliance_status: dict[str, bool]
