"""Model card generator following Google's Model Card format.

Includes ECOA and FCRA compliance checklists for regulatory alignment.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelCardQuantitativeAnalysis:
    """Quantitative performance analysis for the model card.

    Attributes:
        metrics: Dictionary of metric name to value.
        confidence_intervals: Optional CIs per metric.
        slice_metrics: Optional per-slice performance.
    """

    metrics: dict[str, float]
    confidence_intervals: dict[str, tuple] | None = None
    slice_metrics: dict[str, dict[str, float]] | None = None


@dataclass
class ModelCardFairnessAnalysis:
    """Fairness analysis section for the model card.

    Attributes:
        protected_attributes: List of analyzed attributes.
        metrics_by_group: Metrics per protected group.
        bias_mitigation_applied: Whether mitigation was applied.
        mitigation_method: Name of mitigation method if applied.
    """

    protected_attributes: list[str]
    metrics_by_group: dict[str, dict[str, float]]
    bias_mitigation_applied: bool
    mitigation_method: str | None = None


@dataclass
class ModelCard:
    """Complete model card following Google's format.

    Contains model details, intended use, training/evaluation data,
    quantitative analysis, fairness analysis, ethical considerations,
    caveats, recommendations, and compliance checks.
    """

    model_name: str
    model_version: str
    model_type: str
    model_date: str
    primary_intended_uses: list[str]
    primary_intended_users: list[str]
    out_of_scope_uses: list[str]
    training_data_description: str
    training_data_size: int
    training_data_preprocessing: list[str]
    evaluation_data_description: str
    evaluation_data_size: int
    quantitative_analysis: ModelCardQuantitativeAnalysis
    fairness_analysis: ModelCardFairnessAnalysis
    ethical_considerations: list[str]
    caveats: list[str]
    recommendations: list[str]
    compliance_checks: dict[str, bool]

    def to_dict(self) -> dict:
        """Convert model card to dictionary.

        Returns:
            Dictionary representation of the model card.
        """
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize model card to JSON string.

        Args:
            indent: JSON indentation level.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        """Save model card to a JSON file.

        Args:
            path: Output file path.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())
        logger.info("Model card saved to %s", path)


class ModelCardGenerator:
    """Generate model cards following Google's format with ECOA/FCRA compliance."""

    def __init__(self) -> None:
        self.compliance_checklist = self._get_compliance_checklist()

    @staticmethod
    def _get_compliance_checklist() -> dict[str, dict[str, dict[str, Any]]]:
        """Get ECOA and FCRA compliance requirements.

        Returns:
            Nested dictionary of regulation -> requirement -> details.
        """
        return {
            "ECOA": {
                "no_prohibited_factors": {
                    "description": (
                        "Model does not use race, color, religion, national origin, "
                        "sex, marital status, age (if applicant has legal capacity)"
                    ),
                    "checked": False,
                },
                "adverse_action_notice": {
                    "description": "System provides specific reasons for adverse actions",
                    "checked": False,
                },
                "disparate_impact_analysis": {
                    "description": "Disparate impact analysis conducted on protected groups",
                    "checked": False,
                },
            },
            "FCRA": {
                "permissible_purpose": {
                    "description": "Credit information used only for permissible purposes",
                    "checked": False,
                },
                "consumer_disclosure": {
                    "description": "Consumers can request disclosure of information used",
                    "checked": False,
                },
                "dispute_process": {
                    "description": "Process exists for consumers to dispute decisions",
                    "checked": False,
                },
                "accuracy_requirements": {
                    "description": "Reasonable procedures to ensure accuracy of information",
                    "checked": False,
                },
            },
        }

    def generate(
        self,
        model_name: str,
        model_version: str,
        model_type: str,
        training_data_info: dict,
        evaluation_metrics: dict[str, float],
        fairness_results: dict,
        compliance_status: dict[str, bool],
    ) -> ModelCard:
        """Generate a complete model card.

        Args:
            model_name: Name of the model.
            model_version: Version string.
            model_type: Model algorithm type.
            training_data_info: Info about training data.
            evaluation_metrics: Performance metrics dictionary.
            fairness_results: Fairness analysis results.
            compliance_status: Compliance check pass/fail status.

        Returns:
            Populated ModelCard instance.
        """
        card = ModelCard(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            model_date=datetime.now().isoformat(),
            primary_intended_uses=[
                "Credit risk assessment for loan applications",
                "Default probability estimation",
                "Credit score generation",
            ],
            primary_intended_users=[
                "Loan officers",
                "Credit analysts",
                "Automated underwriting systems",
            ],
            out_of_scope_uses=[
                "Criminal justice decisions",
                "Employment decisions",
                "Insurance underwriting without proper validation",
            ],
            training_data_description=training_data_info.get(
                "description", "German Credit Dataset from UCI ML Repository"
            ),
            training_data_size=training_data_info.get("size", 1000),
            training_data_preprocessing=training_data_info.get(
                "preprocessing",
                [
                    "Categorical encoding",
                    "Missing value imputation",
                    "Feature scaling",
                    "WoE transformation",
                ],
            ),
            evaluation_data_description="Stratified 20% holdout from training data",
            evaluation_data_size=int(training_data_info.get("size", 1000) * 0.2),
            quantitative_analysis=ModelCardQuantitativeAnalysis(metrics=evaluation_metrics),
            fairness_analysis=ModelCardFairnessAnalysis(
                protected_attributes=fairness_results.get(
                    "protected_attributes", ["age_group", "personal_status"]
                ),
                metrics_by_group=fairness_results.get("metrics_by_group", {}),
                bias_mitigation_applied=fairness_results.get("mitigation_applied", False),
                mitigation_method=fairness_results.get("mitigation_method"),
            ),
            ethical_considerations=[
                "Model may perpetuate historical lending biases present in training data",
                "Age and gender proxies may exist in correlated features",
                "Regular monitoring for disparate impact is recommended",
                "Human review recommended for borderline decisions",
            ],
            caveats=[
                "Model trained on German Credit data from 1994, patterns may not reflect current conditions",
                "Limited to features available in dataset, may not capture full creditworthiness",
                "Performance may vary across different demographic groups",
            ],
            recommendations=[
                "Validate model on current, representative data before production use",
                "Implement ongoing fairness monitoring",
                "Provide clear adverse action reasons to applicants",
                "Maintain human oversight for high-impact decisions",
            ],
            compliance_checks=compliance_status,
        )

        logger.info("Generated model card for %s v%s", model_name, model_version)
        return card
