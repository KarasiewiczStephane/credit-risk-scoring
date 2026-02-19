"""Tests for model card generator and compliance checklist."""

import json
import os
import tempfile

import pytest

from src.models.registry import (
    ModelCard,
    ModelCardFairnessAnalysis,
    ModelCardGenerator,
    ModelCardQuantitativeAnalysis,
)


@pytest.fixture
def generator() -> ModelCardGenerator:
    """Create a model card generator."""
    return ModelCardGenerator()


@pytest.fixture
def sample_card(generator: ModelCardGenerator) -> ModelCard:
    """Generate a sample model card."""
    return generator.generate(
        model_name="CreditRiskLGBM",
        model_version="1.0.0",
        model_type="LightGBM",
        training_data_info={"size": 800, "description": "German Credit Dataset"},
        evaluation_metrics={"auc_roc": 0.78, "gini": 0.56, "ks_statistic": 0.42},
        fairness_results={
            "protected_attributes": ["age_group"],
            "metrics_by_group": {"age_group": {"<25": 0.7, "25-35": 0.8}},
            "mitigation_applied": True,
            "mitigation_method": "reweighting",
        },
        compliance_status={
            "no_prohibited_factors": True,
            "adverse_action_notice": True,
            "disparate_impact_analysis": True,
        },
    )


class TestModelCardGenerator:
    """Tests for ModelCardGenerator."""

    def test_generate_returns_model_card(self, sample_card: ModelCard) -> None:
        assert isinstance(sample_card, ModelCard)

    def test_model_name_populated(self, sample_card: ModelCard) -> None:
        assert sample_card.model_name == "CreditRiskLGBM"
        assert sample_card.model_version == "1.0.0"
        assert sample_card.model_type == "LightGBM"

    def test_intended_uses_populated(self, sample_card: ModelCard) -> None:
        assert len(sample_card.primary_intended_uses) > 0
        assert len(sample_card.primary_intended_users) > 0
        assert len(sample_card.out_of_scope_uses) > 0

    def test_training_data_populated(self, sample_card: ModelCard) -> None:
        assert sample_card.training_data_size == 800
        assert sample_card.training_data_description == "German Credit Dataset"
        assert len(sample_card.training_data_preprocessing) > 0

    def test_evaluation_size_calculated(self, sample_card: ModelCard) -> None:
        assert sample_card.evaluation_data_size == 160  # 800 * 0.2

    def test_quantitative_analysis_populated(self, sample_card: ModelCard) -> None:
        qa = sample_card.quantitative_analysis
        assert isinstance(qa, ModelCardQuantitativeAnalysis)
        assert qa.metrics["auc_roc"] == 0.78
        assert qa.metrics["gini"] == 0.56

    def test_fairness_analysis_populated(self, sample_card: ModelCard) -> None:
        fa = sample_card.fairness_analysis
        assert isinstance(fa, ModelCardFairnessAnalysis)
        assert "age_group" in fa.protected_attributes
        assert fa.bias_mitigation_applied is True
        assert fa.mitigation_method == "reweighting"

    def test_ethical_considerations_nonempty(self, sample_card: ModelCard) -> None:
        assert len(sample_card.ethical_considerations) > 0

    def test_caveats_nonempty(self, sample_card: ModelCard) -> None:
        assert len(sample_card.caveats) > 0

    def test_recommendations_nonempty(self, sample_card: ModelCard) -> None:
        assert len(sample_card.recommendations) > 0

    def test_compliance_checks_populated(self, sample_card: ModelCard) -> None:
        assert "no_prohibited_factors" in sample_card.compliance_checks
        assert sample_card.compliance_checks["no_prohibited_factors"] is True


class TestModelCard:
    """Tests for ModelCard serialization."""

    def test_to_dict(self, sample_card: ModelCard) -> None:
        d = sample_card.to_dict()
        assert isinstance(d, dict)
        assert "model_name" in d
        assert "quantitative_analysis" in d
        assert "fairness_analysis" in d

    def test_to_json_valid(self, sample_card: ModelCard) -> None:
        j = sample_card.to_json()
        parsed = json.loads(j)
        assert parsed["model_name"] == "CreditRiskLGBM"

    def test_save_creates_file(self, sample_card: ModelCard) -> None:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        sample_card.save(path)
        assert os.path.exists(path)

        with open(path) as f:
            loaded = json.load(f)
        assert loaded["model_name"] == "CreditRiskLGBM"
        os.unlink(path)

    def test_model_date_populated(self, sample_card: ModelCard) -> None:
        assert sample_card.model_date is not None
        assert "T" in sample_card.model_date  # ISO format


class TestComplianceChecklist:
    """Tests for compliance checklist."""

    def test_checklist_has_ecoa(self, generator: ModelCardGenerator) -> None:
        assert "ECOA" in generator.compliance_checklist

    def test_checklist_has_fcra(self, generator: ModelCardGenerator) -> None:
        assert "FCRA" in generator.compliance_checklist

    def test_ecoa_has_required_checks(self, generator: ModelCardGenerator) -> None:
        ecoa = generator.compliance_checklist["ECOA"]
        assert "no_prohibited_factors" in ecoa
        assert "adverse_action_notice" in ecoa
        assert "disparate_impact_analysis" in ecoa

    def test_fcra_has_required_checks(self, generator: ModelCardGenerator) -> None:
        fcra = generator.compliance_checklist["FCRA"]
        assert "permissible_purpose" in fcra
        assert "consumer_disclosure" in fcra
        assert "dispute_process" in fcra
        assert "accuracy_requirements" in fcra

    def test_checklist_items_have_description(self, generator: ModelCardGenerator) -> None:
        for regulation in generator.compliance_checklist.values():
            for check in regulation.values():
                assert "description" in check
                assert len(check["description"]) > 0
