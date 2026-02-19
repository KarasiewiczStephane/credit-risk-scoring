"""Tests for project documentation completeness."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


class TestReadme:
    """Tests for README.md content."""

    def test_readme_exists(self) -> None:
        assert (PROJECT_ROOT / "README.md").exists()

    def test_readme_has_title(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "# Credit Risk Scoring" in content

    def test_readme_has_overview(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "## Overview" in content

    def test_readme_has_architecture(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "## Architecture" in content
        assert "FastAPI" in content

    def test_readme_has_quick_start(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "## Quick Start" in content
        assert "pip install" in content

    def test_readme_has_scorecard_example(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "## Scorecard Example" in content
        assert "WoE" in content
        assert "Points" in content
        assert "Base Score" in content

    def test_readme_has_score_interpretation(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "Score Interpretation" in content
        assert "Excellent" in content
        assert "Very Poor" in content

    def test_readme_has_model_performance(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "## Model Performance" in content
        assert "AUC-ROC" in content
        assert "Gini" in content
        assert "KS Statistic" in content

    def test_readme_has_fairness_analysis(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "Fairness" in content
        assert "Before Mitigation" in content
        assert "After Mitigation" in content
        assert "Demographic Parity" in content

    def test_readme_has_api_usage(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "## API Usage" in content
        assert "/score" in content
        assert "/explain" in content
        assert "curl" in content

    def test_readme_has_docker_instructions(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "## Docker" in content
        assert "docker-build" in content

    def test_readme_has_project_structure(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "## Project Structure" in content
        assert "src/" in content
        assert "tests/" in content

    def test_readme_has_regulatory_compliance(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "Regulatory Compliance" in content
        assert "ECOA" in content
        assert "FCRA" in content

    def test_readme_has_ci_badge(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "badge.svg" in content

    def test_readme_has_license(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "MIT" in content

    def test_readme_no_todo_placeholders(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "TODO" not in content

    def test_readme_no_your_username_placeholder(self) -> None:
        content = (PROJECT_ROOT / "README.md").read_text()
        assert "YOUR_USERNAME" not in content


class TestProjectFiles:
    """Tests for required project files."""

    def test_requirements_exists(self) -> None:
        assert (PROJECT_ROOT / "requirements.txt").exists()

    def test_config_exists(self) -> None:
        assert (PROJECT_ROOT / "configs" / "config.yaml").exists()

    def test_pyproject_exists(self) -> None:
        assert (PROJECT_ROOT / "pyproject.toml").exists()

    def test_dockerfile_exists(self) -> None:
        assert (PROJECT_ROOT / "Dockerfile").exists()

    def test_makefile_exists(self) -> None:
        assert (PROJECT_ROOT / "Makefile").exists()

    def test_gitignore_exists(self) -> None:
        assert (PROJECT_ROOT / ".gitignore").exists()

    def test_pre_commit_config_exists(self) -> None:
        assert (PROJECT_ROOT / ".pre-commit-config.yaml").exists()
