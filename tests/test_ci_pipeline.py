"""Tests for GitHub Actions CI pipeline configuration."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = PROJECT_ROOT / ".github" / "workflows"


class TestCIWorkflow:
    """Tests for ci.yml workflow."""

    def test_ci_workflow_exists(self) -> None:
        assert (WORKFLOWS_DIR / "ci.yml").exists()

    def test_ci_workflow_valid_yaml(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)

    def test_ci_triggers_on_push_and_pr(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        parsed = yaml.safe_load(content)
        assert "push" in parsed[True]
        assert "pull_request" in parsed[True]

    def test_ci_triggers_on_main_branch(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        parsed = yaml.safe_load(content)
        assert "main" in parsed[True]["push"]["branches"]

    def test_ci_has_lint_job(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        parsed = yaml.safe_load(content)
        assert "lint" in parsed["jobs"]

    def test_ci_has_test_job(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        parsed = yaml.safe_load(content)
        assert "test" in parsed["jobs"]

    def test_ci_has_docker_job(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        parsed = yaml.safe_load(content)
        assert "docker" in parsed["jobs"]

    def test_ci_has_security_job(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        parsed = yaml.safe_load(content)
        assert "security" in parsed["jobs"]

    def test_test_job_depends_on_lint(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        parsed = yaml.safe_load(content)
        assert "lint" in parsed["jobs"]["test"]["needs"]

    def test_docker_job_depends_on_test(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        parsed = yaml.safe_load(content)
        assert "test" in parsed["jobs"]["docker"]["needs"]

    def test_lint_job_uses_ruff(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        assert "ruff check" in content
        assert "ruff format" in content

    def test_test_job_uses_pytest(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        assert "pytest" in content

    def test_test_job_has_coverage(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        assert "--cov=src" in content
        assert "--cov-fail-under=80" in content

    def test_security_job_uses_bandit(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        assert "bandit" in content

    def test_ci_uses_python_311(self) -> None:
        content = (WORKFLOWS_DIR / "ci.yml").read_text()
        assert "3.11" in content


class TestReleaseWorkflow:
    """Tests for release.yml workflow."""

    def test_release_workflow_exists(self) -> None:
        assert (WORKFLOWS_DIR / "release.yml").exists()

    def test_release_workflow_valid_yaml(self) -> None:
        content = (WORKFLOWS_DIR / "release.yml").read_text()
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)

    def test_release_triggers_on_tags(self) -> None:
        content = (WORKFLOWS_DIR / "release.yml").read_text()
        parsed = yaml.safe_load(content)
        assert "push" in parsed[True]
        assert "tags" in parsed[True]["push"]

    def test_release_uses_docker_buildx(self) -> None:
        content = (WORKFLOWS_DIR / "release.yml").read_text()
        assert "docker/setup-buildx-action" in content

    def test_release_uses_docker_login(self) -> None:
        content = (WORKFLOWS_DIR / "release.yml").read_text()
        assert "docker/login-action" in content

    def test_release_builds_and_pushes(self) -> None:
        content = (WORKFLOWS_DIR / "release.yml").read_text()
        assert "docker/build-push-action" in content
        assert "push: true" in content
