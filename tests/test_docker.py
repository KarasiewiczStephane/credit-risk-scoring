"""Tests for Docker configuration files."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent


class TestDockerfile:
    """Tests for Dockerfile."""

    def test_dockerfile_exists(self) -> None:
        assert (PROJECT_ROOT / "Dockerfile").exists()

    def test_dockerfile_uses_multistage_build(self) -> None:
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "AS builder" in content
        assert "COPY --from=builder" in content

    def test_dockerfile_uses_slim_base(self) -> None:
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "python:3.11-slim" in content

    def test_dockerfile_has_healthcheck(self) -> None:
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "HEALTHCHECK" in content

    def test_dockerfile_creates_nonroot_user(self) -> None:
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "useradd" in content
        assert "USER appuser" in content

    def test_dockerfile_exposes_port(self) -> None:
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "EXPOSE 8000" in content

    def test_dockerfile_sets_unbuffered(self) -> None:
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "PYTHONUNBUFFERED=1" in content

    def test_dockerfile_uses_uvicorn_factory(self) -> None:
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "--factory" in content
        assert "src.api.app:create_app" in content

    def test_dockerfile_copies_source(self) -> None:
        content = (PROJECT_ROOT / "Dockerfile").read_text()
        assert "COPY src/ src/" in content
        assert "COPY configs/ configs/" in content


class TestDockerCompose:
    """Tests for docker-compose.yml."""

    def test_compose_file_exists(self) -> None:
        assert (PROJECT_ROOT / "docker-compose.yml").exists()

    def test_compose_valid_yaml(self) -> None:
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        parsed = yaml.safe_load(content)
        assert isinstance(parsed, dict)

    def test_compose_has_api_service(self) -> None:
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        parsed = yaml.safe_load(content)
        assert "services" in parsed
        assert "credit-risk-api" in parsed["services"]

    def test_compose_port_mapping(self) -> None:
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        parsed = yaml.safe_load(content)
        service = parsed["services"]["credit-risk-api"]
        assert "8000:8000" in service["ports"]

    def test_compose_has_healthcheck(self) -> None:
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        parsed = yaml.safe_load(content)
        service = parsed["services"]["credit-risk-api"]
        assert "healthcheck" in service

    def test_compose_has_volumes(self) -> None:
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        parsed = yaml.safe_load(content)
        service = parsed["services"]["credit-risk-api"]
        assert "volumes" in service
        assert any("models" in v for v in service["volumes"])

    def test_compose_restart_policy(self) -> None:
        content = (PROJECT_ROOT / "docker-compose.yml").read_text()
        parsed = yaml.safe_load(content)
        service = parsed["services"]["credit-risk-api"]
        assert service["restart"] == "unless-stopped"


class TestDockerIgnore:
    """Tests for .dockerignore."""

    def test_dockerignore_exists(self) -> None:
        assert (PROJECT_ROOT / ".dockerignore").exists()

    def test_dockerignore_excludes_pycache(self) -> None:
        content = (PROJECT_ROOT / ".dockerignore").read_text()
        assert "__pycache__" in content

    def test_dockerignore_excludes_git(self) -> None:
        content = (PROJECT_ROOT / ".dockerignore").read_text()
        assert ".git" in content

    def test_dockerignore_excludes_tests(self) -> None:
        content = (PROJECT_ROOT / ".dockerignore").read_text()
        assert "tests" in content

    def test_dockerignore_excludes_venv(self) -> None:
        content = (PROJECT_ROOT / ".dockerignore").read_text()
        assert ".venv" in content or "venv" in content


class TestMakefile:
    """Tests for Makefile Docker targets."""

    def test_makefile_exists(self) -> None:
        assert (PROJECT_ROOT / "Makefile").exists()

    def test_makefile_has_docker_build(self) -> None:
        content = (PROJECT_ROOT / "Makefile").read_text()
        assert "docker-build:" in content

    def test_makefile_has_docker_run(self) -> None:
        content = (PROJECT_ROOT / "Makefile").read_text()
        assert "docker-run:" in content

    def test_makefile_has_docker_test(self) -> None:
        content = (PROJECT_ROOT / "Makefile").read_text()
        assert "docker-test:" in content

    def test_makefile_has_compose_targets(self) -> None:
        content = (PROJECT_ROOT / "Makefile").read_text()
        assert "docker-compose-up:" in content
        assert "docker-compose-down:" in content
