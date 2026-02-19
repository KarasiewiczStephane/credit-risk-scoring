"""Tests for project setup, configuration, and logging."""

import logging
import os
import tempfile

import pytest
import yaml

from src.utils.config import Config, DataConfig, load_config
from src.utils.logger import get_logger


class TestDataConfig:
    """Tests for DataConfig model."""

    def test_default_values(self) -> None:
        config = DataConfig()
        assert config.source == "german_credit"
        assert config.sample_size is None
        assert config.test_size == 0.2
        assert config.random_state == 42

    def test_custom_values(self) -> None:
        config = DataConfig(source="sample", sample_size=200, test_size=0.3, random_state=99)
        assert config.source == "sample"
        assert config.sample_size == 200
        assert config.test_size == 0.3
        assert config.random_state == 99

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(ValueError):
            DataConfig(test_size="not_a_float")


class TestConfig:
    """Tests for root Config model."""

    def test_minimal_config(self) -> None:
        config = Config(data=DataConfig())
        assert config.data.source == "german_credit"
        assert config.model == {}
        assert config.scorecard == {}

    def test_full_config(self) -> None:
        config = Config(
            data=DataConfig(),
            model={"lightgbm": {"n_estimators": 100}},
            scorecard={"pdo": 20},
            fairness={"protected_attributes": ["age_group"]},
            api={"host": "0.0.0.0", "port": 8000},
        )
        assert config.model["lightgbm"]["n_estimators"] == 100
        assert config.scorecard["pdo"] == 20
        assert config.api["port"] == 8000


class TestLoadConfig:
    """Tests for config file loading."""

    def test_load_valid_config(self) -> None:
        config = load_config("configs/config.yaml")
        assert config.data.source == "german_credit"
        assert config.data.test_size == 0.2
        assert config.data.random_state == 42
        assert "lightgbm" in config.model
        assert config.scorecard["pdo"] == 20

    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_load_custom_config(self) -> None:
        custom = {
            "data": {"source": "sample", "test_size": 0.3, "random_state": 7},
            "model": {},
            "scorecard": {},
            "fairness": {},
            "api": {},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(custom, f)
            f.flush()
            config = load_config(f.name)

        assert config.data.source == "sample"
        assert config.data.test_size == 0.3
        os.unlink(f.name)

    def test_load_invalid_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("data:\n  source: 'test'\n  test_size: invalid\n")
            f.flush()

            with pytest.raises(ValueError):
                load_config(f.name)

        os.unlink(f.name)


class TestLogger:
    """Tests for logger configuration."""

    def test_logger_creation(self) -> None:
        log = get_logger("test_logger")
        assert isinstance(log, logging.Logger)
        assert log.name == "test_logger"
        assert log.level == logging.INFO

    def test_logger_custom_level(self) -> None:
        log = get_logger("test_debug", level=logging.DEBUG)
        assert log.level == logging.DEBUG

    def test_logger_has_handler(self) -> None:
        log = get_logger("test_handler")
        assert len(log.handlers) > 0

    def test_logger_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        log = get_logger("test_format_check")
        log.info("test message")
        captured = capsys.readouterr()
        assert "test_format_check" in captured.out
        assert "INFO" in captured.out
        assert "test message" in captured.out

    def test_logger_no_duplicate_handlers(self) -> None:
        log1 = get_logger("test_dup")
        handler_count = len(log1.handlers)
        log2 = get_logger("test_dup")
        assert len(log2.handlers) == handler_count


class TestImports:
    """Test that all module imports resolve correctly."""

    def test_config_imports(self) -> None:
        from src.utils.config import Config, DataConfig, load_config

        assert Config is not None
        assert DataConfig is not None
        assert load_config is not None

    def test_logger_imports(self) -> None:
        from src.utils.logger import get_logger

        assert get_logger is not None

    def test_main_imports(self) -> None:
        from src.main import main

        assert main is not None

    def test_api_imports(self) -> None:
        from src.api.app import create_app

        assert create_app is not None
