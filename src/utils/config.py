"""Configuration management for the credit risk scoring system.

Loads and validates YAML configuration using Pydantic models.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Configuration for data loading and splitting."""

    source: str = Field(default="german_credit", description="Dataset source identifier")
    sample_size: int | None = Field(default=None, description="Number of samples to use")
    test_size: float = Field(default=0.2, description="Fraction of data for testing")
    random_state: int = Field(default=42, description="Random seed for reproducibility")


class Config(BaseModel):
    """Root configuration model for the credit risk scoring system."""

    data: DataConfig
    model: dict[str, Any] = Field(default_factory=dict)
    scorecard: dict[str, Any] = Field(default_factory=dict)
    fairness: dict[str, Any] = Field(default_factory=dict)
    api: dict[str, Any] = Field(default_factory=dict)


def load_config(path: str = "configs/config.yaml") -> Config:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return Config(**raw)
