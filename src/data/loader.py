"""Convenience data loading functions for the credit risk scoring system.

Provides a unified interface for loading and preprocessing data
based on configuration settings.
"""

import pandas as pd

from src.data.downloader import download_german_credit
from src.data.preprocessor import CreditDataPreprocessor
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and preprocess data based on configuration.

    Args:
        config: Application configuration with data source settings.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If the data source is not recognized.
    """
    source = config.data.source

    if source == "german_credit":
        logger.info("Loading full German Credit dataset")
        df = download_german_credit()
    elif source == "sample":
        logger.info("Loading sample dataset for testing")
        df = pd.read_csv("data/sample/german_credit_sample.csv")
    else:
        raise ValueError(f"Unknown data source: {source}")

    if config.data.sample_size is not None:
        df = df.head(config.data.sample_size)
        logger.info("Sampled %d rows from dataset", config.data.sample_size)

    preprocessor = CreditDataPreprocessor(
        test_size=config.data.test_size,
        random_state=config.data.random_state,
    )

    return preprocessor.fit_transform(df)
