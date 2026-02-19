"""Dataset download utilities for the credit risk scoring system.

Handles downloading and caching the German Credit dataset from UCI ML Repository.
"""

import urllib.request
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

GERMAN_CREDIT_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
)

COLUMN_NAMES = [
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_rate",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
    "class",
]


def download_german_credit(data_dir: str = "data/raw") -> pd.DataFrame:
    """Download German Credit dataset from UCI repository.

    Downloads the dataset if not already cached locally. Converts the target
    variable so that 1=Good -> 0 (non-default), 2=Bad -> 1 (default).

    Args:
        data_dir: Directory to store the downloaded data.

    Returns:
        DataFrame with the German Credit dataset.
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(data_dir) / "german_credit.csv"

    if not filepath.exists():
        logger.info("Downloading German Credit dataset from UCI repository...")
        raw_path = filepath.with_suffix(".data")
        urllib.request.urlretrieve(GERMAN_CREDIT_URL, raw_path)  # noqa: S310

        df = pd.read_csv(raw_path, sep=" ", header=None, names=COLUMN_NAMES)
        df["class"] = df["class"].map({1: 0, 2: 1})  # 1=Good->0, 2=Bad->1
        df.to_csv(filepath, index=False)
        logger.info("Dataset saved to %s (%d rows, %d columns)", filepath, len(df), len(df.columns))
    else:
        logger.info("Loading cached dataset from %s", filepath)

    return pd.read_csv(filepath)
