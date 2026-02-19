"""Generate sample dataset for CI testing.

Creates a 200-row stratified sample of the German Credit dataset format.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic sample data matching German Credit dataset structure.

    Args:
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with sample credit data.
    """
    rng = np.random.RandomState(random_state)

    checking_options = ["A11", "A12", "A13", "A14"]
    credit_history_options = ["A30", "A31", "A32", "A33", "A34"]
    purpose_options = ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A48", "A49", "A410"]
    savings_options = ["A61", "A62", "A63", "A64", "A65"]
    employment_options = ["A71", "A72", "A73", "A74", "A75"]
    personal_status_options = ["A91", "A92", "A93", "A94"]
    other_parties_options = ["A101", "A102", "A103"]
    property_options = ["A121", "A122", "A123", "A124"]
    other_payment_options = ["A141", "A142", "A143"]
    housing_options = ["A151", "A152", "A153"]
    job_options = ["A171", "A172", "A173", "A174"]
    telephone_options = ["A191", "A192"]
    foreign_worker_options = ["A201", "A202"]

    data = {
        "checking_status": rng.choice(checking_options, n_samples),
        "duration": rng.choice([6, 12, 18, 24, 36, 48, 60], n_samples),
        "credit_history": rng.choice(credit_history_options, n_samples),
        "purpose": rng.choice(purpose_options, n_samples),
        "credit_amount": rng.randint(250, 20000, n_samples),
        "savings_status": rng.choice(savings_options, n_samples),
        "employment": rng.choice(employment_options, n_samples),
        "installment_rate": rng.choice([1, 2, 3, 4], n_samples),
        "personal_status": rng.choice(personal_status_options, n_samples),
        "other_parties": rng.choice(other_parties_options, n_samples),
        "residence_since": rng.choice([1, 2, 3, 4], n_samples),
        "property_magnitude": rng.choice(property_options, n_samples),
        "age": rng.randint(19, 75, n_samples),
        "other_payment_plans": rng.choice(other_payment_options, n_samples),
        "housing": rng.choice(housing_options, n_samples),
        "existing_credits": rng.choice([1, 2, 3, 4], n_samples),
        "job": rng.choice(job_options, n_samples),
        "num_dependents": rng.choice([1, 2], n_samples),
        "own_telephone": rng.choice(telephone_options, n_samples),
        "foreign_worker": rng.choice(foreign_worker_options, n_samples),
        "class": rng.choice([0, 1], n_samples, p=[0.7, 0.3]),
    }

    return pd.DataFrame(data)


if __name__ == "__main__":
    output_dir = Path("data/sample")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_sample_data()
    output_path = output_dir / "german_credit_sample.csv"
    df.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path} ({len(df)} rows)")
