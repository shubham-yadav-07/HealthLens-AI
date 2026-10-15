"""Shared test fixtures. Tests run from anywhere: `pytest` in the project root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.config import ALL_COLUMNS


@pytest.fixture(scope="session")
def small_df():
    """Synthetic frame used ONLY to test code paths (never shown to users or used for results)."""
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "Pregnancies": rng.integers(0, 12, n),
        "Glucose": rng.normal(120, 30, n).clip(50, 200).round(),
        "BloodPressure": rng.normal(70, 10, n).clip(40, 110).round(),
        "SkinThickness": rng.normal(25, 8, n).clip(5, 60).round(),
        "Insulin": rng.normal(100, 50, n).clip(10, 400).round(),
        "BMI": rng.normal(32, 6, n).clip(18, 55).round(1),
        "DiabetesPedigreeFunction": rng.uniform(0.1, 1.5, n).round(3),
        "Age": rng.integers(21, 70, n),
    })
    score = (df["Glucose"] - 120) / 30 + rng.normal(0, 1, n)
    df["Outcome"] = (score > 0.3).astype(int)
    df.loc[:10, "Insulin"] = 0       # impossible zeros
    df.loc[5:8, "Glucose"] = 0
    return df[ALL_COLUMNS]


@pytest.fixture(scope="session")
def quick_results(small_df):
    from src.train_models import train_all
    return train_all(small_df, tune=False, nested_cv=False, save=False)
