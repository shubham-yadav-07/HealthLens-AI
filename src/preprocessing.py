"""
Preprocessing building blocks.

KEY IDEA - preventing data leakage
----------------------------------
Everything that LEARNS something from the data (the median used for
imputation, the mean/std used for scaling) lives inside an sklearn Pipeline.
When we call ``pipeline.fit(X_train, y_train)`` those values are computed from
the TRAINING rows only. The test rows are only ever *transformed* with the
values learned on the training rows. Cross-validation also re-fits the whole
pipeline inside every fold, so validation folds never leak either.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURES, RANDOM_STATE, TARGET, TEST_SIZE, ZERO_AS_MISSING,
)


class ZeroAsMissing(BaseEstimator, TransformerMixin):
    """Replace impossible zeros (e.g. Glucose = 0) with NaN.

    This transformer learns nothing from the data, so it cannot leak. It simply
    converts "0 = not measured" into proper missing values so that the imputer
    (next pipeline step) can handle them.
    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        self.columns_ = list(self.columns) if self.columns is not None else list(ZERO_AS_MISSING)
        self.feature_names_in_ = np.array(list(X.columns)) if hasattr(X, "columns") else None
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ZeroAsMissing expects a pandas DataFrame with named columns.")
        X = X.copy()
        for col in self.columns_:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_)


def build_pipeline(estimator, scale: bool = False) -> Pipeline:
    """Create  zero->NaN  ->  median imputation  ->  (optional scaling)  ->  model."""
    steps = [
        ("zero_as_missing", ZeroAsMissing()),
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:  # Logistic Regression needs scaled inputs; tree models do not.
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def split_data(df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Stratified train/test split (keeps the diabetes share equal in both parts)."""
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply every pipeline step except the final model -> DataFrame with names."""
    transformed = pipeline[:-1].transform(X)
    return pd.DataFrame(transformed, columns=FEATURES, index=X.index)


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count real NaNs and 'hidden' zero-as-missing values per feature."""
    rows = []
    for col in FEATURES:
        nan_count = int(df[col].isna().sum())
        zero_count = int((df[col] == 0).sum()) if col in ZERO_AS_MISSING else 0
        total = nan_count + zero_count
        rows.append({
            "Feature": col,
            "NaN values": nan_count,
            "Zeros treated as missing": zero_count,
            "Total missing": total,
            "Missing %": round(100 * total / len(df), 2),
        })
    return pd.DataFrame(rows)
