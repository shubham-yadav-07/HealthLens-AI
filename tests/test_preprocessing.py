import numpy as np
import pandas as pd

from src.config import FEATURES, ZERO_AS_MISSING
from src.preprocessing import (
    ZeroAsMissing, build_pipeline, missing_summary, split_data, transform_features,
)
from sklearn.linear_model import LogisticRegression


def test_zero_as_missing_converts_only_selected_columns(small_df):
    out = ZeroAsMissing().fit(small_df[FEATURES]).transform(small_df[FEATURES])
    assert out["Insulin"].isna().sum() == (small_df["Insulin"] == 0).sum()
    assert out["Pregnancies"].equals(small_df["Pregnancies"])  # zero is valid here
    assert (small_df["Insulin"] == 0).any()                    # original untouched


def test_pipeline_leaves_no_missing_values(small_df):
    X = small_df[FEATURES]
    pipe = build_pipeline(LogisticRegression(max_iter=1000), scale=True).fit(X, small_df["Outcome"])
    assert not transform_features(pipe, X).isna().any().any()


def test_imputer_learns_from_training_data_only(small_df):
    """Data-leakage check: the median must come from the training rows, not the test rows."""
    X_train, X_test, y_train, _ = split_data(small_df)
    pipe = build_pipeline(LogisticRegression(max_iter=1000), scale=True).fit(X_train, y_train)
    train_median = X_train["Insulin"].replace(0, np.nan).median()
    learned = pipe.named_steps["imputer"].statistics_[FEATURES.index("Insulin")]
    assert np.isclose(learned, train_median)
    all_median = small_df["Insulin"].replace(0, np.nan).median()
    if not np.isclose(train_median, all_median):
        assert not np.isclose(learned, all_median)


def test_split_is_stratified_and_reproducible(small_df):
    a = split_data(small_df)
    b = split_data(small_df)
    assert a[0].index.equals(b[0].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.03
    assert set(a[0].index).isdisjoint(a[1].index)


def test_missing_summary_counts_zeros(small_df):
    ms = missing_summary(small_df).set_index("Feature")
    assert ms.loc["Insulin", "Zeros treated as missing"] == (small_df["Insulin"] == 0).sum()
    assert ms.loc["Pregnancies", "Total missing"] == 0
