import io

import pandas as pd
import pytest

from src.config import ALL_COLUMNS
from src.data_loader import (
    DatasetError, load_dataset, load_dataset_with_report, load_default_dataset, validate_dataframe,
)


def test_default_dataset_loads_with_expected_shape():
    df = load_default_dataset()
    assert list(df.columns) == ALL_COLUMNS
    assert len(df) == 768
    assert set(df["Outcome"].unique()) == {0, 1}


def test_valid_dataframe_passes(small_df):
    result = validate_dataframe(small_df)
    assert result.ok and result.errors == []


def test_missing_column_is_rejected(small_df):
    result = validate_dataframe(small_df.drop(columns=["Glucose"]))
    assert not result.ok
    assert "Glucose" in result.errors[0]


def test_column_names_are_matched_loosely(small_df):
    renamed = small_df.rename(columns={"BloodPressure": "blood pressure", "Outcome": "outcome"})
    assert validate_dataframe(renamed).ok


def test_non_binary_target_is_rejected(small_df):
    bad = small_df.copy()
    bad.loc[0, "Outcome"] = 5
    assert not validate_dataframe(bad).ok


def test_single_class_is_rejected(small_df):
    bad = small_df.copy()
    bad["Outcome"] = 0
    assert not validate_dataframe(bad).ok


def test_too_few_rows_rejected(small_df):
    assert not validate_dataframe(small_df.head(30)).ok


def test_negative_values_rejected(small_df):
    bad = small_df.copy()
    bad.loc[3, "BMI"] = -4
    assert not validate_dataframe(bad).ok


def test_headerless_csv_is_accepted(small_df):
    text = small_df.to_csv(index=False, header=False)
    result = load_dataset_with_report(io.BytesIO(text.encode()))
    assert result.ok and list(result.df.columns) == ALL_COLUMNS


def test_empty_upload_reports_error_instead_of_crashing():
    result = load_dataset_with_report(io.BytesIO(b""))
    assert not result.ok and result.errors


def test_missing_file_raises_friendly_error(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "does_not_exist.csv")


def test_text_garbage_is_reported_not_crashed():
    result = load_dataset_with_report(io.BytesIO(b"hello,world\n1,2\n"))
    assert not result.ok
