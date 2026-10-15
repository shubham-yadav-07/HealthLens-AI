"""
Data loading and validation.

Responsibilities
----------------
* load the default Pima Indians Diabetes dataset (or download it),
* load a user-uploaded CSV (with or without a header row),
* validate that all required columns exist and the values make sense.

Nothing in this file depends on Streamlit, so it is easy to unit-test.
"""
from __future__ import annotations

import io
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

from src.config import (
    ALL_COLUMNS, DATA_DIR, DATASET_URL, DEFAULT_DATASET, FEATURES, TARGET,
)

Source = Union[str, Path, BinaryIO]


class DatasetError(Exception):
    """Raised when a dataset cannot be loaded or is not compatible."""


@dataclass
class ValidationResult:
    """Outcome of validate_dataframe()."""
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    df: pd.DataFrame | None = None


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _canon(name: str) -> str:
    """'Blood Pressure', 'blood_pressure' and 'BloodPressure' all -> 'bloodpressure'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_CANON_TO_STANDARD = {_canon(c): c for c in ALL_COLUMNS}
# a few common aliases found in public copies of the dataset
_CANON_TO_STANDARD.update({
    "diabetespedigree": "DiabetesPedigreeFunction",
    "dpf": "DiabetesPedigreeFunction",
    "class": TARGET,
    "diabetes": TARGET,
    "target": TARGET,
    "label": TARGET,
})


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the standard names when they match case-insensitively."""
    mapping = {c: _CANON_TO_STANDARD[_canon(c)] for c in df.columns if _canon(c) in _CANON_TO_STANDARD}
    return df.rename(columns=mapping)


def _looks_headerless(df: pd.DataFrame) -> bool:
    """True when the 'header' row is really numeric data (9 columns of numbers)."""
    if df.shape[1] != len(ALL_COLUMNS):
        return False
    try:
        [float(c) for c in df.columns]
        return True
    except (TypeError, ValueError):
        return False


def _read_csv(source: Source) -> pd.DataFrame:
    """Read a CSV from a path or file-like object, handling header-less files."""
    try:
        if hasattr(source, "read"):
            raw = source.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            if not raw.strip():
                raise DatasetError("The uploaded file is empty.")
            buffer = io.BytesIO(raw)
            df = pd.read_csv(buffer)
            if _looks_headerless(df):
                df = pd.read_csv(io.BytesIO(raw), header=None, names=ALL_COLUMNS)
        else:
            path = Path(source)
            if not path.exists():
                raise DatasetError(f"File not found: {path}")
            df = pd.read_csv(path)
            if _looks_headerless(df):
                df = pd.read_csv(path, header=None, names=ALL_COLUMNS)
    except DatasetError:
        raise
    except pd.errors.EmptyDataError as exc:
        raise DatasetError("The file is empty or is not a valid CSV.") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Could not parse the file as CSV: {exc}") from exc
    return df


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------
def validate_dataframe(df: pd.DataFrame) -> ValidationResult:
    """Check that a dataframe is compatible with HealthLens AI.

    Returns a ValidationResult. ``ok`` is False when there is at least one
    blocking error. Warnings are informative only.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if df is None or df.empty:
        return ValidationResult(False, ["The dataset has no rows."], [], None)

    df = standardise_columns(df.copy())

    missing_cols = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(
            "Missing required column(s): " + ", ".join(missing_cols)
            + ". Required columns: " + ", ".join(ALL_COLUMNS) + "."
        )
        return ValidationResult(False, errors, warnings, None)

    extra = [c for c in df.columns if c not in ALL_COLUMNS]
    if extra:
        warnings.append("Ignoring extra column(s): " + ", ".join(map(str, extra)))
    df = df[ALL_COLUMNS].copy()

    # every column must be numeric (strings such as '?' are coerced to NaN and reported)
    for col in ALL_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            bad = int(coerced.isna().sum() - df[col].isna().sum())
            if bad > 0:
                warnings.append(f"Column '{col}': {bad} non-numeric value(s) were converted to missing.")
            df[col] = coerced

    if df[TARGET].isna().any():
        n = int(df[TARGET].isna().sum())
        warnings.append(f"Dropped {n} row(s) with a missing '{TARGET}' value.")
        df = df.dropna(subset=[TARGET])

    if df.empty:
        return ValidationResult(False, ["No usable rows remain after cleaning."], warnings, None)

    labels = set(df[TARGET].unique())
    if not labels.issubset({0, 1}):
        errors.append(f"'{TARGET}' must contain only 0 and 1, but found: {sorted(labels)[:8]}.")
    else:
        df[TARGET] = df[TARGET].astype(int)
        counts = df[TARGET].value_counts()
        if len(counts) < 2:
            errors.append(f"'{TARGET}' contains only one class; both 0 and 1 are needed.")
        elif counts.min() < 20:
            errors.append(
                f"The minority class has only {int(counts.min())} rows; at least 20 are needed "
                "for a stratified split and cross-validation."
            )

    if len(df) < 100:
        errors.append(f"Only {len(df)} rows found; at least 100 are required.")

    negatives = [c for c in FEATURES if (df[c] < 0).any()]
    if negatives:
        errors.append("Negative values found in: " + ", ".join(negatives) + ".")

    if errors:
        return ValidationResult(False, errors, warnings, None)

    all_nan = [c for c in FEATURES if df[c].isna().all()]
    if all_nan:
        return ValidationResult(False, ["Column(s) contain no values: " + ", ".join(all_nan)], warnings, None)

    return ValidationResult(True, [], warnings, df.reset_index(drop=True))


# ----------------------------------------------------------------------------
# Public loading functions
# ----------------------------------------------------------------------------
def load_dataset(source: Source) -> pd.DataFrame:
    """Load + validate a dataset. Raises DatasetError with a friendly message."""
    df = _read_csv(source)
    result = validate_dataframe(df)
    if not result.ok:
        raise DatasetError(" ".join(result.errors))
    return result.df


def load_dataset_with_report(source: Source) -> ValidationResult:
    """Like load_dataset() but returns the full ValidationResult (never raises
    for incompatible content - used by the upload widget)."""
    try:
        df = _read_csv(source)
    except DatasetError as exc:
        return ValidationResult(False, [str(exc)], [], None)
    return validate_dataframe(df)


def download_default_dataset(destination: Path = DEFAULT_DATASET) -> Path:
    """Download the Pima dataset from a public mirror and save it WITH a header."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(DATASET_URL, timeout=30) as response:
            raw = response.read()
    except Exception as exc:  # network errors are many and varied
        raise DatasetError(
            f"Could not download the dataset ({exc}). Download it manually from "
            f"{DATASET_URL} (or Kaggle: 'Pima Indians Diabetes Database') and save it as "
            f"{destination.name} inside the data/ folder."
        ) from exc
    df = pd.read_csv(io.BytesIO(raw), header=None, names=ALL_COLUMNS)
    result = validate_dataframe(df)
    if not result.ok:
        raise DatasetError("Downloaded file failed validation: " + " ".join(result.errors))
    result.df.to_csv(destination, index=False)
    return destination


def load_default_dataset() -> pd.DataFrame:
    """Load the bundled Pima dataset, downloading it first if it is missing."""
    if not DEFAULT_DATASET.exists():
        download_default_dataset()
    return load_dataset(DEFAULT_DATASET)


if __name__ == "__main__":
    # `python -m src.data_loader` downloads the dataset into data/
    path = download_default_dataset()
    print(f"Saved dataset to {path}")
    print(load_dataset(path).describe().T)
