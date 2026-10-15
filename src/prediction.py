"""
Prediction logic: input validation, probability estimation, reliability checks.

No Streamlit imports -> easy to unit-test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import FEATURES, FEATURE_INFO, MODEL_NAMES, ZERO_AS_MISSING


class InputValidationError(ValueError):
    """Raised when user-supplied feature values are invalid."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def dataset_ranges(df: pd.DataFrame) -> dict:
    """Observed min / 5th / 95th percentile / max per feature (zeros ignored where they mean missing)."""
    ranges = {}
    for col in FEATURES:
        s = df[col].replace(0, np.nan) if col in ZERO_AS_MISSING else df[col]
        s = s.dropna()
        ranges[col] = {
            "min": float(s.min()), "max": float(s.max()),
            "p05": float(s.quantile(0.05)), "p95": float(s.quantile(0.95)),
            "median": float(s.median()),
        }
    return ranges


def validate_inputs(values: dict, ranges: dict | None = None) -> tuple[dict, list[str], list[str]]:
    """Validate a dict {feature: value}.

    A value of ``None`` / NaN means "not measured" and is allowed only for the
    features where zero marks missing data in the Pima dataset.

    Returns (clean_values, errors, warnings). If ``errors`` is non-empty the
    values must not be used for prediction.
    """
    errors: list[str] = []
    warnings: list[str] = []
    clean: dict = {}

    unknown = [k for k in values if k not in FEATURES]
    if unknown:
        errors.append("Unknown feature(s): " + ", ".join(map(str, unknown)))
    missing_keys = [f for f in FEATURES if f not in values]
    if missing_keys:
        errors.append("Missing required feature(s): " + ", ".join(missing_keys))
    if errors:
        return {}, errors, warnings

    for feat in FEATURES:
        info = FEATURE_INFO[feat]
        v = values[feat]
        is_blank = v is None or (isinstance(v, float) and np.isnan(v))
        if is_blank:
            if feat in ZERO_AS_MISSING:
                clean[feat] = np.nan
                warnings.append(f"{info['label']} was not provided; a typical value from the training data will be used.")
                continue
            errors.append(f"{info['label']} is required.")
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errors.append(f"{info['label']} must be a number.")
            continue
        if not np.isfinite(v):
            errors.append(f"{info['label']} must be a finite number.")
            continue
        if v < info["hard_min"] or v > info["hard_max"]:
            errors.append(
                f"{info['label']} = {v:g} {info['unit']} is outside the plausible range "
                f"({info['hard_min']:g} to {info['hard_max']:g} {info['unit']})."
            )
            continue
        clean[feat] = v
        if ranges and feat in ranges:
            r = ranges[feat]
            if v < r["min"] or v > r["max"]:
                warnings.append(
                    f"{info['label']} = {v:g} is outside the range seen in the training data "
                    f"({r['min']:g} to {r['max']:g}); the model is extrapolating, so the estimate is less reliable."
                )
    return clean, errors, warnings


def to_frame(clean_values: dict) -> pd.DataFrame:
    """One-row DataFrame in the exact column order the models expect."""
    return pd.DataFrame([[clean_values[f] for f in FEATURES]], columns=FEATURES)


def predict_all(models: dict, clean_values: dict) -> dict:
    """Probability of the 'diabetes recorded' class from every model."""
    X = to_frame(clean_values)
    out = {}
    for name in MODEL_NAMES:
        if name in models:
            out[name] = float(models[name].predict_proba(X)[0, 1])
    return out


def risk_band(p: float) -> tuple[str, str]:
    """Descriptive band for a model probability. These cut-offs are arbitrary
    presentation choices, NOT clinical thresholds."""
    if p < 0.30:
        return "Lower model-estimated risk", "low"
    if p < 0.60:
        return "Moderate model-estimated risk", "mid"
    return "Higher model-estimated risk", "high"


def reliability_notes(probas: dict, warnings: list[str], n_unmeasured: int) -> list[str]:
    """Reasons why this particular estimate should be treated with extra caution."""
    notes = []
    if probas:
        spread = max(probas.values()) - min(probas.values())
        if spread >= 0.25:
            notes.append(
                f"The four models disagree noticeably (estimates range from {min(probas.values()):.0%} "
                f"to {max(probas.values()):.0%}), which signals real uncertainty."
            )
    if n_unmeasured >= 3:
        notes.append(f"{n_unmeasured} measurements were left blank and replaced by typical values, "
                     "so this estimate rests mostly on the remaining inputs.")
    elif n_unmeasured > 0:
        notes.append(f"{n_unmeasured} measurement(s) were left blank and replaced by typical values.")
    if any("outside the range seen" in w for w in warnings):
        notes.append("At least one input lies outside the range of the training data.")
    return notes


def percentile_of(df: pd.DataFrame, feature: str, value: float) -> float:
    """Where a value sits within the dataset (0-100 percentile)."""
    s = df[feature].replace(0, np.nan) if feature in ZERO_AS_MISSING else df[feature]
    s = s.dropna()
    return float((s < value).mean() * 100)
