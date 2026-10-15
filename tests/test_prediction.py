import numpy as np
import pytest

from src import prediction as pred
from src.config import FEATURES, MODEL_NAMES, ZERO_AS_MISSING

VALID = {"Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 25,
         "Insulin": 90, "BMI": 30.0, "DiabetesPedigreeFunction": 0.4, "Age": 35}


def test_valid_input_has_no_errors(small_df):
    clean, errors, _ = pred.validate_inputs(VALID, pred.dataset_ranges(small_df))
    assert errors == [] and set(clean) == set(FEATURES)


def test_missing_feature_is_an_error():
    bad = dict(VALID)
    bad.pop("Age")
    _, errors, _ = pred.validate_inputs(bad)
    assert errors and "Age" in errors[0]


def test_unknown_feature_is_an_error():
    _, errors, _ = pred.validate_inputs({**VALID, "Height": 170})
    assert errors


def test_out_of_range_value_is_an_error():
    _, errors, _ = pred.validate_inputs({**VALID, "Glucose": 900})
    assert any("Glucose" in e or "glucose" in e for e in errors)


def test_non_numeric_value_is_an_error():
    _, errors, _ = pred.validate_inputs({**VALID, "BMI": "abc"})
    assert errors


def test_nan_is_not_allowed_for_required_fields():
    _, errors, _ = pred.validate_inputs({**VALID, "Age": float("nan")})
    assert errors


def test_unmeasured_lab_value_is_allowed_with_warning():
    clean, errors, warns = pred.validate_inputs({**VALID, "Insulin": None})
    assert errors == [] and np.isnan(clean["Insulin"]) and warns


def test_outside_training_range_warns(small_df):
    ranges = pred.dataset_ranges(small_df)
    _, errors, warns = pred.validate_inputs({**VALID, "Age": 95}, ranges)
    assert errors == [] and any("outside the range" in w for w in warns)


def test_prediction_returns_probabilities_for_all_models(quick_results):
    clean, errors, _ = pred.validate_inputs(VALID)
    probs = pred.predict_all(quick_results["models"], clean)
    assert set(probs) == set(MODEL_NAMES)
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_prediction_works_with_unmeasured_values(quick_results):
    clean, errors, _ = pred.validate_inputs({**VALID, "Insulin": None, "SkinThickness": None})
    assert errors == []
    probs = pred.predict_all(quick_results["models"], clean)
    assert all(np.isfinite(p) for p in probs.values())


def test_higher_glucose_does_not_lower_risk_for_logistic_regression(quick_results):
    lo, _, _ = pred.validate_inputs({**VALID, "Glucose": 80})
    hi, _, _ = pred.validate_inputs({**VALID, "Glucose": 180})
    m = {"Logistic Regression": quick_results["models"]["Logistic Regression"]}
    assert pred.predict_all(m, hi)["Logistic Regression"] >= pred.predict_all(m, lo)["Logistic Regression"]


def test_risk_band_boundaries():
    assert pred.risk_band(0.1)[1] == "low"
    assert pred.risk_band(0.45)[1] == "mid"
    assert pred.risk_band(0.8)[1] == "high"


def test_reliability_notes_flag_disagreement():
    notes = pred.reliability_notes({"a": 0.1, "b": 0.7}, [], 0)
    assert any("disagree" in n for n in notes)
