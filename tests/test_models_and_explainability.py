import numpy as np
import pytest

from src import explainability as xai
from src.config import MODEL_NAMES
from src.evaluate_models import age_group_performance, compute_metrics, threshold_table
from src.train_models import load_results, save_results
from src import train_models


def test_all_models_trained_and_scored(quick_results):
    assert set(quick_results["models"]) == set(MODEL_NAMES)
    m = quick_results["metrics"]
    assert ((m["ROC-AUC"] >= 0) & (m["ROC-AUC"] <= 1)).all()


def test_test_set_is_disjoint_from_training(quick_results):
    assert set(quick_results["X_train"].index).isdisjoint(quick_results["X_test"].index)


def test_metrics_perfect_and_worst_cases():
    y = np.array([0, 0, 1, 1])
    perfect = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["Accuracy"] == 1 and perfect["Recall"] == 1 and perfect["ROC-AUC"] == 1
    worst = compute_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))
    assert worst["ROC-AUC"] == 0


def test_lower_threshold_never_reduces_recall(quick_results):
    tt = threshold_table(quick_results["y_test"].values, quick_results["probas"]["Random Forest"])
    assert tt["Recall"].is_monotonic_decreasing


def test_age_group_performance_flags_small_groups(quick_results):
    out = age_group_performance(quick_results["X_test"], quick_results["y_test"],
                                quick_results["probas"]["XGBoost"], min_group=1000)
    assert (out["Reliable?"] == "too few cases").all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_shap_values_add_up_to_model_output(quick_results, name):
    """SHAP additivity: base value + sum of contributions == model output."""
    X = quick_results["X_test"].head(20)
    values, base, _ = xai.compute_shap(name, quick_results["models"][name], quick_results["X_train"], X)
    assert values.shape == (20, 8)
    proba = quick_results["models"][name].predict_proba(X)[:, 1]
    raw = values.sum(axis=1) + base
    recon = 1 / (1 + np.exp(-raw)) if xai.UNITS[name] == "log-odds" else raw
    assert np.allclose(recon, proba, atol=1e-4)


def test_save_and_load_roundtrip(quick_results, tmp_path, monkeypatch):
    monkeypatch.setattr(train_models, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train_models, "RESULTS_FILE", tmp_path / "results.joblib")
    save_results(quick_results)
    loaded = load_results()
    assert set(loaded["models"]) == set(MODEL_NAMES)


def test_loading_missing_models_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(train_models, "RESULTS_FILE", tmp_path / "nope.joblib")
    with pytest.raises(FileNotFoundError):
        load_results()
