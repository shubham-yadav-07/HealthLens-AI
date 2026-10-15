"""
Model training.

Run from the project root:

    python -m src.train_models              # full run: tuning + nested CV (~2-4 min)
    python -m src.train_models --fast       # quick run: no nested CV

What happens
------------
1. load + validate the dataset
2. stratified 80/20 train/test split (random seed fixed -> reproducible)
3. for each of the 4 algorithms:
      * tune hyper-parameters with 5-fold stratified cross-validation (ROC-AUC),
        using ONLY the training set
      * (optional) nested cross-validation for an honest performance estimate
      * refit the best pipeline on the whole training set
4. evaluate on the untouched test set
5. save models + results into models/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS, MODELS_DIR, MODEL_NAMES, RANDOM_STATE, RESULTS_FILE, TARGET, TEST_SIZE,
)
from src.data_loader import load_default_dataset
from src.evaluate_models import compute_metrics
from src.preprocessing import build_pipeline, split_data

CV_SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def get_model_specs() -> dict:
    """The four algorithms, their (small) hyper-parameter grids and scaling need."""
    return {
        "Logistic Regression": dict(
            estimator=LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
            grid={"model__C": [0.01, 0.1, 1, 10, 100]},
            scale=True,
        ),
        "Decision Tree": dict(
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            grid={"model__max_depth": [2, 3, 4, 5, 6, 8],
                  "model__min_samples_leaf": [1, 5, 10, 20]},
            scale=False,
        ),
        "Random Forest": dict(
            estimator=RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=1),
            grid={"model__max_depth": [4, 6, 8, None],
                  "model__min_samples_leaf": [1, 3, 5]},
            scale=False,
        ),
        "XGBoost": dict(
            estimator=XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE,
                                    n_jobs=1, tree_method="hist"),
            grid={"model__n_estimators": [100, 200],
                  "model__max_depth": [2, 3, 4],
                  "model__learning_rate": [0.03, 0.1],
                  "model__subsample": [0.8, 1.0]},
            scale=False,
        ),
    }


def _dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def train_all(df: pd.DataFrame, tune: bool = True, nested_cv: bool = False,
              save: bool = False, progress=None) -> dict:
    """Train all models and return a results dictionary.

    Parameters
    ----------
    df         validated dataframe (see data_loader)
    tune       run GridSearchCV; if False use sensible default hyper-parameters
    nested_cv  also run nested CV (outer folds re-run the whole tuning) - slower
    save       write models/results to the models/ folder
    progress   optional callable(message, fraction) for UI progress bars
    """
    X_train, X_test, y_train, y_test = split_data(df)
    outer_cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    specs = get_model_specs()

    models, probas, best_params, cv_tables = {}, {}, {}, {}
    metrics_rows = []
    started = time.time()

    for i, name in enumerate(MODEL_NAMES):
        if progress:
            progress(f"Training {name} ...", i / len(MODEL_NAMES))
        spec = specs[name]
        pipe = build_pipeline(spec["estimator"], scale=spec["scale"])

        if tune:
            search = GridSearchCV(pipe, spec["grid"], scoring="roc_auc", cv=outer_cv,
                                  n_jobs=-1, refit=True)
            search.fit(X_train, y_train)
            best = search.best_estimator_
            best_params[name] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
            tuning_cv_auc = float(search.best_score_)
            cv_kind = "Nested CV (tuning repeated inside every fold)" if nested_cv else \
                      "CV of the best configuration (slightly optimistic)"
            if nested_cv:
                scores = cross_validate(
                    GridSearchCV(pipe, spec["grid"], scoring="roc_auc", cv=StratifiedKFold(
                        CV_FOLDS, shuffle=True, random_state=RANDOM_STATE + 1), n_jobs=1),
                    X_train, y_train, cv=outer_cv, scoring=CV_SCORING, n_jobs=-1)
            else:
                scores = cross_validate(best, X_train, y_train, cv=outer_cv, scoring=CV_SCORING)
        else:
            best = pipe.fit(X_train, y_train)
            best_params[name] = {"note": "default hyper-parameters (tuning skipped)"}
            tuning_cv_auc = float("nan")
            cv_kind = "CV with default hyper-parameters"
            scores = cross_validate(pipe, X_train, y_train, cv=outer_cv, scoring=CV_SCORING)

        cv_df = pd.DataFrame({m: scores[f"test_{m}"] for m in CV_SCORING})
        cv_df.insert(0, "Fold", np.arange(1, len(cv_df) + 1))
        cv_df.insert(0, "Model", name)
        cv_tables[name] = cv_df

        proba = best.predict_proba(X_test)[:, 1]
        models[name] = best
        probas[name] = proba

        row = compute_metrics(y_test.values, proba, threshold=0.5, with_ci=True)
        row.update({
            "Model": name,
            "CV ROC-AUC (mean)": float(cv_df["roc_auc"].mean()),
            "CV ROC-AUC (std)": float(cv_df["roc_auc"].std(ddof=1)),
            "Tuning CV ROC-AUC": tuning_cv_auc,
        })
        metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows).set_index("Model")
    cv_results = pd.concat(cv_tables.values(), ignore_index=True)

    meta = {
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - started, 1),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "cv_type": cv_kind,
        "tuned": tune,
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_all": float(df[TARGET].mean()),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "dataset_hash": _dataset_hash(df),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    results = {
        "models": models,
        "metrics": metrics,
        "cv_results": cv_results,
        "best_params": best_params,
        "probas": probas,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "meta": meta,
    }
    if progress:
        progress("Done", 1.0)
    if save:
        save_results(results)
    return results


def save_results(results: dict) -> None:
    """Persist everything to the models/ folder (one joblib bundle + readable CSV/JSON)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(results, RESULTS_FILE, compress=3)
    for name, model in results["models"].items():
        joblib.dump(model, MODELS_DIR / f"{name.lower().replace(' ', '_')}.joblib", compress=3)
    results["metrics"].round(4).to_csv(MODELS_DIR / "model_comparison.csv")
    results["cv_results"].round(4).to_csv(MODELS_DIR / "cv_results.csv", index=False)
    with open(MODELS_DIR / "training_meta.json", "w", encoding="utf-8") as fh:
        json.dump({"meta": results["meta"], "best_params": results["best_params"]},
                  fh, indent=2, default=str)


def load_results() -> dict:
    """Load the saved bundle. Raises FileNotFoundError / other errors if unusable."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            "No trained models found. Run `python -m src.train_models` first.")
    return joblib.load(RESULTS_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HealthLens AI models")
    parser.add_argument("--fast", action="store_true", help="skip nested CV (quicker)")
    parser.add_argument("--no-tune", action="store_true", help="skip hyper-parameter tuning")
    args = parser.parse_args()

    df = load_default_dataset()
    print(f"Dataset: {len(df)} rows. Training ...")
    results = train_all(
        df, tune=not args.no_tune, nested_cv=not (args.fast or args.no_tune), save=True,
        progress=lambda msg, frac: print(f"[{int(frac * 100):3d}%] {msg}"),
    )
    cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV ROC-AUC (mean)"]
    print("\nTest-set results (threshold 0.5):")
    print(results["metrics"][cols].round(3).to_string())
    print(f"\nSaved to {MODELS_DIR}  (took {results['meta']['seconds']} s)")


if __name__ == "__main__":
    main()
