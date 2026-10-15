# Models folder

Created by `python -m src.train_models`:

| File | Content |
|---|---|
| `results.joblib` | All fitted pipelines + test data + probabilities + CV results (used by the app) |
| `logistic_regression.joblib`, `decision_tree.joblib`, `random_forest.joblib`, `xgboost.joblib` | Individual fitted pipelines |
| `model_comparison.csv` | Test-set metrics |
| `cv_results.csv` | Per-fold cross-validation scores |
| `training_meta.json` | Seed, split sizes, library versions, tuned hyper-parameters |

The files are small (a few MB) and are committed so the app starts instantly. If they are missing, or were saved with an
incompatible library version, the app retrains automatically on first start (about one minute).
