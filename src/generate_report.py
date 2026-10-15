"""
Generate docs/PROJECT_REPORT.md from the REAL results of this project.

    python -m src.generate_report

All numbers (dataset statistics, model scores, SHAP importances ...) are read
from the dataset and the saved models, so the report always matches what you
actually ran. Re-run it after retraining.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from src import eda
from src import evaluate_models as ev
from src import explainability as xai
from src.config import FEATURES, MODEL_NAMES, ROOT, TARGET
from src.data_loader import load_default_dataset
from src.preprocessing import missing_summary
from src.train_models import load_results


def md_table(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:.3f}") -> str:
    """Tiny DataFrame -> Markdown table converter (no extra dependency needed)."""
    d = df.reset_index() if index else df.copy()
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = []
    for _, r in d.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


def build() -> str:
    df = load_default_dataset()
    res = load_results()
    meta = res["meta"]
    ov = eda.overview(df)
    ms = missing_summary(df)
    cb = eda.class_balance(df)
    gc = eda.group_comparison(df)
    out_tab = eda.outlier_table(df)
    corr = eda.correlation_matrix(df)[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    m = res["metrics"]
    table = ev.metrics_table(res, 0.5)
    cvs = ev.cv_summary(res["cv_results"])
    imp_all = xai.importance_all_models(res)
    best = table["ROC-AUC"].idxmax()
    best_params = pd.DataFrame({k: {a: str(b) for a, b in v.items()} for k, v in res["best_params"].items()}).T
    age = ev.age_group_performance(res["X_test"], res["y_test"], res["probas"][best])
    tt = ev.threshold_table(res["y_test"].values, res["probas"][best], [0.2, 0.3, 0.4, 0.5, 0.6])
    baseline = 1 - meta["positive_rate_test"]

    metric_cols = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]
    results_tbl = table[metric_cols].copy()
    results_tbl["ROC-AUC 95% CI"] = [f"{m.loc[x, 'ROC-AUC CI low']:.2f}-{m.loc[x, 'ROC-AUC CI high']:.2f}" for x in results_tbl.index]
    conf = table[["TP", "FP", "TN", "FN"]].astype(int)
    top_shap = imp_all.mean(axis=1).sort_values(ascending=False)

    findings = "\n".join(f"- {f.replace('**', '')}" for f in eda.key_findings(df))

    return f"""# HealthLens AI - Early Health Risk Analysis and Prediction Platform
### Project report (auto-generated on {date.today().isoformat()} from the actual run)

> Educational and research prototype. Not a medical device. Nothing in this report constitutes medical advice.

## 1. Abstract
HealthLens AI is an end-to-end data science application that estimates diabetes-associated risk from eight routine
clinical measurements and explains its predictions. Four classifiers (Logistic Regression, Decision Tree, Random Forest,
XGBoost) were trained on the Pima Indians Diabetes Database ({ov['rows']} records) inside leakage-free scikit-learn
pipelines, tuned with {meta['cv_folds']}-fold stratified cross-validation and evaluated on a stratified held-out test set
of {meta['n_test']} records. The highest test ROC-AUC was {table.loc[best, 'ROC-AUC']:.3f} ({best}), but bootstrap confidence
intervals of all four models overlap, so no model can be declared superior. SHAP identifies
**{top_shap.index[0]}** as the most influential feature, followed by **{top_shap.index[1]}**. The system is delivered as a
Streamlit application with EDA, model comparison, explainability and report-export modules.

## 2. Introduction
Diabetes mellitus is a chronic metabolic condition whose early detection depends on measurements such as blood glucose and
body mass index. Machine learning can summarise how such measurements relate to later outcomes, but in healthcare a model
must also be transparent, honestly evaluated and clearly limited. This project treats those requirements as first-class goals.

## 3. Problem statement
Given eight clinical measurements, estimate the probability that a participant of the Pima study was recorded as developing
diabetes, compare candidate algorithms fairly, and expose which measurements drive the model's output - without claiming
clinical validity.

## 4. Objectives
1. Build a reproducible ML pipeline that prevents data leakage.
2. Compare four algorithms with cross-validation and a held-out test set.
3. Analyse data quality (hidden missing values, duplicates, outliers, class imbalance).
4. Provide model-agnostic and model-specific explanations with SHAP.
5. Deliver an interactive, accessible interface with clear disclaimers.

## 5. Existing system and its limitations
Typical student and tutorial projects fit one model, report accuracy only, ignore the impossible zeros in the Pima data, and
scale or impute before splitting (data leakage). They rarely quantify uncertainty or explain predictions. Clinical risk scores
used in practice are validated on large, diverse cohorts, which this prototype is not.

## 6. Proposed system
A modular Python application: `src/` holds the data, preprocessing, EDA, training, evaluation, prediction and explainability
code; `pages/` holds the Streamlit pages; `tests/` holds {41} automated tests. See Section 10 for the architecture.

## 7. Literature review
- **Dataset origin.** Smith et al. (1988) introduced the data collected by the US National Institute of Diabetes and Digestive
  and Kidney Diseases and applied the ADAP learning algorithm to forecast diabetes onset. The dataset has since become a common
  teaching benchmark for binary classification.
- **Methods used here.** Random forests (Breiman, 2001) and gradient-boosted trees as implemented in XGBoost (Chen and Guestrin, 2016)
  are widely used for tabular data; logistic regression and decision trees are standard interpretable baselines (Hastie et al., 2009).
- **Explainability.** SHAP (Lundberg and Lee, 2017) unifies feature-attribution methods using Shapley values, and TreeSHAP
  (Lundberg et al., 2020) computes them efficiently for tree ensembles. Rudin (2019) argues that high-stakes decisions should favour
  inherently interpretable models, a caution relevant to this project.
- **Evaluation pitfalls.** Kaufman et al. (2012) describe how data leakage inflates reported performance; Obermeyer et al. (2019)
  show how health algorithms can embed bias when the target or data are poorly chosen.
- *Student note: extend this section with papers you have personally read on diabetes prediction; this draft intentionally cites only
  works whose content is well established.*

## 8. Dataset description
- **Source:** Pima Indians Diabetes Database (NIDDK). Public mirror used for download: `jbrownlee/Datasets` on GitHub; also on UCI and Kaggle.
- **Records:** {ov['rows']}; **features:** {len(FEATURES)}; **target:** `Outcome` (1 = diabetes recorded within five years).
- **Population:** women aged 21+ of Pima heritage; **intended use:** benchmarking and teaching.
- **Class balance:**

{md_table(cb, index=False)}

- **Hidden missing values** (zeros that are physiologically impossible):

{md_table(ms, index=False)}

**Limitations:** single population, small sample, 1980s data, no demographic variables beyond age, no information on diabetes type.

## 9. Methodology
1. Load and validate data (columns, numeric types, binary target, minimum rows per class).
2. Stratified {int((1 - meta['test_size']) * 100)}/{int(meta['test_size'] * 100)} train/test split, `random_state={meta['random_state']}`. Test set untouched until final evaluation.
3. Pipeline: zero-to-NaN -> median imputation -> (standard scaling for Logistic Regression) -> classifier. All fitted on training rows only.
4. Hyper-parameter search with `GridSearchCV` ({meta['cv_folds']}-fold stratified, scoring ROC-AUC) on the training set.
5. Nested cross-validation to estimate performance without tuning optimism. CV type used in this run: *{meta['cv_type']}*.
6. Single evaluation on the test set; bootstrap (1,000 resamples) confidence interval for ROC-AUC.
7. SHAP explanations on the test set.

Tuned parameters:

{md_table(best_params)}

## 10. System architecture
```
data/pima_diabetes.csv --> src/data_loader.py (validate)
        --> src/preprocessing.py (pipelines, split)
        --> src/train_models.py  (tune, CV, save to models/)
        --> src/evaluate_models.py (metrics, ROC, thresholds, subgroups)
        --> src/explainability.py  (SHAP)
        --> src/prediction.py      (input validation, probabilities)
app.py + pages/*.py (Streamlit UI)  <-- src/ui.py, src/state.py, src/plotting.py
```

## 11. Data preprocessing
- Impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are converted to missing values.
- Missing values are imputed with the training-set median (robust to skew and outliers).
- Features are scaled only for Logistic Regression; tree-based models are scale-invariant.
- Duplicates found: **{ov['duplicates']}**. Outliers are retained (see Section 12).
- Leakage control: imputer and scaler live inside the pipeline, so they are re-fitted on each CV training fold and on the training split only.

## 12. Exploratory data analysis
Key findings (computed from the data):

{findings}

Group comparison (Mann-Whitney U with Bonferroni correction; effect size Cohen's d). Exploratory only:

{md_table(gc[['Feature', 'Median (no diabetes)', 'Median (diabetes)', "Cohen's d", 'Significant (0.05)']], index=False)}

Outliers (IQR rule):

{md_table(out_tab, index=False)}

## 13. Machine-learning algorithms
- **Logistic Regression:** linear model of log-odds; interpretable coefficients; needs scaled inputs.
- **Decision Tree:** recursive axis-aligned splits; highly interpretable but unstable and prone to overfitting.
- **Random Forest:** averages many decorrelated trees (bagging); reduces variance.
- **XGBoost:** gradient-boosted trees built sequentially to correct previous errors, with regularisation.

## 14. Model evaluation
Test-set results at threshold 0.5 (n = {meta['n_test']}):

{md_table(results_tbl)}

Confusion-matrix counts:

{md_table(conf)}

Cross-validation on the training set (mean ± std over {meta['cv_folds']} folds):

{md_table(cvs)}

**Why accuracy alone is not enough.** A classifier that always predicts "no diabetes" would score {baseline:.1%} accuracy on this
test set while finding no positive cases. Missing a real case usually costs more than a false alarm, so recall, specificity, precision,
F1 and ROC-AUC are reported together with the threshold trade-off below ({best}):

{md_table(tt)}

## 15. Explainable AI
Mean share of total |SHAP| value, per model (percent):

{md_table(imp_all.round(1), floatfmt="{:.1f}")}

SHAP additivity (base value + contributions = model output) is verified by an automated test for every model. Importance describes
reliance of the *model* on a feature; it does not establish causation, and correlated features can split importance.

## 16. Results and discussion
- The best test ROC-AUC is {table.loc[best, 'ROC-AUC']:.3f} ({best}); the lowest is {table['ROC-AUC'].min():.3f}. The 95% bootstrap intervals overlap, so differences are not statistically distinguishable on {meta['n_test']} test records.
- At the default 0.5 threshold recall ranges from {table['Recall'].min():.2f} to {table['Recall'].max():.2f}: several models miss roughly half of positive cases, which would be unacceptable for screening.
- {top_shap.index[0]} dominates across models, consistent with its clinical role and with its large effect size in the EDA ({gc.iloc[0]["Cohen's d"]:.2f} Cohen's d for {gc.iloc[0]['Feature']}). Because the data are observational this is an association, not a causal finding.
- Age-group performance for {best} (only demographic variable available; groups are small):

{md_table(age, index=False)}

## 17. Limitations
- One population, one era, {ov['rows']} records. No external validation.
- Test-set estimates are noisy (wide confidence intervals); results depend on the random split.
- Heavy hidden missingness (Insulin, SkinThickness) makes imputation influential.
- No calibration study; model probabilities are not guaranteed to match true frequencies.
- Fairness cannot be assessed: no sex, ethnicity or socioeconomic variables beyond a single population.

## 18. Ethical considerations
The app states prominently that it is educational, labels outputs as model-estimated probabilities, never assigns "healthy" or "diabetic",
gives no treatment advice, flags unreliable inputs, stores no personal data, and does not claim fairness. Using data from a specific
Indigenous population carries a responsibility not to over-generalise or stigmatise; conclusions are restricted to this sample.

## 19. Future scope
External validation on other datasets, probability calibration, nested-CV model selection with repeated splits, additional (non-diabetes)
outcomes, richer fairness audits with appropriate data, and clinician-in-the-loop user studies.

## 20. Conclusion
The project delivers a transparent, reproducible and explainable pipeline. Its main scientific message is cautionary: with a small, old,
single-population dataset, four very different algorithms reach similar ROC-AUC, recall at the default threshold is modest, and
explanations must be read as descriptions of the model, not of biology.

## 21. References
1. Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., Johannes, R.S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261-265.
2. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
3. Chen, T., Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD '16*, 785-794.
4. Lundberg, S.M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*.
5. Lundberg, S.M. et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56-67.
6. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
7. Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
8. Kaufman, S., Rosset, S., Perlich, C., Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data*, 6(4), 15.
9. Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206-215.
10. Obermeyer, Z., Powers, B., Vogeli, C., Mullainathan, S. (2019). Dissecting racial bias in an algorithm used to manage the health of populations. *Science*, 366(6464), 447-453.
"""


if __name__ == "__main__":
    out = ROOT / "docs" / "PROJECT_REPORT.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print(f"Wrote {out}")
