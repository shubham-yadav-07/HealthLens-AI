# HealthLens AI - Early Health Risk Analysis and Prediction Platform
### Project report (auto-generated on 2026-10-14 from the actual run)

> Educational and research prototype. Not a medical device. Nothing in this report constitutes medical advice.

## 1. Abstract
HealthLens AI is an end-to-end data science application that estimates diabetes-associated risk from eight routine
clinical measurements and explains its predictions. Four classifiers (Logistic Regression, Decision Tree, Random Forest,
XGBoost) were trained on the Pima Indians Diabetes Database (768 records) inside leakage-free scikit-learn
pipelines, tuned with 5-fold stratified cross-validation and evaluated on a stratified held-out test set
of 154 records. The highest test ROC-AUC was 0.816 (XGBoost), but bootstrap confidence
intervals of all four models overlap, so no model can be declared superior. SHAP identifies
**Glucose** as the most influential feature, followed by **BMI**. The system is delivered as a
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
code; `pages/` holds the Streamlit pages; `tests/` holds 41 automated tests. See Section 10 for the architecture.

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
- **Records:** 768; **features:** 8; **target:** `Outcome` (1 = diabetes recorded within five years).
- **Population:** women aged 21+ of Pima heritage; **intended use:** benchmarking and teaching.
- **Class balance:**

| Class | Count | Share % |
|---|---|---|
| No diabetes recorded | 500 | 65.100 |
| Diabetes recorded | 268 | 34.900 |

- **Hidden missing values** (zeros that are physiologically impossible):

| Feature | NaN values | Zeros treated as missing | Total missing | Missing % |
|---|---|---|---|---|
| Pregnancies | 0 | 0 | 0 | 0.000 |
| Glucose | 0 | 5 | 5 | 0.650 |
| BloodPressure | 0 | 35 | 35 | 4.560 |
| SkinThickness | 0 | 227 | 227 | 29.560 |
| Insulin | 0 | 374 | 374 | 48.700 |
| BMI | 0 | 11 | 11 | 1.430 |
| DiabetesPedigreeFunction | 0 | 0 | 0 | 0.000 |
| Age | 0 | 0 | 0 | 0.000 |

**Limitations:** single population, small sample, 1980s data, no demographic variables beyond age, no information on diabetes type.

## 9. Methodology
1. Load and validate data (columns, numeric types, binary target, minimum rows per class).
2. Stratified 80/20 train/test split, `random_state=42`. Test set untouched until final evaluation.
3. Pipeline: zero-to-NaN -> median imputation -> (standard scaling for Logistic Regression) -> classifier. All fitted on training rows only.
4. Hyper-parameter search with `GridSearchCV` (5-fold stratified, scoring ROC-AUC) on the training set.
5. Nested cross-validation to estimate performance without tuning optimism. CV type used in this run: *Nested CV (tuning repeated inside every fold)*.
6. Single evaluation on the test set; bootstrap (1,000 resamples) confidence interval for ROC-AUC.
7. SHAP explanations on the test set.

Tuned parameters:

| index | C | max_depth | min_samples_leaf | learning_rate | n_estimators | subsample |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.1 | nan | nan | nan | nan | nan |
| Decision Tree | nan | 4 | 1 | nan | nan | nan |
| Random Forest | nan | 4 | 5 | nan | nan | nan |
| XGBoost | nan | 2 | nan | 0.03 | 100 | 0.8 |

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
- Duplicates found: **0**. Outliers are retained (see Section 12).
- Leakage control: imputer and scaler live inside the pipeline, so they are re-fitted on each CV training fold and on the training split only.

## 12. Exploratory data analysis
Key findings (computed from the data):

- The dataset has 768 rows and 8 features; 34.9% of records have a recorded diabetes outcome (imbalance ratio 1.87 : 1).
- Hidden missing data: impossible zeros mean the most incomplete features are Insulin (48.7%), SkinThickness (29.56%), BloodPressure (4.56%).
- Duplicate rows: 0.
- The largest group difference is in Glucose (Cohen's d = 1.19); median 107.0 (no diabetes) vs 140.0 (diabetes).
- 8 of 8 features differ significantly between groups after Bonferroni correction: Glucose, BMI, Insulin, SkinThickness, Age, Pregnancies, DiabetesPedigreeFunction, BloodPressure. This is an association, not proof of cause.
- Strongest linear correlation with the outcome: Glucose (r = 0.49), then BMI (r = 0.31).
- Most IQR outliers: DiabetesPedigreeFunction with 29 values (3.78%); outliers were kept because they may be genuine clinical values.

Group comparison (Mann-Whitney U with Bonferroni correction; effect size Cohen's d). Exploratory only:

| Feature | Median (no diabetes) | Median (diabetes) | Cohen's d | Significant (0.05) |
|---|---|---|---|---|
| Glucose | 107.000 | 140.000 | 1.193 | yes |
| BMI | 30.100 | 34.300 | 0.691 | yes |
| Insulin | 102.500 | 169.500 | 0.676 | yes |
| SkinThickness | 27.000 | 32.000 | 0.569 | yes |
| Age | 27.000 | 36.000 | 0.514 | yes |
| Pregnancies | 2.000 | 4.000 | 0.477 | yes |
| DiabetesPedigreeFunction | 0.336 | 0.449 | 0.370 | yes |
| BloodPressure | 70.000 | 74.500 | 0.364 | yes |

Outliers (IQR rule):

| Feature | Lower fence | Upper fence | Outliers | Outlier % | Max value |
|---|---|---|---|---|---|
| Pregnancies | -6.500 | 13.500 | 4 | 0.520 | 17.000 |
| Glucose | 36.000 | 204.000 | 0 | 0.000 | 199.000 |
| BloodPressure | 40.000 | 104.000 | 14 | 1.910 | 122.000 |
| SkinThickness | 1.000 | 57.000 | 3 | 0.550 | 99.000 |
| Insulin | -94.375 | 360.625 | 24 | 6.090 | 846.000 |
| BMI | 13.850 | 50.250 | 8 | 1.060 | 67.100 |
| DiabetesPedigreeFunction | -0.330 | 1.200 | 29 | 3.780 | 2.420 |
| Age | -1.500 | 66.500 | 9 | 1.170 | 81.000 |

## 13. Machine-learning algorithms
- **Logistic Regression:** linear model of log-odds; interpretable coefficients; needs scaled inputs.
- **Decision Tree:** recursive axis-aligned splits; highly interpretable but unstable and prone to overfitting.
- **Random Forest:** averages many decorrelated trees (bagging); reduces variance.
- **XGBoost:** gradient-boosted trees built sequentially to correct previous errors, with regularisation.

## 14. Model evaluation
Test-set results at threshold 0.5 (n = 154):

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | ROC-AUC 95% CI |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.688 | 0.565 | 0.481 | 0.800 | 0.520 | 0.810 | 0.74-0.87 |
| Decision Tree | 0.786 | 0.698 | 0.685 | 0.840 | 0.692 | 0.789 | 0.71-0.86 |
| Random Forest | 0.734 | 0.667 | 0.481 | 0.870 | 0.559 | 0.801 | 0.73-0.87 |
| XGBoost | 0.740 | 0.667 | 0.519 | 0.860 | 0.583 | 0.816 | 0.75-0.88 |

Confusion-matrix counts:

| Model | TP | FP | TN | FN |
|---|---|---|---|---|
| Logistic Regression | 26 | 20 | 80 | 28 |
| Decision Tree | 37 | 16 | 84 | 17 |
| Random Forest | 26 | 13 | 87 | 28 |
| XGBoost | 28 | 14 | 86 | 26 |

Cross-validation on the training set (mean ± std over 5 folds):

| Model | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|
| Logistic Regression | 0.780 ± 0.019 | 0.752 ± 0.072 | 0.561 ± 0.024 | 0.640 ± 0.016 | 0.844 ± 0.018 |
| Decision Tree | 0.723 ± 0.042 | 0.659 ± 0.145 | 0.548 ± 0.150 | 0.572 ± 0.070 | 0.772 ± 0.041 |
| Random Forest | 0.770 ± 0.015 | 0.724 ± 0.070 | 0.570 ± 0.060 | 0.633 ± 0.018 | 0.835 ± 0.021 |
| XGBoost | 0.770 ± 0.019 | 0.714 ± 0.063 | 0.579 ± 0.044 | 0.637 ± 0.024 | 0.836 ± 0.022 |

**Why accuracy alone is not enough.** A classifier that always predicts "no diabetes" would score 64.9% accuracy on this
test set while finding no positive cases. Missing a real case usually costs more than a false alarm, so recall, specificity, precision,
F1 and ROC-AUC are reported together with the threshold trade-off below (XGBoost):

| index | Threshold | Accuracy | Precision | Recall | Specificity | F1 | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 0.200 | 0.688 | 0.533 | 0.907 | 0.570 | 0.671 | 49.000 | 43.000 | 57.000 | 5.000 |
| 1.000 | 0.300 | 0.747 | 0.600 | 0.833 | 0.700 | 0.698 | 45.000 | 30.000 | 70.000 | 9.000 |
| 2.000 | 0.400 | 0.734 | 0.614 | 0.648 | 0.780 | 0.631 | 35.000 | 22.000 | 78.000 | 19.000 |
| 3.000 | 0.500 | 0.740 | 0.667 | 0.519 | 0.860 | 0.583 | 28.000 | 14.000 | 86.000 | 26.000 |
| 4.000 | 0.600 | 0.740 | 0.719 | 0.426 | 0.910 | 0.535 | 23.000 | 9.000 | 91.000 | 31.000 |

## 15. Explainable AI
Mean share of total |SHAP| value, per model (percent):

| Feature | Logistic Regression | Decision Tree | Random Forest | XGBoost |
|---|---|---|---|---|
| Pregnancies | 14.3 | 0.0 | 5.8 | 4.8 |
| Glucose | 44.7 | 48.4 | 33.9 | 43.2 |
| BloodPressure | 0.1 | 0.0 | 3.1 | 0.3 |
| SkinThickness | 2.1 | 0.0 | 4.1 | 0.5 |
| Insulin | 0.1 | 5.3 | 10.0 | 5.9 |
| BMI | 24.1 | 26.0 | 19.9 | 24.5 |
| DiabetesPedigreeFunction | 8.0 | 0.0 | 8.1 | 7.1 |
| Age | 6.5 | 20.3 | 15.1 | 13.8 |

SHAP additivity (base value + contributions = model output) is verified by an automated test for every model. Importance describes
reliance of the *model* on a feature; it does not establish causation, and correlated features can split importance.

## 16. Results and discussion
- The best test ROC-AUC is 0.816 (XGBoost); the lowest is 0.789. The 95% bootstrap intervals overlap, so differences are not statistically distinguishable on 154 test records.
- At the default 0.5 threshold recall ranges from 0.48 to 0.69: several models miss roughly half of positive cases, which would be unacceptable for screening.
- Glucose dominates across models, consistent with its clinical role and with its large effect size in the EDA (1.19 Cohen's d for Glucose). Because the data are observational this is an association, not a causal finding.
- Age-group performance for XGBoost (only demographic variable available; groups are small):

| Age group | n | Diabetes recorded | Reliable? | Accuracy | Recall | Specificity | ROC-AUC |
|---|---|---|---|---|---|---|---|
| 21-29 | 85 | 16 | yes | 0.812 | 0.375 | 0.913 | 0.759 |
| 30-39 | 30 | 20 | yes | 0.633 | 0.550 | 0.800 | 0.802 |
| 40-49 | 24 | 12 | yes | 0.750 | 0.667 | 0.833 | 0.861 |
| 50+ | 15 | 6 | too few cases | 0.533 | 0.500 | 0.556 | 0.685 |

## 17. Limitations
- One population, one era, 768 records. No external validation.
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
