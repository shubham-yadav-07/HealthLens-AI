# Methodology (short form)

## 1. Experimental design
| Item | Choice | Reason |
|---|---|---|
| Split | Stratified 80/20, seed 42 | Keeps the 35% positive share in both parts; reproducible |
| Missing values | Zero -> NaN in 5 columns, then median imputation | Zeros are impossible; median is robust to skew |
| Scaling | StandardScaler for Logistic Regression only | Linear model with regularisation is scale-sensitive; trees are not |
| Tuning | GridSearchCV, 5-fold stratified, ROC-AUC | Threshold-independent metric suited to imbalance |
| Honest estimate | Nested CV (tuning repeated inside each outer fold) | Tuning on the same folds you report on is optimistic |
| Final test | Used once, after all choices | Prevents test-set overfitting |
| Uncertainty | Bootstrap 95% CI for test ROC-AUC | A 154-record test set is noisy |
| Imbalance | No resampling or class weights; threshold analysis instead | Keeps probabilities closer to real frequencies and shows the recall/precision trade-off explicitly |

## 2. Preventing data leakage
Imputation medians and scaling parameters are learned *inside* an sklearn `Pipeline`. `fit()` sees training rows only; `predict()` only
applies stored values. Cross-validation clones and refits the whole pipeline per fold. A unit test checks that the learned median equals
the training median and not the global one.

## 3. Statistics used in EDA
- **Mann-Whitney U**: compares two groups without assuming normality.
- **Bonferroni correction**: multiplies p-values by the number of tests (8) to control false positives.
- **Cohen's d**: standardised mean difference (0.2 small, 0.5 medium, 0.8 large).
- **IQR rule**: outlier if outside Q1 - 1.5*IQR or Q3 + 1.5*IQR.
These are exploratory. Significance in one sample is not validation, and not causation.

## 4. Explainability
`TreeExplainer` for Decision Tree, Random Forest and XGBoost; `LinearExplainer` for Logistic Regression on scaled inputs.
Units: log-odds (LR, XGBoost) or probability (trees). Additivity (base + sum of contributions = model output) is tested.

## 5. Reproduce everything
```bash
python -m src.data_loader        # dataset
python -m src.train_models       # full run with nested CV (~3 min)
python -m src.generate_report    # docs/PROJECT_REPORT.md from real numbers
pytest                           # 41 tests
streamlit run app.py
```
