# Viva questions and answers

Answers are written in simple language. Where numbers are mentioned, check them against the Model Comparison page of *your* run.

## A. Project overview
**Q1. What does HealthLens AI do?**
It trains four ML models on the Pima diabetes data, compares them, estimates a model-based risk probability for entered measurements, and explains predictions with SHAP. It is educational, not diagnostic.

**Q2. Why Pima Indians data?**
It is public, small enough to run anywhere, widely used as a benchmark, and has real clinical features. Its flaws (hidden missing values, one population) make it a good teaching case for honest analysis.

**Q3. Why did you not claim the model is accurate?**
Test scores come from ~154 records of one population. Confidence intervals are wide and there is no external validation, so a score is not clinical accuracy.

## B. Python and Pandas
**Q4. Difference between `loc` and `iloc`?** `loc` selects by label, `iloc` by integer position.
**Q5. How do you find missing values?** `df.isna().sum()`. In this project zeros also count as missing for five columns, so I also count `(df[col] == 0)`.
**Q6. What does `df.describe()` show?** Count, mean, std, min, quartiles, max for numeric columns.
**Q7. How do you detect duplicates?** `df.duplicated().sum()`; the Pima data has none.
**Q8. What is a pipeline in scikit-learn?** A chain of steps (impute, scale, model) applied in order; `fit` learns each step from training data, `predict` reuses them.
**Q9. Why use `random_state`?** To get the same split and model every run (reproducibility).

## C. EDA
**Q10. What hidden problem does Pima have?** Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are impossible, so they mean "not measured". Insulin is worst (about half).
**Q11. Why median imputation?** Medians are not distorted by skew and outliers like means.
**Q12. How do you handle outliers?** Detect with the IQR rule, keep them if they can be genuine clinical values, and prefer robust models.
**Q13. What does the correlation heatmap tell you?** Linear relationships between pairs of columns. Glucose has the strongest correlation with the outcome. It does not prove causation.
**Q14. Why Mann-Whitney U instead of a t-test?** It does not assume normal distributions, and many features here are skewed.
**Q15. Why Bonferroni?** Running 8 tests increases the chance of a false positive; Bonferroni multiplies p-values by 8.
**Q16. What is Cohen's d?** The difference between group means in units of standard deviation: an effect size independent of sample size.
**Q17. Statistically significant vs important?** With enough data tiny differences become significant; effect size tells whether it matters.

## D. ML algorithms
**Q18. How does Logistic Regression work?** It models log-odds of the positive class as a linear combination of features and converts it to a probability with the sigmoid.
**Q19. Why scale for Logistic Regression only?** Regularised linear models depend on feature scale; trees only compare values within a feature.
**Q20. How does a Decision Tree split?** It picks the feature/threshold that best reduces impurity (Gini). Easy to read but overfits when deep.
**Q21. Random Forest vs Decision Tree?** A forest averages many trees trained on bootstrap samples with random feature subsets, reducing variance.
**Q22. Bagging vs boosting?** Bagging trains models independently and averages; boosting trains sequentially, each model correcting the previous errors (XGBoost).
**Q23. What is overfitting and how did you control it?** Memorising training noise. Controls: cross-validation, limiting tree depth/min leaf size, regularisation in LR, held-out test set.
**Q24. Which hyper-parameters did you tune?** LR: C; Tree: max_depth, min_samples_leaf; RF: max_depth, min_samples_leaf; XGBoost: n_estimators, max_depth, learning_rate, subsample.

## E. Evaluation
**Q25. Define precision, recall, F1, specificity.** Precision = TP/(TP+FP); recall = TP/(TP+FN); specificity = TN/(TN+FP); F1 = harmonic mean of precision and recall.
**Q26. What is ROC-AUC?** The probability that a random positive gets a higher score than a random negative; 0.5 is chance, 1.0 is perfect. Independent of the threshold.
**Q27. Why is accuracy not enough?** Class imbalance, unequal cost of errors, threshold dependence and noisy small test sets. An "always no diabetes" model gets about 65% accuracy.
**Q28. Which error is worse in screening?** Usually false negatives (missed cases), so recall matters; but too many false positives cause unnecessary tests. The threshold page shows the trade-off.
**Q29. Why a bootstrap confidence interval?** To show how much the test AUC would vary with a different sample of 154 patients.
**Q30. Which model is best?** XGBoost has the highest test ROC-AUC, but the intervals overlap, so the data cannot show it is truly better. Decision Tree has higher recall at 0.5 but lower AUC; recall depends on the threshold.

## F. Data leakage and cross-validation
**Q31. What is data leakage?** Information from the test data influencing training, e.g. computing the mean on the full dataset before splitting. It makes results look better than they are.
**Q32. How did you prevent it?** Split first; imputer and scaler inside a Pipeline fitted on training rows only; test set used once at the end; CV refits the pipeline in every fold.
**Q33. What is stratified k-fold CV?** Data are split into k folds each preserving the class ratio; each fold serves once as validation.
**Q34. What is nested CV and why?** The outer loop estimates performance while an inner loop does the tuning, so tuning does not inflate the estimate.
**Q35. Why stratify the train/test split?** To keep about 35% positives in both sets, otherwise evaluation could be unrepresentative.

## G. SHAP and explainability
**Q36. What is SHAP?** A method based on Shapley values from game theory that splits a prediction among features. Contributions add up to the prediction minus the average prediction.
**Q37. Global vs local explanation?** Global: average absolute SHAP per feature. Local: contributions for one person.
**Q38. Does a high SHAP value mean the feature causes diabetes?** No. It shows what the model uses. Correlated features share credit, and the data are observational.
**Q39. Which explainer for which model?** TreeExplainer for trees (exact, fast); LinearExplainer for Logistic Regression.
**Q40. What units are SHAP values in?** Log-odds for LR and XGBoost; probability for Decision Tree and Random Forest.

## H. Healthcare AI limitations and ethics
**Q41. Is the predicted probability a person's real risk?** No. It is the model's output learned from a specific historical sample. It is not calibrated for the individual.
**Q42. Why does the app not say "healthy" or "diabetic"?** A diagnosis needs clinical tests and a professional. A model output cannot decide it.
**Q43. What are the dataset's biases?** One population of women aged 21+, 1980s clinical practice, small size, imputed values.
**Q44. Can you claim the model is fair?** No. There are no variables (sex, ethnicity, income) or enough records for a fairness audit; the age-group table is only a sanity check.
**Q45. How would you make this clinically useful?** External validation on larger diverse data, calibration, prospective evaluation, regulatory review and clinician involvement.
**Q46. What if a user enters a value outside the training range?** The app warns that the model is extrapolating; impossible values are rejected with an error.

## I. Engineering
**Q47. How is the app structured?** `src/` modules for logic (testable without Streamlit), `pages/` for UI, `tests/` for pytest.
**Q48. How did you test it?** 41 pytest tests: data loading/validation, preprocessing and leakage, prediction and invalid input, metrics, SHAP additivity, model save/load.
**Q49. What happens if model files are missing?** The app retrains automatically on first start and saves them.
**Q50. What would you improve next?** Calibration curves, repeated nested CV, more datasets, and a proper fairness study with suitable data.
