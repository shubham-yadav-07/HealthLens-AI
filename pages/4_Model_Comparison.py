"""Model comparison: metrics, ROC, confusion matrices, thresholds, CV, subgroups."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src import evaluate_models as ev
from src.config import MODEL_NAMES
from src.state import get_active_dataset, get_artifacts
from src.train_models import train_all
from src.ui import info, kpi_row, note, page_header

page_header("Model Comparison",
            "Four algorithms, judged on the same held-out test set and on cross-validation.")

df, ds_name, uploaded = get_active_dataset()
default_results = get_artifacts()

# ------------------------------------------------------- choose results source
results, source_label = default_results, "Default Pima models"
if uploaded:
    st.markdown(f"You are using an uploaded dataset (**{ds_name}**). You can train the four models on it "
                "inside this session; nothing is saved to disk.")
    c1, c2 = st.columns([1, 2])
    if c1.button("Train models on uploaded data", type="primary"):
        bar = st.progress(0.0, text="Starting...")
        st.session_state["custom_results"] = train_all(
            df, tune=True, nested_cv=False, save=False,
            progress=lambda msg, frac: bar.progress(min(frac, 1.0), text=msg))
        bar.empty()
    if "custom_results" in st.session_state:
        choice = c2.radio("Show results for", ["Uploaded dataset (trained now)", "Default Pima models"], horizontal=True)
        if choice.startswith("Uploaded"):
            results, source_label = st.session_state["custom_results"], f"Uploaded: {ds_name}"

metrics_default = results["metrics"]
meta = results["meta"]
st.caption(f"Source: {source_label}. Trained {meta['trained_at_utc']} UTC, seed {meta['random_state']}, "
           f"{meta['n_train']} training / {meta['n_test']} test rows (stratified split). "
           f"Cross-validation: {meta['cv_type']}.")

threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05,
                      help="A record is predicted positive when its probability is at or above this value.")
table = ev.metrics_table(results, threshold)
best = table["ROC-AUC"].idxmax()

kpi_row([
    ("Highest ROC-AUC", f"{table.loc[best, 'ROC-AUC']:.3f}", best, ""),
    ("Highest recall", f"{table['Recall'].max():.2f}", table["Recall"].idxmax(), "blue"),
    ("Highest accuracy", f"{table['Accuracy'].max():.2f}", table["Accuracy"].idxmax(), "green"),
    ("Test set size", str(meta["n_test"]), f"{meta['positive_rate_test']:.0%} positive", ""),
])
st.write("")

tab_main, tab_cm, tab_thr, tab_cv, tab_sub, tab_why = st.tabs(
    ["Metrics and ROC", "Confusion matrices", "Threshold analysis", "Cross-validation", "Age groups", "Why accuracy is not enough"])

with tab_main:
    show = table[["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]].round(3)
    ci = metrics_default[["ROC-AUC CI low", "ROC-AUC CI high"]]
    show["ROC-AUC 95% CI (bootstrap)"] = [f"{ci.loc[m, 'ROC-AUC CI low']:.2f} to {ci.loc[m, 'ROC-AUC CI high']:.2f}" for m in show.index]
    st.dataframe(show, width="stretch")
    st.caption("Confidence intervals come from 1,000 bootstrap resamples of the test set at threshold 0.5.")
    c1, c2 = st.columns(2, gap="large")
    c1.plotly_chart(ev.fig_metric_bars(table), width="stretch")
    c2.plotly_chart(ev.fig_roc(results), width="stretch")
    note("<b>Read the intervals before ranking models.</b> The ROC-AUC intervals of all four models overlap heavily, "
         "so the test set cannot show that one model is truly better than another. Higher test scores do not make "
         "any of these models clinically accurate.")
    st.markdown("**Tuned hyper-parameters (5-fold CV on training data)**")
    st.dataframe(pd.DataFrame({m: {k: v for k, v in results["best_params"][m].items()} for m in MODEL_NAMES}).T.astype(str),
                 width="stretch")

with tab_cm:
    cols = st.columns(2, gap="large")
    for i, name in enumerate(MODEL_NAMES):
        row = table.loc[name]
        cols[i % 2].plotly_chart(ev.fig_confusion(row.to_dict(), f"{name} (threshold {threshold:.2f})"),
                                 width="stretch")
    info("<b>False negatives</b> (actual positive, predicted negative) are usually the costliest error in health "
         "screening because a person who needs follow-up is missed. <b>False positives</b> cause unnecessary worry and testing.")

with tab_thr:
    pick = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(best))
    tt = ev.threshold_table(results["y_test"].values, results["probas"][pick])
    st.plotly_chart(ev.fig_threshold(tt, f"{pick}: metrics at different thresholds"), width="stretch")
    st.dataframe(tt.round(3), width="stretch", hide_index=True)
    st.caption("Lowering the threshold finds more true cases (higher recall) at the price of more false alarms.")
    st.download_button("Download threshold table (CSV)", tt.round(4).to_csv(index=False).encode("utf-8"),
                       file_name=f"threshold_analysis_{pick.lower().replace(' ', '_')}.csv", mime="text/csv")

with tab_cv:
    st.dataframe(ev.cv_summary(results["cv_results"]), width="stretch")
    metric = st.selectbox("Metric to plot", ["roc_auc", "recall", "precision", "f1", "accuracy"])
    st.plotly_chart(ev.fig_cv_box(results["cv_results"], metric), width="stretch")
    st.caption("Each point is one validation fold. Wide spreads mean the score depends heavily on which records "
               "land in the fold; with only ~120 records per fold this is expected.")
    with st.expander("Per-fold values"):
        st.dataframe(results["cv_results"].round(4), width="stretch", hide_index=True)

with tab_sub:
    pick2 = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(best), key="sub_model")
    sub = ev.age_group_performance(results["X_test"], results["y_test"], results["probas"][pick2], threshold)
    st.dataframe(sub.round(3), width="stretch", hide_index=True)
    note("Age is the <b>only</b> demographic variable in this dataset and every participant is a woman from one population. "
         "Subgroups here contain very few records, so these numbers are unstable. They are a sanity check, "
         "<b>not evidence that the model is fair</b>. A real fairness audit needs far more data and variables.")

with tab_why:
    st.markdown(
        """
**Why accuracy alone can mislead in healthcare**

1. **Class imbalance.** If 65% of people have no diabetes recorded, a model that always says "no diabetes" is 65% accurate yet useless.
2. **Errors have different costs.** Missing a real case (false negative) usually matters more than a false alarm, but accuracy treats them equally.
3. **Threshold dependence.** Accuracy at one cut-off hides the recall/precision trade-off, which you can see on the threshold tab.
4. **Uncertainty.** With a test set of about 150 records, one extra correct prediction moves accuracy by roughly 0.7 points.
5. **Generalisation.** A high score on a held-out sample from the same population does not prove the model works in a new hospital, region or decade.

That is why this page reports recall, precision, specificity, F1, ROC-AUC with confidence intervals, and cross-validation.
""")
    baseline = 1 - meta["positive_rate_test"]
    st.metric("Accuracy of always predicting 'no diabetes' on this test set", f"{baseline:.1%}")

st.divider()
st.download_button("Download model comparison (CSV)", table.round(4).to_csv().encode("utf-8"),
                   file_name="model_comparison.csv", mime="text/csv")
