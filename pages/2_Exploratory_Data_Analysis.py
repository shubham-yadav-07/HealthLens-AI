"""Exploratory Data Analysis with interactive charts and statistics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src import eda
from src.config import FEATURES, FEATURE_INFO, TARGET
from src.preprocessing import missing_summary
from src.state import get_active_dataset
from src.ui import info, kpi_row, note, page_header

page_header("Exploratory Data Analysis",
            "Every figure and table on this page is calculated from the dataset you have loaded.")

df, name, _ = get_active_dataset()
ov = eda.overview(df)
use_clean = st.toggle("Treat impossible zeros as missing in charts", value=True,
                      help="Recommended. A glucose of 0 is a missing measurement, not a real value.")

tabs = st.tabs(["Overview", "Missing and duplicates", "Distributions", "Outcome comparison",
                "Correlations", "Outliers", "Key findings"])

with tabs[0]:
    kpi_row([
        ("Rows", f"{ov['rows']:,}", "", ""),
        ("Features", str(ov["features"]), "plus 1 target", "blue"),
        ("Diabetes recorded", f"{ov['positive_rate'] * 100:.1f}%", f"imbalance {eda.imbalance_ratio(df):.2f} : 1", "green"),
        ("Duplicates", str(ov["duplicates"]), "", ""),
    ])
    st.write("")
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        st.plotly_chart(eda.fig_class_distribution(df), width="stretch")
        info("The classes are moderately imbalanced. Accuracy can look decent even for a model that "
             "misses many positive cases, so recall and ROC-AUC matter too.")
    with c2:
        st.markdown("**Descriptive statistics** (raw values, including impossible zeros)")
        st.dataframe(eda.describe_table(df), width="stretch")

with tabs[1]:
    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        st.dataframe(missing_summary(df), width="stretch", hide_index=True)
    with c2:
        st.plotly_chart(eda.fig_missing(df), width="stretch")
    info("Missing values are filled with the <b>median of the training rows only</b>, inside the model pipeline, "
         "so no information from the test set leaks into training.")
    dup = eda.duplicate_report(df)
    st.markdown(f"**Duplicate records:** {dup['n_duplicates']}")
    if dup["n_duplicates"]:
        st.dataframe(dup["rows_involved"], width="stretch")
    else:
        st.caption("No exact duplicate rows were found.")

with tabs[2]:
    c1, c2 = st.columns([1, 1])
    feat = c1.selectbox("Feature", FEATURES, format_func=lambda f: f"{f} ({FEATURE_INFO[f]['unit']})")
    split = c2.checkbox("Split by outcome", value=True)
    st.plotly_chart(eda.fig_histogram(df, feat, use_clean, split), width="stretch")
    st.caption(FEATURE_INFO[feat]["help"])
    st.plotly_chart(eda.fig_box_by_class(df, feat, use_clean), width="stretch", key="dist_box")

with tabs[3]:
    gc = eda.group_comparison(df)
    st.plotly_chart(eda.fig_violin_grid(df), width="stretch")
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        show = gc.copy()
        show["p-value"] = show["p-value"].map(lambda p: f"{p:.2e}")
        show["p (Bonferroni)"] = show["p (Bonferroni)"].map(lambda p: f"{p:.2e}")
        st.dataframe(show, width="stretch", hide_index=True)
    with c2:
        st.plotly_chart(eda.fig_effect_sizes(gc), width="stretch")
    note("<b>Exploratory, not confirmatory.</b> Mann-Whitney U tests show that the groups differ in this sample. "
         "They do not show that a feature causes diabetes, and the data are observational from one population. "
         "Cohen's d around 0.2 is small, 0.5 medium and 0.8 large.")
    st.subheader("Relationship between two features")
    c1, c2 = st.columns(2)
    x = c1.selectbox("X axis", FEATURES, index=1)
    y = c2.selectbox("Y axis", FEATURES, index=5)
    if x == y:
        st.warning("Choose two different features.")
    else:
        st.plotly_chart(eda.fig_scatter(df, x, y, use_clean), width="stretch")

with tabs[4]:
    method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True,
                      help="Spearman uses ranks and is less sensitive to outliers.")
    st.plotly_chart(eda.fig_correlation(df, method), width="stretch")
    corr = eda.correlation_matrix(df, method)[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    st.markdown("**Correlation with the outcome**")
    st.dataframe(corr.rename("r").round(3).to_frame(), width="stretch")
    info("Correlation measures how two columns move together. It is not causation, and a low correlation "
         "does not rule out a useful non-linear relationship.")

with tabs[5]:
    st.dataframe(eda.outlier_table(df), width="stretch", hide_index=True)
    st.caption("IQR rule on values with impossible zeros removed. Outliers are kept in model training: "
               "tree models are robust to them, and extreme values can be genuine clinical measurements.")
    out_feat = st.selectbox("Inspect outliers for", FEATURES, key="out_feat")
    st.plotly_chart(eda.fig_box_by_class(df, out_feat, use_clean), width="stretch", key="outlier_box")

with tabs[6]:
    for f in eda.key_findings(df):
        st.markdown(f"- {f}")
    st.caption("These sentences are generated from the loaded data each time, so they change if you upload a different file.")
