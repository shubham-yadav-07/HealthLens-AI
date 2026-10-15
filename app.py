"""
HealthLens AI - application entry point.

Run with:   streamlit run app.py

This file (1) configures the page, (2) injects the brand CSS, (3) defines the
navigation with icons and (4) contains the Home / landing page.
"""
import sys
from pathlib import Path

# make `import src...` work no matter where streamlit is started from
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plotly.graph_objects as go
import streamlit as st

from src import eda
from src.config import CLASS_LABELS, COLORS, DISCLAIMER, FEATURES, MODEL_NAMES, TARGET
from src.plotting import style_fig
from src.state import get_active_dataset, get_artifacts
from src.ui import (
    card, disclaimer, inject_css, kpi_row, sidebar_brand, sidebar_footer,
)

st.set_page_config(page_title="HealthLens AI", page_icon="🩺", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()


def home():
    df, name, uploaded = get_active_dataset()
    artifacts = get_artifacts()
    metrics = artifacts["metrics"]
    best = metrics["ROC-AUC"].idxmax()

    # ------------------------------------------------------------------ hero
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        st.markdown(
            "<div class='hl-hero'>"
            "<span class='hl-chip'>Educational research prototype</span>"
            "<h1>See which health measurements move a diabetes risk model.</h1>"
            "<p class='lead'>HealthLens AI trains four machine-learning models on the Pima Indians "
            "Diabetes data, compares them honestly, and explains every prediction with SHAP.</p>"
            "</div>", unsafe_allow_html=True)
        if st.button("Open the analytics dashboard", type="primary"):
            st.switch_page("pages/1_Dataset_Explorer.py")
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.page_link("pages/3_Health_Risk_Prediction.py", label="Or try the risk estimator",
                     icon=":material/monitor_heart:")
    with right:
        d = eda.clean_view(df)
        fig = go.Figure()
        for cls, color in [(0, COLORS["class0"]), (1, COLORS["class1"])]:
            vals = d.loc[d[TARGET] == cls, "Glucose"].dropna()
            fig.add_trace(go.Histogram(x=vals, name=CLASS_LABELS[cls], marker_color=color, opacity=0.7,
                                       histnorm="probability density", nbinsx=28))
        fig.update_layout(barmode="overlay", title="Live from the data: glucose by outcome",
                          xaxis_title="Plasma glucose (mg/dL)", yaxis_title="Density")
        st.plotly_chart(style_fig(fig, 360, legend_below=True), width="stretch")

    disclaimer()
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ------------------------------------------------------------ live KPIs
    ov = eda.overview(df)
    st.subheader("Platform snapshot")
    kpi_row([
        ("Patient records", f"{ov['rows']:,}", name, ""),
        ("Diabetes recorded", f"{ov['positive_rate'] * 100:.1f}%", "class share in the data", "blue"),
        ("Impossible zeros found", f"{ov['missing_cells']:,}", "treated as missing values", "green"),
        ("Best test ROC-AUC", f"{metrics.loc[best, 'ROC-AUC']:.3f}", f"{best} (default models)", ""),
    ])
    if uploaded:
        st.caption("Dataset figures follow your uploaded CSV. Prediction and SHAP pages always use the "
                   "models trained on the default Pima data.")

    # --------------------------------------------------------- capabilities
    st.markdown("### What you can do here")
    caps = [
        ("Explore the data", "Browse records, check validity, or upload a compatible CSV. Impossible zeros, duplicates and outliers are flagged automatically."),
        ("Analyse it statistically", "Distributions, class comparisons with effect sizes, correlation heatmaps. All numbers are computed from the loaded data."),
        ("Compare four models", "Logistic Regression, Decision Tree, Random Forest and XGBoost, tuned with cross-validation and judged on more than accuracy."),
        ("Estimate a risk score", "Enter measurements, get a model-estimated probability with its limits spelled out. Never a diagnosis."),
        ("Understand predictions", "SHAP shows how each measurement pushed a prediction up or down, in plain English."),
        ("Export your findings", "Download the EDA report, model comparison, cross-validation folds and SHAP importances."),
    ]
    for row in (caps[:3], caps[3:]):
        cols = st.columns(3, gap="medium")
        for c, (t, b) in zip(cols, row):
            c.markdown(card(t, b), unsafe_allow_html=True)
        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)

    # ------------------------------------------------- dataset + model info
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("### Dataset")
        st.markdown(
            "- **Source:** Pima Indians Diabetes Database, National Institute of Diabetes and Digestive "
            "and Kidney Diseases (Smith et al., 1988), via the UCI / Kaggle copy.\n"
            "- **Population:** women aged 21+ of Pima Indian heritage. Results do **not** generalise to other groups.\n"
            f"- **Features:** {', '.join(FEATURES)}.\n"
            "- **Target:** `Outcome` (1 = diabetes recorded within five years).\n"
            "- **Known issue:** zeros in five columns really mean *not measured*; they are converted to missing values.")
    with b:
        st.markdown("### Models")
        mt = metrics[["ROC-AUC", "Recall", "F1"]].copy().round(3)
        mt["Tuned parameters"] = [", ".join(f"{k}={v}" for k, v in artifacts["best_params"][m].items())
                                  for m in mt.index]
        st.dataframe(mt, width="stretch")
        st.caption(f"Held-out test set of {artifacts['meta']['n_test']} records, decision threshold 0.5. "
                   "Differences between models are small relative to the uncertainty; see Model Comparison.")


pages = {
    "": [st.Page(home, title="Home", icon=":material/home:", default=True)],
    "Analyse": [
        st.Page("pages/1_Dataset_Explorer.py", title="Dataset Explorer", icon=":material/table_view:"),
        st.Page("pages/2_Exploratory_Data_Analysis.py", title="Exploratory Analysis", icon=":material/monitoring:"),
    ],
    "Model": [
        st.Page("pages/3_Health_Risk_Prediction.py", title="Health Risk Prediction", icon=":material/monitor_heart:"),
        st.Page("pages/4_Model_Comparison.py", title="Model Comparison", icon=":material/compare_arrows:"),
        st.Page("pages/5_Explainable_AI.py", title="Explainable AI", icon=":material/lightbulb:"),
    ],
    "Project": [
        st.Page("pages/6_Reports.py", title="Insights and Reports", icon=":material/download:"),
        st.Page("pages/7_About.py", title="About the Project", icon=":material/info:"),
    ],
}
navigation = st.navigation(pages)
sidebar_brand()
_, _active_name, _ = get_active_dataset()
sidebar_footer(_active_name)
navigation.run()
