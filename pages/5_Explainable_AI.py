"""Explainable AI with SHAP: global importance, summary plot, single-prediction explanation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import explainability as xai
from src.config import FEATURES, MODEL_NAMES
from src.state import get_artifacts
from src.ui import info, note, page_header


@st.cache_data(show_spinner=False)
def _shap_for(model_name: str, _results_key: str):
    r = get_artifacts()
    values, base, Xt = xai.compute_shap(model_name, r["models"][model_name], r["X_train"], r["X_test"])
    return values, base, Xt


@st.cache_data(show_spinner=False)
def _importance_all(_results_key: str):
    return xai.importance_all_models(get_artifacts())


page_header("Explainable AI",
            "SHAP shows how much each measurement pushed a model's output up or down.")

artifacts = get_artifacts()
key = artifacts["meta"]["dataset_hash"]
model_name = st.selectbox("Model to explain", MODEL_NAMES, index=MODEL_NAMES.index("XGBoost"))

with st.spinner("Computing SHAP values on the test set..."):
    values, base, Xt = _shap_for(model_name, key)

info(f"SHAP values for <b>{model_name}</b> are in <b>{xai.UNITS[model_name]}</b>. A positive value pushes the output "
     "towards <i>diabetes recorded</i>, a negative value towards <i>no diabetes recorded</i>. "
     "Explanations are computed on the held-out test set.")
st.write("")

tab_global, tab_summary, tab_single, tab_compare, tab_read = st.tabs(
    ["Global importance", "SHAP summary plot", "Single prediction", "Compare models", "How to read this"])

with tab_global:
    imp = xai.global_importance(values)
    c1, c2 = st.columns([1.4, 1], gap="large")
    c1.plotly_chart(xai.fig_global_importance(imp, model_name), width="stretch")
    with c2:
        top = imp.iloc[0]
        st.markdown(f"For this model, **{top['Feature']}** carries the largest share ({top['Share %']:.0f}%) "
                    f"of the total average impact, followed by **{imp.iloc[1]['Feature']}** ({imp.iloc[1]['Share %']:.0f}%) "
                    f"and **{imp.iloc[2]['Feature']}** ({imp.iloc[2]['Share %']:.0f}%).")
        st.dataframe(imp.round(3), width="stretch", hide_index=True)
    note("Importance means the <b>model relies</b> on a feature. It does not mean the feature causes diabetes, "
         "and correlated features can share or steal importance from each other.")

with tab_summary:
    fig = xai.fig_beeswarm(values, Xt, model_name)
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption("Each dot is one test record. Position shows the SHAP value, colour shows whether the feature value "
               "was high (red) or low (blue). Features are sorted by overall importance.")

with tab_single:
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    idx_pos = st.selectbox("Choose a test-set record", list(range(len(X_test))),
                           format_func=lambda i: f"Record #{X_test.index[i]} (actual outcome: {'diabetes' if y_test.iloc[i] == 1 else 'no diabetes'})")
    row_raw = X_test.iloc[idx_pos]
    contrib = xai.contribution_table(values[idx_pos], row_raw, Xt.iloc[idx_pos])
    proba = artifacts["probas"][model_name][idx_pos]
    st.markdown(f"Model-estimated probability for this record: **{proba:.0%}**")
    c1, c2 = st.columns([1.2, 1], gap="large")
    c1.plotly_chart(xai.fig_contributions(contrib, model_name, base), width="stretch")
    with c2:
        st.markdown("**In plain English**")
        for line in xai.plain_english(contrib, model_name):
            st.markdown(f"- {line}")
        st.dataframe(contrib.round(3), width="stretch", hide_index=True)
    st.caption("A value shown as 0 in the table for Glucose, BloodPressure, SkinThickness, Insulin or BMI means it was "
               "unmeasured in the data; the model used the training median instead (see 'Value used').")

with tab_compare:
    imp_all = _importance_all(key)
    st.plotly_chart(xai.fig_importance_comparison(imp_all), width="stretch")
    st.dataframe(imp_all.round(1), width="stretch")
    st.caption("Share of each model's total mean |SHAP|, so models with different units can be compared. "
               "Agreement across very different algorithms makes a pattern more credible, but it is still not causal evidence.")
    st.download_button("Download importance table (CSV)", imp_all.round(4).to_csv().encode("utf-8"),
                       file_name="shap_importance_all_models.csv", mime="text/csv")

with tab_read:
    st.markdown(
        """
- **SHAP** comes from game theory (Shapley values): every feature gets a fair share of the credit for moving a prediction away from the average prediction.
- The contributions of all features **add up exactly** to the model output minus the base value. This app checks that property in its tests.
- **Global importance** averages the absolute contributions over many records. **Local explanations** show one record.
- Trees use **TreeExplainer** (exact), Logistic Regression uses **LinearExplainer** on the scaled inputs.
- Explanations can change with the model, the data split and the preprocessing. Treat them as a way to *inspect* a model, not as medical reasoning.
""")
