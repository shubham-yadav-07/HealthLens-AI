"""Health risk prediction: educational, model-estimated probability with explanations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import prediction as pred
from src.config import COLORS, FEATURES, FEATURE_INFO, MODEL_COLORS, MODEL_NAMES, ZERO_AS_MISSING
from src.explainability import UNITS, compute_shap, contribution_table, fig_contributions, plain_english
from src.plotting import style_fig
from src.preprocessing import transform_features
from src.state import get_artifacts, load_default_cached
from src.ui import disclaimer, info, note, page_header

page_header("Health Risk Prediction",
            "Enter measurements to see what the trained models estimate, and why. This is a learning tool, not a screening test.")
disclaimer()
st.write("")

artifacts = get_artifacts()
reference = load_default_cached()          # predictions always use the default Pima training data
ranges = pred.dataset_ranges(reference)
models = artifacts["models"]

# ------------------------------------------------------------------ input form
left, right = st.columns([1, 1], gap="large")
values = {}
with left:
    st.subheader("Measurements")
    for feat in FEATURES:
        meta = FEATURE_INFO[feat]
        r = ranges[feat]
        c_in, c_chk = st.columns([3, 1.4])
        unmeasured = False
        if feat in ZERO_AS_MISSING:
            unmeasured = c_chk.checkbox("Not measured", key=f"nm_{feat}", help="Use a typical value from the training data.")
        label = f"{meta['label']} ({meta['unit']})"
        hint = f"{meta['help']} Seen in training data: {r['min']:g} to {r['max']:g}."
        if isinstance(meta["step"], int):
            v = c_in.number_input(label, value=int(meta["default"]), step=1, key=f"in_{feat}",
                                  help=hint, disabled=unmeasured)
        else:
            v = c_in.number_input(label, value=float(meta["default"]), step=float(meta["step"]),
                                  key=f"in_{feat}", help=hint, disabled=unmeasured, format="%.2f")
        values[feat] = None if unmeasured else v

    model_name = st.selectbox("Model used for the explanation", MODEL_NAMES, index=MODEL_NAMES.index("XGBoost"))
    run = st.button("Estimate risk", type="primary")

if run:
    clean, errors, warns = pred.validate_inputs(values, ranges)
    st.session_state["pred_state"] = {"clean": clean, "errors": errors, "warns": warns}

state = st.session_state.get("pred_state")

with right:
    st.subheader("Result")
    if state is None:
        info("Fill in the form and press <b>Estimate risk</b>. Tick <i>Not measured</i> for any lab value you do not have.")
    elif state["errors"]:
        for e in state["errors"]:
            st.error(e)
        info("Correct the highlighted values and try again.")
    else:
        clean, warns = state["clean"], state["warns"]
        with st.spinner("Calculating..."):
            probas = pred.predict_all(models, clean)
        p = probas[model_name]
        label, tone = pred.risk_band(p)
        st.markdown(
            f"<div class='hl-result'><span class='hl-pill {tone}'>{label}</span>"
            f"<div class='big' style='margin-top:.6rem'>{p:.0%}</div>"
            f"<div style='color:{COLORS['muted']};margin-top:.3rem'>model-estimated probability of the "
            f"<i>diabetes recorded</i> class ({model_name})</div></div>", unsafe_allow_html=True)
        st.write("")
        note("This number is what the model outputs for people with similar measurements in a 1980s sample of "
             "Pima women. It is <b>not</b> your actual chance of developing diabetes, and it says nothing about "
             "whether you have or do not have the condition. Speak to a clinician about health concerns.")

        n_unmeasured = sum(1 for f in ZERO_AS_MISSING if np.isnan(clean[f]))
        for w in warns:
            st.warning(w)
        for n in pred.reliability_notes(probas, warns, n_unmeasured):
            st.warning(n)

        fig = go.Figure(go.Bar(x=list(probas.values()), y=list(probas.keys()), orientation="h",
                               marker_color=[MODEL_COLORS[m] for m in probas],
                               text=[f"{v:.0%}" for v in probas.values()], textposition="outside",
                               cliponaxis=False))
        fig.update_layout(title="All four models on the same input", xaxis_range=[0, 1.1],
                          xaxis_tickformat=".0%")
        st.plotly_chart(style_fig(fig, 260), width="stretch")

# ------------------------------------------------------------ explanation area
if state is not None and not state["errors"]:
    clean = state["clean"]
    st.divider()
    st.subheader("Why did the model say this?")
    X_row = pred.to_frame(clean)
    with st.spinner("Computing SHAP explanation..."):
        vals, base, Xt = compute_shap(model_name, models[model_name], artifacts["X_train"], X_row)
    contrib = contribution_table(vals[0], X_row.iloc[0], Xt.iloc[0])
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        st.plotly_chart(fig_contributions(contrib, model_name, base), width="stretch")
    with c2:
        st.markdown("**In plain English**")
        for line in plain_english(contrib, model_name):
            st.markdown(f"- {line}")
        st.caption(f"Blue bars push the estimate up, teal bars push it down. Units are {UNITS[model_name]}. "
                   "These describe how the *model* reacts to inputs. They do not show that a measurement causes diabetes.")

    st.subheader("Where your values sit in the training data")
    rows = []
    for f in FEATURES:
        if np.isnan(clean[f]):
            continue
        rows.append({"Feature": f, "Percentile": pred.percentile_of(reference, f, clean[f])})
    pc = pd.DataFrame(rows)
    fig = go.Figure(go.Bar(x=pc["Percentile"], y=pc["Feature"], orientation="h",
                           marker_color=COLORS["teal"], text=[f"{v:.0f}th" for v in pc["Percentile"]],
                           textposition="outside", cliponaxis=False))
    fig.add_vline(x=50, line_dash="dash", line_color="#9FB3B8", annotation_text="median")
    fig.update_layout(title="Percentile of each input within the 768 training records", xaxis_range=[0, 112],
                      xaxis_title="Percentile")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style_fig(fig, 380), width="stretch")

    with st.expander("Why can't this tool tell you more?"):
        st.markdown(
            "- The models learned from **768 women of one population**; they are not validated for anyone else.\n"
            "- Cross-validated ROC-AUC is around 0.8, meaning many people are ranked incorrectly.\n"
            "- Risk bands (under 30%, 30-60%, over 60%) are presentation choices, not clinical thresholds.\n"
            "- Real risk assessment uses many more factors and professional judgement.")
