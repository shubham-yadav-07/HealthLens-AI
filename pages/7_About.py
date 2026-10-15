"""About the project: purpose, method, stack, ethics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.state import get_artifacts
from src.ui import card, disclaimer, note, page_header

page_header("About the Project",
            "HealthLens AI is an Honours-level data science project on explainable health risk modelling.")
disclaimer()
st.write("")

meta = get_artifacts()["meta"]

c1, c2, c3 = st.columns(3, gap="medium")
c1.markdown(card("Objective", "Estimate diabetes-associated risk from routine measurements with real machine learning, "
                 "and make the reasons visible."), unsafe_allow_html=True)
c2.markdown(card("Method", "Stratified split, leakage-free pipelines, cross-validated tuning, four algorithms, "
                 "bootstrap confidence intervals and SHAP."), unsafe_allow_html=True)
c3.markdown(card("Principle", "Scientific transparency over spectacle: limitations are stated beside every result.",
                 tinted=True), unsafe_allow_html=True)

st.markdown("### Pipeline")
st.code(
    "CSV (768 records)\n"
    "  -> validation (columns, types, binary target)\n"
    "  -> stratified 80/20 split (seed 42)\n"
    "  -> Pipeline: zeros->NaN -> median imputer -> (scaler) -> model     <- fitted on training rows only\n"
    "  -> GridSearchCV, 5-fold stratified, scoring = ROC-AUC\n"
    "  -> nested CV for an honest estimate\n"
    "  -> one-time evaluation on the untouched test set\n"
    "  -> SHAP explanations, reports, Streamlit interface", language="text")

st.markdown("### Technology")
st.markdown("Python, Pandas, NumPy, SciPy, scikit-learn, XGBoost, SHAP, Plotly, Matplotlib, Seaborn (available for notebooks), "
            "Streamlit, Joblib.")
v = meta["versions"]
st.caption("Models in this deployment were trained with " + ", ".join(f"{k} {x}" for k, x in v.items()) + ".")

st.markdown("### Ethics and limitations")
st.markdown(
    "- Not a medical device. Never use it to decide on diagnosis, treatment or medication.\n"
    "- Trained on one 1980s sample of Pima women; not valid for other groups.\n"
    "- Estimated probabilities are model outputs, not personal risk.\n"
    "- SHAP and correlations describe the model and the sample, not causes.\n"
    "- Fairness cannot be assessed because the data lack the required demographic variables.\n"
    "- No personal data are stored; uploaded files stay in your session memory only.")

st.markdown("### References")
st.markdown(
    "- Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., Johannes, R.S. (1988). Using the ADAP learning algorithm "
    "to forecast the onset of diabetes mellitus. *Proc. Annual Symposium on Computer Application in Medical Care*, 261-265.\n"
    "- Lundberg, S.M., Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*.\n"
    "- Chen, T., Guestrin, C. (2016). XGBoost: a scalable tree boosting system. *KDD '16*, 785-794.\n"
    "- Breiman, L. (2001). Random forests. *Machine Learning, 45*, 5-32.\n"
    "- Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *JMLR, 12*, 2825-2830.")
note("Full report, methodology and viva questions are in the <code>docs/</code> folder of the repository.")
