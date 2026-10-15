"""Insights and downloadable reports."""
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src import eda
from src import evaluate_models as ev
from src import explainability as xai
from src.config import MODEL_NAMES
from src.state import get_active_dataset, get_artifacts
from src.ui import info, note, page_header

page_header("Insights and Reports",
            "Plain-language findings from your data, plus downloads for your report or viva.")

df, ds_name, uploaded = get_active_dataset()
artifacts = get_artifacts()
metrics = ev.metrics_table(artifacts, 0.5)
meta = artifacts["meta"]

st.subheader("Data insights")
for f in eda.key_findings(df):
    st.markdown(f"- {f}")

st.subheader("Model insights (default Pima models, threshold 0.5)")
best_auc = metrics["ROC-AUC"].idxmax()
best_rec = metrics["Recall"].idxmax()
ci = artifacts["metrics"]
st.markdown(f"- Highest test ROC-AUC: **{best_auc}** ({metrics.loc[best_auc, 'ROC-AUC']:.3f}, 95% bootstrap CI "
            f"{ci.loc[best_auc, 'ROC-AUC CI low']:.2f} to {ci.loc[best_auc, 'ROC-AUC CI high']:.2f}).")
st.markdown(f"- Highest recall at threshold 0.5: **{best_rec}** ({metrics.loc[best_rec, 'Recall']:.2f}); "
            f"lowest: **{metrics['Recall'].idxmin()}** ({metrics['Recall'].min():.2f}). "
            "Missing positive cases is the main weakness at the default threshold.")
lo, hi = metrics["ROC-AUC"].min(), metrics["ROC-AUC"].max()
st.markdown(f"- Test ROC-AUC across models spans only {lo:.3f} to {hi:.3f}, "
            "within the confidence intervals, so the ranking should not be over-interpreted.")

note("<b>Exploratory vs validated.</b> Everything here is a description of one historical sample. "
     "Nothing has been validated prospectively, externally, or for fairness across groups.")

st.divider()
st.subheader("Downloads")


@st.cache_data(show_spinner=False)
def _eda_html(ds_hash: str, name: str, _df):
    return eda.build_html_report(_df, name)


ds_hash = str(pd.util.hash_pandas_object(df, index=False).sum())
with st.spinner("Building reports..."):
    html_report = _eda_html(ds_hash, ds_name, df)
    shap_imp = xai.importance_all_models(artifacts)
    thr_all = pd.concat(
        [ev.threshold_table(artifacts["y_test"].values, artifacts["probas"][m]).assign(Model=m) for m in MODEL_NAMES])
    summary = {"dataset": ds_name, "training_meta": meta,
               "best_params": {k: {a: str(b) for a, b in v.items()} for k, v in artifacts["best_params"].items()},
               "test_metrics_threshold_0.5": metrics.round(4).to_dict(orient="index")}
    summary_json = json.dumps(summary, indent=2, default=str)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("eda_report.html", html_report)
        z.writestr("dataset_used.csv", df.to_csv(index=False))
        z.writestr("model_comparison.csv", metrics.round(4).to_csv())
        z.writestr("cross_validation_folds.csv", artifacts["cv_results"].round(4).to_csv(index=False))
        z.writestr("threshold_analysis_all_models.csv", thr_all.round(4).to_csv(index=False))
        z.writestr("shap_importance_all_models.csv", shap_imp.round(4).to_csv())
        z.writestr("summary.json", summary_json)
        z.writestr("group_comparison.csv", eda.group_comparison(df).to_csv(index=False))
        z.writestr("missing_values.csv", eda.missing_summary(df).to_csv(index=False))

c1, c2, c3 = st.columns(3)
c1.download_button("EDA report (HTML)", html_report.encode("utf-8"), "eda_report.html", "text/html")
c2.download_button("Model comparison (CSV)", metrics.round(4).to_csv().encode("utf-8"), "model_comparison.csv", "text/csv")
c3.download_button("Cross-validation folds (CSV)", artifacts["cv_results"].round(4).to_csv(index=False).encode("utf-8"),
                   "cross_validation_folds.csv", "text/csv")
c4, c5, c6 = st.columns(3)
c4.download_button("Threshold analysis (CSV)", thr_all.round(4).to_csv(index=False).encode("utf-8"),
                   "threshold_analysis_all_models.csv", "text/csv")
c5.download_button("SHAP importance (CSV)", shap_imp.round(4).to_csv().encode("utf-8"),
                   "shap_importance_all_models.csv", "text/csv")
c6.download_button("Everything (ZIP)", buffer.getvalue(), "healthlens_analysis_bundle.zip", "application/zip")
info("The HTML report is self-contained (charts load Plotly from a CDN). Open it in a browser and use Print to save as PDF.")
