"""Dataset Explorer: preview, upload, validate and download the dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src import eda
from src.config import ALL_COLUMNS, FEATURE_INFO, TARGET, ZERO_AS_MISSING
from src.data_loader import load_dataset_with_report
from src.state import get_active_dataset, reset_upload
from src.ui import info, kpi_row, note, page_header

page_header("Dataset Explorer",
            "Inspect the records, check data quality, or bring your own compatible CSV.")

df, name, uploaded = get_active_dataset()
ov = eda.overview(df)

kpi_row([
    ("Rows", f"{ov['rows']:,}", name, ""),
    ("Columns", str(ov["columns"]), f"{ov['features']} features + target", "blue"),
    ("Duplicate rows", str(ov["duplicates"]), "exact copies", "green"),
    ("Missing / impossible zeros", f"{ov['missing_cells']:,}", "cells across features", ""),
])
st.write("")

tab_data, tab_dict, tab_upload, tab_source = st.tabs(
    ["Browse records", "Column dictionary", "Upload your own CSV", "Source and limitations"])

with tab_data:
    c1, c2, c3 = st.columns([1, 1, 1])
    outcome = c1.selectbox("Outcome", ["All", "Diabetes recorded (1)", "No diabetes recorded (0)"])
    hide_zero = c2.checkbox("Show impossible zeros as blank", value=False,
                            help="Replaces 0 with a blank in Glucose, BloodPressure, SkinThickness, Insulin and BMI.")
    n_show = c3.slider("Rows to show", 10, min(500, len(df)), 50, step=10)
    view = eda.clean_view(df) if hide_zero else df.copy()
    if outcome.endswith("(1)"):
        view = view[view[TARGET] == 1]
    elif outcome.endswith("(0)"):
        view = view[view[TARGET] == 0]
    st.dataframe(view.head(n_show), width="stretch", height=380)
    st.caption(f"Showing {min(n_show, len(view))} of {len(view):,} matching rows.")
    st.download_button("Download this dataset as CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="healthlens_dataset.csv", mime="text/csv")

with tab_dict:
    rows = []
    for col in ALL_COLUMNS:
        if col == TARGET:
            rows.append({"Column": col, "Meaning": "1 = diabetes recorded, 0 = not recorded", "Unit": "class",
                         "Zero means missing?": "no"})
        else:
            i = FEATURE_INFO[col]
            rows.append({"Column": col, "Meaning": i["help"], "Unit": i["unit"],
                         "Zero means missing?": "yes" if col in ZERO_AS_MISSING else "no"})
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.subheader("Data types and uniqueness")
    st.dataframe(eda.dtypes_table(df), width="stretch", hide_index=True)

with tab_upload:
    st.markdown("Upload a CSV with the columns below. Names are matched without regard to case, spaces or "
                "underscores, and files without a header row are accepted if they have 9 numeric columns in this order.")
    st.code(", ".join(ALL_COLUMNS), language="text")
    file = st.file_uploader("Choose a CSV file", type=["csv"])
    if file is not None:
        with st.spinner("Validating file..."):
            result = load_dataset_with_report(file)
        if result.ok:
            for w in result.warnings:
                st.warning(w)
            st.success(f"File accepted: {len(result.df):,} rows, {result.df.shape[1]} columns.")
            if st.button("Use this dataset across the app", type="primary"):
                st.session_state["uploaded_df"] = result.df
                st.session_state["uploaded_name"] = file.name
                st.session_state.pop("custom_results", None)
                st.rerun()
        else:
            for e in result.errors:
                st.error(e)
            for w in result.warnings:
                st.warning(w)
            info("Fix the problems above and upload again. Nothing was changed.")
    if uploaded:
        st.write("")
        info(f"Currently using your upload: <b>{name}</b>. Dataset Explorer, EDA and Reports follow it. "
             "To compare models on it, open Model Comparison and train in-session.")
        if st.button("Go back to the default Pima dataset"):
            reset_upload()
            st.rerun()

with tab_source:
    st.markdown(
        """
**Origin.** The Pima Indians Diabetes Database was collected by the US National Institute of Diabetes and
Digestive and Kidney Diseases and described by Smith et al. (1988). Public copies exist on the UCI
Machine Learning Repository and Kaggle. This project downloads a public mirror of the same 768 records
(`python -m src.data_loader`).

**Target.** `Outcome` is 1 when the participant was recorded as developing diabetes within five years of the baseline exam.

**Intended use.** A benchmark for teaching and for comparing classification methods.

**Limitations you should state in your viva**
- **One population.** All 768 participants are women, aged 21 or over, of Pima heritage. Predictions for men, children or other populations are not supported by this data.
- **Small sample.** Test-set metrics from about 150 records have wide confidence intervals.
- **Hidden missing values.** Zeros stand in for unmeasured values, most severely in Insulin and SkinThickness.
- **Old data.** Collected in the 1980s, so clinical practice and diagnostic criteria have changed.
- **No demographic variety.** Only age is available for subgroup checks, so fairness cannot be assessed.
""")
    note("This app never invents or augments records. Every statistic comes from the file you see here.")
