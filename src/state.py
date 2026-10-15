"""
Streamlit-side state and caching.

* get_active_dataset()  -> the dataset all pages should analyse
                           (default Pima data, or the CSV the user uploaded)
* get_artifacts()       -> trained models + results (loaded from models/, or
                           trained automatically on first run if missing/incompatible)
"""
from __future__ import annotations

import streamlit as st

from src.config import DEFAULT_DATASET
from src.data_loader import DatasetError, load_default_dataset
from src.train_models import load_results, save_results, train_all


@st.cache_data(show_spinner=False)
def load_default_cached():
    return load_default_dataset()


def get_active_dataset():
    """Return (dataframe, display_name, is_uploaded)."""
    if st.session_state.get("uploaded_df") is not None:
        return st.session_state["uploaded_df"], st.session_state.get("uploaded_name", "Uploaded CSV"), True
    try:
        return load_default_cached(), "Pima Indians Diabetes (default)", False
    except DatasetError as exc:
        st.error(f"The default dataset could not be loaded. {exc}")
        st.code("python -m src.data_loader   # downloads the dataset into data/", language="bash")
        st.stop()


@st.cache_resource(show_spinner=False)
def get_artifacts():
    """Load saved models; if they are missing or were saved with an incompatible
    library version, retrain quickly (no nested CV) and save again."""
    try:
        return load_results()
    except Exception:  # missing file, version mismatch, corrupted pickle ...
        with st.spinner("Preparing models for the first time (about a minute)..."):
            df = load_default_cached()
            results = train_all(df, tune=True, nested_cv=False, save=False)
            try:
                save_results(results)
            except OSError:
                pass  # read-only filesystem is fine; keep results in memory
            return results


def reset_upload():
    for key in ("uploaded_df", "uploaded_name", "custom_results"):
        st.session_state.pop(key, None)
