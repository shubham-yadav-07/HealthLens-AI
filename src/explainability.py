"""
Explainable AI with SHAP.

SHAP assigns every feature a contribution (a "SHAP value") to one prediction.
Positive value -> pushes the model output UP (towards "diabetes recorded"),
negative value -> pushes it DOWN. The contributions add up to the difference
between this prediction and the average prediction.

Units depend on the model:
  * Logistic Regression and XGBoost -> log-odds (a technical scale)
  * Decision Tree and Random Forest  -> probability (percentage points / 100)

Important: SHAP explains the MODEL, not the world. A large SHAP value means the
model relies on that feature, NOT that the feature causes diabetes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from src.config import COLORS, FEATURES, FEATURE_INFO, MODEL_COLORS, MODEL_NAMES
from src.plotting import style_fig
from src.preprocessing import transform_features

UNITS = {
    "Logistic Regression": "log-odds",
    "XGBoost": "log-odds",
    "Decision Tree": "probability",
    "Random Forest": "probability",
}


def _positive_class(values):
    """Different SHAP/model versions return (n, f), (n, f, 2) or a list of 2 arrays.
    Always return the array for class 1 with shape (n, f)."""
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def _expected_value(explainer):
    ev = explainer.expected_value
    ev = np.asarray(ev).ravel()
    return float(ev[-1])  # class-1 value when two are given


def compute_shap(model_name: str, pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame):
    """Return (shap_values [n x f], base_value, transformed X_explain).

    ``X_background`` is used by the linear explainer only (training data).
    All data are passed through the fitted preprocessing steps first, so SHAP
    sees exactly what the model sees.
    """
    model = pipeline[-1]
    Xt = transform_features(pipeline, X_explain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model_name == "Logistic Regression":
            background = transform_features(pipeline, X_background)
            masker = shap.maskers.Independent(background, max_samples=len(background))
            explainer = shap.LinearExplainer(model, masker)
            values = _positive_class(explainer.shap_values(Xt))
        else:
            explainer = shap.TreeExplainer(model)
            values = _positive_class(explainer.shap_values(Xt))
    return values, _expected_value(explainer), Xt


def global_importance(values: np.ndarray) -> pd.DataFrame:
    """Mean absolute SHAP value per feature (global importance)."""
    imp = pd.DataFrame({"Feature": FEATURES, "Mean |SHAP|": np.abs(values).mean(axis=0)})
    imp["Share %"] = 100 * imp["Mean |SHAP|"] / imp["Mean |SHAP|"].sum()
    return imp.sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)


def importance_all_models(results: dict) -> pd.DataFrame:
    """Normalised global importance (share %) of each feature for every model."""
    frames = {}
    for name in MODEL_NAMES:
        values, _, _ = compute_shap(name, results["models"][name], results["X_train"], results["X_test"])
        imp = global_importance(values).set_index("Feature")["Share %"]
        frames[name] = imp.reindex(FEATURES)
    return pd.DataFrame(frames)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_global_importance(imp: pd.DataFrame, model_name: str) -> go.Figure:
    d = imp.sort_values("Mean |SHAP|")
    fig = go.Figure(go.Bar(x=d["Mean |SHAP|"], y=d["Feature"], orientation="h",
                           marker_color=MODEL_COLORS.get(model_name, COLORS["teal"]),
                           text=[f"{v:.1f}%" for v in d["Share %"]], textposition="outside",
                           cliponaxis=False))
    fig.update_layout(title=f"Global feature importance - {model_name}",
                      xaxis_title=f"Mean |SHAP value| ({UNITS[model_name]})")
    return style_fig(fig, 400)


def fig_beeswarm(values: np.ndarray, Xt: pd.DataFrame, model_name: str):
    """Classic SHAP summary (beeswarm) plot as a matplotlib figure."""
    import matplotlib.pyplot as plt
    plt.close("all")
    explanation = shap.Explanation(values=values, data=Xt.values, feature_names=list(Xt.columns))
    shap.plots.beeswarm(explanation, show=False, max_display=len(FEATURES), plot_size=(9, 5))
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_xlabel(f"SHAP value  (impact on model output, {UNITS[model_name]})")
    fig.tight_layout()
    return fig


def fig_importance_comparison(imp_all: pd.DataFrame) -> go.Figure:
    order = imp_all.mean(axis=1).sort_values().index
    fig = go.Figure()
    for name in imp_all.columns:
        fig.add_trace(go.Bar(y=imp_all.loc[order].index, x=imp_all.loc[order, name], name=name,
                             orientation="h", marker_color=MODEL_COLORS[name]))
    fig.update_layout(barmode="group", title="Feature importance across models (share of total SHAP, %)",
                      xaxis_title="Share of total mean |SHAP| (%)")
    return style_fig(fig, 520, legend_below=True)


def fig_contributions(contrib: pd.DataFrame, model_name: str, base: float, title: str | None = None) -> go.Figure:
    """Horizontal bar of per-feature contributions for ONE prediction."""
    d = contrib.sort_values("SHAP value")
    colors = [COLORS["blue"] if v > 0 else COLORS["teal_light"] for v in d["SHAP value"]]
    labels = [f"{f} = {v:g}" for f, v in zip(d["Feature"], d["Value used"])]
    fig = go.Figure(go.Bar(x=d["SHAP value"], y=labels, orientation="h", marker_color=colors,
                           text=[f"{v:+.3f}" for v in d["SHAP value"]], textposition="outside",
                           cliponaxis=False))
    fig.update_layout(title=title or "What pushed this prediction up or down",
                      xaxis_title=f"Contribution ({UNITS[model_name]}; base value {base:.3f})")
    fig.add_vline(x=0, line_color="#9FB3B8")
    return style_fig(fig, 420)


# ----------------------------------------------------------------------------
# Plain-English explanation for ONE prediction
# ----------------------------------------------------------------------------
def contribution_table(values_row: np.ndarray, x_row_raw: pd.Series, x_row_used: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "Feature": FEATURES,
        "Value entered": [x_row_raw[f] for f in FEATURES],
        "Value used": [round(float(x_row_used[f]), 3) for f in FEATURES],
        "SHAP value": values_row,
    })


def plain_english(contrib: pd.DataFrame, model_name: str, top_n: int = 3) -> list[str]:
    """Turn SHAP values into short sentences a non-expert can read."""
    lines = []
    ordered = contrib.reindex(contrib["SHAP value"].abs().sort_values(ascending=False).index)
    for _, r in ordered.head(top_n).iterrows():
        label = FEATURE_INFO[r["Feature"]]["label"].lower()
        unit = FEATURE_INFO[r["Feature"]]["unit"]
        direction = "raised" if r["SHAP value"] > 0 else "lowered"
        was_missing = pd.isna(r["Value entered"]) or r["Value entered"] == 0 and r["Feature"] in (
            "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")
        used = f"{r['Value used']:g} {unit}"
        extra = " (a typical value was filled in because it was not provided)" if was_missing else ""
        lines.append(f"**{label.capitalize()}** ({used}{extra}) {direction} the model's estimate "
                     f"({r['SHAP value']:+.3f} {UNITS[model_name]}).")
    return lines
