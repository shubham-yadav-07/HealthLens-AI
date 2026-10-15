"""
Model evaluation: metrics, confidence intervals, ROC, confusion matrices,
threshold analysis, age-group performance, and the Plotly figures for them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from src.config import COLORS, MODEL_COLORS, MODEL_NAMES, RANDOM_STATE
from src.plotting import style_fig


# ----------------------------------------------------------------------------
# Numbers
# ----------------------------------------------------------------------------
def compute_metrics(y_true, proba, threshold: float = 0.5, with_ci: bool = False,
                    n_boot: int = 1000) -> dict:
    """Accuracy / Precision / Recall / F1 / ROC-AUC (+ specificity) at a threshold.

    ``with_ci`` adds a bootstrap 95% confidence interval for ROC-AUC. With a
    small test set (~150 rows) these intervals are wide - which is exactly why a
    single test-set number should never be over-interpreted.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, proba),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }
    if with_ci:
        rng = np.random.default_rng(RANDOM_STATE)
        aucs = []
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            aucs.append(roc_auc_score(y_true[idx], proba[idx]))
        out["ROC-AUC CI low"], out["ROC-AUC CI high"] = np.percentile(aucs, [2.5, 97.5])
    return out


def metrics_table(results: dict, threshold: float = 0.5) -> pd.DataFrame:
    """Test-set metrics for every model at the chosen decision threshold."""
    y = results["y_test"].values
    rows = []
    for name in MODEL_NAMES:
        m = compute_metrics(y, results["probas"][name], threshold)
        m["Model"] = name
        rows.append(m)
    return pd.DataFrame(rows).set_index("Model")


def threshold_table(y_true, proba, thresholds=None) -> pd.DataFrame:
    """Metrics for a range of decision thresholds (precision/recall trade-off)."""
    thresholds = thresholds if thresholds is not None else np.round(np.arange(0.10, 0.91, 0.05), 2)
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, proba, float(t))
        rows.append({"Threshold": float(t), **{k: m[k] for k in
                     ["Accuracy", "Precision", "Recall", "Specificity", "F1", "TP", "FP", "TN", "FN"]}})
    return pd.DataFrame(rows)


def age_group_performance(X_test: pd.DataFrame, y_test, proba, threshold: float = 0.5,
                          min_group: int = 20) -> pd.DataFrame:
    """Performance by age band.

    Age is the ONLY demographic variable in the Pima data (all participants are
    women of one population), so this is the only subgroup check possible.
    Small groups give unreliable numbers and are flagged.
    """
    bands = pd.cut(X_test["Age"], bins=[20, 29, 39, 49, 120],
                   labels=["21-29", "30-39", "40-49", "50+"])
    y = np.asarray(y_test)
    proba = np.asarray(proba)
    rows = []
    for band in bands.cat.categories:
        mask = (bands == band).values
        n = int(mask.sum())
        if n == 0:
            continue
        pos = int(y[mask].sum())
        row = {"Age group": band, "n": n, "Diabetes recorded": pos,
               "Reliable?": "yes" if (n >= min_group and 0 < pos < n) else "too few cases"}
        if 0 < pos < n:
            m = compute_metrics(y[mask], proba[mask], threshold)
            row.update({k: m[k] for k in ["Accuracy", "Recall", "Specificity", "ROC-AUC"]})
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_metric_bars(metrics: pd.DataFrame, metric_names=None) -> go.Figure:
    metric_names = metric_names or ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    long = metrics[metric_names].reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Metric", y="Score", color="Model", barmode="group",
                 color_discrete_map=MODEL_COLORS, range_y=[0, 1], text_auto=".2f")
    fig.update_traces(textfont_size=10, textposition="outside", cliponaxis=False)
    fig.update_layout(title="Test-set metrics by model", yaxis_title=None, xaxis_title=None)
    return style_fig(fig, 400, legend_below=True)


def fig_roc(results: dict, models=None) -> go.Figure:
    fig = go.Figure()
    y = results["y_test"].values
    for name in (models or MODEL_NAMES):
        fpr, tpr, _ = roc_curve(y, results["probas"][name])
        auc = roc_auc_score(y, results["probas"][name])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})",
                                 line=dict(color=MODEL_COLORS[name], width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guessing",
                             line=dict(color="#9FB3B8", dash="dash")))
    fig.update_layout(title="ROC curves (test set)", xaxis_title="False positive rate (1 - specificity)",
                      yaxis_title="True positive rate (recall)")
    fig.update_yaxes(scaleanchor=None)
    return style_fig(fig, 430, legend_below=True)


def fig_confusion(cm_row: dict, title: str) -> go.Figure:
    z = [[cm_row["TN"], cm_row["FP"]], [cm_row["FN"], cm_row["TP"]]]
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    text = [[f"{labels[i][j]}<br><b>{z[i][j]}</b>" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted: no diabetes", "Predicted: diabetes"],
        y=["Actual: no diabetes", "Actual: diabetes"], text=text, texttemplate="%{text}",
        colorscale=[[0, "#F1F8F7"], [1, COLORS["teal"]]], showscale=False,
        hovertemplate="%{y}<br>%{x}<br>count=%{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title)
    return style_fig(fig, 320)


def fig_threshold(tt: pd.DataFrame, title: str = "Precision, recall and specificity vs. threshold") -> go.Figure:
    fig = go.Figure()
    for col, color in [("Precision", COLORS["blue"]), ("Recall", COLORS["teal"]),
                       ("Specificity", COLORS["green"]), ("F1", "#6D5BD0")]:
        fig.add_trace(go.Scatter(x=tt["Threshold"], y=tt[col], mode="lines+markers", name=col,
                                 line=dict(color=color, width=2.5), marker=dict(size=5)))
    fig.update_layout(title=title, xaxis_title="Decision threshold", yaxis_title=None, yaxis_range=[0, 1.02])
    return style_fig(fig, 400, legend_below=True)


def fig_cv_box(cv_results: pd.DataFrame, metric: str = "roc_auc") -> go.Figure:
    fig = px.box(cv_results, x="Model", y=metric, color="Model", points="all",
                 color_discrete_map=MODEL_COLORS)
    fig.update_layout(title=f"Cross-validation spread: {metric} across folds",
                      showlegend=False, xaxis_title=None, yaxis_title=metric)
    return style_fig(fig, 380)


def cv_summary(cv_results: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- std of each metric over the CV folds."""
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    mean = cv_results.groupby("Model")[cols].mean()
    std = cv_results.groupby("Model")[cols].std(ddof=1)
    out = pd.DataFrame(index=mean.index)
    for c in cols:
        out[c] = [f"{m:.3f} ± {s:.3f}" for m, s in zip(mean[c], std[c])]
    return out.reindex([m for m in MODEL_NAMES if m in out.index])
