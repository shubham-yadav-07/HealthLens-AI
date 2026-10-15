"""
Exploratory Data Analysis helpers.

Every number shown in the app is computed here from the dataset that is
currently loaded - nothing is hard-coded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from src.config import (
    CLASS_LABELS, COLORS, FEATURES, FEATURE_INFO, TARGET, ZERO_AS_MISSING,
)
from src.plotting import style_fig
from src.preprocessing import missing_summary

CLASS_COLOR_MAP = {CLASS_LABELS[0]: COLORS["class0"], CLASS_LABELS[1]: COLORS["class1"]}


# ----------------------------------------------------------------------------
# Data views
# ----------------------------------------------------------------------------
def clean_view(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df where impossible zeros are replaced by NaN (for honest plots)."""
    out = df.copy()
    for col in ZERO_AS_MISSING:
        out[col] = out[col].replace(0, np.nan)
    return out


def with_class_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Group"] = out[TARGET].map(CLASS_LABELS)
    return out


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def overview(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "features": len(FEATURES),
        "positive_rate": float(df[TARGET].mean()),
        "duplicates": int(df.duplicated().sum()),
        "missing_cells": int(missing_summary(df)["Total missing"].sum()),
    }


def dtypes_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Column": df.columns,
        "Type": [str(t) for t in df.dtypes],
        "Unique values": [int(df[c].nunique()) for c in df.columns],
        "Role": ["Target" if c == TARGET else "Feature" for c in df.columns],
    })


def describe_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.describe().T
    d["skew"] = df.skew()
    return d.round(3)


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    counts = df[TARGET].value_counts().sort_index()
    return pd.DataFrame({
        "Class": [CLASS_LABELS[i] for i in counts.index],
        "Count": counts.values,
        "Share %": (100 * counts / counts.sum()).round(1).values,
    })


def imbalance_ratio(df: pd.DataFrame) -> float:
    counts = df[TARGET].value_counts()
    return float(counts.max() / counts.min())


def duplicate_report(df: pd.DataFrame) -> dict:
    dup_mask = df.duplicated(keep=False)
    return {"n_duplicates": int(df.duplicated().sum()), "rows_involved": df[dup_mask]}


def outlier_table(df: pd.DataFrame) -> pd.DataFrame:
    """IQR rule: values beyond 1.5 * IQR outside Q1/Q3 (computed on the cleaned view)."""
    data = clean_view(df)
    rows = []
    for col in FEATURES:
        s = data[col].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"Feature": col, "Lower fence": round(lo, 3), "Upper fence": round(hi, 3),
                     "Outliers": n_out, "Outlier %": round(100 * n_out / len(s), 2),
                     "Max value": round(s.max(), 3)})
    return pd.DataFrame(rows)


def group_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare each feature between the two outcome groups.

    Mann-Whitney U (no normality assumption) + Cohen's d effect size.
    p-values are corrected with Bonferroni because 8 tests are run.
    These are EXPLORATORY associations - not proof of cause.
    """
    data = clean_view(df)
    rows = []
    for col in FEATURES:
        a = data.loc[data[TARGET] == 0, col].dropna()
        b = data.loc[data[TARGET] == 1, col].dropna()
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        d = (b.mean() - a.mean()) / pooled if pooled > 0 else np.nan
        rows.append({
            "Feature": col,
            "Mean (no diabetes)": round(a.mean(), 3),
            "Mean (diabetes)": round(b.mean(), 3),
            "Median (no diabetes)": round(a.median(), 3),
            "Median (diabetes)": round(b.median(), 3),
            "Cohen's d": round(d, 3),
            "p-value": p,
            "p (Bonferroni)": min(p * len(FEATURES), 1.0),
            "Significant (0.05)": "yes" if min(p * len(FEATURES), 1.0) < 0.05 else "no",
        })
    return pd.DataFrame(rows).sort_values("Cohen's d", key=np.abs, ascending=False).reset_index(drop=True)


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return clean_view(df)[FEATURES + [TARGET]].corr(method=method)


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_class_distribution(df: pd.DataFrame) -> go.Figure:
    cb = class_balance(df)
    fig = px.bar(cb, x="Class", y="Count", color="Class", color_discrete_map=CLASS_COLOR_MAP,
                 text=[f"{c} ({s}%)" for c, s in zip(cb["Count"], cb["Share %"])])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(title="Class distribution", showlegend=False, xaxis_title=None)
    return style_fig(fig, 340)


def fig_missing(df: pd.DataFrame) -> go.Figure:
    ms = missing_summary(df)
    fig = px.bar(ms, x="Feature", y="Missing %", text="Missing %",
                 color_discrete_sequence=[COLORS["teal"]])
    fig.update_traces(texttemplate="%{text}%", textposition="outside", cliponaxis=False)
    fig.update_layout(title="Missing values per feature (NaN + impossible zeros)", xaxis_title=None)
    return style_fig(fig, 340)


def fig_histogram(df: pd.DataFrame, feature: str, use_clean: bool = True, show_classes: bool = True) -> go.Figure:
    data = with_class_names(clean_view(df) if use_clean else df)
    fig = px.histogram(data, x=feature, color="Group" if show_classes else None, nbins=30,
                       barmode="overlay", opacity=0.65, marginal="box",
                       color_discrete_map=CLASS_COLOR_MAP,
                       color_discrete_sequence=[COLORS["teal"]])
    fig.update_layout(title=f"Distribution of {FEATURE_INFO[feature]['label'].lower()}",
                      xaxis_title=f"{feature} ({FEATURE_INFO[feature]['unit']})", yaxis_title="Count")
    return style_fig(fig, 420, legend_below=True)


def fig_box_by_class(df: pd.DataFrame, feature: str, use_clean: bool = True) -> go.Figure:
    data = with_class_names(clean_view(df) if use_clean else df)
    fig = px.box(data, x="Group", y=feature, color="Group", points="outliers",
                 color_discrete_map=CLASS_COLOR_MAP)
    fig.update_layout(title=f"{feature} by outcome", showlegend=False, xaxis_title=None)
    return style_fig(fig, 380)


def fig_violin_grid(df: pd.DataFrame) -> go.Figure:
    """Small multiples: one violin per feature, split by class (standardised)."""
    data = clean_view(df)
    z = (data[FEATURES] - data[FEATURES].mean()) / data[FEATURES].std()
    z[TARGET] = data[TARGET]
    long = z.melt(id_vars=TARGET, var_name="Feature", value_name="Standardised value")
    long["Group"] = long[TARGET].map(CLASS_LABELS)
    fig = px.violin(long, x="Feature", y="Standardised value", color="Group", box=True,
                    color_discrete_map=CLASS_COLOR_MAP)
    fig.update_layout(title="All features by outcome (standardised so they share one axis)", xaxis_title=None)
    return style_fig(fig, 450, legend_below=True)


def fig_correlation(df: pd.DataFrame, method: str = "pearson") -> go.Figure:
    corr = correlation_matrix(df, method)
    fig = px.imshow(corr.round(2), text_auto=True, zmin=-1, zmax=1, aspect="auto",
                    color_continuous_scale=[[0, COLORS["blue"]], [0.5, "#FFFFFF"], [1, COLORS["teal"]]])
    fig.update_layout(title=f"{method.title()} correlation heatmap (zeros treated as missing)")
    return style_fig(fig, 520)


def fig_scatter(df: pd.DataFrame, x: str, y: str, use_clean: bool = True) -> go.Figure:
    data = with_class_names(clean_view(df) if use_clean else df)
    fig = px.scatter(data, x=x, y=y, color="Group", opacity=0.7, color_discrete_map=CLASS_COLOR_MAP,
                     marginal_x="histogram", marginal_y="histogram")
    fig.update_layout(title=f"{y} vs {x}")
    return style_fig(fig, 520, legend_below=True)


def fig_effect_sizes(gc: pd.DataFrame) -> go.Figure:
    g = gc.sort_values("Cohen's d")
    fig = go.Figure(go.Bar(x=g["Cohen's d"], y=g["Feature"], orientation="h",
                           marker_color=[COLORS["blue"] if v > 0 else COLORS["teal"] for v in g["Cohen's d"]]))
    fig.update_layout(title="Effect size of group difference (Cohen's d)", xaxis_title="Cohen's d (diabetes − no diabetes)")
    return style_fig(fig, 380)


# ----------------------------------------------------------------------------
# Auto-generated key findings (computed from the loaded data)
# ----------------------------------------------------------------------------
def key_findings(df: pd.DataFrame) -> list[str]:
    ov = overview(df)
    ms = missing_summary(df)
    gc = group_comparison(df)
    corr = correlation_matrix(df)[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
    out_tab = outlier_table(df)

    findings = [
        f"The dataset has **{ov['rows']:,} rows** and **{len(FEATURES)} features**; "
        f"**{ov['positive_rate'] * 100:.1f}%** of records have a recorded diabetes outcome "
        f"(imbalance ratio {imbalance_ratio(df):.2f} : 1).",
    ]
    worst = ms.sort_values("Total missing", ascending=False).head(3)
    worst = worst[worst["Total missing"] > 0]
    if len(worst):
        parts = ", ".join(f"{r.Feature} ({r['Missing %']}%)" for _, r in worst.iterrows())
        findings.append(f"Hidden missing data: impossible zeros mean the most incomplete features are {parts}.")
    else:
        findings.append("No missing or impossible-zero values were detected.")
    findings.append(f"Duplicate rows: **{ov['duplicates']}**.")
    top = gc.iloc[0]
    d_value = top["Cohen's d"]
    findings.append(
        f"The largest group difference is in **{top['Feature']}** (Cohen's d = {d_value:.2f}); "
        f"median {top['Median (no diabetes)']} (no diabetes) vs {top['Median (diabetes)']} (diabetes)."
    )
    sig = gc[gc["Significant (0.05)"] == "yes"]["Feature"].tolist()
    findings.append(
        f"{len(sig)} of {len(FEATURES)} features differ significantly between groups after Bonferroni correction: "
        + (", ".join(sig) if sig else "none") + ". This is an association, not proof of cause."
    )
    findings.append(
        f"Strongest linear correlation with the outcome: **{corr.index[0]}** (r = {corr.iloc[0]:.2f}), "
        f"then **{corr.index[1]}** (r = {corr.iloc[1]:.2f})."
    )
    worst_out = out_tab.sort_values("Outliers", ascending=False).iloc[0]
    findings.append(
        f"Most IQR outliers: **{worst_out['Feature']}** with {worst_out['Outliers']} values "
        f"({worst_out['Outlier %']}%); outliers were kept because they may be genuine clinical values."
    )
    return findings


# ----------------------------------------------------------------------------
# Downloadable report
# ----------------------------------------------------------------------------
def build_html_report(df: pd.DataFrame, dataset_name: str = "dataset") -> str:
    """Self-contained HTML EDA report (open in any browser, print to PDF)."""
    ov = overview(df)
    parts = [
        "<html><head><meta charset='utf-8'><title>HealthLens AI - EDA report</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:980px;margin:30px auto;color:#0B2530;line-height:1.5}"
        "h1,h2{color:#0E7C86}table{border-collapse:collapse;margin:10px 0;font-size:13px}"
        "td,th{border:1px solid #DCE8E8;padding:4px 8px;text-align:right}th{background:#F1F8F7}"
        ".note{background:#FFF8E6;border-left:4px solid #B7791F;padding:8px 12px}</style></head><body>",
        "<h1>HealthLens AI - Exploratory Data Analysis report</h1>",
        f"<p>Dataset: <b>{dataset_name}</b> &middot; {ov['rows']} rows &middot; {len(FEATURES)} features</p>",
        "<p class='note'>Educational and research prototype. Findings are exploratory associations, "
        "not diagnoses and not proof of causation.</p>",
        "<h2>Key findings</h2><ul>",
    ]
    for f in key_findings(df):
        parts.append("<li>" + f.replace("**", "") + "</li>")
    parts.append("</ul>")
    for title, table in [
        ("Descriptive statistics", describe_table(df)),
        ("Missing values (NaN + impossible zeros)", missing_summary(df)),
        ("Class balance", class_balance(df)),
        ("Outliers (IQR rule)", outlier_table(df)),
        ("Group comparison (Mann-Whitney U, Cohen's d)", group_comparison(df)),
        ("Correlation matrix (Pearson)", correlation_matrix(df).round(3).reset_index()),
    ]:
        parts.append(f"<h2>{title}</h2>" + table.to_html(index=not isinstance(table.index, pd.RangeIndex) and title.startswith("Desc"), border=0))
    first = True
    for title, fig in [
        ("Class distribution", fig_class_distribution(df)),
        ("Missing values", fig_missing(df)),
        ("Correlation heatmap", fig_correlation(df)),
        ("Features by outcome", fig_violin_grid(df)),
    ]:
        parts.append(f"<h2>{title}</h2>" + fig.to_html(full_html=False,
                     include_plotlyjs="cdn" if first else False))
        first = False
    parts.append("</body></html>")
    return "\n".join(parts)
