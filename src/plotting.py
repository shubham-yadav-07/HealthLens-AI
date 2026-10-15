"""Shared Plotly styling so every chart matches the HealthLens AI brand."""
from __future__ import annotations

import plotly.graph_objects as go

from src.config import COLORS

FONT = "Source Sans 3, Segoe UI, Helvetica, Arial, sans-serif"


def style_fig(fig: go.Figure, height: int | None = 380, legend_below: bool = False) -> go.Figure:
    """Apply the brand look: white background, soft grid, brand fonts."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, size=13, color=COLORS["ink"]),
        title=dict(font=dict(family="Manrope, " + FONT, size=16, color=COLORS["ink"]),
                   x=0.0, xanchor="left"),
        margin=dict(l=10, r=10, t=50, b=10),
        colorway=[COLORS["teal"], COLORS["blue"], COLORS["green"], "#6D5BD0", COLORS["teal_light"]],
        hoverlabel=dict(font_family=FONT),
    )
    if height:
        fig.update_layout(height=height)
    if legend_below:
        fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0))
    fig.update_xaxes(gridcolor="#EEF3F3", zerolinecolor="#DCE8E8", linecolor="#DCE8E8")
    fig.update_yaxes(gridcolor="#EEF3F3", zerolinecolor="#DCE8E8", linecolor="#DCE8E8")
    return fig
