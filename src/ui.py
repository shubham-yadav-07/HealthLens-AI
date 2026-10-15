"""
Shared Streamlit UI helpers: brand CSS, page headers, KPI cards, disclaimers.
Keeping the look in ONE file means every page stays consistent.
"""
from __future__ import annotations

import html

import streamlit as st

from src.config import COLORS, DISCLAIMER

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;600;700;800&family=Source+Sans+3:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, label, p, li, td, th {{
    font-family: 'Source Sans 3', 'Segoe UI', Helvetica, Arial, sans-serif;
}}
h1, h2, h3, h4, .hl-title, [data-testid="stMetricValue"] {{
    font-family: 'Manrope', 'Segoe UI', Helvetica, Arial, sans-serif !important;
    color: {COLORS['ink']};
    letter-spacing: -0.01em;
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }}
header[data-testid="stHeader"] {{ background: rgba(255,255,255,0.0); }}

/* sidebar */
section[data-testid="stSidebar"] {{
    background: {COLORS['surface']};
    border-right: 1px solid {COLORS['line']};
}}
.hl-brand {{ display:flex; align-items:center; gap:.55rem; padding:.2rem 0 .8rem 0; }}
.hl-mark {{
    width:34px; height:34px; border-radius:10px;
    background: linear-gradient(135deg, {COLORS['teal']} 0%, {COLORS['blue']} 100%);
    display:flex; align-items:center; justify-content:center; color:white; font-weight:800;
    font-family:'Manrope'; font-size:1rem;
}}
.hl-brand-name {{ font-family:'Manrope'; font-weight:800; font-size:1.15rem; color:{COLORS['ink']}; line-height:1.05; }}
.hl-brand-sub {{ font-size:.74rem; color:{COLORS['muted']}; }}
.hl-side-note {{
    font-size:.78rem; color:{COLORS['muted']}; border-top:1px solid {COLORS['line']};
    padding-top:.7rem; margin-top:.8rem; line-height:1.35;
}}

/* page header */
.hl-page-head {{ margin-bottom: 1.3rem; }}
.hl-page-head h1 {{ font-size: 2rem; font-weight: 800; margin: 0 0 .25rem 0; padding:0; }}
.hl-page-head p {{ color: {COLORS['muted']}; font-size: 1.05rem; margin: 0; max-width: 760px; }}

/* cards */
.hl-card {{
    background: #fff; border:1px solid {COLORS['line']}; border-radius: 16px;
    padding: 1.05rem 1.2rem; height: 100%;
}}
.hl-card h4 {{ margin:0 0 .35rem 0; font-size:1.02rem; font-weight:700; }}
.hl-card p {{ margin:0; color:{COLORS['muted']}; font-size:.95rem; line-height:1.45; }}
.hl-card-tinted {{ background:{COLORS['surface']}; border-color: transparent; }}

.hl-kpi {{
    background:#fff; border:1px solid {COLORS['line']}; border-radius: 16px;
    padding: .95rem 1.1rem; border-left: 5px solid {COLORS['teal']};
}}
.hl-kpi .v {{ font-family:'Manrope'; font-weight:800; font-size:1.75rem; color:{COLORS['ink']}; line-height:1.1; }}
.hl-kpi .l {{ font-size:.85rem; color:{COLORS['muted']}; margin-top:.15rem; }}
.hl-kpi .n {{ font-size:.78rem; color:{COLORS['muted']}; margin-top:.25rem; }}
.hl-kpi.blue {{ border-left-color:{COLORS['blue']}; }}
.hl-kpi.green {{ border-left-color:{COLORS['green']}; }}

/* callouts */
.hl-note {{
    border-radius: 12px; padding: .8rem 1rem; font-size: .93rem; line-height:1.45;
    background: #FFF8E6; border:1px solid #F0DDAE; color:#5B4410;
}}
.hl-info {{
    border-radius: 12px; padding: .8rem 1rem; font-size: .93rem; line-height:1.45;
    background: {COLORS['surface']}; border:1px solid {COLORS['line']}; color:{COLORS['ink']};
}}

/* risk result */
.hl-result {{ border-radius:18px; padding:1.2rem 1.4rem; border:1px solid {COLORS['line']}; background:#fff; }}
.hl-result .big {{ font-family:'Manrope'; font-weight:800; font-size:2.6rem; line-height:1; }}
.hl-pill {{
    display:inline-block; padding:.18rem .7rem; border-radius:999px; font-size:.82rem; font-weight:600;
}}
.hl-pill.low {{ background:#E3F6EC; color:#176B42; }}
.hl-pill.mid {{ background:#FFF1D6; color:#8A5A0B; }}
.hl-pill.high {{ background:#E5ECFD; color:#1F3FA3; }}

/* hero */
.hl-hero h1 {{ font-size: 3rem; line-height:1.05; font-weight:800; margin:0 0 .8rem 0; padding:0; }}
.hl-hero p.lead {{ font-size:1.18rem; color:{COLORS['muted']}; max-width: 560px; margin-bottom:1.2rem; }}
.hl-chip {{
    display:inline-block; font-size:.8rem; font-weight:600; color:{COLORS['teal']};
    background:{COLORS['surface']}; border:1px solid {COLORS['line']}; padding:.15rem .65rem;
    border-radius:999px; margin:0 .35rem .35rem 0;
}}

/* streamlit widgets */
div.stButton > button, div.stDownloadButton > button, a[data-testid="stPageLink-NavLink"] {{
    border-radius: 12px; font-weight: 600; padding: .55rem 1.1rem;
}}
div.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {COLORS['teal']} 0%, #136C9C 100%); border: none; color:white;
}}
div.stButton > button[kind="primary"]:hover {{ filter: brightness(1.07); }}
button:focus-visible, a:focus-visible, input:focus-visible {{ outline: 3px solid {COLORS['blue']} !important; outline-offset:2px; }}
div[data-baseweb="tab-list"] {{ gap: .3rem; }}
button[data-baseweb="tab"] {{ border-radius: 10px 10px 0 0; font-weight:600; }}
[data-testid="stDataFrame"] {{ border: 1px solid {COLORS['line']}; border-radius: 12px; overflow:hidden; }}
[data-testid="stExpander"] {{ border-radius: 12px; border:1px solid {COLORS['line']}; }}
hr {{ border-color: {COLORS['line']}; }}
@media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}
@media (max-width: 760px) {{ .hl-hero h1 {{ font-size: 2.1rem; }} .hl-page-head h1 {{ font-size:1.6rem; }} }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_brand() -> None:
    with st.sidebar:
        st.markdown(
            "<div class='hl-brand'><div class='hl-mark'>H</div><div>"
            "<div class='hl-brand-name'>HealthLens AI</div>"
            "<div class='hl-brand-sub'>Early health risk analytics</div></div></div>",
            unsafe_allow_html=True,
        )


def sidebar_footer(dataset_name: str) -> None:
    with st.sidebar:
        st.markdown(
            f"<div class='hl-side-note'><b>Active dataset</b><br>{html.escape(dataset_name)}<br><br>"
            "Educational prototype. Not a medical device.</div>",
            unsafe_allow_html=True,
        )


def page_header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f"<div class='hl-page-head'><h1>{title}</h1>{sub}</div>", unsafe_allow_html=True)


def kpi(label: str, value: str, note: str = "", tone: str = "") -> str:
    n = f"<div class='n'>{note}</div>" if note else ""
    return f"<div class='hl-kpi {tone}'><div class='v'>{value}</div><div class='l'>{label}</div>{n}</div>"


def kpi_row(items: list[tuple]) -> None:
    """items = [(label, value, note, tone), ...]"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        label, value, *rest = item
        note = rest[0] if rest else ""
        tone = rest[1] if len(rest) > 1 else ""
        col.markdown(kpi(label, value, note, tone), unsafe_allow_html=True)


def card(title: str, body: str, tinted: bool = False) -> str:
    cls = "hl-card hl-card-tinted" if tinted else "hl-card"
    return f"<div class='{cls}'><h4>{title}</h4><p>{body}</p></div>"


def note(text: str) -> None:
    st.markdown(f"<div class='hl-note'>{text}</div>", unsafe_allow_html=True)


def info(text: str) -> None:
    st.markdown(f"<div class='hl-info'>{text}</div>", unsafe_allow_html=True)


def disclaimer() -> None:
    note("<b>Educational use only.</b> " + html.escape(DISCLAIMER))
