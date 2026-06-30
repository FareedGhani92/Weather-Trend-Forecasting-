"""
🌦️  Weather Trend Forecasting — Interactive Dashboard
=====================================================

A fully interactive Streamlit + Plotly dashboard for the Global Weather
Repository assessment (PM Accelerator).

Run:
    streamlit run dashboard.py

Every tab below uses interactive Plotly charts with written explanations:
EDA (deep), multi-model forecasting with time-series diagnostics, multi-method
anomaly detection, climate analysis, air-quality / environmental impact,
feature importance, and an interactive world map.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.inspection import permutation_importance

from src import config, data_cleaning, forecasting

warnings.filterwarnings("ignore")

# ===========================================================================
# Page setup & theme
# ===========================================================================
st.set_page_config(
    page_title="Weather Trend Forecasting | PM Accelerator",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_TEMPLATE = "plotly_white"
# NOTE: these drive the LAYOUT-neutral chart series colours. The app *chrome*
# (background/cards/sidebar) is blue+white via CSS, but charts use whichever
# scheme communicates the data best — a clear multi-hue qualitative palette for
# categorical series, and meaningful continuous scales chosen per chart below.
ACCENT = "#2563eb"      # blue-600  (primary / 1st series)
ACCENT2 = "#f97316"     # orange-500 (2nd series — high contrast vs blue)
WARM = "#ef4444"        # red-500   (anomalies / alerts)
COOL = "#3b82f6"        # blue-500  (normal points)
GREEN = "#10b981"       # emerald-500 (3rd series)
GOLD = "#d97706"        # amber-600 (trendlines — strong on white)
PURPLE = "#8b5cf6"      # violet-500 (4th series)
# Continuous scales — chosen for interpretability, not theme:
SEQ = "RdYlBu_r"        # temperature: diverging cool(blue)→warm(red), intuitive
DIVERGE = "RdBu_r"      # correlations (-1..1)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Sora:wght@600;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

/* ---- App canvas: slate-50 with a faint blue wash (shadcn/Tailwind) ---- */
.stApp {
    background:
        radial-gradient(1100px 520px at 100% -8%, #eff6ff 0%, rgba(239,246,255,0) 60%),
        radial-gradient(900px 480px at -5% 0%, #f0f9ff 0%, rgba(240,249,255,0) 55%),
        linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    color: #0f172a;
}
h1, h2, h3, h4, h5, h6 { color: #0f172a !important; }
[data-testid="stHeader"] { background: transparent; }

/* ---- Hero banner ---- */
.hero {
    border-radius: 20px; padding: 32px 38px; margin-bottom: 6px;
    background: linear-gradient(120deg, #ffffff 0%, #eff6ff 60%, #e0f2fe 100%);
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 18px 40px rgba(37,99,235,0.07);
}
.hero h1 {
    font-family: 'Sora', sans-serif; font-weight: 800; font-size: 2.45rem;
    margin: 0; letter-spacing: -0.6px;
    background: linear-gradient(90deg, #1d4ed8, #2563eb, #0ea5e9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: #475569; font-size: 1.0rem; margin: 8px 0 0 0; max-width: 920px; }

/* ---- Pill badges ---- */
.badge {
    display:inline-block; padding:4px 12px; border-radius:999px; font-size:0.7rem;
    font-weight:600; letter-spacing:0.4px; text-transform:uppercase;
    background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; margin-right:8px;
}

/* ---- KPI cards (white, subtle border + shadow, blue hover) ---- */
.kpi {
    border-radius: 16px; padding: 18px 20px; height: 100%;
    background: #ffffff; border: 1px solid #e2e8f0;
    box-shadow: 0 1px 2px rgba(15,23,42,0.05);
    transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}
.kpi:hover { transform: translateY(-3px); border-color: #93c5fd;
             box-shadow: 0 12px 26px rgba(37,99,235,0.12); }
.kpi .label { color:#64748b; font-size:0.74rem; text-transform:uppercase;
              letter-spacing:0.6px; font-weight:600; }
.kpi .value { font-family:'Sora',sans-serif; font-size:1.55rem; font-weight:800;
              margin-top:6px; color:#0f172a; line-height:1.15; }
.kpi .value.sm { font-size:1.12rem; line-height:1.2; word-break:break-word; }
.kpi .sub { color:#2563eb; font-size:0.78rem; margin-top:2px; font-weight:600; }

/* ---- Mission / insight / explain callouts ---- */
.mission {
    border-radius:14px; padding:16px 18px; margin-top:10px;
    background: linear-gradient(160deg, #eff6ff, #f0f9ff);
    border:1px solid #bfdbfe; font-size:0.84rem; color:#334155;
}
.insight {
    border-left:4px solid #2563eb; background:#eff6ff;
    padding:12px 16px; border-radius:0 10px 10px 0; margin:10px 0;
    font-size:0.9rem; color:#1e293b;
}
.explain {
    border-left:4px solid #4f46e5; background:#eef2ff;
    padding:12px 16px; border-radius:0 10px 10px 0; margin:10px 0;
    font-size:0.9rem; color:#1e293b;
}

/* ---- Layout: comfortable content width & padding ---- */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1480px; }

/* ===== Sidebar ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border-right: 1px solid #e2e8f0;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 10px; }

/* Brand block */
.brand { display:flex; align-items:center; gap:12px; padding:4px 2px 6px 2px; }
.brand-logo {
    width:44px; height:44px; border-radius:13px; flex:none;
    display:flex; align-items:center; justify-content:center; font-size:23px;
    background:linear-gradient(135deg, #1d4ed8, #0ea5e9);
    box-shadow:0 8px 18px rgba(37,99,235,0.32);
}
.brand-title { font-family:'Sora',sans-serif; font-weight:800; font-size:1.06rem;
               color:#0f172a; line-height:1.12; }
.brand-sub { font-size:0.72rem; color:#64748b; font-weight:600; letter-spacing:0.3px; }

/* Group labels */
.nav-label { font-size:0.68rem; font-weight:700; letter-spacing:1.3px; text-transform:uppercase;
             color:#94a3b8; margin:16px 4px 8px 4px; }

/* Nav menu (built from st.radio) */
section[data-testid="stSidebar"] [role="radiogroup"] { gap:3px; }
section[data-testid="stSidebar"] [role="radiogroup"] > label {
    display:flex; align-items:center; gap:9px;
    padding:9px 13px; margin:0; border-radius:11px; cursor:pointer;
    border:1px solid transparent; background:transparent;
    font-weight:600; font-size:0.92rem; color:#475569;
    transition: all .13s ease;
}
section[data-testid="stSidebar"] [role="radiogroup"] > label:hover {
    background:#eff6ff; color:#1d4ed8;
}
/* hide the native radio circle */
section[data-testid="stSidebar"] [role="radiogroup"] > label > div:first-child { display:none !important; }
/* active (checked) item */
section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) {
    background:linear-gradient(90deg, #2563eb, #3b82f6); border-color:#2563eb;
    box-shadow:0 6px 16px rgba(37,99,235,0.22);
}
section[data-testid="stSidebar"] [role="radiogroup"] > label:has(input:checked) * { color:#ffffff !important; }

/* Footer stat card */
.sidestat { margin-top:14px; padding:11px 13px; border-radius:11px;
            background:#f1f5f9; border:1px solid #e2e8f0; font-size:0.74rem; color:#475569; }
.sidestat b { color:#0f172a; }
.sidestat .row { display:flex; justify-content:space-between; padding:2px 0; }

/* ---- Dataframes / containers: soft cards ---- */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden;
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.stTabs [aria-selected="true"] { color:#2563eb; }
footer {visibility:hidden;} #MainMenu {visibility:hidden;}

/* ---- High-quality full-screen loading overlay (white card, crisp ring) ---- */
[data-testid="stSpinner"] {
    position: fixed; inset: 0; margin: auto;
    width: 340px; height: 226px;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 20px;
    background: #ffffff;
    border: 1px solid #e2e8f0; border-radius: 24px;
    box-shadow: 0 0 0 100vmax rgba(15,23,42,0.34), 0 26px 64px rgba(15,23,42,0.24);
    z-index: 999999;
    color: #0f172a; font-family: 'Inter', sans-serif; font-weight: 600;
    font-size: 0.98rem; text-align: center; padding: 0 26px;
}
/* hide Streamlit's native spinner icon (a nested <i>), keep the message <p> */
[data-testid="stSpinner"] i,
[data-testid="stSpinner"] svg { display: none !important; }
[data-testid="stSpinner"] p { margin: 0; color: #334155; }
/* crisp dual-tone ring spinner with a soft glow */
[data-testid="stSpinner"]::before {
    content: ""; width: 68px; height: 68px; border-radius: 50%;
    border: 6px solid #dbeafe;            /* track: blue-100 */
    border-top-color: #2563eb;            /* arc: blue-600 */
    border-right-color: #3b82f6;          /* arc fade: blue-500 */
    box-shadow: 0 8px 22px rgba(37,99,235,0.22);
    animation: wxspin 0.8s cubic-bezier(0.55, 0.15, 0.45, 0.85) infinite;
}
[data-testid="stSpinner"]::after {
    content: "🌦️ Weather Forecasting"; font-family: 'Sora', sans-serif; font-weight: 700;
    font-size: 0.74rem; letter-spacing: 0.6px; color: #2563eb; opacity: 0.9;
    animation: wxpulse 1.6s ease-in-out infinite;
}
@keyframes wxspin { to { transform: rotate(360deg); } }
@keyframes wxpulse { 0%,100% { opacity: 0.5; } 50% { opacity: 1; } }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ===========================================================================
# Cached data & computation
# ===========================================================================
@st.cache_resource(show_spinner="Loading & cleaning the Global Weather Repository…")
def load_data() -> pd.DataFrame:
    if not config.DATASET_CSV.exists():
        return pd.DataFrame()
    return data_cleaning.clean_pipeline()


def _subset(df: pd.DataFrame, country_key: tuple) -> pd.DataFrame:
    if not country_key:
        return df
    return df[df[config.COL_COUNTRY].isin(country_key)]


@st.cache_data(show_spinner="Training & comparing forecasting models…")
def compute_forecast(country_key: tuple):
    df = _subset(load_data(), country_key)
    series = forecasting.build_daily_series(df)
    if len(series) < 80:
        return None
    metrics_df, bundle = forecasting.evaluate_models(series)
    future = forecasting.forecast_future(series)
    return {"series": series, "metrics": metrics_df, "bundle": bundle, "future": future}


@st.cache_data(show_spinner="Decomposing the time series…")
def compute_decomposition(country_key: tuple):
    from statsmodels.tsa.seasonal import seasonal_decompose

    series = forecasting.build_daily_series(_subset(load_data(), country_key))
    period = 365 if len(series) >= 730 else 30
    res = seasonal_decompose(series, model="additive", period=period)
    return series, res, period


@st.cache_data(show_spinner="Computing autocorrelations…")
def compute_acf_pacf(country_key: tuple):
    from statsmodels.tsa.stattools import acf, pacf

    series = forecasting.build_daily_series(_subset(load_data(), country_key))
    diff = series.diff().dropna()
    n = len(diff)
    a = acf(diff, nlags=40)
    p = pacf(diff, nlags=min(40, n // 2 - 1))
    conf = 1.96 / np.sqrt(n)
    return a, p, conf


@st.cache_data(show_spinner="Detecting anomalies (3 methods)…")
def compute_anomalies(country_key: tuple) -> pd.DataFrame:
    df = _subset(load_data(), country_key)
    cols = [c for c in (config.COL_TEMP, config.COL_PRECIP, config.COL_HUMIDITY,
                        config.COL_WIND, config.COL_PRESSURE) if c in df.columns]
    work = df[cols].dropna()
    iso = IsolationForest(contamination=0.02, random_state=config.RANDOM_STATE,
                          n_estimators=200)
    labels = iso.fit_predict(work)
    out = df.loc[work.index].copy()
    out["iso_anomaly"] = labels == -1
    out["iso_score"] = iso.score_samples(work)        # higher = more normal
    # Univariate methods on temperature for comparison.
    z = np.abs(stats.zscore(out[config.COL_TEMP]))
    out["z_anomaly"] = z > 3
    q1, q3 = out[config.COL_TEMP].quantile([0.25, 0.75])
    iqr = q3 - q1
    out["iqr_anomaly"] = (out[config.COL_TEMP] < q1 - 1.5 * iqr) | (
        out[config.COL_TEMP] > q3 + 1.5 * iqr)
    return out


@st.cache_data(show_spinner="Computing feature importance…")
def compute_importance(country_key: tuple):
    df = _subset(load_data(), country_key)
    candidate = [c for c in (config.COL_HUMIDITY, config.COL_PRECIP, config.COL_WIND,
                config.COL_PRESSURE, "cloud", "uv_index", "visibility_km",
                config.COL_LAT, config.COL_LON, "month", "dayofyear") if c in df.columns]
    data = df[candidate + [config.COL_TEMP]].dropna()
    if len(data) > 20000:
        data = data.sample(20000, random_state=config.RANDOM_STATE)
    X, y = data[candidate], data[config.COL_TEMP]
    rf = RandomForestRegressor(n_estimators=200, max_depth=14,
                               random_state=config.RANDOM_STATE, n_jobs=-1).fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=5,
                                  random_state=config.RANDOM_STATE, n_jobs=-1)
    out = pd.DataFrame({
        "permutation": perm.importances_mean,
        "gini": rf.feature_importances_,
        "abs_corr": [abs(np.corrcoef(X[c], y)[0, 1]) for c in candidate],
    }, index=candidate)
    return out.sort_values("permutation", ascending=False), rf.score(X, y)


# ===========================================================================
# Load
# ===========================================================================
df_all = load_data()
if df_all.empty:
    st.error(
        "Dataset not found. Run `python -m src.download_data` (or place "
        "`GlobalWeatherRepository.csv` in the `data/` folder), then refresh."
    )
    st.stop()


# ===========================================================================
# Sidebar — branding, mission & filters
# ===========================================================================
with st.sidebar:
    # Brand
    st.markdown(
        '<div class="brand">'
        '<div class="brand-logo">🌦️</div>'
        '<div><div class="brand-title">Weather Forecasting</div>'
        '<div class="brand-sub">PM Accelerator · Analytics</div></div>'
        '</div>', unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="nav-label">Navigate</div>', unsafe_allow_html=True)
    PAGES = ["🌍 Overview", "📊 EDA & Trends", "🔮 Forecasting", "🚨 Anomalies",
             "🌡️ Climate", "💨 Air Quality", "⭐ Feature Importance", "🗺️ Geo Explorer"]
    page = st.radio("Section", PAGES, label_visibility="collapsed",
                    help="Only the selected section renders — keeps it fast.")

    # Filters
    st.markdown('<div class="nav-label">Filters</div>', unsafe_allow_html=True)
    countries = sorted(df_all[config.COL_COUNTRY].dropna().unique())
    sel_countries = st.multiselect("Countries", countries, default=[],
        placeholder="All countries", label_visibility="collapsed",
        help="Filter every chart to specific countries.")
    country_key = tuple(sorted(sel_countries))
    var = st.selectbox("Primary variable",
        [config.COL_TEMP, config.COL_PRECIP, config.COL_HUMIDITY,
         config.COL_WIND, config.COL_PRESSURE, "uv_index"],
        format_func=lambda c: c.replace("_", " ").title())

    # Mission (footer)
    st.markdown('<div class="nav-label">About</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mission"><b>🎯 PM Accelerator Mission</b><br>'
        "Supporting PM professionals through every career stage — from "
        "entry-level to leadership — via certified training, hands-on projects "
        "and career services, while fostering a diverse tech industry. Through "
        "<b>PMA Kids</b> it offers free PM education to underserved teens."
        '<br><a href="https://www.pmaccelerator.io/" style="color:#2563eb;font-weight:600;">'
        "pmaccelerator.io ↗</a></div>",
        unsafe_allow_html=True)

    # Stats
    st.markdown(
        f'<div class="sidestat">'
        f'<div class="row"><span>Records</span><b>{len(df_all):,}</b></div>'
        f'<div class="row"><span>Countries</span><b>{df_all[config.COL_COUNTRY].nunique()}</b></div>'
        f'<div class="row"><span>Cities</span><b>{df_all[config.COL_LOCATION].nunique()}</b></div>'
        f'</div>', unsafe_allow_html=True)

df = _subset(df_all, country_key)
VARNAME = var.replace("_", " ").title()


# ===========================================================================
# Hero header
# ===========================================================================
st.markdown(
    """
    <div class="hero">
      <span class="badge">Data Science</span>
      <span class="badge">Time-Series Forecasting</span>
      <span class="badge">150K+ Records</span>
      <h1>🌦️ Weather Trend Forecasting</h1>
      <p>An end-to-end analysis of worldwide daily weather — cleaning, deep
      exploratory analysis, multi-model forecasting with diagnostics, and
      advanced climate / air-quality / spatial intelligence — in one interactive
      dashboard.</p>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# KPI cards
# ===========================================================================
def kpi(label, value, sub="", small=False):
    cls = "value sm" if small else "value"
    return (f'<div class="kpi"><div class="label">{label}</div>'
            f'<div class="{cls}">{value}</div><div class="sub">{sub}</div></div>')


avg_temp = df[config.COL_TEMP].mean()
ctry_mean = df.groupby(config.COL_COUNTRY)[config.COL_TEMP].mean()
hottest, coolest = ctry_mean.idxmax(), ctry_mean.idxmin()
span_days = (df[config.COL_TIME].max() - df[config.COL_TIME].min()).days

c1, c2, c3, c4, c5 = st.columns(5)
c1.markdown(kpi("Records", f"{len(df):,}", f"{df[config.COL_COUNTRY].nunique()} countries"), unsafe_allow_html=True)
c2.markdown(kpi("Avg Temperature", f"{avg_temp:.1f}°C", f"over {span_days} days"), unsafe_allow_html=True)
c3.markdown(kpi("Cities", f"{df[config.COL_LOCATION].nunique()}", "tracked daily"), unsafe_allow_html=True)
c4.markdown(kpi("Hottest", f"{hottest[:18]}", f"{ctry_mean.max():.1f}°C avg", small=True), unsafe_allow_html=True)
c5.markdown(kpi("Coolest", f"{coolest[:18]}", f"{ctry_mean.min():.1f}°C avg", small=True), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ===========================================================================
# Tabs
# ===========================================================================
# Only the section selected in the sidebar renders below — this keeps the
# page light (≈5 charts instead of ~33) and eliminates browser lag.


def style_fig(fig, h=420, title=None):
    # Title sits alone at the top; the legend goes to the BOTTOM so the two
    # never overlap (a top horizontal legend collides with the title).
    # automargin lets axes grow to fit long tick labels instead of clipping.
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=h,
        title=dict(text=title or "", x=0.012, xanchor="left", y=0.98, yanchor="top",
                   font=dict(family="Sora", color="#0f172a", size=16)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=58, b=64),
        font=dict(family="Inter", color="#334155"),
        xaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0", automargin=True),
        yaxis=dict(gridcolor="#e2e8f0", zerolinecolor="#e2e8f0", automargin=True),
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"))
    return fig


def sample_df(d, n=2500):
    return d.sample(min(n, len(d)), random_state=1) if len(d) > n else d


# --- Overview --------------------------------------------------------------
if page == "🌍 Overview":
    st.subheader("Dataset at a glance")
    left, right = st.columns([1.3, 1])
    with left:
        daily = df.set_index(config.COL_TIME).resample("D")[config.COL_TEMP].mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily.index, y=daily.values, mode="lines",
            line=dict(color=ACCENT, width=2), fill="tozeroy",
            fillcolor="rgba(37,99,235,0.10)", name="Global mean temp"))
        style_fig(fig, 380, "Daily global-mean temperature")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown('<div class="insight">📈 <b>What this dashboard covers</b><br>'
            "Cleaning → deep EDA → Forecasting (5 models + ensemble + diagnostics) "
            "→ Anomaly detection (3 methods) → Climate → Air quality → Feature "
            "importance → Interactive world map.</div>", unsafe_allow_html=True)
        if "condition_text" in df.columns:
            top_cond = df["condition_text"].value_counts().head(8).sort_values()
            fig = px.bar(x=top_cond.values, y=top_cond.index, orientation="h",
                         color=top_cond.values, color_continuous_scale="Blues")
            fig.update_layout(coloraxis_showscale=False)
            style_fig(fig, 300, "Most common conditions")
            st.plotly_chart(fig, width="stretch")
    st.markdown("#### Summary statistics")
    stat_cols = [c for c in (config.COL_TEMP, config.COL_PRECIP, config.COL_HUMIDITY,
                 config.COL_WIND, config.COL_PRESSURE, "uv_index") if c in df.columns]
    st.dataframe(df[stat_cols].describe().T.round(2), width="stretch")
    st.markdown('<div class="explain">🧹 <b>Data cleaning recap:</b> timestamps parsed '
        "from <code>last_updated</code>; duplicate (location, time) rows removed; "
        "numeric gaps median-imputed and categoricals mode-imputed; outliers "
        "winsorized to the IQR fence; calendar + z-score features engineered. "
        "The raw file is already complete (0 missing), so cleaning mainly adds "
        "structure and robustness.</div>", unsafe_allow_html=True)


# --- EDA (deep) ------------------------------------------------------------
if page == "📊 EDA & Trends":
    st.subheader(f"Exploratory Data Analysis — {VARNAME}")

    sk = stats.skew(df[var].dropna()); ku = stats.kurtosis(df[var].dropna())
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi("Mean", f"{df[var].mean():.2f}"), unsafe_allow_html=True)
    k2.markdown(kpi("Median", f"{df[var].median():.2f}"), unsafe_allow_html=True)
    k3.markdown(kpi("Skewness", f"{sk:.2f}", "0 = symmetric"), unsafe_allow_html=True)
    k4.markdown(kpi("Kurtosis", f"{ku:.2f}", "0 = normal tails"), unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        fig = px.histogram(df, x=var, nbins=50, marginal="box",
                           color_discrete_sequence=[ACCENT])
        fig.update_traces(marker_line_width=0)
        style_fig(fig, 380, f"Distribution + box — {VARNAME}")
        st.plotly_chart(fig, width="stretch")
    with b:
        if "season" in df.columns:
            order = ["Winter", "Spring", "Summer", "Autumn"]
            fig = px.violin(df, x="season", y=var, category_orders={"season": order},
                            color="season", box=True, points=False,
                            color_discrete_sequence=px.colors.qualitative.Set2)
            style_fig(fig, 380, f"{VARNAME} by season (violin)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch")
    st.markdown(f'<div class="explain">📊 <b>Distribution shape.</b> Skewness '
        f"<b>{sk:.2f}</b> and kurtosis <b>{ku:.2f}</b> describe how far {VARNAME} "
        "departs from a normal bell curve — positive skew means a long right tail "
        "(common for precipitation/wind), heavy kurtosis means frequent extremes. "
        "The violin shows the full per-season density, not just quartiles.</div>",
        unsafe_allow_html=True)

    st.markdown("#### Relationships")
    c, d2 = st.columns(2)
    with c:
        smp = sample_df(df)
        yv = config.COL_TEMP if var != config.COL_TEMP else config.COL_HUMIDITY
        fig = px.scatter(smp, x=var, y=yv, trendline="ols", opacity=0.4,
                         color_discrete_sequence=[ACCENT2],
                         trendline_color_override=GOLD)
        style_fig(fig, 360, f"{VARNAME} vs {yv.replace('_',' ').title()} (+OLS fit)")
        st.plotly_chart(fig, width="stretch")
    with d2:
        smp = sample_df(df)
        fig = px.scatter(smp, x=config.COL_LAT, y=config.COL_TEMP, opacity=0.35,
                         color=config.COL_TEMP, color_continuous_scale=SEQ)
        style_fig(fig, 360, "Temperature vs latitude")
        st.plotly_chart(fig, width="stretch")

    st.markdown("#### Correlation matrix")
    corr_cols = [c for c in (config.COL_TEMP, "feels_like_celsius", config.COL_HUMIDITY,
                 config.COL_PRECIP, config.COL_WIND, config.COL_PRESSURE, "cloud",
                 "uv_index", "visibility_km") if c in df.columns]
    corr = df[corr_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    style_fig(fig, 460, "Weather variable correlations (Pearson r)")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Multivariate scatter matrix & normality")
    c3, c4 = st.columns([1.4, 1])
    with c3:
        pair_cols = [c for c in (config.COL_TEMP, config.COL_HUMIDITY,
                     config.COL_PRESSURE, "uv_index") if c in df.columns]
        fig = px.scatter_matrix(sample_df(df, 1500), dimensions=pair_cols,
            color=config.COL_TEMP, color_continuous_scale=SEQ,
            labels={c: c.replace("_", " ").title()[:10] for c in pair_cols})
        fig.update_traces(diagonal_visible=False, showupperhalf=False, marker_size=3)
        style_fig(fig, 460, "Pairwise relationships")
        st.plotly_chart(fig, width="stretch")
    with c4:
        s = df[var].dropna(); s = s.sample(min(5000, len(s)), random_state=1)
        qq = stats.probplot(s, dist="norm")
        theo, obs = qq[0][0], qq[0][1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=theo, y=obs, mode="markers",
                      marker=dict(color=ACCENT, size=4, opacity=0.5), name="Data"))
        fig.add_trace(go.Scatter(x=theo, y=qq[1][1] + qq[1][0]*theo, mode="lines",
                      line=dict(color=GOLD, width=2), name="Normal"))
        style_fig(fig, 460, f"Q–Q plot — {VARNAME}")
        st.plotly_chart(fig, width="stretch")
    st.markdown('<div class="insight">🔎 Temperature tracks <b>feels-like</b> almost '
        "perfectly and rises with <b>UV</b>, while <b>humidity</b> and <b>pressure</b> "
        "pull it down. Points hugging the diagonal in the Q–Q plot indicate "
        "near-normality; systematic curves reveal skew/heavy tails.</div>",
        unsafe_allow_html=True)

    st.markdown("#### Rolling trend & volatility")
    daily = df.set_index(config.COL_TIME).resample("D")[var].mean()
    roll_m = daily.rolling(14, min_periods=3).mean()
    roll_s = daily.rolling(14, min_periods=3).std()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily.values, mode="lines",
        line=dict(color="rgba(148,163,184,0.45)", width=1), name="Daily"))
    fig.add_trace(go.Scatter(x=roll_m.index, y=roll_m.values, mode="lines",
        line=dict(color=ACCENT, width=2.5), name="14-day mean"))
    fig.add_trace(go.Scatter(x=roll_s.index, y=roll_s.values, mode="lines",
        line=dict(color=WARM, width=1.6, dash="dot"), name="14-day std (volatility)", yaxis="y2"))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False,
                                  title="Std"))
    style_fig(fig, 360, f"{VARNAME}: rolling mean & volatility")
    st.plotly_chart(fig, width="stretch")


# --- Forecasting (deep) ----------------------------------------------------
if page == "🔮 Forecasting":
    st.subheader("Time-Series Forecasting — models, diagnostics & forecast")
    fc = compute_forecast(country_key)
    if fc is None:
        st.warning("Not enough history in the current filter to forecast. "
                   "Select more countries or clear the filter.")
    else:
        metrics, bundle, future, series = (
            fc["metrics"], fc["bundle"], fc["future"], fc["series"])
        best = metrics.index[0]
        y_test, preds = bundle["y_test"], bundle["preds"]
        best_pred = preds[best]
        resid = y_test.values - best_pred

        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(kpi("Best model", best, f"R² = {metrics.iloc[0]['R2']:.3f}"), unsafe_allow_html=True)
        m2.markdown(kpi("Best RMSE", f"{metrics.iloc[0]['RMSE']:.3f}°C", "lower is better"), unsafe_allow_html=True)
        m3.markdown(kpi("Best MAE", f"{metrics.iloc[0]['MAE']:.3f}°C", "avg error"), unsafe_allow_html=True)
        m4.markdown(kpi("30-day outlook", f"{future.mean():.1f}°C", "avg forecast"), unsafe_allow_html=True)

        st.markdown("##### 1 · Time-series decomposition")
        series_d, decomp, period = compute_decomposition(country_key)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
            subplot_titles=("Observed", "Trend", f"Seasonal (period={period}d)", "Residual"),
            vertical_spacing=0.06)
        fig.add_trace(go.Scatter(x=series_d.index, y=series_d.values, line=dict(color=ACCENT)), 1, 1)
        fig.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend.values, line=dict(color=GOLD)), 2, 1)
        fig.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal.values, line=dict(color=GREEN)), 3, 1)
        fig.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid.values, mode="markers",
                      marker=dict(color=WARM, size=3)), 4, 1)
        fig.update_layout(showlegend=False)
        style_fig(fig, 520, None)
        st.plotly_chart(fig, width="stretch")
        st.markdown('<div class="explain">🧩 <b>Decomposition</b> splits the series into '
            "<b>trend</b> (long-run direction), <b>seasonal</b> (repeating annual cycle) "
            "and <b>residual</b> (irregular noise). A clean seasonal band and small "
            "residuals mean the series is highly structured and therefore "
            "forecastable.</div>", unsafe_allow_html=True)

        st.markdown("##### 2 · Autocorrelation (ACF / PACF)")
        acf_v, pacf_v, conf = compute_acf_pacf(country_key)
        cc1, cc2 = st.columns(2)
        for col, vals, name in ((cc1, acf_v, "ACF"), (cc2, pacf_v, "PACF")):
            fig = go.Figure()
            fig.add_trace(go.Bar(x=list(range(len(vals))), y=vals, marker_color=ACCENT))
            fig.add_hline(y=conf, line_dash="dot", line_color=WARM)
            fig.add_hline(y=-conf, line_dash="dot", line_color=WARM)
            style_fig(fig, 300, f"{name} of 1st-differenced series")
            col.plotly_chart(fig, width="stretch")
        st.markdown('<div class="explain">📈 <b>ACF/PACF</b> reveal how strongly each day '
            "depends on previous days (after differencing). Bars beyond the dotted "
            "bands are statistically significant lags — they justify the AR/MA "
            "orders used by SARIMAX.</div>", unsafe_allow_html=True)

        st.markdown("##### 3 · Forecast vs. actuals + 30-day future")
        sigma = float(np.std(resid))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=series.index[-120:], y=series.values[-120:],
                      mode="lines", name="History", line=dict(color="#64748b", width=1.5)))
        fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, mode="lines",
                      name="Actual", line=dict(color="#0f172a", width=3)))
        palette = [ACCENT, ACCENT2, PURPLE, GREEN, "#0ea5e9"]
        for i, (name, p) in enumerate([(k, v) for k, v in preds.items() if k != "Naive (lag-1)"]):
            fig.add_trace(go.Scatter(x=y_test.index, y=p, mode="lines", name=name,
                          line=dict(width=1.8, color=palette[i % len(palette)])))
        # 95% band around the future forecast.
        fig.add_trace(go.Scatter(x=list(future.index)+list(future.index[::-1]),
            y=list(future.values+1.96*sigma)+list((future.values-1.96*sigma)[::-1]),
            fill="toself", fillcolor="rgba(37,99,235,0.13)", line=dict(width=0),
            name="95% interval", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=future.index, y=future.values, mode="lines",
                      name="Future forecast", line=dict(color=WARM, width=2.5, dash="dot")))
        style_fig(fig, 460, "Held-out test + 30-day future forecast (95% band)")
        st.plotly_chart(fig, width="stretch")

        st.markdown("##### 4 · Model leaderboard & error profile")
        cc1, cc2 = st.columns([1.1, 1])
        with cc1:
            st.dataframe(metrics.round(3), width="stretch")
            fig = px.bar(metrics.reset_index(), x="index", y=["MAE", "RMSE"],
                         barmode="group", color_discrete_sequence=[ACCENT, ACCENT2])
            style_fig(fig, 300, "Error by model (°C)"); fig.update_layout(xaxis_title="")
            st.plotly_chart(fig, width="stretch")
        with cc2:
            # Normalised radar across metrics (lower better -> invert).
            mt = metrics.copy()
            norm = pd.DataFrame(index=mt.index)
            for c in ["MAE", "RMSE", "MAPE"]:
                norm[c] = 1 - (mt[c] - mt[c].min()) / (mt[c].max() - mt[c].min() + 1e-9)
            norm["R2"] = (mt["R2"] - mt["R2"].min()) / (mt["R2"].max() - mt["R2"].min() + 1e-9)
            fig = go.Figure()
            for m in norm.index:
                fig.add_trace(go.Scatterpolar(r=norm.loc[m].values, theta=norm.columns,
                              fill="toself", name=m, opacity=0.5))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
            style_fig(fig, 360, "Skill radar (outer = better)")
            st.plotly_chart(fig, width="stretch")

        st.markdown("##### 5 · Residual diagnostics (best model)")
        r1, r2, r3 = st.columns(3)
        with r1:
            fig = go.Figure(go.Scatter(x=y_test.index, y=resid, mode="markers+lines",
                            marker=dict(color=WARM, size=5)))
            fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
            style_fig(fig, 300, "Residuals over time"); st.plotly_chart(fig, width="stretch")
        with r2:
            fig = px.histogram(x=resid, nbins=20, color_discrete_sequence=[ACCENT])
            style_fig(fig, 300, "Residual distribution"); st.plotly_chart(fig, width="stretch")
        with r3:
            fig = px.scatter(x=best_pred, y=y_test.values, opacity=0.7,
                             color_discrete_sequence=[GREEN])
            lo, hi = float(min(best_pred.min(), y_test.min())), float(max(best_pred.max(), y_test.max()))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                          line=dict(color=GOLD, dash="dash"), name="Perfect"))
            style_fig(fig, 300, "Predicted vs actual"); st.plotly_chart(fig, width="stretch")
        st.markdown('<div class="insight">🔮 Models are evaluated <b>1-step-ahead '
            "walk-forward</b> so the comparison is fair. Residuals scattered "
            "around zero with no pattern = a well-specified model. On a planetary "
            "daily mean the naive baseline is strong; SARIMAX & the ensemble track "
            "within ~0.02 RMSE.</div>", unsafe_allow_html=True)


# --- Anomalies (multi-method) ----------------------------------------------
if page == "🚨 Anomalies":
    st.subheader("Anomaly Detection — three complementary methods")
    an = compute_anomalies(country_key)
    n_iso, n_z, n_iqr = int(an["iso_anomaly"].sum()), int(an["z_anomaly"].sum()), int(an["iqr_anomaly"].sum())
    a1, a2, a3, a4 = st.columns(4)
    a1.markdown(kpi("IsolationForest", f"{n_iso:,}", f"{an['iso_anomaly'].mean()*100:.1f}% (multivariate)"), unsafe_allow_html=True)
    a2.markdown(kpi("Z-score |z|>3", f"{n_z:,}", "temp, univariate"), unsafe_allow_html=True)
    a3.markdown(kpi("IQR rule", f"{n_iqr:,}", "temp, univariate"), unsafe_allow_html=True)
    a4.markdown(kpi("Scanned", f"{len(an):,}", "records"), unsafe_allow_html=True)
    st.markdown('<div class="explain">🚨 <b>Why three methods?</b> '
        "<b>IsolationForest</b> finds multivariate anomalies (odd <i>combinations</i> "
        "of temp/humidity/wind/pressure/precip). <b>Z-score</b> flags points >3 "
        "standard deviations from the mean. The <b>IQR rule</b> flags points beyond "
        "1.5×IQR. Agreement across methods raises confidence; disagreement shows "
        "each captures a different notion of 'unusual'.</div>", unsafe_allow_html=True)

    # Downsample for responsiveness: keep ALL anomalies, sample the (huge)
    # normal set so the scatter plots stay light (~7k points, not ~150k).
    anom = an[an["iso_anomaly"]]
    normal = an[~an["iso_anomaly"]]
    normal_s = normal.sample(min(4000, len(normal)), random_state=1)
    plot_df = pd.concat([normal_s, anom])

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(plot_df, x=config.COL_TEMP, y=config.COL_HUMIDITY,
            color="iso_anomaly", opacity=0.55,
            color_discrete_map={False: COOL, True: WARM},
            hover_data=[config.COL_COUNTRY, config.COL_LOCATION],
            labels={"iso_anomaly": "Anomaly"})
        style_fig(fig, 380, "IsolationForest: temp vs humidity")
        st.plotly_chart(fig, width="stretch")
    with c2:
        ts = plot_df.sort_values(config.COL_TIME)
        nn, aa = ts[~ts["iso_anomaly"]], ts[ts["iso_anomaly"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nn[config.COL_TIME], y=nn[config.COL_TEMP],
            mode="markers", marker=dict(size=4, color=COOL, opacity=0.35), name="Normal"))
        fig.add_trace(go.Scatter(x=aa[config.COL_TIME], y=aa[config.COL_TEMP],
            mode="markers", marker=dict(size=7, color=WARM, symbol="x"), name="Anomaly"))
        style_fig(fig, 380, "Anomalies over time (temperature)")
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        overlap = pd.DataFrame({"Method": ["IsolationForest", "Z-score", "IQR"],
                                "Count": [n_iso, n_z, n_iqr]})
        fig = px.bar(overlap, x="Method", y="Count", color="Method",
                     color_discrete_sequence=[ACCENT, GOLD, PURPLE], text="Count")
        style_fig(fig, 360, "Anomalies flagged per method"); fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c4:
        by_ctry = (anom.groupby(config.COL_COUNTRY).size()
                   .sort_values(ascending=False).head(12).sort_values())
        if len(by_ctry):
            fig = px.bar(x=by_ctry.values, y=by_ctry.index, orientation="h",
                         color=by_ctry.values, color_continuous_scale="Reds")
            fig.update_layout(coloraxis_showscale=False)
            style_fig(fig, 360, "Top countries by anomaly count")
            st.plotly_chart(fig, width="stretch")

    st.caption(f"Scatter plots show all {len(anom):,} flagged anomalies plus a "
               f"{len(normal_s):,}-point sample of normal records (of {len(normal):,}) "
               "for fast, smooth rendering.")

    st.markdown("##### Sample of flagged anomalies (IsolationForest)")
    show = [c for c in (config.COL_COUNTRY, config.COL_LOCATION, config.COL_TIME,
            config.COL_TEMP, config.COL_HUMIDITY, config.COL_WIND, config.COL_PRECIP,
            "iso_score") if c in an.columns]
    st.dataframe(an[an["iso_anomaly"]].sort_values("iso_score")[show].head(15), width="stretch")


# --- Climate ---------------------------------------------------------------
if page == "🌡️ Climate":
    st.subheader("Climate Analysis — seasonality, zones & long-term patterns")
    cdf = df.copy()
    cdf["lat_band"] = pd.cut(cdf[config.COL_LAT],
        bins=[-90, -60, -30, 0, 30, 60, 90],
        labels=["S Polar", "S Temperate", "S Tropics", "N Tropics", "N Temperate", "N Polar"])
    cdf["hemisphere"] = np.where(cdf[config.COL_LAT] >= 0, "Northern", "Southern")

    st.markdown('<div class="explain">🌡️ <b>Climate vs weather.</b> Weather is the '
        "day-to-day state; climate is the long-run average. Below we average over "
        "many cities and days to expose seasonal cycles, how they flip between "
        "hemispheres, and how temperature is organised by latitude zone.</div>",
        unsafe_allow_html=True)

    band = cdf.groupby(["lat_band", "month"], observed=True)[config.COL_TEMP].mean().reset_index()
    fig = px.line(band, x="month", y=config.COL_TEMP, color="lat_band", markers=True,
                  color_discrete_sequence=px.colors.sequential.RdBu_r)
    style_fig(fig, 400, "Monthly temperature by latitude band")
    st.plotly_chart(fig, width="stretch")
    st.markdown('<div class="insight">🧭 Tropical bands stay warm year-round; temperate '
        "and polar bands swing strongly with the seasons — and the two hemispheres "
        "move in <b>opposite</b> phase.</div>", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        hemi = cdf.groupby(["hemisphere", "month"], observed=True)[config.COL_TEMP].mean().reset_index()
        fig = px.line(hemi, x="month", y=config.COL_TEMP, color="hemisphere", markers=True,
                      color_discrete_map={"Northern": WARM, "Southern": COOL})
        style_fig(fig, 340, "Hemisphere seasonality (mirror image)")
        st.plotly_chart(fig, width="stretch")
    with g2:
        monthly = df.groupby("month")[config.COL_TEMP].agg(["mean", "std"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["mean"]+monthly["std"],
                      mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["mean"]-monthly["std"],
                      mode="lines", line=dict(width=0), fill="tonexty",
                      fillcolor="rgba(37,99,235,0.13)", showlegend=False))
        fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["mean"], mode="lines+markers",
                      line=dict(color=ACCENT, width=2.5), name="Mean"))
        style_fig(fig, 340, "Global monthly climatology (mean ± std)")
        st.plotly_chart(fig, width="stretch")

    g3, g4 = st.columns(2)
    with g3:
        pivot = cdf.pivot_table(values=config.COL_TEMP, index="lat_band",
                                columns="month", observed=True, aggfunc="mean")
        fig = px.imshow(pivot, color_continuous_scale=SEQ, aspect="auto", text_auto=".0f")
        style_fig(fig, 340, "Heatmap: latitude band × month")
        st.plotly_chart(fig, width="stretch")
    with g4:
        fig = px.violin(cdf.dropna(subset=["lat_band"]), x="lat_band", y=config.COL_TEMP,
                        color="lat_band", box=True, points=False,
                        color_discrete_sequence=px.colors.sequential.RdBu_r)
        style_fig(fig, 340, "Temperature distribution by zone")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

    st.markdown("##### Long-term trend across the record")
    monthly_ts = df.set_index(config.COL_TIME).resample("ME")[config.COL_TEMP].mean()
    fig = px.scatter(x=monthly_ts.index, y=monthly_ts.values, trendline="ols",
                     color_discrete_sequence=[ACCENT], trendline_color_override=GOLD)
    style_fig(fig, 320, "Monthly global mean temperature + linear trend")
    st.plotly_chart(fig, width="stretch")


# --- Air Quality -----------------------------------------------------------
if page == "💨 Air Quality":
    st.subheader("Environmental Impact — air quality vs. weather")
    aq = [c for c in config.AIR_QUALITY_COLS if c in df.columns]
    weather = [c for c in (config.COL_TEMP, config.COL_HUMIDITY, config.COL_WIND,
               config.COL_PRESSURE, config.COL_PRECIP, "uv_index") if c in df.columns]
    if not aq:
        st.info("Air-quality columns not present in this dataset.")
    else:
        st.markdown('<div class="explain">💨 <b>Why weather drives air quality.</b> '
            "Wind <i>disperses</i> pollutants, rain <i>washes them out</i>, and "
            "sunlight/UV <i>creates</i> ground-level ozone photochemically. The "
            "charts below quantify each of these mechanisms.</div>",
            unsafe_allow_html=True)

        corr = df[weather + aq].corr().loc[weather, aq]
        nice = [c.replace("air_quality_", "").replace("_", " ") for c in aq]
        fig = px.imshow(corr.values, x=nice, y=[w.replace("_", " ").title() for w in weather],
                        text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        aspect="auto")
        style_fig(fig, 380, "Weather ↔ pollutant correlations")
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns(2)
        smp = sample_df(df)
        pm = "air_quality_PM2.5"
        if pm in df.columns:
            with c1:
                fig = px.scatter(smp, x=config.COL_WIND, y=pm, color=config.COL_TEMP,
                                 color_continuous_scale=SEQ, opacity=0.5,
                                 trendline="ols", labels={pm: "PM2.5"},
                                 trendline_color_override=GOLD)
                style_fig(fig, 340, "Wind disperses PM2.5 (↓ with wind)")
                st.plotly_chart(fig, width="stretch")
        if "air_quality_Ozone" in df.columns:
            with c2:
                fig = px.scatter(smp, x="uv_index", y="air_quality_Ozone",
                                 color=config.COL_HUMIDITY, color_continuous_scale="Plasma",
                                 opacity=0.5, trendline="ols",
                                 labels={"air_quality_Ozone": "Ozone"},
                                 trendline_color_override=GOLD)
                style_fig(fig, 340, "UV creates Ozone (↑ with sunlight)")
                st.plotly_chart(fig, width="stretch")

        c3, c4 = st.columns(2)
        if "air_quality_PM10" in df.columns and pm in df.columns:
            with c3:
                fig = px.scatter(smp, x=pm, y="air_quality_PM10", opacity=0.4,
                                 color_discrete_sequence=[ACCENT2], trendline="ols",
                                 labels={pm: "PM2.5", "air_quality_PM10": "PM10"},
                                 trendline_color_override=GOLD)
                style_fig(fig, 340, "PM2.5 vs PM10 (co-emitted particulates)")
                st.plotly_chart(fig, width="stretch")
        epa = "air_quality_us-epa-index"
        if epa in df.columns:
            with c4:
                counts = df[epa].value_counts().sort_index()
                labels = {1: "1 Good", 2: "2 Moderate", 3: "3 Unhealthy(S)",
                          4: "4 Unhealthy", 5: "5 V.Unhealthy", 6: "6 Hazardous"}
                fig = px.bar(x=[labels.get(int(i), str(i)) for i in counts.index],
                             y=counts.values, color=counts.values,
                             color_continuous_scale="RdYlGn_r")
                fig.update_layout(coloraxis_showscale=False)
                style_fig(fig, 340, "US EPA air-quality index distribution")
                st.plotly_chart(fig, width="stretch")

        if pm in df.columns and config.COL_COUNTRY in df.columns:
            st.markdown("##### Most polluted countries (avg PM2.5)")
            top = df.groupby(config.COL_COUNTRY)[pm].mean().sort_values(ascending=False).head(12).sort_values()
            fig = px.bar(x=top.values, y=top.index, orientation="h",
                         color=top.values, color_continuous_scale="OrRd")
            fig.update_layout(coloraxis_showscale=False)
            style_fig(fig, 360, "Average PM2.5 by country")
            st.plotly_chart(fig, width="stretch")

        st.markdown('<div class="insight">💨 Humidity is the strongest weather correlate '
            "of pollution — it anti-correlates with ozone and the EPA/DEFRA indices, "
            "while UV/sunlight pushes ozone up and wind clears particulates.</div>",
            unsafe_allow_html=True)


# --- Feature Importance ----------------------------------------------------
if page == "⭐ Feature Importance":
    st.subheader("Feature Importance — what drives temperature?")
    imp, r2 = compute_importance(country_key)
    st.markdown(f'<div class="explain">⭐ <b>Three lenses on importance.</b> '
        "<b>Permutation</b> importance measures how much error rises when a feature "
        "is shuffled (model-agnostic, most reliable). <b>Gini</b> importance is the "
        "tree's built-in impurity reduction. <b>|Correlation|</b> is the raw linear "
        f"association with temperature. The Random Forest explains <b>R² = {r2:.3f}</b> "
        "of temperature variance.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        s = imp["permutation"].sort_values()
        fig = px.bar(x=s.values, y=s.index, orientation="h", color=s.values,
                     color_continuous_scale="Viridis")
        fig.update_layout(coloraxis_showscale=False)
        style_fig(fig, 420, "Permutation importance")
        st.plotly_chart(fig, width="stretch")
    with c2:
        comp = imp[["permutation", "gini"]].copy()
        comp = comp / comp.max()  # normalise each to [0,1] for comparison
        comp = comp.sort_values("permutation")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=comp.index, x=comp["permutation"], orientation="h",
                      name="Permutation", marker_color=ACCENT))
        fig.add_trace(go.Bar(y=comp.index, x=comp["gini"], orientation="h",
                      name="Gini", marker_color=ACCENT2))
        fig.update_layout(barmode="group")
        style_fig(fig, 420, "Permutation vs Gini (normalised)")
        st.plotly_chart(fig, width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        cum = imp["permutation"].sort_values(ascending=False)
        cum = (cum / cum.sum()).cumsum()
        fig = go.Figure(go.Scatter(x=list(range(1, len(cum)+1)), y=cum.values,
                        mode="lines+markers", line=dict(color=GOLD, width=2)))
        fig.add_hline(y=0.9, line_dash="dot", line_color=WARM)
        fig.update_xaxes(tickvals=list(range(1, len(cum)+1)),
                         ticktext=list(cum.index), tickangle=-40)
        style_fig(fig, 360, "Cumulative importance (90% line)")
        st.plotly_chart(fig, width="stretch")
    with c4:
        s = imp["abs_corr"].sort_values()
        fig = px.bar(x=s.values, y=s.index, orientation="h", color=s.values,
                     color_continuous_scale="Tealgrn")
        fig.update_layout(coloraxis_showscale=False)
        style_fig(fig, 360, "|Correlation| with temperature")
        st.plotly_chart(fig, width="stretch")

    top = imp.index.tolist()
    st.markdown(f'<div class="insight">⭐ <b>{top[0]}</b> dominates, followed by '
        f"<b>{top[1]}</b> and <b>{top[2]}</b> — i.e. <i>where</i> (latitude) and "
        "<i>when</i> (season) explain most of the temperature variation, far more "
        "than instantaneous weather like wind or precipitation.</div>",
        unsafe_allow_html=True)


# --- Geo Explorer ----------------------------------------------------------
if page == "🗺️ Geo Explorer":
    st.subheader("🗺️ Interactive World Map")
    metric = st.radio("Color cities by", [config.COL_TEMP, config.COL_PRECIP,
                      config.COL_HUMIDITY, "uv_index", "air_quality_PM2.5"],
                      horizontal=True, format_func=lambda c: c.replace("_", " ").title())
    latest = df.sort_values(config.COL_TIME).groupby(config.COL_LOCATION).tail(1)
    scale = SEQ if metric == config.COL_TEMP else "Viridis"
    fig = px.scatter_geo(latest, lat=config.COL_LAT, lon=config.COL_LON, color=metric,
        size=np.abs(latest[config.COL_TEMP]) + 5, size_max=14, hover_name=config.COL_LOCATION,
        hover_data={config.COL_COUNTRY: True, config.COL_LAT: False, config.COL_LON: False,
                    metric: ":.1f"},
        color_continuous_scale=scale, projection="natural earth")
    fig.update_geos(bgcolor="rgba(0,0,0,0)", showland=True, landcolor="#e7edf5",
        showocean=True, oceancolor="#eff6ff", showcountries=True,
        countrycolor="#cbd5e1", coastlinecolor="#94a3b8", showframe=False)
    style_fig(fig, 560, f"Latest {metric.replace('_',' ').title()} per city")
    st.plotly_chart(fig, width="stretch")

    st.markdown("##### Country comparison")
    top = df[config.COL_COUNTRY].value_counts().head(15).index
    sub = df[df[config.COL_COUNTRY].isin(top)]
    fig = px.box(sub, x=config.COL_COUNTRY, y=config.COL_TEMP, color=config.COL_COUNTRY)
    fig.update_traces(boxpoints=False)  # don't ship thousands of outlier points
    style_fig(fig, 400, "Temperature spread across most-sampled countries")
    fig.update_layout(showlegend=False, xaxis_title="")
    st.plotly_chart(fig, width="stretch")


# ===========================================================================
# Footer
# ===========================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#7d8ca3;font-size:0.82rem;'>"
    "Built for the <b>PM Accelerator</b> Weather Trend Forecasting assessment · "
    "Data: Kaggle Global Weather Repository · "
    "Stack: Python · Pandas · scikit-learn · statsmodels · SciPy · Plotly · Streamlit"
    "</div>", unsafe_allow_html=True)
