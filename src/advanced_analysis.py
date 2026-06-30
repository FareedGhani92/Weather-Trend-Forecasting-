"""Advanced analyses for the Global Weather Repository.

Covers the Advanced Assessment "Unique Analyses" plus anomaly detection:
  * Anomaly detection (IsolationForest + z-score) on weather variables
  * Climate analysis - long-term patterns & variation across regions
  * Environmental impact - air quality vs weather correlations
  * Feature importance - what drives temperature (RF + permutation)
  * Spatial / geographical patterns - maps and continent comparisons

Figures are written to outputs/figures/.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.inspection import permutation_importance

from . import config

sns.set_theme(style="whitegrid")


def _save(fig, name: str) -> str:
    config.ensure_dirs()
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------
# 1. Anomaly detection
# --------------------------------------------------------------------------
def detect_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Flag anomalous records with IsolationForest over core weather vars."""
    cols = [
        c
        for c in (
            config.COL_TEMP, config.COL_PRECIP, config.COL_HUMIDITY,
            config.COL_WIND, config.COL_PRESSURE,
        )
        if c in df.columns
    ]
    work = df[cols].dropna()
    iso = IsolationForest(
        contamination=0.02, random_state=config.RANDOM_STATE, n_estimators=200
    )
    labels = iso.fit_predict(work)
    out = df.loc[work.index].copy()
    out["anomaly"] = labels == -1

    fig, ax = plt.subplots(figsize=(11, 5))
    normal = out[~out["anomaly"]]
    anom = out[out["anomaly"]]
    ax.scatter(
        normal[config.COL_TEMP], normal[config.COL_HUMIDITY],
        s=8, alpha=0.3, color="#2e86ab", label="Normal",
    )
    ax.scatter(
        anom[config.COL_TEMP], anom[config.COL_HUMIDITY],
        s=22, color="#d1495b", label="Anomaly",
    )
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Humidity (%)")
    ax.set_title(
        f"IsolationForest anomalies "
        f"({out['anomaly'].sum()} of {len(out)} records, "
        f"{out['anomaly'].mean()*100:.1f}%)"
    )
    ax.legend()
    fig.tight_layout()
    return out, _save(fig, "anomaly_detection.png")


# --------------------------------------------------------------------------
# 2. Climate analysis
# --------------------------------------------------------------------------
def climate_analysis(df: pd.DataFrame) -> str:
    """Monthly temperature climatology per season and latitude band."""
    df = df.copy()
    if config.COL_LAT in df.columns:
        df["lat_band"] = pd.cut(
            df[config.COL_LAT],
            bins=[-90, -60, -30, 0, 30, 60, 90],
            labels=["S Polar", "S Temperate", "S Tropics",
                    "N Tropics", "N Temperate", "N Polar"],
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    monthly = df.groupby("month")[config.COL_TEMP].agg(["mean", "std"])
    axes[0].plot(monthly.index, monthly["mean"], marker="o", color="#d1495b")
    axes[0].fill_between(
        monthly.index, monthly["mean"] - monthly["std"],
        monthly["mean"] + monthly["std"], alpha=0.2, color="#d1495b",
    )
    axes[0].set_title("Monthly temperature climatology (mean ± std)")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Temperature (°C)")

    if "lat_band" in df.columns:
        band = df.groupby(["lat_band", "month"], observed=True)[config.COL_TEMP].mean().unstack(0)
        for col in band.columns:
            axes[1].plot(band.index, band[col], marker=".", label=str(col))
        axes[1].set_title("Temperature seasonality by latitude band")
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Temperature (°C)")
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "climate_analysis.png")


# --------------------------------------------------------------------------
# 3. Environmental impact - air quality
# --------------------------------------------------------------------------
def air_quality_analysis(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Correlate air-quality pollutants with weather parameters."""
    aq = [c for c in config.AIR_QUALITY_COLS if c in df.columns]
    weather = [
        c for c in (
            config.COL_TEMP, config.COL_HUMIDITY, config.COL_WIND,
            config.COL_PRESSURE, config.COL_PRECIP, "uv_index",
        ) if c in df.columns
    ]
    if not aq:
        return "", pd.DataFrame()

    corr = df[weather + aq].corr().loc[weather, aq]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Air quality vs. weather parameter correlations")
    fig.tight_layout()
    return _save(fig, "air_quality_correlation.png"), corr


# --------------------------------------------------------------------------
# 4. Feature importance
# --------------------------------------------------------------------------
def feature_importance(df: pd.DataFrame) -> tuple[str, pd.Series]:
    """Random-forest + permutation importance for predicting temperature."""
    candidate = [
        c for c in (
            config.COL_HUMIDITY, config.COL_PRECIP, config.COL_WIND,
            config.COL_PRESSURE, "cloud", "uv_index", "visibility_km",
            config.COL_LAT, config.COL_LON, "month", "dayofyear",
        ) if c in df.columns
    ]
    data = df[candidate + [config.COL_TEMP]].dropna()
    X, y = data[candidate], data[config.COL_TEMP]

    rf = RandomForestRegressor(
        n_estimators=250, max_depth=14, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X, y)
    # Permutation importance is more reliable than impurity-based on its own.
    perm = permutation_importance(
        rf, X, y, n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    imp = pd.Series(perm.importances_mean, index=candidate).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    imp.plot(kind="barh", ax=ax, color="#2e86ab")
    ax.set_title("Permutation feature importance for temperature")
    ax.set_xlabel("Mean importance (increase in error when shuffled)")
    fig.tight_layout()
    return _save(fig, "feature_importance.png"), imp.sort_values(ascending=False)


# --------------------------------------------------------------------------
# 5. Spatial / geographical patterns
# --------------------------------------------------------------------------
def spatial_analysis(df: pd.DataFrame) -> str:
    """Scatter the globe coloured by temperature (poor-man's map)."""
    if config.COL_LAT not in df.columns or config.COL_LON not in df.columns:
        return ""
    latest = (
        df.sort_values(config.COL_TIME)
        .groupby(config.COL_LOCATION)
        .tail(1)
    )
    fig, ax = plt.subplots(figsize=(13, 6.5))
    sc = ax.scatter(
        latest[config.COL_LON], latest[config.COL_LAT],
        c=latest[config.COL_TEMP], cmap="coolwarm", s=18, alpha=0.8,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Global temperature distribution (latest reading per city)")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    fig.colorbar(sc, ax=ax, label="Temperature (°C)", shrink=0.8)
    fig.tight_layout()
    return _save(fig, "spatial_temperature_map.png")


def geographical_patterns(df: pd.DataFrame) -> str:
    """Temperature & precipitation spread for the most-sampled countries."""
    if config.COL_COUNTRY not in df.columns:
        return ""
    top = df[config.COL_COUNTRY].value_counts().head(12).index
    sub = df[df[config.COL_COUNTRY].isin(top)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    order = (
        sub.groupby(config.COL_COUNTRY)[config.COL_TEMP].median()
        .sort_values().index
    )
    sns.boxplot(
        data=sub, y=config.COL_COUNTRY, x=config.COL_TEMP, order=order,
        ax=axes[0], palette="coolwarm",
    )
    axes[0].set_title("Temperature spread by country")
    sns.barplot(
        data=sub, y=config.COL_COUNTRY, x=config.COL_PRECIP, order=order,
        ax=axes[1], palette="Blues", estimator=np.mean,
    )
    axes[1].set_title("Mean precipitation by country")
    fig.tight_layout()
    return _save(fig, "geographical_patterns.png")


def run_all(df: pd.DataFrame) -> dict:
    results = {}
    _, results["anomaly"] = detect_anomalies(df)
    results["climate"] = climate_analysis(df)
    aq_fig, _ = air_quality_analysis(df)
    if aq_fig:
        results["air_quality"] = aq_fig
    fi_fig, _ = feature_importance(df)
    results["feature_importance"] = fi_fig
    sp = spatial_analysis(df)
    if sp:
        results["spatial"] = sp
    gp = geographical_patterns(df)
    if gp:
        results["geographical"] = gp
    return results
