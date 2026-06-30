"""Exploratory Data Analysis for the Global Weather Repository.

Covers the Basic Assessment EDA requirements:
  * uncover trends, correlations and patterns
  * generate visualizations for temperature and precipitation

All figures are saved to outputs/figures/ so they can be embedded in the
report and viewed by reviewers without re-running the code.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid")


def _save(fig, name: str) -> str:
    config.ensure_dirs()
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics for the key weather variables."""
    cols = [
        c
        for c in (
            config.COL_TEMP,
            config.COL_PRECIP,
            config.COL_HUMIDITY,
            config.COL_WIND,
            config.COL_PRESSURE,
        )
        if c in df.columns
    ]
    return df[cols].describe().T


def plot_temperature_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df[config.COL_TEMP], kde=True, ax=axes[0], color="#d1495b")
    axes[0].set_title("Temperature distribution (°C)")
    axes[0].set_xlabel("Temperature (°C)")

    if "season" in df.columns:
        order = ["Winter", "Spring", "Summer", "Autumn"]
        sns.boxplot(
            data=df, x="season", y=config.COL_TEMP, order=order, ax=axes[1],
            palette="coolwarm",
        )
        axes[1].set_title("Temperature by season")
    fig.tight_layout()
    return _save(fig, "temperature_distribution.png")


def plot_precipitation(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df[config.COL_PRECIP], bins=40, ax=axes[0], color="#2e86ab")
    axes[0].set_title("Precipitation distribution (mm)")
    axes[0].set_xlabel("Precipitation (mm)")
    axes[0].set_yscale("log")

    # Monthly mean precipitation pattern.
    if "month" in df.columns:
        monthly = df.groupby("month")[config.COL_PRECIP].mean()
        axes[1].plot(monthly.index, monthly.values, marker="o", color="#2e86ab")
        axes[1].set_title("Mean precipitation by month")
        axes[1].set_xlabel("Month")
        axes[1].set_ylabel("Precipitation (mm)")
    fig.tight_layout()
    return _save(fig, "precipitation_overview.png")


def plot_global_temp_trend(df: pd.DataFrame) -> str:
    """Daily global-mean temperature & precipitation over time."""
    daily = (
        df.set_index(config.COL_TIME)
        .resample("D")[[config.COL_TEMP, config.COL_PRECIP]]
        .mean()
    )
    fig, ax1 = plt.subplots(figsize=(13, 4.5))
    ax1.plot(daily.index, daily[config.COL_TEMP], color="#d1495b", label="Temp (°C)")
    ax1.set_ylabel("Temperature (°C)", color="#d1495b")
    ax1.set_xlabel("Date")
    ax2 = ax1.twinx()
    ax2.bar(
        daily.index, daily[config.COL_PRECIP], alpha=0.3, color="#2e86ab",
        label="Precip (mm)",
    )
    ax2.set_ylabel("Precipitation (mm)", color="#2e86ab")
    ax1.set_title("Daily global mean temperature & precipitation")
    fig.tight_layout()
    return _save(fig, "global_temp_precip_trend.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    cols = [
        c
        for c in (
            config.COL_TEMP, "feels_like_celsius", config.COL_HUMIDITY,
            config.COL_PRECIP, config.COL_WIND, config.COL_PRESSURE,
            "cloud", "uv_index", "visibility_km",
        )
        if c in df.columns
    ]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
        square=True, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlation matrix of weather variables")
    fig.tight_layout()
    return _save(fig, "correlation_heatmap.png")


def plot_top_countries(df: pd.DataFrame, n: int = 15) -> str:
    """Average temperature for the hottest/coolest countries."""
    if config.COL_COUNTRY not in df.columns:
        return ""
    means = df.groupby(config.COL_COUNTRY)[config.COL_TEMP].mean().sort_values()
    coolest = means.head(n)
    hottest = means.tail(n)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x=hottest.values, y=hottest.index, ax=axes[0], palette="Reds_r")
    axes[0].set_title(f"Top {n} hottest countries (avg °C)")
    axes[0].set_xlabel("Avg temperature (°C)")
    sns.barplot(x=coolest.values, y=coolest.index, ax=axes[1], palette="Blues")
    axes[1].set_title(f"Top {n} coolest countries (avg °C)")
    axes[1].set_xlabel("Avg temperature (°C)")
    fig.tight_layout()
    return _save(fig, "country_temperature_ranking.png")


def run_all(df: pd.DataFrame) -> dict:
    """Generate every EDA figure and return a name -> path map."""
    figs = {
        "temperature_distribution": plot_temperature_distribution(df),
        "precipitation_overview": plot_precipitation(df),
        "global_trend": plot_global_temp_trend(df),
        "correlation_heatmap": plot_correlation_heatmap(df),
        "country_ranking": plot_top_countries(df),
    }
    return {k: v for k, v in figs.items() if v}
