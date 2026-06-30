"""Data cleaning & preprocessing for the Global Weather Repository dataset.

Covers the Basic Assessment requirement: "Handle missing values, outliers, and
normalize data." Functions are small and composable so they can be reused from
the notebook or the end-to-end pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw CSV, parsing the time column."""
    path = path or config.DATASET_CSV
    df = pd.read_csv(path)
    if config.COL_TIME in df.columns:
        df[config.COL_TIME] = pd.to_datetime(df[config.COL_TIME], errors="coerce")
    return df


def basic_overview(df: pd.DataFrame) -> dict:
    """Return a small dictionary summarising the dataset shape and quality."""
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "n_countries": df[config.COL_COUNTRY].nunique()
        if config.COL_COUNTRY in df
        else None,
        "n_locations": df[config.COL_LOCATION].nunique()
        if config.COL_LOCATION in df
        else None,
        "time_min": df[config.COL_TIME].min() if config.COL_TIME in df else None,
        "time_max": df[config.COL_TIME].max() if config.COL_TIME in df else None,
        "total_missing": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and percentages, sorted descending."""
    miss = df.isna().sum()
    out = pd.DataFrame(
        {"missing": miss, "missing_pct": (miss / len(df) * 100).round(2)}
    )
    return out[out["missing"] > 0].sort_values("missing", ascending=False)


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values.

    * Numeric columns -> median (robust to outliers).
    * Categorical/object columns -> mode (most frequent value).
    Rows missing the time index or the primary target are dropped because they
    cannot be used for time-series forecasting.
    """
    df = df.copy()

    # Drop rows with no timestamp or no temperature target - unusable.
    critical = [c for c in (config.COL_TIME, config.COL_TEMP) if c in df.columns]
    df = df.dropna(subset=critical)

    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            if not mode.empty:
                df[c] = df[c].fillna(mode.iloc[0])

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows and duplicate (location, timestamp) records."""
    df = df.drop_duplicates()
    keys = [c for c in (config.COL_LOCATION, config.COL_TIME) if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last")
    return df


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """Boolean mask of outliers using the Tukey IQR rule."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)


def cap_outliers(df: pd.DataFrame, cols=None, k: float = 1.5) -> pd.DataFrame:
    """Winsorize (clip) numeric outliers to the IQR fence instead of dropping.

    Clipping keeps every record (important for the per-location time series)
    while neutralising extreme sensor errors.
    """
    df = df.copy()
    if cols is None:
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
    for c in cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - k * iqr, q3 + k * iqr
        df[c] = df[c].clip(lower, upper)
    return df


def add_normalized(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Add z-score normalized copies of key numeric columns (suffix ``_z``).

    Original columns are kept so downstream EDA/plots stay interpretable.
    """
    df = df.copy()
    if cols is None:
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
    scaler = StandardScaler()
    df[[f"{c}_z" for c in cols]] = scaler.fit_transform(df[cols])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar features from the time column for EDA & modelling."""
    df = df.copy()
    t = df[config.COL_TIME]
    df["date"] = t.dt.date
    df["year"] = t.dt.year
    df["month"] = t.dt.month
    df["day"] = t.dt.day
    df["dayofyear"] = t.dt.dayofyear
    df["hour"] = t.dt.hour
    df["season"] = df["month"].map(_month_to_season)
    return df


def _month_to_season(m: int) -> str:
    # Northern-hemisphere meteorological seasons (used as a coarse grouping).
    return {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }[m]


def clean_pipeline(path=None) -> pd.DataFrame:
    """Run the full cleaning pipeline and return an analysis-ready DataFrame."""
    df = load_raw(path)
    df = remove_duplicates(df)
    df = handle_missing(df)
    df = cap_outliers(df)
    df = add_time_features(df)
    df = add_normalized(df)
    return df


if __name__ == "__main__":
    cleaned = clean_pipeline()
    print(basic_overview(cleaned))
    print(cleaned.shape)
