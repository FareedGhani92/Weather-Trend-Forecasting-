"""Time-series forecasting of global mean temperature.

Covers:
  * Basic Assessment - "Build a basic forecasting model and evaluate its
    performance using different metrics. Use last_updated for the time series."
  * Advanced Assessment - "Build and compare multiple forecasting models" and
    "Create an ensemble of models to improve forecast accuracy."

Approach
--------
The raw data has many cities per day, so we aggregate to a single daily global
mean temperature series indexed by ``last_updated``. We then compare:
  1. Seasonal-naive baseline (yesterday's value carried forward)
  2. Linear regression on engineered calendar/lag features
  3. Random forest on the same features
  4. SARIMAX statistical time-series model
  5. An ensemble (mean of the ML + SARIMAX forecasts)

Models are evaluated on a held-out tail of the series with MAE, RMSE, MAPE
and R2, then refit on all data to forecast FORECAST_HORIZON days ahead.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# Series construction & feature engineering
# --------------------------------------------------------------------------
def build_daily_series(df: pd.DataFrame) -> pd.Series:
    """Aggregate to a daily global-mean temperature series (gap-filled)."""
    s = (
        df.set_index(config.COL_TIME)[config.COL_TEMP]
        .resample("D")
        .mean()
    )
    s = s.interpolate("time").ffill().bfill()
    s.name = "temp"
    return s


def _make_features(s: pd.Series) -> pd.DataFrame:
    """Build supervised-learning features (lags + calendar) from a series."""
    d = pd.DataFrame({"temp": s})
    for lag in (1, 2, 3, 7, 14):
        d[f"lag_{lag}"] = d["temp"].shift(lag)
    d["roll7"] = d["temp"].shift(1).rolling(7).mean()
    d["roll14"] = d["temp"].shift(1).rolling(14).mean()
    idx = d.index
    d["dayofyear"] = idx.dayofyear
    d["month"] = idx.month
    # Cyclical encodings so the model sees the year as continuous.
    d["sin_doy"] = np.sin(2 * np.pi * d["dayofyear"] / 365.25)
    d["cos_doy"] = np.cos(2 * np.pi * d["dayofyear"] / 365.25)
    return d.dropna()


def _metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # Guard against divide-by-zero in MAPE.
    denom = np.where(np.abs(y_true) < 1e-6, 1e-6, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


# --------------------------------------------------------------------------
# Model comparison on a held-out test tail
# --------------------------------------------------------------------------
def evaluate_models(s: pd.Series, test_days: int = 30) -> tuple[pd.DataFrame, dict]:
    """Train/test split on the series tail; return metrics table + predictions."""
    feat = _make_features(s)
    X, y = feat.drop(columns="temp"), feat["temp"]

    split = len(feat) - test_days
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    preds: dict[str, np.ndarray] = {}

    # 1. Seasonal-naive baseline: predict with the previous day's value.
    preds["Naive (lag-1)"] = X_test["lag_1"].values

    # 2. Linear regression.
    lin = LinearRegression().fit(X_train, y_train)
    preds["Linear Regression"] = lin.predict(X_test)

    # 3. Random forest.
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=12, random_state=config.RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    preds["Random Forest"] = rf.predict(X_test)

    # 4. SARIMAX, evaluated the SAME way as the ML models: walk-forward
    #    1-step-ahead. We fit once on the training span, then roll through the
    #    test set appending each true observation (refit=False) and forecasting
    #    one step. This keeps the comparison fair (every model sees one real
    #    lag) instead of asking SARIMAX to forecast 30 steps blind.
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        train_series = s.loc[: y_train.index[-1]]
        res = SARIMAX(
            train_series, order=(3, 1, 1),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=200, method="lbfgs")
        onestep = []
        for ts in y_test.index:
            onestep.append(float(np.asarray(res.forecast(steps=1))[0]))
            # Append the realised value so the next 1-step forecast is fair.
            res = res.append(s.loc[[ts]], refit=False)
        preds["SARIMAX"] = np.array(onestep)
    except Exception as exc:  # pragma: no cover - statsmodels edge cases
        print(f"SARIMAX skipped: {exc}")

    # 5. Ensemble: average of the non-naive models.
    members = [k for k in ("Linear Regression", "Random Forest", "SARIMAX") if k in preds]
    if members:
        preds["Ensemble"] = np.mean([preds[m] for m in members], axis=0)

    rows = {name: _metrics(y_test.values, p) for name, p in preds.items()}
    metrics_df = pd.DataFrame(rows).T.sort_values("RMSE")

    bundle = {
        "y_train": y_train, "y_test": y_test, "preds": preds,
        "models": {"linear": lin, "rf": rf},
        "feature_names": list(X.columns),
    }
    return metrics_df, bundle


def plot_forecast_comparison(bundle: dict) -> str:
    y_train, y_test, preds = bundle["y_train"], bundle["y_test"], bundle["preds"]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(y_train.index[-90:], y_train.values[-90:], color="#444", label="History")
    ax.plot(y_test.index, y_test.values, color="black", lw=2.5, label="Actual")
    for name, p in preds.items():
        if name == "Naive (lag-1)":
            continue
        ax.plot(y_test.index, p, lw=1.6, alpha=0.85, label=name)
    ax.set_title("Forecast comparison on held-out test window")
    ax.set_ylabel("Daily mean temperature (°C)")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    config.ensure_dirs()
    path = config.FIG_DIR / "forecast_comparison.png"
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_metrics(metrics_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    metrics_df[["MAE", "RMSE"]].plot(kind="bar", ax=ax, color=["#2e86ab", "#d1495b"])
    ax.set_title("Model error comparison (lower is better)")
    ax.set_ylabel("Error (°C)")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    config.ensure_dirs()
    path = config.FIG_DIR / "model_metrics.png"
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------
# Future forecast (refit on all data)
# --------------------------------------------------------------------------
def forecast_future(s: pd.Series, horizon: int = config.FORECAST_HORIZON) -> pd.Series:
    """Forecast `horizon` days ahead with a SARIMAX model refit on all data."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        model = SARIMAX(
            s, order=(2, 1, 2), seasonal_order=(1, 0, 1, 7),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc = model.forecast(steps=horizon)
        idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series(np.asarray(fc), index=idx, name="forecast")
    except Exception as exc:
        print(f"Future SARIMAX failed ({exc}); falling back to seasonal mean.")
        season = s.groupby(s.index.dayofyear).mean()
        idx = pd.date_range(s.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        return pd.Series([season.get(d.dayofyear, s.mean()) for d in idx], index=idx)


def plot_future(s: pd.Series, fc: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(s.index[-120:], s.values[-120:], color="#444", label="Observed")
    ax.plot(fc.index, fc.values, color="#d1495b", lw=2, label="Forecast")
    ax.axvline(s.index[-1], color="grey", ls="--", alpha=0.6)
    ax.set_title(f"{len(fc)}-day global mean temperature forecast")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    fig.tight_layout()
    config.ensure_dirs()
    path = config.FIG_DIR / "future_forecast.png"
    fig.savefig(path, dpi=config.FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(path)
