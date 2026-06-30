"""End-to-end Weather Trend Forecasting pipeline.

Runs every stage of the assessment and writes all figures + a metrics summary
to outputs/. Run from the project root:

    python main.py

If the dataset is missing it points you at `python -m src.download_data`.
"""
from __future__ import annotations

import json

import pandas as pd

from src import advanced_analysis, config, data_cleaning, eda, forecasting


def main() -> None:
    config.ensure_dirs()

    if not config.DATASET_CSV.exists():
        raise SystemExit(
            f"Dataset not found at {config.DATASET_CSV}.\n"
            "Download it first:  python -m src.download_data\n"
            "(or place GlobalWeatherRepository.csv in the data/ folder)."
        )

    print("=" * 70)
    print("1/5  Cleaning & preprocessing")
    df = data_cleaning.clean_pipeline()
    overview = data_cleaning.basic_overview(df)
    print(json.dumps(overview, indent=2, default=str))

    print("=" * 70)
    print("2/5  Exploratory Data Analysis")
    eda_figs = eda.run_all(df)
    print("  saved:", ", ".join(eda_figs))

    print("=" * 70)
    print("3/5  Forecasting (model comparison + ensemble)")
    series = forecasting.build_daily_series(df)
    metrics_df, bundle = forecasting.evaluate_models(series)
    forecasting.plot_forecast_comparison(bundle)
    forecasting.plot_metrics(metrics_df)
    future = forecasting.forecast_future(series)
    forecasting.plot_future(series, future)
    print(metrics_df.round(3).to_string())

    print("=" * 70)
    print("4/5  Advanced analyses")
    adv_figs = advanced_analysis.run_all(df)
    print("  saved:", ", ".join(adv_figs))

    print("=" * 70)
    print("5/5  Writing summary")
    summary = {
        "overview": overview,
        "model_metrics": metrics_df.round(4).to_dict(orient="index"),
        "best_model": metrics_df.index[0],
        "forecast_horizon_days": len(future),
        "forecast_mean_temp": round(float(future.mean()), 2),
    }
    out_path = config.OUTPUT_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    metrics_df.round(4).to_csv(config.OUTPUT_DIR / "model_metrics.csv")
    future.round(3).to_csv(config.OUTPUT_DIR / "future_forecast.csv")
    print(f"  wrote {out_path}")
    print("Done. See outputs/figures for visualizations.")


if __name__ == "__main__":
    main()
