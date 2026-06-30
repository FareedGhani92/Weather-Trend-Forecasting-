# 🌦️ Weather Trend Forecasting — Global Weather Repository

A data-science project that analyzes the **Global Weather Repository** dataset
(daily weather for cities worldwide, 40+ features) to **forecast future weather
trends** and showcase both basic and advanced analytics. Built for the
**PM Accelerator** technical assessment.

---

## 🎯 PM Accelerator — Mission

> **The Product Manager Accelerator Program** is designed to support PM
> professionals through every stage of their careers. From students looking for
> entry-level jobs to Directors looking to take on a leadership role, our
> program has helped hundreds of students fulfill their career aspirations.
>
> PM Accelerator accelerates product-management careers by providing certified
> training programs, hands-on projects, career services and a supportive
> community — fostering a diverse landscape in the tech industry. Through the
> **PMA Kids** initiative it is also committed to offering free Product
> Management education to teenagers from underserved families, breaking down
> financial barriers and advancing educational fairness.

🔗 Learn more: <https://www.pmaccelerator.io/>

---

## 📋 What this project does

This repository fulfils **both** the Basic and Advanced assessment tracks.

### Basic Assessment
| Requirement | Where |
|---|---|
| Handle missing values, outliers, normalize | [`src/data_cleaning.py`](src/data_cleaning.py) |
| Basic EDA — trends, correlations, patterns | [`src/eda.py`](src/eda.py) |
| Visualizations for temperature & precipitation | [`src/eda.py`](src/eda.py) |
| Forecasting model + evaluation metrics | [`src/forecasting.py`](src/forecasting.py) |
| Use `last_updated` for time-series analysis | [`src/forecasting.py`](src/forecasting.py) |

### Advanced Assessment
| Requirement | Where |
|---|---|
| Anomaly detection & outlier analysis | [`src/advanced_analysis.py`](src/advanced_analysis.py) |
| Multiple forecasting models + **ensemble** | [`src/forecasting.py`](src/forecasting.py) |
| Climate analysis (long-term/regional patterns) | [`src/advanced_analysis.py`](src/advanced_analysis.py) |
| Environmental impact (air quality vs weather) | [`src/advanced_analysis.py`](src/advanced_analysis.py) |
| Feature importance | [`src/advanced_analysis.py`](src/advanced_analysis.py) |
| Spatial analysis & geographical patterns | [`src/advanced_analysis.py`](src/advanced_analysis.py) |

**Deliverables — report & presentation (all analyses, evaluations & visuals):**
- 📄 **Report:** [`report/REPORT.md`](report/REPORT.md) — full write-up with all
  14 figures embedded, the model-evaluation table, and a requirements checklist.
- 📊 **Presentation:** [`report/Weather_Trend_Forecasting.pptx`](report/Weather_Trend_Forecasting.pptx)
  — 13-slide deck (regenerate with `node build_deck.js`).
- 📓 **Notebook:** [`notebooks/Weather_Trend_Forecasting.ipynb`](notebooks/Weather_Trend_Forecasting.ipynb)
- 🖥️ **Interactive dashboard:** [`dashboard.py`](dashboard.py) (`streamlit run dashboard.py`).

## 🖥️ Interactive Dashboard
A Streamlit + Plotly dashboard presents the entire analysis in one place —
the PM Accelerator mission, live country/variable filters, KPI cards, and eight
interactive tabs covering EDA, multi-model forecasting, anomaly detection,
climate, air quality, feature importance and an interactive world map.
Launch it with `streamlit run dashboard.py`.

---

## 📁 Project structure

```
Weather_Project/
├── data/                      # GlobalWeatherRepository.csv (downloaded)
├── src/
│   ├── config.py              # paths, column names, tunables
│   ├── download_data.py       # Kaggle API downloader
│   ├── data_cleaning.py       # missing values, outliers, normalization
│   ├── eda.py                 # exploratory analysis & visualizations
│   ├── forecasting.py         # multi-model forecasting + ensemble
│   └── advanced_analysis.py   # anomaly, climate, air quality, importance, spatial
├── notebooks/
│   └── Weather_Trend_Forecasting.ipynb
├── outputs/
│   ├── figures/               # all generated charts (.png)
│   ├── model_metrics.csv
│   ├── future_forecast.csv
│   └── summary.json
├── report/REPORT.md           # full methodology & insights
├── dashboard.py               # interactive Streamlit + Plotly dashboard
├── main.py                    # end-to-end pipeline
├── requirements.txt
└── README.md
```

---

## 🚀 How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the dataset
**Option A — automatic (Kaggle API):**
1. Create a Kaggle API token: <https://www.kaggle.com/settings> → *API* → *Create New Token*.
2. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
   (or set `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables).
3. Run:
   ```bash
   python -m src.download_data
   ```

**Option B — manual:** download `GlobalWeatherRepository.csv` from
[Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)
and place it in the `data/` folder.

### 3. Run the full pipeline
```bash
python main.py
```
This cleans the data, generates every visualization in `outputs/figures/`,
trains and compares the forecasting models, runs the advanced analyses, and
writes `outputs/summary.json`.

### 4. 🚀 Launch the interactive dashboard (recommended)
```bash
streamlit run dashboard.py
```
Opens a polished web app at <http://localhost:8501> with global filters, KPI
cards and **8 interactive tabs**: Overview, EDA & Trends, Forecasting, Anomalies,
Climate, Air Quality, Feature Importance, and an interactive world map
(Geo Explorer). All charts are Plotly — zoom, pan, hover and filter live.

### 5. (Optional) Explore the notebook
```bash
jupyter notebook notebooks/Weather_Trend_Forecasting.ipynb
```

---

## 🔬 Methodology (summary)

1. **Cleaning** — parse `last_updated` to datetime, drop duplicate
   (location, time) records, median-impute numeric / mode-impute categorical
   missing values, **winsorize** outliers to the IQR fence, and add z-score
   normalized columns plus calendar features.
2. **EDA** — distributions, seasonal boxplots, a global temperature/precip
   trend, a correlation heatmap, and hottest/coolest country rankings.
3. **Forecasting** — aggregate to a **daily global-mean temperature** series
   indexed by `last_updated`; compare a naive baseline, Linear Regression,
   Random Forest and SARIMAX, then build an **ensemble**. Evaluated with
   **MAE, RMSE, MAPE and R²** on a held-out tail; the best model forecasts 30
   days ahead.
4. **Advanced** — IsolationForest anomaly detection, climate climatology by
   latitude band, air-quality↔weather correlations, permutation feature
   importance, and spatial/geographical maps.

See [`report/REPORT.md`](report/REPORT.md) for full details and the resulting
insights.

---

## 📦 Requirements
All Python dependencies are pinned in [`requirements.txt`](requirements.txt)
(pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn, plotly,
kaggle, jupyter).

## 📝 License
Released as open-source for evaluation purposes.
