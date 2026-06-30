"""Central configuration for the Weather Trend Forecasting project.

Keeping paths and a few tunables in one place so every module/notebook
resolves the same locations regardless of where it is launched from.
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

# The canonical dataset filename as distributed on Kaggle.
DATASET_CSV = DATA_DIR / "GlobalWeatherRepository.csv"
KAGGLE_DATASET = "nelgiriyewithana/global-weather-repository"

# --- Column names (as they appear in the Kaggle CSV) -----------------------
# The dataset ships snake_case column names. Central references avoid typos.
COL_TIME = "last_updated"
COL_TEMP = "temperature_celsius"
COL_PRECIP = "precip_mm"
COL_HUMIDITY = "humidity"
COL_WIND = "wind_kph"
COL_PRESSURE = "pressure_mb"
COL_LAT = "latitude"
COL_LON = "longitude"
COL_COUNTRY = "country"
COL_LOCATION = "location_name"

AIR_QUALITY_COLS = [
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5",
    "air_quality_PM10",
    "air_quality_us-epa-index",
    "air_quality_gb-defra-index",
]

# --- Misc tunables ---------------------------------------------------------
RANDOM_STATE = 42
FORECAST_HORIZON = 30  # days to forecast into the future
FIG_DPI = 120


def ensure_dirs() -> None:
    """Create output directories if they do not exist yet."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
