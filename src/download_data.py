"""Download the Global Weather Repository dataset from Kaggle.

Usage:
    python -m src.download_data

Requirements:
    * The `kaggle` package installed (see requirements.txt).
    * Kaggle API credentials available either as ~/.kaggle/kaggle.json or via
      the KAGGLE_USERNAME / KAGGLE_KEY environment variables.

If credentials are missing, the script prints clear instructions instead of
failing cryptically. Once `data/GlobalWeatherRepository.csv` exists the rest of
the pipeline runs without Kaggle.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from . import config


def _have_credentials() -> bool:
    import os

    # Classic username/key credentials.
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    # Newer single-token credentials (KGAT_...).
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True
    for base in (
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / ".kaggle" / "access_token",
        Path(".kaggle/kaggle.json"),
    ):
        if base.exists():
            return True
    return False


def download() -> Path:
    config.ensure_dirs()
    if config.DATASET_CSV.exists():
        print(f"Dataset already present at {config.DATASET_CSV}")
        return config.DATASET_CSV

    if not _have_credentials():
        print(
            "Kaggle credentials not found.\n"
            "  1. Create a token at https://www.kaggle.com/settings -> API -> "
            "Create New Token\n"
            "  2. Save kaggle.json to ~/.kaggle/kaggle.json, OR set the\n"
            "     KAGGLE_USERNAME and KAGGLE_KEY environment variables.\n"
            "  3. Re-run: python -m src.download_data\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # Imported here so the rest of the project does not hard-depend on kaggle.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {config.KAGGLE_DATASET} ...")
    api.dataset_download_files(
        config.KAGGLE_DATASET, path=str(config.DATA_DIR), quiet=False
    )

    # Kaggle delivers a .zip; extract and normalise the CSV name.
    zip_path = config.DATA_DIR / "global-weather-repository.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(config.DATA_DIR)
        zip_path.unlink()

    # The CSV inside is named GlobalWeatherRepository.csv.
    if not config.DATASET_CSV.exists():
        csvs = list(config.DATA_DIR.glob("*.csv"))
        if csvs:
            csvs[0].rename(config.DATASET_CSV)

    print(f"Done -> {config.DATASET_CSV}")
    return config.DATASET_CSV


if __name__ == "__main__":
    download()
