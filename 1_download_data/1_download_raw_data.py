import gdown
import pandas as pd
from pathlib import Path
import numpy as np

# Default is TRUE to DOWNLOAD_RAW and DOWNLOAD_CLEANSED
# These default values make the data download process quicker while still seeing tranformations
DOWNLOAD_RAW = True # This determines if RAW datasets get downloaded locally

# Paths
BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)

RAW_DIR = BASE_DIR/"raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILE = RAW_DIR / "all_flights_2018-2022_raw.parquet"
WEATHER_FILE = RAW_DIR / "weather_airports_2018_2022_CLEAN.parquet"

if DOWNLOAD_RAW == True:
    # Google Drive file
    FILE_ID = "17HSBiO3rKT6rX28r4l4LKgwLqHgtTc0m"
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    # Download raw flight file
    print(f"Downloading raw dataset to {RAW_FILE} ...")
    gdown.download(url, str(RAW_FILE), quiet=False)
    print("Download complete!")

    # Download weather data
    FILE_ID = "1hpqqFArHMMODSXfxU25tVdZSSzm8FDMU"
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print(f"Downloading raw weather dataset to {WEATHER_FILE} ...")
    gdown.download(url, str(WEATHER_FILE), quiet=False)
    print("Raw Weather Download complete!")