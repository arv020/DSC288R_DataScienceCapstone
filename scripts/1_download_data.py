import gdown
import pandas as pd
from pathlib import Path
import numpy as np

# Default is TRUE to DOWNLOAD_RAW and DOWNLOAD_CLEANSED
# These default values make the data download process quicker while still seeing tranformations
DOWNLOAD_RAW = True # This determines if RAW datasets get downloaded locally
MANUAL_CLEANSED = False # This determines if we want to manually tranform dataset
DOWNLOAD_CLEANSED = True # This determines to skip manual, and download cleansed dataset

if DOWNLOAD_RAW == True:
    # Google Drive file
    FILE_ID = "1Ph1LoKHpMHeBOHUKESU7FkNZOZYXkbIz"
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    # Paths
    RAW_DIR = Path("../data/raw")
    CLEAN_DIR = Path("../data/cleansed")
    MODEL_READY_DIR = Path("../data/model_ready")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

    RAW_FILE = RAW_DIR / "all_flights_2018-2022_raw.parquet"
    WEATHER_FILE = RAW_DIR / "flightsweather.parquet"
    CLEAN_FILE = CLEAN_DIR / "all_flights_2018-2022_cleansed.parquet"
    #MODEL_READY_FILE = MODEL_READY_DIR / "flightsweather.parquet"


    # Download raw file
    print(f"Downloading raw dataset to {RAW_FILE} ...")
    gdown.download(url, str(RAW_FILE), quiet=False)
    print("Download complete!")

    # Download weather data
    FILE_ID = "1hpqqFArHMMODSXfxU25tVdZSSzm8FDMU"
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print(f"Downloading raw weather dataset to {WEATHER_FILE} ...")
    gdown.download(url, str(WEATHER_FILE), quiet=False)
    print("Raw Weather Download complete!")

if MANUAL_CLEANSED == True:
    # Load into pandas
    print("Loading dataset into pandas...")
    df = pd.read_parquet(RAW_FILE)

    # Apply "target" column
    conditions = [
        df["Cancelled"] == True,
        df["DepDelayMinutes"] >= 15
    ]

    choices = [
        "Cancelled",
        "Delayed"
    ]

    df["target"] = np.select(
        conditions,
        choices,
        default="On time"
    )

    # Removing columns that may cause data leakage
    df = df.drop(
        columns=[
            "DepTime",
            "DepDelay",
            "DepDelayMinutes",
            "DepDel15",
            "DepartureDelayGroups",
            "ArrTime",
            "ArrDelay",
            "ArrDelayMinutes",
            "ArrDel15",
            "ArrivalDelayGroups",
            "Cancelled",
            "Diverted",
            "DivAirportLandings",
            "TaxiOut",
            "TaxiIn",
            "WheelsOff",
            "WheelsOn",
            "ActualElapsedTime",
            "AirTime",
        ],
        errors="ignore"
    )

    # Removing 2020 due to noise due to global pandemic
    df = df[df['year'] != 2020]

    # Loading weather data
    weather_cols = ["airport_code","valid","tmpf","vsby","sknt","p01i","relh","gust"]
    weather = pd.read_parquet(WEATHER_FILE, columns=weather_cols)
    print("weather:", weather.shape)

    weather["valid"] = pd.to_datetime(weather["valid"], errors="coerce")
    weather = weather.dropna(subset=["valid"]).copy()
    weather["date"] = weather["valid"].dt.date
    weather["hour"] = weather["valid"].dt.hour

    weather_hourly = (
        weather.groupby(["airport_code", "date", "hour"], as_index=False)
        .agg({
            "tmpf": "mean",
            "vsby": "mean",
            "sknt": "mean",
            "p01i": "sum",
            "relh": "mean",
            "gust": "max",
        })
    )

    df["date"] = df["FlightDate"].dt.date
    df['dep_hour'] = df['CRSDepTime'] // 100

    joined_df = df.merge(
        weather_hourly,
        left_on=["Origin", "date", "dep_hour"],
        right_on=["airport_code", "date", "hour"],
        how="left",
    ).drop(columns=["airport_code", "hour"], errors="ignore")
    print(f"Shape after joining: {joined_df.shape}")

    joined_df = joined_df.dropna(subset=["tmpf",
        "vsby",
        "sknt",
        "p01i",
        "relh",
        "gust"])
    print(f"Shape after dropping null values for all weather data: {joined_df.shape}")

    # Save cleaned parquet
    print(f"Saving cleaned dataset to {CLEAN_FILE} ...")
    joined_df.to_parquet(CLEAN_FILE, index=False)


if DOWNLOAD_CLEANSED == True:
# # 5. Download merged weather and flights file (cleansed)
    FILE_ID = "1-dMr8usR2TF1RYgrk0tbStdc07cuP8V9"
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print(f"Downloading flights and weather merged dataset to {CLEAN_FILE} ...")
    gdown.download(url, str(CLEAN_FILE), quiet=False)
    print("Model Ready Download complete!")
    print("All done")




