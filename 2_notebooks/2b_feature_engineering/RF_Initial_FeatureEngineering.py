import gdown
import pandas as pd
from pathlib import Path
import numpy as np
import holidays

# Default is TRUE to DOWNLOAD_RAW and DOWNLOAD_CLEANSED
# These default values make the data download process quicker while still seeing tranformations
DOWNLOAD_RAW = False # This determines if RAW datasets get downloaded locally
MANUAL_CLEANSED = False # This determines if we want to manually tranform dataset
DOWNLOAD_CLEANSED = False # This determines to skip manual, and download cleansed dataset

import os
import requests
from tqdm import tqdm
# print(os.getcwd())
# # Paths
# RAW_DIR = Path("../../1_download_data/raw")
# print(RAW_DIR)
# CLEAN_DIR = Path("../../1_download_data/cleansed")
# MODEL_READY_DIR = Path("../../1_download_data/model_ready")

# RAW_DIR.mkdir(parents=True, exist_ok=True)
# CLEAN_DIR.mkdir(parents=True, exist_ok=True)
# MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

# RAW_FILE = RAW_DIR / "all_flights_2018-2022_raw.parquet"
# WEATHER_FILE = RAW_DIR / "flightsweather.parquet"
# CLEAN_FILE = CLEAN_DIR / "all_flights_2018-2022_cleansed.parquet"

# === Base directory of the repo ===
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # 3 levels up from notebook folder

# === Data folders ===
RAW_DIR = BASE_DIR / "1_download_data" / "raw"
CLEAN_DIR = BASE_DIR / "1_download_data" / "cleansed"
MODEL_READY_DIR = BASE_DIR / "1_download_data" / "model_ready"

# Make sure folders exist
RAW_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)

# === Data files ===
RAW_FILE = RAW_DIR / "all_flights_2018-2022_raw.parquet"
WEATHER_FILE = RAW_DIR / "flightsweather.parquet"
CLEAN_FILE = CLEAN_DIR / "all_flights_2018-2022_cleansed.parquet"
MODEL_READY_FILE = MODEL_READY_DIR / "flights_model_ready.parquet"

MANUAL_CLEANSED = False

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
    print(weather.columns)

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

    # NAN value indicates missing for relh & tempf only
    # although relh has overlap with other nan values, so we want to keep that for now
    weather_hourly= weather_hourly.dropna(subset = ['tmpf'])

    # NAN value indicates dry conditions
    weather_hourly['p01i']= weather_hourly['p01i'].fillna(0)

    # NAN value indicates stable wind conditions
    weather_hourly['gust'] = weather_hourly['gust'].fillna(0)

    # NAN value clear conditions
    weather_hourly['vsby'] = weather_hourly['vsby'].fillna(0)

    # NAN value clear conditions
    weather_hourly['sknt'] = weather_hourly['sknt'].fillna(0)   

    df["date"] = df["FlightDate"].dt.date
    df['dep_hour'] = df['CRSDepTime'] // 100

    joined_df = df.merge(
        weather_hourly,
        left_on=["Origin", "date", "dep_hour"],
        right_on=["airport_code", "date", "hour"],
        how="left",
    ).drop(columns=["airport_code", "hour"], errors="ignore")
    print(f"Shape after joining: {joined_df.shape}")

    # Dropping rows that did not join
    joined_df = joined_df.dropna(subset = ["tmpf","vsby","sknt","p01i","relh","gust"], how='all')
    print(f"Shape after dropping unmerged: {joined_df.shape}")

    # Save cleaned parquet
    print(f"Saving cleaned dataset to {CLEAN_FILE} ...")
    joined_df.to_parquet(CLEAN_FILE, index=False)



DOWNLOAD_MODEL_READY = True # Will only download the data



PROCESS_DATA = False
if PROCESS_DATA == True:
    df = pd.read_parquet(CLEAN_FILE)
    print("cleansed dataframe read")

    #day of week
    df['DayOfWeek'] = df['FlightDate'].dt.day_name()

    day_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6
    }

    df["DayOfWeek_num"] = df["DayOfWeek"].map(day_map)
    print("Day of week feature complete")

    # cyclincal sin cos 
    df["month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    df["dow_sin"] = np.sin(2 * np.pi * df["DayOfWeek_num"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["DayOfWeek_num"] / 7)

    #standardize weather features
    num_cols = ["Distance", "tmpf", "vsby", "sknt", "relh", "gust"]

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(df[num_cols])

    scaled_col_names = [f"{c}_std" for c in num_cols]

    df[scaled_col_names] = scaled_values
    print("Scaling weather features done")


    # normalized distance from the mean of airport origin
    df["distance_origin_norm"] = (
        df["Distance"] - df.groupby("Origin")["Distance"].transform("mean")
    )

    #has rain binary column
    df["has_precip"] = (df["p01i"] > 0).astype(int)


    # holidays
    us_holidays = holidays.US()
    print("Holidays feature started")

    # Is Holiday (0/1)
    df["is_holiday"] = df["FlightDate"].dt.date.apply(
        lambda x: 1 if x in us_holidays else 0
    )

    # # Within ±3 days of a holiday
    # def near_holiday(date, holiday_calendar, window=3):
    #     for offset in range(-window, window + 1):
    #         if (date + pd.Timedelta(days=offset)) in holiday_calendar:
    #             return 1
    #     return 0

    # df["near_holiday_3d"] = df["FlightDate"].dt.date.apply(
    #     lambda x: near_holiday(pd.Timestamp(x), us_holidays)
    # )
    print("Delay or cancellation prior day starting")

    # Feature where 1 if any flights were cancelled the pervious day from the same origin

    # define failure
    df["is_failure"] = df["target"].isin(["Cancelled", "Delayed"]).astype(int)

    # airport-day level: did any failure happen?
    daily_failure = (
        df.groupby(["Origin", "FlightDate"])["is_failure"]
        .max()
        .reset_index()
        .rename(columns={"is_failure": "had_failure"})
    )

    # shift → yesterday
    daily_failure["prev_day_failure_origin"] = (
        daily_failure
            .sort_values("FlightDate")
            .groupby("Origin")["had_failure"]
            .shift(1)
    )

    # first day = no history
    daily_failure["prev_day_failure_origin"] = (
        daily_failure["prev_day_failure_origin"]
            .fillna(0)
            .astype(int)
    )

    # merge back to flights
    df = df.merge(
        daily_failure[["Origin", "FlightDate", "prev_day_failure_origin"]],
        on=["Origin", "FlightDate"],
        how="left"
    )

    df.drop(columns=["is_failure"], inplace=True)

    print("starting dep hour binning features")
    # Morning peak: 6–9
    df["is_morning_peak"] = df["dep_hour"].between(6, 9).astype(int)

    # Evening peak: 17–23
    df["is_evening_peak"] = df["dep_hour"].between(17, 23).astype(int)

    print(F"transformations complete, now uploading to {MODEL_READY_FILE}")

    df.to_parquet(MODEL_READY_FILE, index=False)
    print(f"File successfully updated to {MODEL_READY_FILE}")

if DOWNLOAD_MODEL_READY == True:
# # 5. Download merged weather and flights file (cleansed)

        # Dataset info
    datasets = [
        {
            "url": "https://github.com/arv020/DSC288R_DataScienceCapstone/releases/download/288R-ModelReady/flights_model_ready.parquet",
            "output_dir": MODEL_READY_DIR,
            "output_file": "flights_model_ready.parquet"
        }
    ]
    
    # Download function
    def download_file(url, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024 * 1024  # 1MB
    
        with open(output_path, "wb") as f, tqdm(
            desc=os.path.basename(output_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    
    # Loop over datasets
    for dataset in datasets:
        print(f"\nDownloading {dataset['output_file']}...")
        download_file(dataset["url"], os.path.join(dataset["output_dir"], dataset["output_file"]))

# print("\nAll downloads complete!")
#     FILE_ID = "1ObHzHu0q5OqpXTyWJ5nhnIySasdDVskN"
#     url = f"https://drive.google.com/uc?id={FILE_ID}"

#     print(f"Downloading model ready dataset to {MODEL_READY_FILE} ...")
#     gdown.download(url, str(MODEL_READY_FILE), quiet=False)
#     print("Model Ready Download complete!")
#     print("All done")