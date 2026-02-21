import gdown
import pandas as pd
from pathlib import Path
import numpy as np

# Default is TRUE to DOWNLOAD_RAW and DOWNLOAD_CLEANSED
# These default values make the data download process quicker while still seeing tranformations
DOWNLOAD_RAW = True # This determines if RAW datasets get downloaded locally

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)

RAW_DIR = BASE_DIR/"data"/"raw"

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

# if MANUAL_CLEANSED == True:
#     # Load into pandas
#     print("Loading dataset into pandas...")
#     df = pd.read_parquet(RAW_FILE)

#     # Apply "target" column
#     conditions = [
#         df["Cancelled"] == True,
#         df["DepDelayMinutes"] >= 15
#     ]

#     choices = [
#         "Cancelled",
#         "Delayed"
#     ]

#     df["target"] = np.select(
#         conditions,
#         choices,
#         default="On time"
#     )

#     # Removing columns that may cause data leakage
#     df = df.drop(
#         columns=[
#             "DepTime",
#             "DepDelay",
#             "DepDelayMinutes",
#             "DepDel15",
#             "DepartureDelayGroups",
#             "ArrTime",
#             "ArrDelay",
#             "ArrDelayMinutes",
#             "ArrDel15",
#             "ArrivalDelayGroups",
#             "Cancelled",
#             "Diverted",
#             "DivAirportLandings",
#             "TaxiOut",
#             "TaxiIn",
#             "WheelsOff",
#             "WheelsOn",
#             "ActualElapsedTime",
#             "AirTime",
#         ],
#         errors="ignore"
#     )

#     # Removing 2020 due to noise due to global pandemic
#     df = df[df['year'] != 2020]

#     # Loading weather data
#     weather_cols = ["airport_code","valid","tmpf","vsby","sknt","p01i","relh","gust"]
#     weather = pd.read_parquet(WEATHER_FILE, columns=weather_cols)
#     print("weather:", weather.shape)

#     weather["valid"] = pd.to_datetime(weather["valid"], errors="coerce")
#     weather = weather.dropna(subset=["valid"]).copy()
#     weather["date"] = weather["valid"].dt.date
#     weather["hour"] = weather["valid"].dt.hour
#     print(weather.columns)

#     weather_hourly = (
#         weather.groupby(["airport_code", "date", "hour"], as_index=False)
#         .agg({
#             "tmpf": "mean",
#             "vsby": "mean",
#             "sknt": "mean",
#             "p01i": "sum",
#             "relh": "mean",
#             "gust": "max",
#         })
#     )

#     # NAN value indicates missing for relh & tempf only
#     # although relh has overlap with other nan values, so we want to keep that for now
#     weather_hourly= weather_hourly.dropna(subset = ['tmpf'])

#     # NAN value indicates dry conditions
#     weather_hourly['p01i']= weather_hourly['p01i'].fillna(0)

#     # NAN value indicates stable wind conditions
#     weather_hourly['gust'] = weather_hourly['gust'].fillna(0)

#     # NAN value clear conditions
#     weather_hourly['vsby'] = weather_hourly['vsby'].fillna(0)

#     # NAN value clear conditions
#     weather_hourly['sknt'] = weather_hourly['sknt'].fillna(0)   

#     df["date"] = df["FlightDate"].dt.date
#     df['dep_hour'] = df['CRSDepTime'] // 100

#     joined_df = df.merge(
#         weather_hourly,
#         left_on=["Origin", "date", "dep_hour"],
#         right_on=["airport_code", "date", "hour"],
#         how="left",
#     ).drop(columns=["airport_code", "hour"], errors="ignore")
#     print(f"Shape after joining: {joined_df.shape}")

#     # Dropping rows that did not join
#     joined_df = joined_df.dropna(subset = ["tmpf","vsby","sknt","p01i","relh","gust"], how='all')
#     print(f"Shape after dropping unmerged: {joined_df.shape}")

#     # Save cleaned parquet
#     print(f"Saving cleaned dataset to {CLEAN_FILE} ...")
#     joined_df.to_parquet(CLEAN_FILE, index=False)


# if DOWNLOAD_CLEANSED == True:
# # # 5. Download merged weather and flights file (cleansed)
#     FILE_ID = "1V2rFxN8UAdsHmBPFEwpASPqm38yb0-bZ"
#     url = f"https://drive.google.com/uc?id={FILE_ID}"

#     print(f"Downloading flights and weather merged dataset to {CLEAN_FILE} ...")
#     gdown.download(url, str(CLEAN_FILE), quiet=False)
#     print("All done")
