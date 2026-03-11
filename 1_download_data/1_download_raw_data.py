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
AIRPORTS_FILE = RAW_DIR / "airports.csv"
RUNWAYS_FILE = RAW_DIR / "runways.csv"
DELAY_CAUSE_FILE = RAW_DIR / "Airline_Delay_Cause.csv"
OPSNET_FILE = RAW_DIR / "WEB-Report-74040.xls"

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

    # Download airports data
    FILE_ID = "1K-CccX6SlAIuLmkq_7X5VOMLFb9tB8Sh"
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print(f"Downloading airpoirts data to {AIRPORTS_FILE} ...")
    gdown.download(url, str(AIRPORTS_FILE), quiet=False)
    print("Airports Download complete!")

    # Download runways data 
    FILE_ID = "1qfvRS8Mae3oCtSn-tfp2nhb66fAz529G"
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print(f"Downloading runways data to {RUNWAYS_FILE} ...")
    gdown.download(url, str(RUNWAYS_FILE), quiet=False)
    print("Runways Download complete!")

    # Download BTS airline delay causes (airline_delay_causes.csv)
    FILE_ID = "1ssd2zi5GlAVNVz3l29svIHWiCNzLLiU0"
    url = f"https://drive.google.com/uc?id={FILE_ID}"

    print(f"Downloading airline delay causes to {DELAY_CAUSE_FILE} ...")
    gdown.download(url, str(DELAY_CAUSE_FILE), quiet=False)
    print("Delay Causes Download complete!")

    # Download FAA OPSNET operations report
    FILE_ID = "1V4J_1E4Ona8qhoMWQ2ype11Jva6zFko_"
    url = f"https://drive.google.com/uc?id={FILE_ID}&export=download"

    print(f"Downloading OPSNET report to {OPSNET_FILE} ...")
    gdown.download(url, str(OPSNET_FILE), quiet=False)
    print("OPSNET Download complete!")
