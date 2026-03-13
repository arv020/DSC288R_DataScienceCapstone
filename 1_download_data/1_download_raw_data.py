import gdown
import pandas as pd
from pathlib import Path
import numpy as np
import requests
import os
from tqdm import tqdm

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


# -----------------------------
# Configuration: list of datasets
# -----------------------------
datasets = [
    {
        "url": "https://github.com/arv020/DSC288R_DataScienceCapstone/releases/download/288R/all_flights_2018-2022.parquet",
        "output_dir": RAW_DIR,
        "output_file": "all_flights_2018-2022_raw.parquet"
    },
    {
        "url": "https://github.com/arv020/DSC288R_DataScienceCapstone/releases/download/288R-weather/weather_airports_2018_2022_CLEAN.parquet",
        "output_dir": RAW_DIR,
        "output_file": "weather_airports_2018_2022_CLEAN.parquet"
    }
     ,
    {
        "url": "https://github.com/arv020/DSC288R_DataScienceCapstone/releases/download/288R-Airports/airports.csv",
        "output_dir": RAW_DIR,
        "output_file": "airports.csv"
    },
    {
        "url": "https://github.com/arv020/DSC288R_DataScienceCapstone/releases/download/288R-Runways/runways.csv",
        "output_dir": RAW_DIR,
        "output_file": "runways.csv"
    },
    {
        "url": "https://github.com/arv020/DSC288R_DataScienceCapstone/releases/download/288R-AirlineDelayCause/Airline_Delay_Cause.csv",
        "output_dir": RAW_DIR,
        "output_file": "Airline_Delay_Cause.csv"
    },
    {
        "url": "https://github.com/arv020/DSC288R_DataScienceCapstone/releases/download/288r-Web/WEB-Report-74040.xls",
        "output_dir": RAW_DIR,
        "output_file": "WEB-Report-74040.xls"
    }
]

# -----------------------------
# Function to download a single file with progress bar
# -----------------------------
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

# -----------------------------
# Loop over all datasets
# -----------------------------
for dataset in datasets:
    print(f"\nDownloading {dataset['output_file']}...")
    download_file(dataset["url"], os.path.join(dataset["output_dir"], dataset["output_file"]))

print("\nAll downloads complete!")
