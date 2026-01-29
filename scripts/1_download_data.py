import gdown
import pandas as pd
from pathlib import Path
import numpy as np

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
CLEAN_FILE = CLEAN_DIR / "all_flights_2018-2022.parquet"
MODEL_READY_FILE = MODEL_READY_DIR / "flightsweather.parquet"

# 1. Download raw file
print(f"Downloading raw dataset to {RAW_FILE} ...")
gdown.download(url, str(RAW_FILE), quiet=False)
print("Download complete!")

# 2. Load into pandas
print("Loading dataset into pandas...")
df = pd.read_parquet(RAW_FILE)

# 3. Apply "target" column
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
print("Transformations complete!")

# 4. Save cleaned parquet
print(f"Saving cleaned dataset to {CLEAN_FILE} ...")
df.to_parquet(CLEAN_FILE, index=False)


# 5. Download merged weather and flights file (model ready)
FILE_ID = "1Emu2_VsYBAPa_lbES-hcJb3M8muSFGJX"
url = f"https://drive.google.com/uc?id={FILE_ID}"

print(f"Downloading flights and weather merged dataset to {MODEL_READY_FILE} ...")
gdown.download(url, str(MODEL_READY_FILE), quiet=False)
print("Model Ready Download complete!")

print("All done")


