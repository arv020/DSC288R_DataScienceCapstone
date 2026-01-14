import gdown
import os
from pathlib import Path

# Replace with your file ID from Google Drive
FILE_ID = "1Ph1LoKHpMHeBOHUKESU7FkNZOZYXkbIz"
OUTPUT_DIR = Path("data/cleansed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "all_flights_2018-2022.parquet"
# Construct Google Drive download URL
url = f"https://drive.google.com/uc?id={FILE_ID}"

print(f"Downloading dataset to {OUTPUT_FILE} ...")
gdown.download(url, str(OUTPUT_FILE), quiet=False)
print("Download complete!")

