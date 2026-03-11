import gdown
from pathlib import Path

# Instructions on preprocessing on script "build_flights_weather_merged.py"
# This simply downloads the preprocess to avoid extra runtime

DOWNLOAD_MERGED = True

# Where the merged file will be saved
BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)

OUT_DIR = BASE_DIR/"cleansed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "final_flights_model_dataset.parquet"

# Google drive file ID
FILE_ID = "1DqQrrzIEZWn5uLpnR9zkljum2Zwp9ec3"

if DOWNLOAD_MERGED:
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    print(f'Downloading merged dataset to {OUT_FILE}:')
    gdown.download(url, str(OUT_FILE), quiet=False)
    print('Downloaded Merged Dataset.')
else:
    print('Did not download, make sure DOWNLOAD_MERGED = True')
