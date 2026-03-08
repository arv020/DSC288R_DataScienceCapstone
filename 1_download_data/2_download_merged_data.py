import gdown
from pathlib import Path

DOWNLOAD_MERGED = True

# Where the merged file will be saved
BASE_DIR = Path(__file__).resolve().parent
print(BASE_DIR)

OUT_DIR = BASE_DIR/"cleansed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / 'flight_weather_merged.parquet'

# Google drive file ID
FILE_ID = '1N66s8CR8tpxZpYenNXNYQyZOtnoPCcLS'

if DOWNLOAD_MERGED:
    url = f'https://drive.google.com/uc?id={FILE_ID}'
    print(f'Downloading merged dataset to {OUT_FILE}:')
    gdown.download(url, str(OUT_FILE), quiet=False)
    print('Downloaded Merged Dataset.')
else:
    print('Did not download, make sure DOWNLOAD_MERGED = True')
