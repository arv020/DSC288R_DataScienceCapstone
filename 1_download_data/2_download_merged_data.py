import os
from pathlib import Path
import requests
from tqdm import tqdm

# Define output directory
OUT_DIR = Path(__file__).resolve().parent / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset info
datasets = [
    {
        "url": "https://github.com/arv020/DSC288R_DataScienceCapstone/releases/download/288R-FlightsMerged/flight_weather_merged.parquet",
        "output_dir": OUT_DIR,
        "output_file": "flight_weather_merged.parquet"
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

print("\nAll downloads complete!")
