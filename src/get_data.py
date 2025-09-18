# src/get_data.py
from pathlib import Path
import io
import zipfile
import requests

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"

def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    zip_path = raw_dir / "UCI_HAR_Dataset.zip"
    extract_dir = raw_dir / "UCI HAR Dataset"

    # Download (skip if already present)
    if not zip_path.exists():
        print("Downloading UCI HAR Dataset...")
        resp = requests.get(URL, timeout=180, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        print(f"Saved zip → {zip_path.resolve()}")
    else:
        print("Zip already exists, skipping download.")

    # Extract (idempotent)
    if not extract_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
        print(f"Extracted to → {extract_dir.resolve()}")
    else:
        print("Already extracted.")

if __name__ == "__main__":
    main()
