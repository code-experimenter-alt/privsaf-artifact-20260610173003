from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "air_quality"
ZIP_PATH = DATA_DIR / "air-quality.zip"
RAW_DIR = DATA_DIR / "raw"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading {URL}")
        urlretrieve(URL, ZIP_PATH)
    else:
        print(f"Using existing {ZIP_PATH}")
    with ZipFile(ZIP_PATH) as zf:
        zf.extractall(RAW_DIR)
    for path in sorted(RAW_DIR.iterdir()):
        print(f"{path.name}\t{path.stat().st_size}")


if __name__ == "__main__":
    main()
