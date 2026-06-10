from __future__ import annotations

import csv
import shutil
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"


DATASETS = [
    {
        "id": "household_power",
        "name": "UCI Individual Household Electric Power Consumption",
        "page_url": "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption",
        "download_url": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
        "archive": "household_power_consumption.zip",
        "kind": "zip",
        "extract": True,
        "target": "Global_active_power",
        "role": "must-run energy stream",
    },
    {
        "id": "bike_sharing",
        "name": "UCI Bike Sharing Dataset",
        "page_url": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
        "download_url": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "archive": "bike_sharing_dataset.zip",
        "kind": "zip",
        "extract": True,
        "target": "hour.csv: cnt, temp, hum",
        "role": "must-run seasonal count stream",
    },
    {
        "id": "beijing_air",
        "name": "UCI Beijing Multi-Site Air Quality",
        "page_url": "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data",
        "download_url": "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip",
        "archive": "beijing_multi_site_air_quality.zip",
        "kind": "zip",
        "extract": True,
        "target": "PM2.5, TEMP, WSPM by station",
        "role": "must-run multistation environmental stream",
    },
    {
        "id": "gas_drift",
        "name": "UCI Gas Sensor Array Drift",
        "page_url": "https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset",
        "download_url": "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
        "archive": "gas_sensor_array_drift.zip",
        "kind": "zip",
        "extract": True,
        "target": "sensor channels R1-R16 over batches",
        "role": "optional drift-heavy stress dataset",
    },
]

NAB_FILES = [
    (
        "realKnownCause/machine_temperature_system_failure.csv",
        "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/machine_temperature_system_failure.csv",
    ),
    (
        "realKnownCause/ambient_temperature_system_failure.csv",
        "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/ambient_temperature_system_failure.csv",
    ),
    (
        "realTraffic/occupancy_6005.csv",
        "https://raw.githubusercontent.com/numenta/NAB/master/data/realTraffic/occupancy_6005.csv",
    ),
]


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"Using existing {path}")
        return
    print(f"Downloading {url}")
    request = Request(url, headers={"User-Agent": "PrivSAF-reproducibility-script"})
    with urlopen(request, timeout=120) as response, path.open("wb") as fout:
        shutil.copyfileobj(response, fout)


def count_csv_rows(path: Path, delimiter: str = ",") -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fin:
        return max(0, sum(1 for _ in fin) - 1)


def extract_zip(path: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(path) as zf:
        zf.extractall(raw_dir)
    for nested in sorted(raw_dir.rglob("*.zip")):
        nested_dir = nested.with_suffix("")
        nested_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(nested) as zf:
            zf.extractall(nested_dir)


def inventory_dataset(dataset: dict[str, object]) -> list[dict[str, object]]:
    root = DATA / str(dataset["id"])
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(ROOT).as_posix()
            row_count = 0
            if path.suffix.lower() == ".csv":
                row_count = count_csv_rows(path)
            elif path.suffix.lower() == ".txt" and "household_power_consumption" in path.name:
                row_count = count_csv_rows(path, delimiter=";")
            rows.append(
                {
                    "dataset_id": dataset["id"],
                    "file": rel,
                    "bytes": path.stat().st_size,
                    "row_count_if_tabular": row_count,
                }
            )
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    inventory: list[dict[str, object]] = []

    for dataset in DATASETS:
        target_dir = DATA / str(dataset["id"])
        zip_path = target_dir / str(dataset["archive"])
        download(str(dataset["download_url"]), zip_path)
        if dataset.get("extract", False):
            extract_zip(zip_path, target_dir / "raw")
        inventory.extend(inventory_dataset(dataset))

    nab_root = DATA / "nab" / "raw"
    for rel, url in NAB_FILES:
        download(url, nab_root / rel)
    inventory.extend(inventory_dataset({"id": "nab"}))

    manifest_path = RESULTS / "downloaded_dataset_inventory.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            fieldnames=["dataset_id", "file", "bytes", "row_count_if_tabular"],
        )
        writer.writeheader()
        writer.writerows(inventory)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
