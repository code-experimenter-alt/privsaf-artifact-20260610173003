from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def normalize_preview(values: pd.Series, n: int = 600) -> np.ndarray:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return np.array([])
    preview = clean[: min(n, len(clean))]
    lo, hi = np.nanquantile(preview, [0.01, 0.99])
    scale = max(float(hi - lo), 1e-12)
    return np.clip(2 * (preview - lo) / scale - 1, -1, 1)


def series_stats(dataset_id: str, target: str, values: pd.Series, source_file: Path) -> dict[str, object]:
    raw = pd.to_numeric(values, errors="coerce")
    missing = int(raw.isna().sum())
    valid = raw.dropna()
    return {
        "dataset_id": dataset_id,
        "target": target,
        "source_file": source_file.relative_to(ROOT).as_posix(),
        "rows": int(len(raw)),
        "valid_rows": int(len(valid)),
        "missing_rows": missing,
        "missing_rate": round(missing / max(1, len(raw)), 6),
        "mean": round(float(valid.mean()), 6) if len(valid) else "",
        "std": round(float(valid.std()), 6) if len(valid) else "",
        "p01": round(float(valid.quantile(0.01)), 6) if len(valid) else "",
        "p50": round(float(valid.quantile(0.50)), 6) if len(valid) else "",
        "p99": round(float(valid.quantile(0.99)), 6) if len(valid) else "",
    }


def load_streams() -> tuple[list[dict[str, object]], list[tuple[str, np.ndarray]]]:
    rows: list[dict[str, object]] = []
    previews: list[tuple[str, np.ndarray]] = []

    path = ROOT / "data" / "air_quality" / "raw" / "AirQualityUCI.csv"
    df = pd.read_csv(path, sep=";", decimal=",", usecols=["C6H6(GT)"])
    values = pd.to_numeric(df["C6H6(GT)"], errors="coerce").replace(-200, np.nan)
    rows.append(series_stats("air_quality", "C6H6(GT)", values, path))
    previews.append(("Air Quality C6H6", normalize_preview(values)))

    path = ROOT / "data" / "household_power" / "raw" / "household_power_consumption.txt"
    df = pd.read_csv(path, sep=";", usecols=["Global_active_power"], na_values=["?"], low_memory=False)
    values = pd.to_numeric(df["Global_active_power"], errors="coerce")
    rows.append(series_stats("household_power", "Global_active_power", values, path))
    previews.append(("Household Power", normalize_preview(values)))

    path = ROOT / "data" / "bike_sharing" / "raw" / "hour.csv"
    df = pd.read_csv(path, usecols=["cnt"])
    values = pd.to_numeric(df["cnt"], errors="coerce")
    rows.append(series_stats("bike_sharing", "cnt", values, path))
    previews.append(("Bike cnt", normalize_preview(values)))

    path = (
        ROOT
        / "data"
        / "beijing_air"
        / "raw"
        / "PRSA2017_Data_20130301-20170228"
        / "PRSA_Data_20130301-20170228"
        / "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
    )
    df = pd.read_csv(path, usecols=["PM2.5"])
    values = pd.to_numeric(df["PM2.5"], errors="coerce")
    rows.append(series_stats("beijing_air", "Aotizhongxin PM2.5", values, path))
    previews.append(("Beijing PM2.5", normalize_preview(values)))

    path = ROOT / "data" / "nab" / "raw" / "realKnownCause" / "machine_temperature_system_failure.csv"
    df = pd.read_csv(path, usecols=["value"])
    values = pd.to_numeric(df["value"], errors="coerce")
    rows.append(series_stats("nab", "machine_temperature", values, path))
    previews.append(("NAB machine temp", normalize_preview(values)))

    return rows, previews


def write_profiles(rows: list[dict[str, object]]) -> None:
    path = RESULTS / "downloaded_stream_profiles.csv"
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def plot_previews(previews: list[tuple[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(len(previews), 1, figsize=(9.5, 7.4), sharex=True, sharey=True)
    for ax, (name, values) in zip(axes, previews):
        ax.plot(values, color="#1d3557", linewidth=1.0)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("first available valid samples, normalized to [-1,1] by preview quantiles")
    fig.suptitle("Downloaded Scalar Stream Previews")
    fig.tight_layout()
    out = RESULTS / "fig_downloaded_stream_previews.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows, previews = load_streams()
    write_profiles(rows)
    plot_previews(previews)


if __name__ == "__main__":
    main()
