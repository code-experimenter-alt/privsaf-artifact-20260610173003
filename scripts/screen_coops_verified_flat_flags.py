from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from calendar import monthrange
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RECENT_SCREEN = RESULTS / "coops_recent_flat_flag_screen.csv"

ERDDAP = "https://opendap.co-ops.nos.noaa.gov/erddap/tabledap/IOOS_SixMin_Verified_Water_Level.csv"
DATASET_INFO = "https://opendap.co-ops.nos.noaa.gov/erddap/info/IOOS_SixMin_Verified_Water_Level/index.html"


def read_top_stations(limit: int) -> list[dict[str, str]]:
    rows = list(csv.DictReader(RECENT_SCREEN.open(encoding="utf-8")))
    positives = [row for row in rows if int(row.get("flat_flag_rows") or 0) > 0]
    positives.sort(key=lambda row: int(row["flat_flag_rows"]), reverse=True)
    selected = positives[:limit]
    extra_ids = {"8772985", "9414863", "9414750"}
    seen = {row["station"] for row in selected}
    for row in rows:
        if row["station"] in extra_ids and row["station"] not in seen:
            selected.append(row)
            seen.add(row["station"])
    return selected


def query_url(station: str, year: int, month: int) -> str:
    begin = f"{year}{month:02d}01"
    end = f"{year}{month:02d}{monthrange(year, month)[1]:02d}"
    columns = "STATION_ID,DATUM,BEGIN_DATE,END_DATE,time,WL_VALUE,F,R,T,I"
    params = [
        ("STATION_ID", f'"{station}"'),
        ("DATUM", '"MLLW"'),
        ("BEGIN_DATE", f'"{begin}"'),
        ("END_DATE", f'"{end}"'),
        ("F", "1"),
    ]
    constraints = "&".join(f"{key}={urllib.parse.quote(value, safe='')}" for key, value in params)
    return f"{ERDDAP}?{columns}&{constraints}"


def fetch_csv(url: str) -> tuple[str, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "PrivSAF data audit"})
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            return "ok", response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return f"http_{exc.code}", body[:500]
    except Exception as exc:  # noqa: BLE001
        return "error", repr(exc)


def parse_rows(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    if len(lines) <= 2:
        return []
    return list(csv.DictReader([lines[0], *lines[2:]]))


def numeric(text: str) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    stations = read_top_stations(limit=12)
    years = [2025, 2024, 2023]
    output_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []

    for station in stations:
        station_id = station["station"]
        for year in years:
            for month in range(1, 13):
                print(f"{station_id} {station['name']} {year}-{month:02d}", flush=True)
                url = query_url(station_id, year, month)
                status, text = fetch_csv(url)
                rows = parse_rows(text) if status == "ok" else []
                numeric_rows = [row for row in rows if numeric(row.get("WL_VALUE", ""))]
                output_rows.append(
                    {
                        "station": station_id,
                        "name": station["name"],
                        "state": station["state"],
                        "recent_flat_rows": station.get("flat_flag_rows", ""),
                        "year": year,
                        "month": month,
                        "status": status,
                        "f1_rows": len(rows),
                        "numeric_f1_rows": len(numeric_rows),
                        "first_time": rows[0]["time"] if rows else "",
                        "first_value": rows[0].get("WL_VALUE", "") if rows else "",
                        "url": url,
                        "error_excerpt": "" if status == "ok" else text,
                    }
                )
                for row in numeric_rows[:50]:
                    event_rows.append(
                        {
                            "station": station_id,
                            "name": station["name"],
                            "state": station["state"],
                            "year": year,
                            "month": month,
                            "time": row["time"],
                            "wl_value": row["WL_VALUE"],
                            "f": row["F"],
                            "r": row["R"],
                            "t": row["T"],
                            "i": row["I"],
                            "url": url,
                        }
                    )
                time.sleep(0.05)

    summary = {
        "status": "pass",
        "dataset_info": DATASET_INFO,
        "station_count": len(stations),
        "years": years,
        "station_months_checked": len(output_rows),
        "station_months_with_f1_rows": sum(int(row["f1_rows"]) > 0 for row in output_rows),
        "station_months_with_numeric_f1_rows": sum(int(row["numeric_f1_rows"]) > 0 for row in output_rows),
        "f1_rows": sum(int(row["f1_rows"]) for row in output_rows),
        "numeric_f1_rows": sum(int(row["numeric_f1_rows"]) for row in output_rows),
        "positive_event_rows_saved": len(event_rows),
        "interpretation": (
            "CO-OPS verified six-minute ERDDAP can return value and official flat flag in the same rows. "
            "END_DATE is treated as an inclusive calendar-day bound. Months with numeric_f1_rows > 0 are direct "
            "public value+label candidates for real flatline PM-LDP evaluation."
        ),
    }

    month_csv = RESULTS / "coops_verified_flat_flag_screen.csv"
    with month_csv.open("w", newline="", encoding="utf-8") as fout:
        fieldnames = sorted({key for row in output_rows for key in row})
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    event_csv = RESULTS / "coops_verified_flat_flag_events.csv"
    with event_csv.open("w", newline="", encoding="utf-8") as fout:
        fieldnames = ["station", "name", "state", "year", "month", "time", "wl_value", "f", "r", "t", "i", "url"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(event_rows)

    summary_json = RESULTS / "coops_verified_flat_flag_screen_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(month_csv)


if __name__ == "__main__":
    main()
