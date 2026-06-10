from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCREEN = RESULTS / "coops_recent_flat_flag_screen.csv"

DATAGETTER = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
RESPONSE_HELP = "https://api.tidesandcurrents.noaa.gov/api/prod/responseHelp.html"
FLAT_FLAG_DOC = "https://tidesandcurrents.noaa.gov/publications/NOAA_Technical_Report_NOS_CO-OPS_030_QC_requirements_doc%28revised%29-11102004.pdf"


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "PrivSAF data audit"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def data_url(station: str, product: str) -> str:
    params = {
        "date": "recent",
        "station": station,
        "product": product,
        "datum": "MLLW",
        "time_zone": "gmt",
        "units": "metric",
        "application": "PrivSAF",
        "format": "json",
    }
    return f"{DATAGETTER}?{urllib.parse.urlencode(params)}"


def flag_bits(value: str) -> list[str]:
    bits = [part.strip() for part in value.split(",")]
    while len(bits) < 4:
        bits.append("")
    return bits[:4]


def numeric(value: object) -> int:
    if value is None:
        return 0
    text = str(value).strip()
    if not text:
        return 0
    try:
        float(text)
    except ValueError:
        return 0
    return 1


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    screen_rows = list(csv.DictReader(SCREEN.open(encoding="utf-8")))
    positives = [row for row in screen_rows if int(row.get("flat_flag_rows") or 0) > 0]

    event_rows: list[dict[str, object]] = []
    station_rows: list[dict[str, object]] = []
    for idx, station in enumerate(positives, start=1):
        station_id = station["station"]
        print(f"{idx}/{len(positives)} {station_id} {station['name']}", flush=True)
        one_minute = {}
        one_minute_status = "ok"
        try:
            one_payload = fetch_json(data_url(station_id, "one_minute_water_level"))
            one_minute = {row["t"]: row.get("v", "") for row in one_payload.get("data", [])}
            if "error" in one_payload:
                one_minute_status = one_payload["error"].get("message", "error")
        except Exception as exc:  # noqa: BLE001
            one_minute_status = repr(exc)

        water_payload = fetch_json(data_url(station_id, "water_level"))
        water_rows = water_payload.get("data", [])
        flat_rows = []
        for row in water_rows:
            bits = flag_bits(str(row.get("f", "")))
            if bits[1] != "1":
                continue
            one_value = one_minute.get(row["t"], "")
            event_rows.append(
                {
                    "station": station_id,
                    "name": station["name"],
                    "state": station["state"],
                    "t": row["t"],
                    "water_level_v": row.get("v", ""),
                    "water_level_s": row.get("s", ""),
                    "water_level_f": row.get("f", ""),
                    "water_level_q": row.get("q", ""),
                    "one_minute_v": one_value,
                    "water_level_numeric": numeric(row.get("v", "")),
                    "one_minute_numeric": numeric(one_value),
                }
            )
            flat_rows.append((row, one_value))

        station_rows.append(
            {
                "station": station_id,
                "name": station["name"],
                "state": station["state"],
                "flat_flag_rows": len(flat_rows),
                "water_level_numeric_flat_rows": sum(numeric(row.get("v", "")) for row, _ in flat_rows),
                "one_minute_numeric_at_flat_rows": sum(numeric(value) for _, value in flat_rows),
                "one_minute_rows": len(one_minute),
                "one_minute_status": one_minute_status,
                "water_level_url": data_url(station_id, "water_level"),
                "one_minute_url": data_url(station_id, "one_minute_water_level"),
            }
        )
        time.sleep(0.05)

    events_csv = RESULTS / "coops_recent_flat_flag_event_rows.csv"
    event_fields = [
        "station",
        "name",
        "state",
        "t",
        "water_level_v",
        "water_level_s",
        "water_level_f",
        "water_level_q",
        "one_minute_v",
        "water_level_numeric",
        "one_minute_numeric",
    ]
    with events_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(event_rows)

    station_csv = RESULTS / "coops_recent_flat_flag_value_availability.csv"
    station_fields = [
        "station",
        "name",
        "state",
        "flat_flag_rows",
        "water_level_numeric_flat_rows",
        "one_minute_numeric_at_flat_rows",
        "one_minute_rows",
        "one_minute_status",
        "water_level_url",
        "one_minute_url",
    ]
    with station_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=station_fields)
        writer.writeheader()
        writer.writerows(station_rows)

    status_counts = Counter(row["one_minute_status"] for row in station_rows)
    summary = {
        "status": "pass",
        "source": "NOAA CO-OPS recent water_level flat-flag rows with one_minute value alignment",
        "response_help": RESPONSE_HELP,
        "flat_flag_doc": FLAT_FLAG_DOC,
        "positive_stations_checked": len(station_rows),
        "flat_flag_event_rows": len(event_rows),
        "water_level_numeric_flat_rows": sum(int(row["water_level_numeric"]) for row in event_rows),
        "one_minute_numeric_at_flat_rows": sum(int(row["one_minute_numeric"]) for row in event_rows),
        "stations_with_numeric_water_level_flat_rows": sum(
            int(row["water_level_numeric_flat_rows"]) > 0 for row in station_rows
        ),
        "stations_with_numeric_one_minute_at_flat_rows": sum(
            int(row["one_minute_numeric_at_flat_rows"]) > 0 for row in station_rows
        ),
        "one_minute_status_counts": dict(sorted(status_counts.items())),
        "interpretation": (
            "The CO-OPS flat flag is an official real flat-tolerance label, but in the recent preliminary API snapshot "
            "the F=1 rows have blank water_level values. Aligning one_minute_water_level at the same timestamps recovers "
            "only sparse numeric values, which is not enough for a robust PrivSAF performance panel. This source remains "
            "a strong label source but is not yet a ready value+label benchmark without an earlier/raw value layer."
        ),
    }
    summary_json = RESULTS / "coops_recent_flat_flag_value_availability_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(events_csv)


if __name__ == "__main__":
    main()
