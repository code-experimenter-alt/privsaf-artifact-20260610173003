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

STATIONS_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi/stations.json?type=waterlevels"
DATAGETTER = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
RESPONSE_HELP = "https://api.tidesandcurrents.noaa.gov/api/prod/responseHelp.html"
FLAT_FLAG_DOC = "https://tidesandcurrents.noaa.gov/publications/NOAA_Technical_Report_NOS_CO-OPS_030_QC_requirements_doc%28revised%29-11102004.pdf"


def fetch_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "PrivSAF data audit"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def station_list() -> list[dict[str, str]]:
    payload = fetch_json(STATIONS_URL)
    return [
        {
            "station": str(station["id"]),
            "name": str(station.get("name", "")),
            "state": str(station.get("state", "")),
            "affiliations": str(station.get("affiliations", "")),
        }
        for station in payload.get("stations", [])
    ]


def data_url(station: str) -> str:
    params = {
        "date": "recent",
        "station": station,
        "product": "water_level",
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


def summarize_station(station: dict[str, str]) -> dict[str, object]:
    url = data_url(station["station"])
    payload = fetch_json(url)
    rows = payload.get("data", [])
    flat_rows = []
    any_flag_rows = 0
    q_counts = Counter()
    for row in rows:
        bits = flag_bits(str(row.get("f", "")))
        q_counts[str(row.get("q", ""))] += 1
        any_flag_rows += int(any(bit not in {"", "0"} for bit in bits))
        if bits[1] == "1":
            flat_rows.append(row)

    return {
        **station,
        "status": "ok",
        "row_count": len(rows),
        "first_time": rows[0]["t"] if rows else "",
        "last_time": rows[-1]["t"] if rows else "",
        "any_flag_rows": any_flag_rows,
        "flat_flag_rows": len(flat_rows),
        "flat_first_time": flat_rows[0]["t"] if flat_rows else "",
        "flat_last_time": flat_rows[-1]["t"] if flat_rows else "",
        "flat_first_value": flat_rows[0]["v"] if flat_rows else "",
        "flat_first_flag": flat_rows[0]["f"] if flat_rows else "",
        "q_counts": json.dumps(dict(sorted(q_counts.items())), sort_keys=True),
        "url": url,
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    stations = station_list()
    output_rows: list[dict[str, object]] = []
    for idx, station in enumerate(stations, start=1):
        print(f"{idx}/{len(stations)} {station['station']} {station['name']}", flush=True)
        try:
            output_rows.append(summarize_station(station))
        except Exception as exc:  # noqa: BLE001
            output_rows.append(
                {
                    **station,
                    "status": "error",
                    "row_count": 0,
                    "first_time": "",
                    "last_time": "",
                    "any_flag_rows": 0,
                    "flat_flag_rows": 0,
                    "flat_first_time": "",
                    "flat_last_time": "",
                    "flat_first_value": "",
                    "flat_first_flag": "",
                    "q_counts": "{}",
                    "url": data_url(station["station"]),
                    "error": repr(exc),
                }
            )
        time.sleep(0.05)

    csv_path = RESULTS / "coops_recent_flat_flag_screen.csv"
    fieldnames = sorted({key for row in output_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    positives = [row for row in output_rows if int(row.get("flat_flag_rows", 0))]
    summary = {
        "status": "pass",
        "source": "NOAA CO-OPS preliminary water_level API recent window",
        "stations_url": STATIONS_URL,
        "response_help": RESPONSE_HELP,
        "flat_flag_doc": FLAT_FLAG_DOC,
        "stations_checked": len(output_rows),
        "ok_stations": sum(row["status"] == "ok" for row in output_rows),
        "error_stations": sum(row["status"] != "ok" for row in output_rows),
        "stations_with_flat_flag": len(positives),
        "flat_flag_rows": sum(int(row.get("flat_flag_rows", 0)) for row in output_rows),
        "positive_stations": [
            {
                "station": row["station"],
                "name": row["name"],
                "state": row["state"],
                "flat_flag_rows": row["flat_flag_rows"],
                "flat_first_time": row["flat_first_time"],
                "flat_last_time": row["flat_last_time"],
                "flat_first_value": row["flat_first_value"],
                "flat_first_flag": row["flat_first_flag"],
            }
            for row in positives
        ],
        "interpretation": (
            "CO-OPS preliminary water_level rows expose measured values and data flags in the same API response. "
            "The second f bit is the official flat tolerance flag, so F=1 rows are direct public value+label candidates "
            "for real flatline screening under PM-LDP."
        ),
    }
    json_path = RESULTS / "coops_recent_flat_flag_screen_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(csv_path)


if __name__ == "__main__":
    main()
