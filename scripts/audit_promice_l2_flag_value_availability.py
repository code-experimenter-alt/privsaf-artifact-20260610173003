from __future__ import annotations

import csv
import json
import math
import re
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from sys import stderr


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INVENTORY = RESULTS / "promice_manual_flatline_flag_inventory.csv"

OPENDAP_PREFIX = "https://thredds.geus.dk/thredds/dodsC/aws/l2stations/netcdf/hour"
CATALOG_URL = "https://thredds.geus.dk/thredds/catalog/aws/l2stations/netcdf/hour/catalog.html"

NAN_TOKENS = {"nan", "NaN", "NAN"}
MAX_SEGMENT = 24


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8")


def parse_dt(value: str) -> datetime | None:
    value = value.strip()
    if value.startswith("#"):
        value = value[1:].strip()
    if not value:
        return None
    if value.endswith("+00:00"):
        value = value[:-6]
    return datetime.fromisoformat(value)


def parse_dds(dds: str) -> tuple[list[str], int]:
    variables: list[str] = []
    time_len = 0
    for line in dds.splitlines():
        match = re.match(r"\s+\w+\s+(\w+)\[time = (\d+)\];", line)
        if match:
            variables.append(match.group(1))
            time_len = int(match.group(2))
    return variables, time_len


def parse_base_time(das: str) -> datetime:
    match = re.search(r'String units "hours since ([^"]+)";', das)
    if not match:
        raise ValueError("Could not find time units in DAS response.")
    return datetime.fromisoformat(match.group(1))


def station_metadata(station: str) -> tuple[list[str], int, datetime]:
    dataset = f"{OPENDAP_PREFIX}/{station}_hour.nc"
    dds = fetch_text(f"{dataset}.dds")
    das = fetch_text(f"{dataset}.das")
    variables, time_len = parse_dds(dds)
    return variables, time_len, parse_base_time(das)


def expand_variables(expr: str, variables: list[str]) -> list[str]:
    expanded: list[str] = []
    parts = [part.strip() for part in re.split(r"\s+", expr.strip()) if part.strip()]
    for part in parts:
        if part == "*":
            continue
        if ".*" in part or part.endswith("($)"):
            try:
                pattern = re.compile(part)
            except re.error:
                continue
            expanded.extend(var for var in variables if pattern.fullmatch(var))
        elif part in variables:
            expanded.append(part)
    return sorted(dict.fromkeys(expanded))


def index_at(dt: datetime, base: datetime) -> int:
    return int(math.floor((dt - base).total_seconds() / 3600.0))


def segment_ranges(start_idx: int, end_idx: int, time_len: int) -> list[tuple[int, int]]:
    start_idx = max(0, min(start_idx, time_len - 1))
    end_idx = max(0, min(end_idx, time_len - 1))
    if end_idx < start_idx:
        start_idx, end_idx = end_idx, start_idx
    return [(start_idx, min(start_idx + MAX_SEGMENT - 1, end_idx))]


def parse_ascii_values(text: str, variable: str) -> list[str]:
    marker = f"{variable}["
    lines = text.splitlines()
    values: list[str] = []
    for idx, line in enumerate(lines):
        if line.startswith(marker) and idx + 1 < len(lines):
            values.extend(token.strip() for token in lines[idx + 1].split(",") if token.strip())
    return values


def fetch_values(station: str, variable: str, ranges: list[tuple[int, int]]) -> tuple[list[str], str]:
    constraints = ",".join(f"{variable}[{lo}:1:{hi}]" for lo, hi in ranges)
    url = f"{OPENDAP_PREFIX}/{station}_hour.nc.ascii?{constraints}"
    return parse_ascii_values(fetch_text(url), variable), url


def value_status(values: list[str]) -> str:
    if not values:
        return "no_values_returned"
    non_nan = [value for value in values if value not in NAN_TOKENS]
    if not non_nan:
        return "all_nan_in_l2_samples"
    if len(non_nan) == len(values):
        return "values_preserved_in_l2_samples"
    return "partially_nan_in_l2_samples"


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows = list(csv.DictReader(INVENTORY.open(encoding="utf-8")))
    candidates = [row for row in rows if row.get("usable_for_pmldp_triage") == "1"]

    station_cache: dict[str, tuple[list[str], int, datetime]] = {}
    audit_rows: list[dict[str, object]] = []

    for row in candidates:
        station = row["station"]
        print(f"station={station} expr={row['variable_expr']}", file=stderr, flush=True)
        try:
            if station not in station_cache:
                station_cache[station] = station_metadata(station)
            variables, time_len, base_time = station_cache[station]
        except Exception as exc:  # noqa: BLE001 - this is an audit, so failures are data.
            audit_rows.append(
                {
                    "station": station,
                    "variable_expr": row["variable_expr"],
                    "variable": "",
                    "t0": row["t0"],
                    "t1": row["t1"],
                    "status": "station_product_unavailable",
                    "detail": repr(exc),
                }
            )
            continue

        start = parse_dt(row["t0"])
        end = parse_dt(row["t1"])
        expanded = expand_variables(row["variable_expr"], variables)
        if start is None or end is None:
            status = "missing_time_bound"
        elif not expanded:
            status = "no_matching_l2_variable"
        else:
            status = ""

        if status:
            audit_rows.append(
                {
                    "station": station,
                    "variable_expr": row["variable_expr"],
                    "variable": " ".join(expanded),
                    "t0": row["t0"],
                    "t1": row["t1"],
                    "status": status,
                    "detail": "",
                }
            )
            continue

        assert start is not None and end is not None
        start_idx = index_at(start, base_time)
        end_idx = index_at(end, base_time)
        ranges = segment_ranges(start_idx, end_idx, time_len)

        for variable in expanded:
            try:
                values, url = fetch_values(station, variable, ranges)
                non_nan = [value for value in values if value not in NAN_TOKENS]
                audit_rows.append(
                    {
                        "station": station,
                        "variable_expr": row["variable_expr"],
                        "variable": variable,
                        "t0": row["t0"],
                        "t1": row["t1"],
                        "duration_hours": row["duration_hours"],
                        "issue": row["issue"],
                        "issue_title": row["issue_title"],
                        "flag": row["flag"],
                        "comment": row["comment"],
                        "l2_product_url": f"{OPENDAP_PREFIX}/{station}_hour.nc",
                        "base_time": base_time.isoformat(),
                        "time_len": time_len,
                        "start_index": start_idx,
                        "end_index": end_idx,
                        "sample_ranges": " ".join(f"{lo}:{hi}" for lo, hi in ranges),
                        "sample_count": len(values),
                        "non_nan_count": len(non_nan),
                        "unique_non_nan_count": len(set(non_nan)),
                        "first_non_nan_values": " ".join(non_nan[:8]),
                        "status": value_status(values),
                        "detail": url,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                audit_rows.append(
                    {
                        "station": station,
                        "variable_expr": row["variable_expr"],
                        "variable": variable,
                        "t0": row["t0"],
                        "t1": row["t1"],
                        "duration_hours": row["duration_hours"],
                        "issue": row["issue"],
                        "issue_title": row["issue_title"],
                        "flag": row["flag"],
                        "comment": row["comment"],
                        "status": "value_request_failed",
                        "detail": repr(exc),
                    }
                )

    fieldnames = sorted({key for row in audit_rows for key in row})
    output_csv = RESULTS / "promice_l2_flag_value_availability.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    counts = Counter(str(row["status"]) for row in audit_rows)
    summary = {
        "status": "pass",
        "source_inventory": str(INVENTORY.relative_to(ROOT)),
        "l2_catalog": CATALOG_URL,
        "candidate_flag_rows": len(candidates),
        "audited_station_variable_rows": len(audit_rows),
        "status_counts": dict(sorted(counts.items())),
        "stations_checked": sorted(station_cache),
        "interpretation": (
            "PROMICE L2 station products are public and expose the candidate variables, but L2 is post-manual-flagging. "
            "A candidate interval is usable as a value+label benchmark only when the flagged variable has non-NaN values "
            "inside the sampled interval. all_nan_in_l2_samples means the public L2 product can document the flag interval "
            "but does not preserve the faulty values needed for a PrivSAF detection/repair performance panel."
        ),
    }
    output_json = RESULTS / "promice_l2_flag_value_availability_summary.json"
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(output_csv)


if __name__ == "__main__":
    main()
