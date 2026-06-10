from __future__ import annotations

import csv
import json
import re
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

REPO_API = "https://api.github.com/repos/GEUS-Glaciology-and-Climate/PROMICE-AWS-data-issues"
FLAGS_API = f"{REPO_API}/contents/flags"
ISSUES_API = f"{REPO_API}/issues"
RAW_PREFIX = "https://raw.githubusercontent.com/GEUS-Glaciology-and-Climate/PROMICE-AWS-data-issues/master/flags"
THREDDS_L2 = "https://thredds.geus.dk/thredds/catalog/aws/l2stations/netcdf/hour/catalog.html"
ESSD_2026 = "https://essd.copernicus.org/articles/18/2829/2026/"

EXPLICIT_REPEATED_RE = re.compile(
    r"\b(constant|flat|stuck|persist|zero values|too much zero|give 0degc|same value)\b",
    re.IGNORECASE,
)
SENSOR_FAILURE_RE = re.compile(
    r"\b(not working|malfunction|sensor drift|suspicious|bad data|disfunctional|failure|failed)\b",
    re.IGNORECASE,
)
SCALAR_VARIABLE_RE = re.compile(r"^(t|p|rh|wspd|wdir|t_i_)", re.IGNORECASE)


def fetch_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as response:
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


def duration_hours(t0: str, t1: str) -> float | None:
    start = parse_dt(t0)
    end = parse_dt(t1)
    if start is None or end is None:
        return None
    return (end - start).total_seconds() / 3600.0


def split_variables(expr: str) -> list[str]:
    return [part.strip() for part in re.split(r"\s+", expr.strip()) if part.strip()]


def variable_family(variable: str) -> str:
    if variable.startswith("t_i_") or variable in {"t_i", "t_u", "t_l"}:
        return "temperature"
    if variable in {"p_i", "p_u", "p_l"}:
        return "pressure"
    if variable in {"rh_i", "rh_u", "rh_l"}:
        return "humidity"
    if variable.startswith("wspd"):
        return "wind_speed"
    if variable.startswith("wdir"):
        return "wind_direction"
    return "other"


def relevance(row: dict[str, str]) -> str:
    text = f"{row.get('comment', '')} {row.get('variable', '')} {row.get('flag', '')}"
    if EXPLICIT_REPEATED_RE.search(text):
        return "explicit_repeated_value_or_zero"
    if SENSOR_FAILURE_RE.search(text):
        return "sensor_failure_or_suspicious"
    return "other_manual_flag"


def issue_number(url: str) -> str:
    match = re.search(r"/issues/(\d+)", url)
    return match.group(1) if match else ""


def commented_out(raw: dict[str, str]) -> int:
    return int(raw.get("t0", "").lstrip().startswith("#") or raw.get("t1", "").lstrip().startswith("#"))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    contents = fetch_json(FLAGS_API)
    if not isinstance(contents, list):
        raise TypeError("GitHub flags API did not return a file list.")

    issue_cache: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    source_files: list[dict[str, object]] = []

    for entry in contents:
        if not isinstance(entry, dict) or not str(entry.get("name", "")).endswith(".csv"):
            continue
        station = str(entry["name"]).removesuffix(".csv")
        download_url = str(entry.get("download_url") or f"{RAW_PREFIX}/{entry['name']}")
        text = fetch_text(download_url)
        source_files.append(
            {
                "station": station,
                "name": entry["name"],
                "download_url": download_url,
                "bytes": len(text.encode("utf-8")),
            }
        )
        reader = csv.DictReader(text.splitlines())
        for raw in reader:
            if not raw:
                continue
            raw = {key: (value or "").strip() for key, value in raw.items()}
            category = relevance(raw)
            variables = split_variables(raw.get("variable", ""))
            scalar_variables = [var for var in variables if SCALAR_VARIABLE_RE.search(var)]
            issue = issue_number(raw.get("URL_graphic", ""))
            issue_title = ""
            is_commented_out = commented_out(raw)
            if issue:
                if issue not in issue_cache:
                    issue_data = fetch_json(f"{ISSUES_API}/{issue}")
                    if isinstance(issue_data, dict):
                        issue_cache[issue] = issue_data
                issue_title = str(issue_cache.get(issue, {}).get("title", ""))

            rows.append(
                {
                    "station": station,
                    "t0": raw.get("t0", ""),
                    "t1": raw.get("t1", ""),
                    "duration_hours": duration_hours(raw.get("t0", ""), raw.get("t1", "")),
                    "variable_expr": raw.get("variable", ""),
                    "scalar_variable_count": len(scalar_variables),
                    "variable_families": " ".join(sorted({variable_family(var) for var in scalar_variables})),
                    "flag": raw.get("flag", ""),
                    "comment": raw.get("comment", ""),
                    "issue": issue,
                    "issue_title": issue_title,
                    "issue_url": raw.get("URL_graphic", ""),
                    "relevance": category,
                    "commented_out": is_commented_out,
                    "usable_for_pmldp_triage": int(
                        not is_commented_out and category == "explicit_repeated_value_or_zero" and bool(scalar_variables)
                    ),
                }
            )

    rows.sort(key=lambda row: (str(row["relevance"]), str(row["station"]), str(row["t0"]), str(row["variable_expr"])))
    inventory_path = RESULTS / "promice_manual_flatline_flag_inventory.csv"
    fieldnames = [
        "station",
        "t0",
        "t1",
        "duration_hours",
        "variable_expr",
        "scalar_variable_count",
        "variable_families",
        "flag",
        "comment",
        "issue",
        "issue_title",
        "issue_url",
        "relevance",
        "commented_out",
        "usable_for_pmldp_triage",
    ]
    with inventory_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    active_rows = [row for row in rows if not int(row["commented_out"])]
    families = Counter()
    for row in rows:
        if int(row["usable_for_pmldp_triage"]):
            for family in str(row["variable_families"]).split():
                families[family] += 1
    summary = {
        "status": "pass",
        "source": "PROMICE/GC-Net AWS manual QA/QC flags",
        "flags_api": FLAGS_API,
        "thredds_l2_catalog": THREDDS_L2,
        "essd_2026": ESSD_2026,
        "flag_files": len(source_files),
        "manual_flag_rows": len(rows),
        "active_manual_flag_rows": len(active_rows),
        "commented_out_rows": sum(int(row["commented_out"]) for row in rows),
        "explicit_repeated_value_or_zero_rows": sum(
            row["relevance"] == "explicit_repeated_value_or_zero" for row in active_rows
        ),
        "sensor_failure_or_suspicious_rows": sum(
            row["relevance"] == "sensor_failure_or_suspicious" for row in active_rows
        ),
        "usable_for_pmldp_triage_rows": sum(int(row["usable_for_pmldp_triage"]) for row in rows),
        "usable_by_variable_family": dict(sorted(families.items())),
        "issues_referenced": len(issue_cache),
        "interpretation": (
            "PROMICE/GC-Net supplies public station-level manual QA/QC flag intervals with comments and issue links. "
            "Many rows explicitly identify constant, flat, or too-many-zero scalar sensor periods, making it a strong "
            "candidate real flatline/stuck source. The next gate is to pair these intervals with a value product that "
            "preserves the flagged observations; the sampled L2 NetCDF product records cleaned variables and does not "
            "obviously expose per-sample QC labels for the main meteorological variables."
        ),
    }
    summary_json = RESULTS / "promice_manual_flatline_flag_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary_csv = RESULTS / "promice_manual_flatline_flag_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    sources_path = RESULTS / "promice_manual_flatline_flag_sources.csv"
    with sources_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["station", "name", "download_url", "bytes"])
        writer.writeheader()
        writer.writerows(source_files)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(inventory_path)


if __name__ == "__main__":
    main()
