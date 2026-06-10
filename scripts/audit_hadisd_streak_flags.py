from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
import re
import shutil
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BASE_URL = "https://hadleyserver.metoffice.gov.uk/hadobs/hadisd/v343_2025f"
STATION_ID = "702606-96401"
STATION_FILE = "hadisd.3.4.3.2025f_19310101-20250829_702606-96401.nc.gz"
STATION_URL = f"{BASE_URL}/data/{STATION_FILE}"
STATION_PAGE_URL = f"{BASE_URL}/station_download_7.html"
TEST_CODES_URL = f"{BASE_URL}/files/tests_codes.txt"
FAIL_SUMMARY_URL = f"{BASE_URL}/files/all_fails_summary_20251006.dat"
HADISD_PAGE_URL = "https://hadleyserver.metoffice.gov.uk/hadobs/hadisd/"

STRING_CODES = {
    "TSS": ("temperatures", 0, "temperature straight string"),
    "DSS": ("dewpoints", 1, "dew point straight string"),
    "WSS": ("windspeeds", 4, "wind speed straight string"),
    "PSS": ("slp", 2, "sea-level pressure straight string"),
    "RSS": ("winddirs", 5, "wind direction straight string"),
    "HTS": ("temperatures", 0, "temperature hour string"),
    "HDS": ("dewpoints", 1, "dew point hour string"),
    "HWS": ("windspeeds", 4, "wind speed hour string"),
    "HPS": ("slp", 2, "sea-level pressure hour string"),
    "DTS": ("temperatures", 0, "temperature day string"),
    "DDS": ("dewpoints", 1, "dew point day string"),
    "DWS": ("windspeeds", 4, "wind speed day string"),
    "DPS": ("slp", 2, "sea-level pressure day string"),
    "HRS": ("winddirs", 5, "wind direction hour string"),
    "DRS": ("winddirs", 5, "wind direction day string"),
}

VALUE_VARIABLES = sorted({item[0] for item in STRING_CODES.values()} | {"time", "flagged_obs"})


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def find_ncdump(explicit: str | None) -> Path:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    on_path = shutil.which("ncdump")
    if on_path:
        candidates.append(on_path)
    candidates.append("/tmp/hadisd_tools/usr/bin/ncdump")
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        "ncdump is required. Install netcdf-bin or pass --ncdump. "
        "For the audit run, Ubuntu .deb packages were unpacked under /tmp/hadisd_tools."
    )


def ncdump_env(ncdump: Path) -> dict[str, str] | None:
    local_lib = ncdump.parents[1] / "lib" / "x86_64-linux-gnu"
    if local_lib.exists():
        env = dict(**__import__("os").environ)
        env["LD_LIBRARY_PATH"] = str(local_lib)
        return env
    return None


def run_ncdump(ncdump: Path, nc_path: Path, variables: list[str], output: Path) -> None:
    cmd = [str(ncdump), "-v", ",".join(variables), str(nc_path)]
    with output.open("w", encoding="utf-8") as fout:
        subprocess.run(cmd, check=True, stdout=fout, env=ncdump_env(ncdump))


def gunzip(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return
    with gzip.open(src, "rb") as fin, dst.open("wb") as fout:
        shutil.copyfileobj(fin, fout)


def data_section(text: str) -> str:
    return text.split("\ndata:", 1)[1]


def variable_blob(cdl_text: str, variable: str) -> str:
    data = data_section(cdl_text)
    matches = list(re.finditer(r"\n\s*" + re.escape(variable) + r"\s*=\s*(.*?);", data, re.S))
    if not matches:
        raise ValueError(f"Missing {variable} in ncdump output.")
    return matches[-1].group(1)


def parse_number_array(cdl_text: str, variable: str, fill: float = math.nan) -> list[float]:
    blob = variable_blob(cdl_text, variable).replace("\n", ",")
    out: list[float] = []
    for token in blob.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(fill if token == "_" else float(token))
    return out


def parse_qc_matrix(cdl_text: str, rows: int, tests: int) -> list[list[int]]:
    blob = variable_blob(cdl_text, "quality_control_flags")
    values = [int(float(item)) for item in re.findall(r"-?\d+(?:\.\d*)?", blob)]
    if len(values) != rows * tests:
        raise ValueError(f"QC matrix has {len(values)} values; expected {rows * tests}.")
    return [values[i : i + tests] for i in range(0, len(values), tests)]


def parse_dimensions(header_text: str) -> tuple[int, int]:
    time_match = re.search(r"\btime\s*=\s*(\d+)\s*;", header_text)
    test_match = re.search(r"\btest\s*=\s*(\d+)\s*;", header_text)
    if not time_match or not test_match:
        raise ValueError("Could not read time/test dimensions from ncdump header.")
    return int(time_match.group(1)), int(test_match.group(1))


def parse_test_codes(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        code, description = line.split("=", 1)
        out[code.strip()] = " ".join(description.split())
    return out


def parse_summary_order(path: Path) -> tuple[list[str], dict[str, dict[str, object]]]:
    order: list[str] = []
    summary: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*(\d+)([A-Z_]+)\s+(.*)", line)
        if not match:
            continue
        index = int(match.group(1))
        code = match.group(2)
        parts = match.group(3).split()
        counts = [int(item) for item in parts[:8]]
        percentages = [float(item) for item in parts[8:16]]
        if index < 71:
            order.append(code)
        summary[code] = {
            "summary_index": index,
            "station_count_bins": counts,
            "station_percent_bins": percentages,
        }
    return order, summary


def run_lengths(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = indices[0]
    for index in indices[1:]:
        if index == prev + 1:
            prev = index
        else:
            runs.append((start, prev))
            start = prev = index
    runs.append((start, prev))
    return runs


def iso_time(hours_since_1931: float) -> str:
    base = dt.datetime(1931, 1, 1)
    return (base + dt.timedelta(hours=float(hours_since_1931))).isoformat(timespec="seconds")


def usable_value(variable_values: list[float], flagged_row: list[float], index: int, flag_col: int) -> float:
    flagged_value = flagged_row[flag_col]
    return flagged_value if not math.isnan(flagged_value) else variable_values[index]


def build_outputs(work_dir: Path, ncdump: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    station_gz = work_dir / STATION_FILE
    station_nc = work_dir / STATION_FILE.removesuffix(".gz")
    tests_path = work_dir / "tests_codes.txt"
    fail_summary_path = work_dir / "all_fails_summary_20251006.dat"
    qc_cdl = work_dir / f"{STATION_ID}_qc.cdl"
    values_cdl = work_dir / f"{STATION_ID}_values.cdl"
    header_cdl = work_dir / f"{STATION_ID}_header.cdl"

    fetch(STATION_URL, station_gz)
    fetch(TEST_CODES_URL, tests_path)
    fetch(FAIL_SUMMARY_URL, fail_summary_path)
    gunzip(station_gz, station_nc)

    with header_cdl.open("w", encoding="utf-8") as fout:
        subprocess.run([str(ncdump), "-h", str(station_nc)], check=True, stdout=fout, env=ncdump_env(ncdump))
    run_ncdump(ncdump, station_nc, ["quality_control_flags"], qc_cdl)
    run_ncdump(ncdump, station_nc, VALUE_VARIABLES, values_cdl)

    header_text = header_cdl.read_text(encoding="utf-8")
    qc_text = qc_cdl.read_text(encoding="utf-8")
    values_text = values_cdl.read_text(encoding="utf-8")
    rows, tests = parse_dimensions(header_text)
    qc = parse_qc_matrix(qc_text, rows, tests)
    descriptions = parse_test_codes(tests_path)
    order, fail_summary = parse_summary_order(fail_summary_path)
    if len(order) != tests:
        raise ValueError(f"Expected {tests} test codes from fail summary, got {len(order)}.")

    arrays = {variable: parse_number_array(values_text, variable) for variable in VALUE_VARIABLES if variable != "flagged_obs"}
    flagged_values = parse_number_array(values_text, "flagged_obs")
    flagged_rows = [flagged_values[i : i + 19] for i in range(0, len(flagged_values), 19)]
    if len(flagged_rows) != rows:
        raise ValueError(f"flagged_obs has {len(flagged_rows)} rows; expected {rows}.")

    summary_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for test_index, code in enumerate(order):
        positives = [idx for idx, row in enumerate(qc) if row[test_index] > 0]
        negatives = [idx for idx, row in enumerate(qc) if row[test_index] < 0]
        runs = run_lengths(positives)
        info = STRING_CODES.get(code)
        is_string = int(info is not None)
        variable = info[0] if info else ""
        summary_row = {
            "source": "HadISD v3.4.3.2025f",
            "station_id": STATION_ID,
            "station_file_url": STATION_URL,
            "station_page_url": STATION_PAGE_URL,
            "test_index": test_index,
            "test_code": code,
            "description": descriptions.get(code, ""),
            "rows": rows,
            "positive_flags": len(positives),
            "negative_flags": len(negatives),
            "nonzero_flags": len(positives) + len(negatives),
            "positive_runs": len(runs),
            "max_positive_run_length": max((end - start + 1 for start, end in runs), default=0),
            "string_or_streak_code": is_string,
            "mapped_variable": variable,
            "field_relevance": "direct repeated-value QC flag" if is_string else "other HadISD QC flag",
            "fit_for_privsaf": (
                "semantic candidate; needs multi-station PM-LDP operator run"
                if is_string and positives
                else "not used for PrivSAF real-streak evidence"
            ),
            "all_station_bin_counts": json.dumps(fail_summary.get(code, {}).get("station_count_bins", [])),
            "all_station_bin_percentages": json.dumps(fail_summary.get(code, {}).get("station_percent_bins", [])),
        }
        summary_rows.append(summary_row)
        if not info:
            continue
        variable, flagged_col, label = info
        for start, end in runs:
            for index in range(start, end + 1):
                other = [order[col] for col, value in enumerate(qc[index]) if value > 0]
                value = usable_value(arrays[variable], flagged_rows[index], index, flagged_col)
                event_rows.append(
                    {
                        "source": "HadISD v3.4.3.2025f",
                        "station_id": STATION_ID,
                        "test_code": code,
                        "event_label": label,
                        "row_index": index,
                        "timestamp": iso_time(arrays["time"][index]),
                        "mapped_variable": variable,
                        "usable_raw_or_flagged_value": value,
                        "stored_variable_value": arrays[variable][index],
                        "flagged_obs_value": flagged_rows[index][flagged_col],
                        "positive_run_start": start,
                        "positive_run_end": end,
                        "positive_run_length": end - start + 1,
                        "cooccurring_positive_flags": "|".join(other),
                    }
                )

    metadata = {
        "status": "pass",
        "source": "HadISD v3.4.3.2025f",
        "station_id": STATION_ID,
        "station_url": STATION_URL,
        "hadisd_page_url": HADISD_PAGE_URL,
        "test_codes_url": TEST_CODES_URL,
        "fail_summary_url": FAIL_SUMMARY_URL,
        "rows": rows,
        "qc_tests": tests,
        "station_file_sha256": sha256sum(station_gz),
        "ncdump": str(ncdump),
        "ncdump_sha256": sha256sum(ncdump),
        "string_code_positive_flags": {
            row["test_code"]: row["positive_flags"]
            for row in summary_rows
            if row["string_or_streak_code"] and row["positive_flags"]
        },
        "interpretation": (
            "HadISD exposes public per-station NetCDF4 QC flags with repeated-value string checks. "
            "The audited station contains real positive straight-string flags, but this single-station "
            "audit does not by itself prove PrivSAF is the selected PM-LDP operator."
        ),
    }
    return summary_rows, event_rows, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HadISD real string/streak QC flags.")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/privsaf_hadisd_audit"))
    parser.add_argument("--ncdump", default=None)
    parser.add_argument("--output-prefix", default="hadisd_real_streak")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    ncdump = find_ncdump(args.ncdump)
    summary_rows, event_rows, metadata = build_outputs(args.work_dir, ncdump)

    summary_path = RESULTS / f"{args.output_prefix}_flag_audit.csv"
    events_path = RESULTS / f"{args.output_prefix}_event_detail.csv"
    metadata_path = RESULTS / f"{args.output_prefix}_summary.json"

    with summary_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with events_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(event_rows[0].keys()))
        writer.writeheader()
        writer.writerows(event_rows)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(f"Wrote {events_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
