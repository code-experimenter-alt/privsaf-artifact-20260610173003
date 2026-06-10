from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import time
from pathlib import Path
from urllib.parse import urljoin

from audit_hadisd_streak_flags import (
    BASE_URL,
    FAIL_SUMMARY_URL,
    HADISD_PAGE_URL,
    RESULTS,
    TEST_CODES_URL,
    fetch,
    find_ncdump,
    gunzip,
    ncdump_env,
    parse_number_array,
    parse_qc_matrix,
    parse_summary_order,
    run_ncdump,
    sha256sum,
)
from run_hadisd_streak_pmldp_panel import (
    auprc,
    auroc,
    channel_likelihood_scan,
    discretize,
    histogram_alpha,
    hmm_infer,
    mean,
    mixture_infer,
    normalize,
    pm_matrix,
    pm_sample_one,
    precision_at_k,
    stdev,
    valid_value,
    write_csv,
)


STATION_PAGE_URL = f"{BASE_URL}/station_download_7.html"

DEFAULT_STATION_IDS = [
    "702606-96401",
    "702600-26435",
    "703050-99999",
    "722780-23183",
    "722950-23174",
    "724940-23234",
    "725030-14732",
    "727930-24233",
]

CASES = {
    "TSS": ("temperatures", 0, "temperature straight string"),
    "DSS": ("dewpoints", 1, "dew point straight string"),
    "WSS": ("windspeeds", 4, "wind speed straight string"),
    "PSS": ("slp", 2, "sea-level pressure straight string"),
    "RSS": ("winddirs", 5, "wind direction straight string"),
}

VALUE_VARIABLES = sorted({"time", "flagged_obs"} | {info[0] for info in CASES.values()})


def parse_size_mb(text: str) -> float:
    text = text.strip()
    match = re.match(r"([0-9.]+)\s*([KMG])", text)
    if not match:
        return math.nan
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "K":
        return value / 1024.0
    if unit == "G":
        return value * 1024.0
    return value


def parse_station_page(path: Path, page_url: str) -> dict[str, dict[str, object]]:
    html = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"<tr><td>(?P<station>[0-9]{6}-[0-9]{5})</td><td><a href=\"(?P<href>[^\"]+)\">[^<]+</a></td>"
        r"<td>(?P<size>[^<]+)</td><td>(?P<start>[^<]+)</td><td>(?P<end>[^<]+)</td></tr>"
    )
    stations: dict[str, dict[str, object]] = {}
    for match in pattern.finditer(html):
        station_id = match.group("station")
        href = match.group("href")
        stations[station_id] = {
            "station_id": station_id,
            "station_file": Path(href).name,
            "station_url": urljoin(page_url, href),
            "page_url": page_url,
            "size": match.group("size"),
            "size_mb": parse_size_mb(match.group("size")),
            "start": match.group("start"),
            "end": match.group("end"),
        }
    return stations


def fetch_station_page(work_dir: Path, page_url: str) -> Path:
    page_path = work_dir / Path(page_url).name
    fetch(page_url, page_path)
    return page_path


def run_header(ncdump: Path, nc_path: Path, output: Path) -> None:
    if output.exists() and output.stat().st_size > 0:
        return
    with output.open("w", encoding="utf-8") as fout:
        subprocess.run([str(ncdump), "-h", str(nc_path)], check=True, stdout=fout, env=ncdump_env(ncdump))


def run_ncdump_cached(ncdump: Path, nc_path: Path, variables: list[str], output: Path) -> None:
    if output.exists() and output.stat().st_size > 0:
        return
    run_ncdump(ncdump, nc_path, variables, output)


def parse_dimensions(header_text: str) -> tuple[int, int]:
    time_match = re.search(r"\btime\s*=\s*(\d+)\s*;", header_text)
    test_match = re.search(r"\btest\s*=\s*(\d+)\s*;", header_text)
    if not time_match or not test_match:
        raise ValueError("Could not parse HadISD time/test dimensions.")
    return int(time_match.group(1)), int(test_match.group(1))


def station_work_dir(base: Path, station_id: str) -> Path:
    path = base / station_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_station(
    station: dict[str, object],
    work_dir: Path,
    ncdump: Path,
) -> tuple[list[str], list[list[int]], dict[str, list[float]], list[list[float]], dict[str, object]]:
    station_id = str(station["station_id"])
    station_dir = station_work_dir(work_dir, station_id)
    station_gz = station_dir / str(station["station_file"])
    station_nc = station_dir / str(station["station_file"]).removesuffix(".gz")
    header_cdl = station_dir / f"{station_id}_header.cdl"
    qc_cdl = station_dir / f"{station_id}_qc.cdl"
    values_cdl = station_dir / f"{station_id}_values.cdl"
    tests_path = work_dir / "tests_codes.txt"
    fail_summary_path = work_dir / "all_fails_summary_20251006.dat"

    fetch(str(station["station_url"]), station_gz)
    fetch(TEST_CODES_URL, tests_path)
    fetch(FAIL_SUMMARY_URL, fail_summary_path)
    gunzip(station_gz, station_nc)
    run_header(ncdump, station_nc, header_cdl)
    run_ncdump_cached(ncdump, station_nc, ["quality_control_flags"], qc_cdl)
    run_ncdump_cached(ncdump, station_nc, VALUE_VARIABLES, values_cdl)

    header_text = header_cdl.read_text(encoding="utf-8")
    qc_text = qc_cdl.read_text(encoding="utf-8")
    values_text = values_cdl.read_text(encoding="utf-8")
    rows, tests = parse_dimensions(header_text)
    qc = parse_qc_matrix(qc_text, rows, tests)
    order, _ = parse_summary_order(fail_summary_path)
    if len(order) != tests:
        raise ValueError(f"{station_id}: expected {tests} QC test codes, got {len(order)}.")

    arrays = {variable: parse_number_array(values_text, variable) for variable in VALUE_VARIABLES if variable != "flagged_obs"}
    flagged_values = parse_number_array(values_text, "flagged_obs")
    flagged_rows = [flagged_values[i : i + 19] for i in range(0, len(flagged_values), 19)]
    if len(flagged_rows) != rows:
        raise ValueError(f"{station_id}: flagged_obs has {len(flagged_rows)} rows; expected {rows}.")

    metadata = {
        "rows": rows,
        "qc_tests": tests,
        "station_file_sha256": sha256sum(station_gz),
        "station_file_bytes": station_gz.stat().st_size,
    }
    return order, qc, arrays, flagged_rows, metadata


def row_value(arrays: dict[str, list[float]], flagged_rows: list[list[float]], variable: str, flag_col: int, index: int) -> float:
    flagged = flagged_rows[index][flag_col]
    return flagged if not math.isnan(flagged) else arrays[variable][index]


def build_case(
    code: str,
    order: list[str],
    qc: list[list[int]],
    arrays: dict[str, list[float]],
    flagged_rows: list[list[float]],
) -> tuple[list[float], list[int], str]:
    variable, flag_col, label = CASES[code]
    test_index = order.index(code)
    values: list[float] = []
    labels: list[int] = []
    for index, row in enumerate(qc):
        value = row_value(arrays, flagged_rows, variable, flag_col, index)
        if not valid_value(variable, value):
            continue
        values.append(float(value))
        labels.append(1 if row[test_index] > 0 else 0)
    return values, labels, label


def cap_case_rows(
    values: list[float],
    labels: list[int],
    max_case_rows: int,
    seed: int,
) -> tuple[list[float], list[int], str]:
    if max_case_rows <= 0 or len(labels) <= max_case_rows:
        return values, labels, "full"

    rng = random.Random(seed)
    positives = [index for index, label in enumerate(labels) if label]
    negatives = [index for index, label in enumerate(labels) if not label]
    target_positive = min(len(positives), max(1, round(max_case_rows * len(positives) / len(labels))))
    target_negative = max_case_rows - target_positive
    if target_positive < len(positives):
        positives = sorted(rng.sample(positives, target_positive))
    if target_negative < len(negatives):
        negatives = sorted(rng.sample(negatives, target_negative))
    selected = sorted(positives + negatives)
    return [values[index] for index in selected], [labels[index] for index in selected], f"stratified_cap_{max_case_rows}"


def run_case(
    station_id: str,
    code: str,
    values: list[float],
    labels: list[int],
    label: str,
    epsilon: float,
    seed: int,
    raw_buckets: int,
    output_buckets: int,
    segment_length: int,
    sample_rule: str,
) -> list[dict[str, object]]:
    reference = [value for value, row_label in zip(values, labels) if row_label == 0]
    normalized, lo, hi = normalize(reference, values)
    alpha = histogram_alpha([value for value, row_label in zip(normalized, labels) if row_label == 0], raw_buckets)
    matrix, edges = pm_matrix(epsilon, raw_buckets, output_buckets)
    rng = random.Random(seed)
    observations = [discretize(pm_sample_one(value, epsilon, rng), edges) for value in normalized]
    expected_rate = max(0.005, min(0.70, sum(labels) / len(labels)))
    methods = {
        "pm_column_likelihood_scan": lambda: channel_likelihood_scan(observations, matrix, alpha),
        "privsaf_mixture": lambda: mixture_infer(observations, matrix, alpha),
        "privsaf_hmm": lambda: hmm_infer(observations, matrix, alpha, expected_rate, segment_length),
    }
    rows = []
    for method, fn in methods.items():
        start = time.perf_counter()
        scores, candidate, estimated_rate = fn()
        runtime = time.perf_counter() - start
        rows.append(
            {
                "panel": "hadisd_multistation_streak_pmldp",
                "station_id": station_id,
                "label_code": code,
                "label": label,
                "epsilon": epsilon,
                "seed": seed,
                "method": method,
                "calibration": "all_nonflagged_nonlabel_rows",
                "sample_rule": sample_rule,
                "n_rows": len(labels),
                "positive_rows": sum(labels),
                "fault_rate": sum(labels) / len(labels),
                "normalization_p01": lo,
                "normalization_p99": hi,
                "auroc": auroc(labels, scores),
                "auprc": auprc(labels, scores),
                "precision_at_k": precision_at_k(labels, scores),
                "estimated_bucket": candidate,
                "estimated_fault_rate": estimated_rate,
                "runtime_sec": runtime,
            }
        )
    best_auprc = max(row["auprc"] for row in rows)
    for row in rows:
        row["selected_by_auprc"] = int(row["auprc"] == best_auprc)
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["station_id"]), str(row["label_code"]), str(row["method"])), []).append(row)
    out = []
    for (station_id, code, method), part in groups.items():
        out.append(
            {
                "panel": "hadisd_multistation_streak_pmldp",
                "station_id": station_id,
                "label_code": code,
                "method": method,
                "runs": len(part),
                "epsilons": ",".join(str(item) for item in sorted({row["epsilon"] for row in part})),
                "seeds": ",".join(str(item) for item in sorted({row["seed"] for row in part})),
                "sample_rule": str(part[0]["sample_rule"]),
                "n_rows": int(part[0]["n_rows"]),
                "positive_rows": int(part[0]["positive_rows"]),
                "auroc_mean": mean([float(row["auroc"]) for row in part]),
                "auroc_std": stdev([float(row["auroc"]) for row in part]),
                "auprc_mean": mean([float(row["auprc"]) for row in part]),
                "auprc_std": stdev([float(row["auprc"]) for row in part]),
                "precision_at_k_mean": mean([float(row["precision_at_k"]) for row in part]),
                "selected_runs": sum(int(row["selected_by_auprc"]) for row in part),
                "runtime_sec_mean": mean([float(row["runtime_sec"]) for row in part]),
            }
        )
    best_by_case: dict[tuple[str, str], float] = {}
    for row in out:
        key = (str(row["station_id"]), str(row["label_code"]))
        best_by_case[key] = max(best_by_case.get(key, -math.inf), float(row["auprc_mean"]))
    for row in out:
        key = (str(row["station_id"]), str(row["label_code"]))
        row["selected_by_case_mean_auprc"] = int(float(row["auprc_mean"]) == best_by_case[key])
    out.sort(key=lambda row: (row["station_id"], row["label_code"], -float(row["auprc_mean"])))
    return out


def rollup(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in summary_rows:
        groups.setdefault((str(row["label_code"]), str(row["method"])), []).append(row)
    out = []
    for (code, method), part in groups.items():
        auprcs = [float(row["auprc_mean"]) for row in part]
        out.append(
            {
                "panel": "hadisd_multistation_streak_pmldp",
                "label_code": code,
                "method": method,
                "station_cases": len(part),
                "stations": len({row["station_id"] for row in part}),
                "rows_total": sum(int(row["n_rows"]) for row in part),
                "positive_rows_total": sum(int(row["positive_rows"]) for row in part),
                "auroc_case_mean": mean([float(row["auroc_mean"]) for row in part]),
                "auprc_case_mean": mean(auprcs),
                "auprc_case_median": statistics.median(auprcs),
                "auprc_case_min": min(auprcs),
                "precision_at_k_case_mean": mean([float(row["precision_at_k_mean"]) for row in part]),
                "selected_case_count": sum(int(row["selected_by_case_mean_auprc"]) for row in part),
                "selected_run_count": sum(int(row["selected_runs"]) for row in part),
                "run_count": sum(int(row["runs"]) for row in part),
            }
        )
    out.sort(key=lambda row: (row["label_code"], -float(row["auprc_case_mean"])))
    return out


def write_empty_csv(path: Path, fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fout:
        csv.DictWriter(fout, fieldnames=fieldnames).writeheader()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HadISD multi-station PM-LDP checks on straight-string QC labels.")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/privsaf_hadisd_multistation"))
    parser.add_argument("--ncdump", default=None)
    parser.add_argument("--station-page-url", default=STATION_PAGE_URL)
    parser.add_argument("--station-ids", default=",".join(DEFAULT_STATION_IDS))
    parser.add_argument("--station-sort", choices=["page", "size", "station_id"], default="page")
    parser.add_argument("--station-offset", type=int, default=0)
    parser.add_argument("--station-limit", type=int, default=0)
    parser.add_argument("--max-station-size-mb", type=float, default=0.0)
    parser.add_argument("--case-codes", default=",".join(CASES))
    parser.add_argument("--epsilons", default="2,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--raw-buckets", type=int, default=32)
    parser.add_argument("--output-buckets", type=int, default=32)
    parser.add_argument("--segment-length", type=int, default=8)
    parser.add_argument("--min-positive", type=int, default=20)
    parser.add_argument("--min-negative", type=int, default=100)
    parser.add_argument("--max-fault-rate", type=float, default=0.70)
    parser.add_argument("--max-case-rows", type=int, default=50000)
    parser.add_argument("--max-station-cases", type=int, default=30)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--output-prefix", default="hadisd_multistation_streak")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    ncdump = find_ncdump(args.ncdump)
    page_path = fetch_station_page(args.work_dir, args.station_page_url)
    station_index = parse_station_page(page_path, args.station_page_url)
    if args.station_ids.strip().lower() == "all":
        stations = list(station_index.values())
        if args.max_station_size_mb > 0:
            stations = [station for station in stations if float(station["size_mb"]) <= args.max_station_size_mb]
        if args.station_sort == "size":
            stations.sort(key=lambda station: (float(station["size_mb"]), str(station["station_id"])))
        elif args.station_sort == "station_id":
            stations.sort(key=lambda station: str(station["station_id"]))
        start = max(args.station_offset, 0)
        end = start + args.station_limit if args.station_limit > 0 else None
        station_ids = [str(station["station_id"]) for station in stations[start:end]]
    else:
        station_ids = [item.strip() for item in args.station_ids.split(",") if item.strip()]
        missing = [station_id for station_id in station_ids if station_id not in station_index]
        if missing:
            raise ValueError(f"Station ids are not listed on {args.station_page_url}: {missing}")
    case_codes = [item.strip() for item in args.case_codes.split(",") if item.strip()]
    invalid_case_codes = [code for code in case_codes if code not in CASES]
    if invalid_case_codes:
        raise ValueError(f"Unknown case codes: {invalid_case_codes}; expected one of {sorted(CASES)}")

    epsilons = [float(item) for item in args.epsilons.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    inventory: list[dict[str, object]] = []
    station_metadata: list[dict[str, object]] = []
    pending_cases: list[tuple[str, str, list[float], list[int], str, str]] = []

    for station_id in station_ids:
        station = station_index[station_id]
        order, qc, arrays, flagged_rows, metadata = prepare_station(station, args.work_dir, ncdump)
        station_metadata.append({**station, **metadata})
        for code in case_codes:
            values, labels, label = build_case(code, order, qc, arrays, flagged_rows)
            positives = sum(labels)
            negatives = len(labels) - positives
            eligible = (
                positives >= args.min_positive
                and negatives >= args.min_negative
                and positives / max(len(labels), 1) <= args.max_fault_rate
            )
            sample_values, sample_labels, sample_rule = cap_case_rows(
                values,
                labels,
                args.max_case_rows,
                seed=int(hashlib.sha256(f"{station_id}:{code}".encode("utf-8")).hexdigest()[:8], 16),
            )
            inventory.append(
                {
                    "station_id": station_id,
                    "label_code": code,
                    "label": label,
                    "n_rows_full": len(labels),
                    "positive_rows_full": positives,
                    "fault_rate_full": positives / len(labels) if labels else math.nan,
                    "negative_rows_full": negatives,
                    "eligible_for_pmldp": int(eligible),
                    "sample_rule": sample_rule,
                    "n_rows_sample": len(sample_labels),
                    "positive_rows_sample": sum(sample_labels),
                    "fault_rate_sample": sum(sample_labels) / len(sample_labels) if sample_labels else math.nan,
                    "station_url": station["station_url"],
                }
            )
            if eligible:
                pending_cases.append((station_id, code, sample_values, sample_labels, label, sample_rule))

    pending_cases = pending_cases[: args.max_station_cases]
    runs: list[dict[str, object]] = []
    if not args.screen_only:
        for station_id, code, values, labels, label, sample_rule in pending_cases:
            for epsilon in epsilons:
                for seed in seeds:
                    runs.extend(
                        run_case(
                            station_id,
                            code,
                            values,
                            labels,
                            label,
                            epsilon,
                            seed,
                            args.raw_buckets,
                            args.output_buckets,
                            args.segment_length,
                            sample_rule,
                        )
                    )

    inventory_path = RESULTS / f"{args.output_prefix}_inventory.csv"
    write_csv(inventory_path, inventory)
    station_metadata_path = RESULTS / f"{args.output_prefix}_station_metadata.csv"
    write_csv(station_metadata_path, station_metadata)

    run_path = RESULTS / f"{args.output_prefix}_pmldp_runs.csv"
    summary_path = RESULTS / f"{args.output_prefix}_pmldp_summary.csv"
    rollup_path = RESULTS / f"{args.output_prefix}_pmldp_rollup.csv"
    summary: list[dict[str, object]] = []
    rollup_rows: list[dict[str, object]] = []
    if runs:
        summary = summarize(runs)
        rollup_rows = rollup(summary)
        write_csv(run_path, runs)
        write_csv(summary_path, summary)
        write_csv(rollup_path, rollup_rows)
    else:
        write_empty_csv(run_path, ["panel", "station_id", "label_code", "method"])
        write_empty_csv(summary_path, ["panel", "station_id", "label_code", "method"])
        write_empty_csv(rollup_path, ["panel", "label_code", "method"])

    source = {
        "status": "pass",
        "panel": "hadisd_multistation_streak_pmldp",
        "source": "HadISD v3.4.3.2025f",
        "hadisd_page_url": HADISD_PAGE_URL,
        "station_page_url": args.station_page_url,
        "test_codes_url": TEST_CODES_URL,
        "fail_summary_url": FAIL_SUMMARY_URL,
        "station_ids": station_ids,
        "station_selection": args.station_ids,
        "station_sort": args.station_sort,
        "station_offset": args.station_offset,
        "station_limit": args.station_limit,
        "max_station_size_mb": args.max_station_size_mb,
        "station_cases_screened": len(inventory),
        "eligible_station_cases": len([row for row in inventory if int(row["eligible_for_pmldp"])]),
        "pmldp_station_cases_run": len(pending_cases) if not args.screen_only else 0,
        "cases": case_codes,
        "epsilons": epsilons,
        "seeds": seeds,
        "min_positive": args.min_positive,
        "min_negative": args.min_negative,
        "max_fault_rate": args.max_fault_rate,
        "max_case_rows": args.max_case_rows,
        "screen_only": args.screen_only,
        "ncdump": str(ncdump),
        "ncdump_sha256": sha256sum(ncdump),
        "interpretation": (
            "Deterministic multi-station HadISD straight-string replication over selected station ids. "
            "Rows are screened by positive/negative support before PM-LDP evaluation; large cases are "
            "stratified to a fixed cap and reported with an explicit sample_rule."
        ),
    }
    metadata_path = RESULTS / f"{args.output_prefix}_pmldp_source_metadata.json"
    metadata_path.write_text(json.dumps(source, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {inventory_path}")
    print(f"Wrote {run_path}")
    print(f"Eligible station-cases: {source['eligible_station_cases']}; PM-LDP station-cases run: {source['pmldp_station_cases_run']}")


if __name__ == "__main__":
    main()
