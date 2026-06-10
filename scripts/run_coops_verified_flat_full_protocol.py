from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
import time
import urllib.error
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCREEN = RESULTS / "coops_verified_flat_flag_screen.csv"
CACHE = RESULTS / "coops_verified_erddap_cache"
BASE_PATH = ROOT / "scripts" / "run_coops_verified_flat_pmldp.py"
DATASET_INFO = "https://opendap.co-ops.nos.noaa.gov/erddap/info/IOOS_SixMin_Verified_Water_Level/index.html"


spec = importlib.util.spec_from_file_location("coops_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


def read_screen_cases(min_positive: int, max_rate_hint: float) -> list[dict[str, str]]:
    rows = list(csv.DictReader(SCREEN.open(encoding="utf-8")))
    cases = []
    for row in rows:
        positives = int(row["numeric_f1_rows"])
        if positives < min_positive:
            continue
        cases.append(row)
    cases.sort(key=lambda row: (-int(row["numeric_f1_rows"]), row["station"], int(row["year"]), int(row["month"])))
    return cases


def cache_key(station: str, year: int, month: int) -> Path:
    return CACHE / f"{station}_{year}_{month:02d}.csv"


def fetch_month_cached(station: str, year: int, month: int) -> tuple[list[dict[str, str]], str]:
    CACHE.mkdir(exist_ok=True)
    path = cache_key(station, year, month)
    begin, end = base.month_bounds(year, month)
    url = base.query_url(station, begin, end)
    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        try:
            text = base.fetch_csv(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return [], url
            raise
        path.write_text(text, encoding="utf-8")
    return base.parse_erddap_csv(text), url


def months_back(year: int, month: int, count: int) -> list[tuple[int, int]]:
    out = []
    y, m = year, month
    for _ in range(count):
        y, m = base.previous_month(y, m)
        out.append((y, m))
    return out


def historical_calibration_rows(station: str, year: int, month: int, min_rows: int = 100) -> tuple[list[dict[str, str]], str]:
    rows: list[dict[str, str]] = []
    urls = []
    for cy, cm in months_back(year, month, 12):
        month_rows, url = fetch_month_cached(station, cy, cm)
        numeric = base.numeric_rows(month_rows)
        rows.extend(numeric)
        urls.append(url)
        if len(rows) >= min_rows:
            break
    return rows, " | ".join(urls)


def prepare_case(case: dict[str, str], buckets: int) -> dict[str, object] | None:
    station = case["station"]
    year = int(case["year"])
    month = int(case["month"])
    rows, source_url = fetch_month_cached(station, year, month)
    rows = base.numeric_rows(rows)
    calibration_rows, calibration_url = historical_calibration_rows(station, year, month)
    if len(calibration_rows) < 100:
        return None

    values = [float(row["WL_VALUE"]) for row in rows]
    labels = [1 if row.get("F") == "1" else 0 for row in rows]
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        return None
    calibration_values = [float(row["WL_VALUE"]) for row in calibration_rows]
    lo = base.quantile(calibration_values, 0.01)
    hi = base.quantile(calibration_values, 0.99)
    raw = base.discretize(values, lo, hi, buckets)
    calibration_raw = base.discretize(calibration_values, lo, hi, buckets)
    alpha = base.histogram_alpha(calibration_raw, buckets)
    return {
        "case": f"{station}_{year}_{month:02d}",
        "station": station,
        "station_name": case["name"],
        "state": case["state"],
        "year": year,
        "month": month,
        "values": values,
        "labels": labels,
        "raw": raw,
        "calibration_raw": calibration_raw,
        "alpha": alpha,
        "lo": lo,
        "hi": hi,
        "source_url": source_url,
        "calibration_url": calibration_url,
        "calibration_rows": len(calibration_rows),
        "first_time": rows[0]["time"] if rows else "",
        "last_time": rows[-1]["time"] if rows else "",
    }


def raw_local_range_score(values: list[float], radius: int = 2) -> list[float]:
    out = []
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        window = values[lo:hi]
        out.append(-(max(window) - min(window)))
    return out


def raw_zero_slope_score(values: list[float], radius: int = 2) -> list[float]:
    out = []
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        diffs = [abs(values[j] - values[j - 1]) for j in range(lo + 1, hi)]
        out.append(-sum(diffs) / max(len(diffs), 1))
    return out


def range_emission(matrix: list[list[float]], cand: int, radius: int) -> list[float]:
    buckets = len(matrix)
    cols = [x for x in range(max(0, cand - radius), min(buckets, cand + radius + 1))]
    return [sum(matrix[y][x] for x in cols) / len(cols) for y in range(buckets)]


def hmm_scan_with_emissions(
    obs: list[int],
    b0: list[float],
    fault_emissions: list[list[float]],
    expected_rate: float,
    seg_len: int,
) -> list[float]:
    b0 = [max(v, 1e-15) for v in b0]
    p11 = max(0.55, 1.0 - 1.0 / max(seg_len, 2))
    p01 = min(0.20, max(1e-4, expected_rate * (1.0 - p11) / max(1.0 - expected_rate, 1e-6)))
    la00 = math.log(1.0 - p01)
    la01 = math.log(p01)
    la10 = math.log(1.0 - p11)
    la11 = math.log(p11)
    n = len(obs)
    best = [0.0] * n
    for emission in fault_emissions:
        fault = [max(v, 1e-15) for v in emission]
        f0 = [0.0] * n
        f1 = [0.0] * n
        f0[0] = math.log(max(1.0 - expected_rate, 1e-15)) + math.log(b0[obs[0]])
        f1[0] = math.log(max(expected_rate, 1e-15)) + math.log(fault[obs[0]])
        scale = base.logsumexp2(f0[0], f1[0])
        f0[0] -= scale
        f1[0] -= scale
        for i in range(1, n):
            y = obs[i]
            nf0 = math.log(b0[y]) + base.logsumexp2(f0[i - 1] + la00, f1[i - 1] + la10)
            nf1 = math.log(fault[y]) + base.logsumexp2(f0[i - 1] + la01, f1[i - 1] + la11)
            scale = base.logsumexp2(nf0, nf1)
            f0[i] = nf0 - scale
            f1[i] = nf1 - scale
        beta0 = 0.0
        beta1 = 0.0
        for i in range(n - 1, -1, -1):
            gamma_log = f1[i] + beta1
            denom = base.logsumexp2(f0[i] + beta0, gamma_log)
            score = math.exp(gamma_log - denom)
            if score > best[i]:
                best[i] = score
            if i > 0:
                y = obs[i]
                nb0 = base.logsumexp2(la00 + math.log(b0[y]) + beta0, la01 + math.log(fault[y]) + beta1)
                nb1 = base.logsumexp2(la10 + math.log(b0[y]) + beta0, la11 + math.log(fault[y]) + beta1)
                scale = base.logsumexp2(nb0, nb1)
                beta0 = nb0 - scale
                beta1 = nb1 - scale
    return best


def precision_at_fraction(labels: list[int], scores: list[float], fraction: float) -> float:
    return base.precision_at_k(labels, scores, max(1, int(round(len(labels) * fraction))))


def run_private_methods(prepared: dict[str, object], eps: float, seed: int, buckets: int, prior: float) -> list[dict[str, object]]:
    labels = list(prepared["labels"])
    raw = list(prepared["raw"])
    calibration_raw = list(prepared["calibration_raw"])
    alpha = list(prepared["alpha"])
    matrix = base.krr_matrix(eps, buckets)
    obs = base.krr_sample(raw, eps, buckets, random.Random(seed))
    cal_obs = base.krr_sample(calibration_raw, eps, buckets, random.Random(10_000 + seed))
    b0 = base.b0_distribution(matrix, alpha)
    point_scores = base.channel_point_glr(obs, matrix, alpha)
    single_emissions = [[matrix[y][cand] for y in range(buckets)] for cand in range(buckets)]
    range1_emissions = [range_emission(matrix, cand, radius=1) for cand in range(buckets)]
    range2_emissions = [range_emission(matrix, cand, radius=2) for cand in range(buckets)]
    methods = {
        "report_frequency": lambda: base.report_frequency_score(obs, cal_obs, buckets),
        "channel_point_glr": lambda: point_scores,
        "channel_window_glr_5": lambda: base.rolling_mean(point_scores, 5),
        "privsaf_hmm_fixed_prior": lambda: hmm_scan_with_emissions(obs, b0, single_emissions, prior, seg_len=5),
        "privsaf_range_hmm_r1_fixed_prior": lambda: hmm_scan_with_emissions(obs, b0, range1_emissions, prior, seg_len=5),
        "privsaf_range_hmm_r2_fixed_prior": lambda: hmm_scan_with_emissions(obs, b0, range2_emissions, prior, seg_len=5),
    }
    rows = []
    for method, fn in methods.items():
        start = time.perf_counter()
        scores = fn()
        runtime = time.perf_counter() - start
        rows.append(metric_row(prepared, method, labels, scores, runtime, "k_ary_randomized_response", eps, seed, buckets, prior))
    return rows


def run_raw_methods(prepared: dict[str, object]) -> list[dict[str, object]]:
    labels = list(prepared["labels"])
    values = list(prepared["values"])
    rows = []
    for method, scores in {
        "raw_local_range_radius2": raw_local_range_score(values, radius=2),
        "raw_zero_slope_radius2": raw_zero_slope_score(values, radius=2),
    }.items():
        rows.append(metric_row(prepared, method, labels, scores, 0.0, "raw_nonprivate_upper_bound", 0.0, -1, 0, 0.0))
    return rows


def metric_row(
    prepared: dict[str, object],
    method: str,
    labels: list[int],
    scores: list[float],
    runtime: float,
    mechanism: str,
    eps: float,
    seed: int,
    buckets: int,
    prior: float,
) -> dict[str, object]:
    positives = sum(labels)
    return {
        "panel": "coops_verified_flat_full_protocol",
        "dataset": "NOAA CO-OPS verified six-minute water level",
        "station": prepared["station"],
        "station_name": prepared["station_name"],
        "state": prepared["state"],
        "case": prepared["case"],
        "year": prepared["year"],
        "month": prepared["month"],
        "epsilon": eps,
        "seed": seed,
        "method": method,
        "mechanism": mechanism,
        "buckets": buckets,
        "fault_prior": prior,
        "calibration": "previous_1_to_12_months_all_numeric_rows_no_label_filter",
        "label_rule": "verified_six_minute_F_flat_tolerance_flag_equals_1",
        "n_calibration": prepared["calibration_rows"],
        "n_test": len(labels),
        "fault_rows": positives,
        "fault_rate": positives / max(len(labels), 1),
        "normalization_p01": prepared["lo"],
        "normalization_p99": prepared["hi"],
        "auroc": base.auroc(labels, scores),
        "auprc": base.auprc(labels, scores),
        "recall_at_1pct_fpr": base.recall_at_fpr(labels, scores, 0.01),
        "recall_at_5pct_fpr": base.recall_at_fpr(labels, scores, 0.05),
        "precision_at_top_1pct": precision_at_fraction(labels, scores, 0.01),
        "precision_at_top_5pct": precision_at_fraction(labels, scores, 0.05),
        "precision_at_k": base.precision_at_k(labels, scores, positives),
        "runtime_sec": runtime,
        "source_url": prepared["source_url"],
        "calibration_url": prepared["calibration_url"],
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(values: list[float], seed: int = 8675309, rounds: int = 500) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(mean(sample))
    means.sort()
    return means[int(0.025 * (rounds - 1))], means[int(0.975 * (rounds - 1))]


def summarize_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, float, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["method"]), float(row["epsilon"]), int(row["buckets"]))].append(row)
    out = []
    for (method, eps, buckets), group in groups.items():
        case_values: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in group:
            case_values[str(row["case"])].append(row)
        case_auprc = [mean([float(row["auprc"]) for row in vals]) for vals in case_values.values()]
        case_auroc = [mean([float(row["auroc"]) for row in vals]) for vals in case_values.values()]
        lo, hi = bootstrap_ci(case_auprc)
        out.append(
            {
                "panel": "coops_verified_flat_full_protocol",
                "method": method,
                "epsilon": eps,
                "buckets": buckets,
                "cases": len(case_values),
                "runs": len(group),
                "case_mean_auroc": mean(case_auroc),
                "case_mean_auprc": mean(case_auprc),
                "case_mean_auprc_ci95_low": lo,
                "case_mean_auprc_ci95_high": hi,
                "row_weighted_auroc": mean([float(row["auroc"]) for row in group]),
                "row_weighted_auprc": mean([float(row["auprc"]) for row in group]),
                "recall_at_1pct_fpr_mean": mean([float(row["recall_at_1pct_fpr"]) for row in group]),
                "recall_at_5pct_fpr_mean": mean([float(row["recall_at_5pct_fpr"]) for row in group]),
                "precision_at_top_1pct_mean": mean([float(row["precision_at_top_1pct"]) for row in group]),
                "precision_at_top_5pct_mean": mean([float(row["precision_at_top_5pct"]) for row in group]),
                "runtime_sec_mean": mean([float(row["runtime_sec"]) for row in group]),
            }
        )
    out.sort(key=lambda row: (float(row["epsilon"]), int(row["buckets"]), -float(row["case_mean_auprc"])))
    return out


def summarize_cases(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, float, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if str(row["mechanism"]) == "raw_nonprivate_upper_bound":
            continue
        groups[(str(row["case"]), float(row["epsilon"]), int(row["buckets"]))].append(row)
    out = []
    for (case, eps, buckets), group in groups.items():
        by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in group:
            by_method[str(row["method"])].append(row)
        scores = []
        for method, vals in by_method.items():
            scores.append((mean([float(row["auprc"]) for row in vals]), mean([float(row["auroc"]) for row in vals]), method, vals[0]))
        scores.sort(reverse=True)
        best_auprc, best_auroc, best_method, exemplar = scores[0]
        out.append(
            {
                "case": case,
                "station": exemplar["station"],
                "station_name": exemplar["station_name"],
                "state": exemplar["state"],
                "year": exemplar["year"],
                "month": exemplar["month"],
                "epsilon": eps,
                "buckets": buckets,
                "n_test": exemplar["n_test"],
                "fault_rows": exemplar["fault_rows"],
                "fault_rate": exemplar["fault_rate"],
                "best_private_method": best_method,
                "best_private_auroc": best_auroc,
                "best_private_auprc": best_auprc,
            }
        )
    out.sort(key=lambda row: (float(row["epsilon"]), int(row["buckets"]), -float(row["best_private_auprc"])))
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    min_positive = 5
    max_rate_hint = 0.70
    primary_buckets = 24
    sensitivity_buckets = [16, 32]
    sensitivity_case_count = 7
    epsilons = [2.0, 4.0]
    seeds = [0, 1]
    fixed_prior = 0.02
    cases = read_screen_cases(min_positive=min_positive, max_rate_hint=max_rate_hint)
    all_runs: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"{idx}/{len(cases)} case {case['station']} {case['year']}-{int(case['month']):02d} positives={case['numeric_f1_rows']}", flush=True)
        prepared_for_inventory = None
        buckets_grid = [primary_buckets]
        if idx <= sensitivity_case_count:
            buckets_grid.extend(sensitivity_buckets)
        for buckets in buckets_grid:
            prepared = prepare_case(case, buckets=buckets)
            if prepared is None:
                continue
            if prepared_for_inventory is None:
                prepared_for_inventory = prepared
                all_runs.extend(run_raw_methods(prepared))
            for eps in epsilons:
                for seed in seeds:
                    all_runs.extend(run_private_methods(prepared, eps=eps, seed=seed, buckets=buckets, prior=fixed_prior))
        if prepared_for_inventory is not None:
            labels = list(prepared_for_inventory["labels"])
            inventory.append(
                {
                    "case": prepared_for_inventory["case"],
                    "station": prepared_for_inventory["station"],
                    "station_name": prepared_for_inventory["station_name"],
                    "state": prepared_for_inventory["state"],
                    "year": prepared_for_inventory["year"],
                    "month": prepared_for_inventory["month"],
                    "rows": len(labels),
                    "f1_rows": sum(labels),
                    "f1_rate": sum(labels) / max(len(labels), 1),
                    "calibration_rows": prepared_for_inventory["calibration_rows"],
                    "first_time": prepared_for_inventory["first_time"],
                    "last_time": prepared_for_inventory["last_time"],
                    "source_url": prepared_for_inventory["source_url"],
                    "calibration_url": prepared_for_inventory["calibration_url"],
                }
            )

    write_csv(RESULTS / "coops_verified_flat_full_protocol_runs.csv", all_runs)
    write_csv(RESULTS / "coops_verified_flat_full_protocol_summary.csv", summarize_runs(all_runs))
    write_csv(RESULTS / "coops_verified_flat_full_protocol_case_summary.csv", summarize_cases(all_runs))
    write_csv(RESULTS / "coops_verified_flat_full_protocol_inventory.csv", inventory)
    metadata = {
        "status": "pass",
        "dataset_info": DATASET_INFO,
        "selection_rule": "all station-months from coops_verified_flat_flag_screen.csv with numeric_f1_rows >= 5",
        "eligible_cases": len(cases),
        "cases_run": len(inventory),
        "runs": len(all_runs),
        "mechanism": "k_ary_randomized_response",
        "epsilons": epsilons,
        "seeds": seeds,
        "primary_buckets": primary_buckets,
        "sensitivity_buckets": sensitivity_buckets,
        "sensitivity_case_count": sensitivity_case_count,
        "fault_prior": fixed_prior,
        "calibration": "previous 1 to 12 months, all numeric rows, no same-month F-label filtering",
        "raw_upper_bounds": ["raw_local_range_radius2", "raw_zero_slope_radius2"],
        "private_methods": [
            "report_frequency",
            "channel_point_glr",
            "channel_window_glr_5",
            "privsaf_hmm_fixed_prior",
            "privsaf_range_hmm_r1_fixed_prior",
            "privsaf_range_hmm_r2_fixed_prior",
        ],
    }
    (RESULTS / "coops_verified_flat_full_protocol_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
