from __future__ import annotations

import csv
import json
import math
import random
import time
import urllib.parse
import urllib.request
from calendar import monthrange
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCREEN = RESULTS / "coops_verified_flat_flag_screen.csv"

ERDDAP = "https://opendap.co-ops.nos.noaa.gov/erddap/tabledap/IOOS_SixMin_Verified_Water_Level.csv"
DATASET_INFO = "https://opendap.co-ops.nos.noaa.gov/erddap/info/IOOS_SixMin_Verified_Water_Level/index.html"


def query_url(station: str, begin: str, end: str, only_f1: bool = False) -> str:
    columns = "STATION_ID,DATUM,BEGIN_DATE,END_DATE,time,WL_VALUE,F,R,T,I"
    params = [
        ("STATION_ID", f'"{station}"'),
        ("DATUM", '"MLLW"'),
        ("BEGIN_DATE", f'"{begin}"'),
        ("END_DATE", f'"{end}"'),
    ]
    if only_f1:
        params.append(("F", "1"))
    constraints = "&".join(f"{key}={urllib.parse.quote(value, safe='')}" for key, value in params)
    return f"{ERDDAP}?{columns}&{constraints}"


def month_bounds(year: int, month: int) -> tuple[str, str]:
    begin = f"{year}{month:02d}01"
    end = f"{year}{month:02d}{monthrange(year, month)[1]:02d}"
    return begin, end


def previous_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def fetch_csv(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "PrivSAF data audit"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def parse_erddap_csv(text: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    if len(lines) <= 2:
        return []
    return list(csv.DictReader([lines[0], *lines[2:]]))


def fetch_month(station: str, year: int, month: int) -> tuple[list[dict[str, str]], str]:
    begin, end = month_bounds(year, month)
    url = query_url(station, begin, end)
    rows = parse_erddap_csv(fetch_csv(url))
    return rows, url


def numeric_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        try:
            float(row.get("WL_VALUE", ""))
        except ValueError:
            continue
        out.append(row)
    return out


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def discretize(values: list[float], lo: float, hi: float, buckets: int) -> list[int]:
    width = max(hi - lo, 1e-12)
    out = []
    for value in values:
        scaled = (value - lo) / width
        out.append(max(0, min(buckets - 1, int(math.floor(scaled * buckets)))))
    return out


def krr_matrix(eps: float, buckets: int) -> list[list[float]]:
    ee = math.exp(eps)
    same = ee / (ee + buckets - 1)
    other = 1.0 / (ee + buckets - 1)
    return [[same if y == x else other for x in range(buckets)] for y in range(buckets)]


def krr_sample(raw: list[int], eps: float, buckets: int, rng: random.Random) -> list[int]:
    ee = math.exp(eps)
    same = ee / (ee + buckets - 1)
    out = []
    for value in raw:
        if rng.random() < same:
            out.append(value)
        else:
            alt = rng.randrange(buckets - 1)
            out.append(alt if alt < value else alt + 1)
    return out


def histogram_alpha(raw: list[int], buckets: int, smooth: float = 1.0) -> list[float]:
    counts = [smooth] * buckets
    for value in raw:
        counts[value] += 1.0
    total = sum(counts)
    return [count / total for count in counts]


def auroc(labels: list[int], scores: list[float]) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def auprc(labels: list[int], scores: list[float]) -> float:
    pos = sum(labels)
    if pos == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for idx in order:
        if labels[idx]:
            tp += 1
        else:
            fp += 1
        recall = tp / pos
        precision = tp / max(tp + fp, 1)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def precision_at_k(labels: list[int], scores: list[float], k: int) -> float:
    if k <= 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:k]
    return sum(labels[idx] for idx in order) / k


def recall_at_fpr(labels: list[int], scores: list[float], target_fpr: float) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    tp = 0
    fp = 0
    best = 0.0
    for idx in order:
        if labels[idx]:
            tp += 1
        else:
            fp += 1
        if fp / neg <= target_fpr:
            best = max(best, tp / pos)
    return best


def b0_distribution(matrix: list[list[float]], alpha: list[float]) -> list[float]:
    buckets = len(alpha)
    return [sum(matrix[y][x] * alpha[x] for x in range(buckets)) for y in range(buckets)]


def report_frequency_score(obs: list[int], calibration_obs: list[int], buckets: int) -> list[float]:
    counts = Counter(calibration_obs)
    total = len(calibration_obs) + buckets
    probs = [(counts.get(k, 0) + 1) / total for k in range(buckets)]
    return [-math.log(probs[y]) for y in obs]


def channel_point_glr(obs: list[int], matrix: list[list[float]], alpha: list[float]) -> list[float]:
    b0 = b0_distribution(matrix, alpha)
    return [max(math.log(matrix[y][x]) for x in range(len(alpha))) - math.log(max(b0[y], 1e-15)) for y in obs]


def rolling_mean(scores: list[float], window: int) -> list[float]:
    if window <= 1:
        return scores
    out = []
    running = 0.0
    q: list[float] = []
    for value in scores:
        q.append(value)
        running += value
        if len(q) > window:
            running -= q.pop(0)
        out.append(running / len(q))
    return out


def logsumexp2(a: float, b: float) -> float:
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def hmm_scan(obs: list[int], matrix: list[list[float]], alpha: list[float], expected_rate: float, seg_len: int) -> list[float]:
    buckets = len(alpha)
    b0 = [max(v, 1e-15) for v in b0_distribution(matrix, alpha)]
    p11 = max(0.55, 1.0 - 1.0 / max(seg_len, 2))
    p01 = min(0.20, max(1e-4, expected_rate * (1.0 - p11) / max(1.0 - expected_rate, 1e-6)))
    la00 = math.log(1.0 - p01)
    la01 = math.log(p01)
    la10 = math.log(1.0 - p11)
    la11 = math.log(p11)
    n = len(obs)
    best = [0.0] * n
    for cand in range(buckets):
        f0 = [0.0] * n
        f1 = [0.0] * n
        f0[0] = math.log(max(1.0 - expected_rate, 1e-15)) + math.log(b0[obs[0]])
        f1[0] = math.log(max(expected_rate, 1e-15)) + math.log(max(matrix[obs[0]][cand], 1e-15))
        scale = logsumexp2(f0[0], f1[0])
        f0[0] -= scale
        f1[0] -= scale
        for i in range(1, n):
            y = obs[i]
            nf0 = math.log(b0[y]) + logsumexp2(f0[i - 1] + la00, f1[i - 1] + la10)
            nf1 = math.log(max(matrix[y][cand], 1e-15)) + logsumexp2(f0[i - 1] + la01, f1[i - 1] + la11)
            scale = logsumexp2(nf0, nf1)
            f0[i] = nf0 - scale
            f1[i] = nf1 - scale
        beta0 = 0.0
        beta1 = 0.0
        cand_scores = [0.0] * n
        for i in range(n - 1, -1, -1):
            gamma_log = f1[i] + beta1
            denom = logsumexp2(f0[i] + beta0, gamma_log)
            cand_scores[i] = math.exp(gamma_log - denom)
            if i > 0:
                y = obs[i]
                nb0 = logsumexp2(la00 + math.log(b0[y]) + beta0, la01 + math.log(max(matrix[y][cand], 1e-15)) + beta1)
                nb1 = logsumexp2(la10 + math.log(b0[y]) + beta0, la11 + math.log(max(matrix[y][cand], 1e-15)) + beta1)
                scale = logsumexp2(nb0, nb1)
                beta0 = nb0 - scale
                beta1 = nb1 - scale
        for i, value in enumerate(cand_scores):
            if value > best[i]:
                best[i] = value
    return best


def select_cases(limit: int, min_positive: int, max_rate: float) -> list[dict[str, str]]:
    rows = list(csv.DictReader(SCREEN.open(encoding="utf-8")))
    candidates = [row for row in rows if int(row["numeric_f1_rows"]) >= min_positive]
    candidates.sort(key=lambda row: int(row["numeric_f1_rows"]), reverse=True)
    selected = []
    for row in candidates:
        if len(selected) >= limit:
            break
        selected.append(row)
    return selected


def prepare_case(case: dict[str, str], buckets: int) -> dict[str, object]:
    station = case["station"]
    year = int(case["year"])
    month = int(case["month"])
    rows, test_url = fetch_month(station, year, month)
    rows = numeric_rows(rows)
    py, pm = previous_month(year, month)
    calibration_rows, calibration_url = fetch_month(station, py, pm)
    calibration_rows = [row for row in numeric_rows(calibration_rows) if row.get("F") == "0"]
    if len(calibration_rows) < 100:
        calibration_rows = [row for row in rows if row.get("F") == "0"]
    if len(calibration_rows) < 100:
        calibration_rows = [row for row in rows if row.get("F") != "1"] or rows

    values = [float(row["WL_VALUE"]) for row in rows]
    labels = [1 if row.get("F") == "1" else 0 for row in rows]
    calibration_values = [float(row["WL_VALUE"]) for row in calibration_rows]
    lo = quantile(calibration_values, 0.01)
    hi = quantile(calibration_values, 0.99)
    raw = discretize(values, lo, hi, buckets)
    calibration_raw = discretize(calibration_values, lo, hi, buckets)
    alpha = histogram_alpha(calibration_raw, buckets)
    inventory = {
        "case": f"{station}_{year}_{month:02d}",
        "station": station,
        "station_name": case["name"],
        "state": case["state"],
        "year": year,
        "month": month,
        "rows": len(labels),
        "f1_rows": sum(labels),
        "f1_rate": sum(labels) / max(len(labels), 1),
        "calibration_rows": len(calibration_rows),
        "source_url": test_url,
        "calibration_url": calibration_url,
        "first_time": rows[0]["time"] if rows else "",
        "last_time": rows[-1]["time"] if rows else "",
    }
    return {
        "case": case,
        "station": station,
        "year": year,
        "month": month,
        "labels": labels,
        "raw": raw,
        "calibration_raw": calibration_raw,
        "alpha": alpha,
        "lo": lo,
        "hi": hi,
        "test_url": test_url,
        "calibration_url": calibration_url,
        "inventory": inventory,
    }


def run_prepared_case(prepared: dict[str, object], eps: float, seed: int, buckets: int) -> list[dict[str, object]]:
    case = prepared["case"]
    station = str(prepared["station"])
    year = int(prepared["year"])
    month = int(prepared["month"])
    labels = list(prepared["labels"])
    raw = list(prepared["raw"])
    calibration_raw = list(prepared["calibration_raw"])
    alpha = list(prepared["alpha"])
    lo = float(prepared["lo"])
    hi = float(prepared["hi"])
    test_url = str(prepared["test_url"])
    calibration_url = str(prepared["calibration_url"])
    matrix = krr_matrix(eps, buckets)
    rng = random.Random(seed)
    obs = krr_sample(raw, eps, buckets, rng)
    cal_obs = krr_sample(calibration_raw, eps, buckets, random.Random(10_000 + seed))
    rate = min(0.70, max(0.005, sum(labels) / max(len(labels), 1)))
    seg_len = 5

    methods = {
        "report_frequency": lambda: report_frequency_score(obs, cal_obs, buckets),
        "channel_point_glr": lambda: channel_point_glr(obs, matrix, alpha),
        "channel_window_glr_5": lambda: rolling_mean(channel_point_glr(obs, matrix, alpha), 5),
        "privsaf_hmm_scan": lambda: hmm_scan(obs, matrix, alpha, rate, seg_len),
    }
    run_rows = []
    for method, fn in methods.items():
        start = time.perf_counter()
        scores = fn()
        runtime = time.perf_counter() - start
        positives = sum(labels)
        run_rows.append(
            {
                "panel": "coops_verified_flat_qc_ldp",
                "dataset": "NOAA CO-OPS verified six-minute water level",
                "station": station,
                "station_name": case["name"],
                "state": case["state"],
                "case": f"{station}_{year}_{month:02d}",
                "year": year,
                "month": month,
                "epsilon": eps,
                "seed": seed,
                "method": method,
                "mechanism": "k_ary_randomized_response",
                "buckets": buckets,
                "calibration": "previous_month_nonflat_rows_else_same_month_nonflat",
                "label_rule": "verified_six_minute_F_flat_tolerance_flag_equals_1",
                "n_calibration": len(calibration_raw),
                "n_test": len(labels),
                "fault_rows": positives,
                "fault_rate": positives / max(len(labels), 1),
                "normalization_p01": lo,
                "normalization_p99": hi,
                "auroc": auroc(labels, scores),
                "auprc": auprc(labels, scores),
                "recall_at_5pct_fpr": recall_at_fpr(labels, scores, 0.05),
                "precision_at_k": precision_at_k(labels, scores, positives),
                "runtime_sec": runtime,
                "source_url": test_url,
                "calibration_url": calibration_url,
            }
        )
    return run_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fout:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in runs:
        groups.setdefault((str(row["method"]), float(row["epsilon"])), []).append(row)
    out = []
    for (method, eps), rows in groups.items():
        def mean(key: str) -> float:
            vals = [float(row[key]) for row in rows if str(row[key]) != "nan"]
            return sum(vals) / len(vals) if vals else float("nan")
        out.append(
            {
                "panel": "coops_verified_flat_qc_ldp",
                "method": method,
                "epsilon": eps,
                "cases": len({row["case"] for row in rows}),
                "runs": len(rows),
                "auroc_mean": mean("auroc"),
                "auprc_mean": mean("auprc"),
                "recall_at_5pct_fpr_mean": mean("recall_at_5pct_fpr"),
                "precision_at_k_mean": mean("precision_at_k"),
                "runtime_sec_mean": mean("runtime_sec"),
            }
        )
    out.sort(key=lambda row: (float(row["epsilon"]), -float(row["auprc_mean"])))
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    cases = select_cases(limit=6, min_positive=25, max_rate=0.70)
    epsilons = [2.0, 4.0]
    seeds = [0, 1]
    all_runs: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for case in cases:
        print(f"case {case['station']} {case['year']}-{int(case['month']):02d} positives={case['numeric_f1_rows']}", flush=True)
        prepared = prepare_case(case, buckets=24)
        for eps in epsilons:
            for seed in seeds:
                all_runs.extend(run_prepared_case(prepared, eps, seed, buckets=24))
        inventory.append(prepared["inventory"])

    write_csv(RESULTS / "coops_verified_flat_pmldp_runs.csv", all_runs)
    write_csv(RESULTS / "coops_verified_flat_pmldp_summary.csv", summarize(all_runs))
    write_csv(RESULTS / "coops_verified_flat_pmldp_inventory.csv", inventory)
    metadata = {
        "status": "pass",
        "dataset_info": DATASET_INFO,
        "cases": len(inventory),
        "runs": len(all_runs),
        "label_rule": "verified_six_minute_F_flat_tolerance_flag_equals_1",
        "mechanism": "k_ary_randomized_response",
        "epsilons": epsilons,
        "seeds": seeds,
        "buckets": 24,
        "interpretation": (
            "This panel uses public NOAA CO-OPS verified six-minute rows where WL_VALUE and the official flat-tolerance "
            "flag F are present in the same ERDDAP product. It is a first public real value+label flatline PM-LDP-style "
            "screen; the mechanism is k-ary randomized response over discretized water levels. ERDDAP END_DATE is "
            "handled as an inclusive calendar-day bound."
        ),
    }
    (RESULTS / "coops_verified_flat_pmldp_source_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
