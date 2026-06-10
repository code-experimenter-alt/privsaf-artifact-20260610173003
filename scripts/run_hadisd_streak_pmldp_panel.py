from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import time
from pathlib import Path

from audit_hadisd_streak_flags import (
    RESULTS,
    STATION_ID,
    STATION_URL,
    VALUE_VARIABLES,
    build_outputs,
    find_ncdump,
    parse_number_array,
    parse_qc_matrix,
    parse_summary_order,
)


CASES = {
    "TSS": ("temperatures", 0, "temperature straight string"),
    "DSS": ("dewpoints", 1, "dew point straight string"),
    "RSS": ("winddirs", 5, "wind direction straight string"),
}


def variable_blob(cdl_text: str, variable: str) -> str:
    data = cdl_text.split("\ndata:", 1)[1]
    matches = list(re.finditer(r"\n\s*" + re.escape(variable) + r"\s*=\s*(.*?);", data, re.S))
    if not matches:
        raise ValueError(f"Missing {variable}.")
    return matches[-1].group(1)


def centers(d: int) -> list[float]:
    return [-1.0 + 1.0 / d + 2.0 * i / d for i in range(d)]


def pm_matrix(epsilon: float, raw_buckets: int, output_buckets: int) -> tuple[list[list[float]], list[float]]:
    s = math.exp(epsilon / 2.0)
    c = (s + 1.0) / (s - 1.0)
    raw_centers = centers(raw_buckets)
    edges = [-c + 2.0 * c * i / output_buckets for i in range(output_buckets + 1)]
    f_in = s / (s + 1.0) / (c - 1.0)
    f_out = 1.0 / (s + 1.0) / (c + 1.0)
    matrix: list[list[float]] = []
    for j in range(output_buckets):
        a, b = edges[j], edges[j + 1]
        row: list[float] = []
        for value in raw_centers:
            left = (s * value - 1.0) / (s - 1.0)
            right = (s * value + 1.0) / (s - 1.0)
            overlap = max(0.0, min(b, right) - max(a, left))
            width = b - a
            row.append(f_in * overlap + f_out * (width - overlap))
        matrix.append(row)
    for col in range(raw_buckets):
        total = sum(matrix[row][col] for row in range(output_buckets))
        for row in range(output_buckets):
            matrix[row][col] /= max(total, 1e-12)
    return matrix, edges


def pm_sample_one(value: float, epsilon: float, rng: random.Random) -> float:
    s = math.exp(epsilon / 2.0)
    c = (s + 1.0) / (s - 1.0)
    left = (s * value - 1.0) / (s - 1.0)
    right = (s * value + 1.0) / (s - 1.0)
    if rng.random() < s / (s + 1.0):
        return rng.uniform(left, right)
    left_len = max(left + c, 0.0)
    right_len = max(c - right, 0.0)
    if rng.random() < left_len / max(left_len + right_len, 1e-12):
        return rng.uniform(-c, left)
    return rng.uniform(right, c)


def discretize(value: float, edges: list[float]) -> int:
    if value <= edges[0]:
        return 0
    for index in range(len(edges) - 1):
        if value <= edges[index + 1]:
            return index
    return len(edges) - 2


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def normalize(reference: list[float], values: list[float]) -> tuple[list[float], float, float]:
    lo = quantile(reference, 0.01)
    hi = quantile(reference, 0.99)
    scale = max(hi - lo, 1e-12)
    return [max(-1.0, min(1.0, 2.0 * (value - lo) / scale - 1.0)) for value in values], lo, hi


def histogram_alpha(values: list[float], buckets: int) -> list[float]:
    counts = [1e-3] * buckets
    for value in values:
        index = max(0, min(buckets - 1, int((value + 1.0) * buckets / 2.0)))
        counts[index] += 1.0
    total = sum(counts)
    return [count / total for count in counts]


def matvec(matrix: list[list[float]], alpha: list[float]) -> list[float]:
    return [sum(row[col] * alpha[col] for col in range(len(alpha))) for row in matrix]


def auroc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan
    order = sorted(range(len(scores)), key=lambda index: scores[index])
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
    positive_rank_sum = sum(ranks[index] for index, label in enumerate(labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def auprc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return math.nan
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    hits = 0
    area = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index]:
            hits += 1
            area += hits / rank
    return area / positives


def precision_at_k(labels: list[int], scores: list[float]) -> float:
    k = sum(labels)
    if k == 0:
        return math.nan
    top = set(sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:k])
    return sum(labels[index] for index in top) / k


def logadd(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    if b < -745:
        return a
    return a + math.log1p(math.exp(b - a))


def forward_backward_candidate(
    observations: list[int],
    normal_emission: list[float],
    fault_emission: list[float],
    p01: float,
    p11: float,
    pi1: float,
    return_gamma: bool,
) -> tuple[float, list[float]]:
    log_a00 = math.log(1.0 - p01)
    log_a01 = math.log(p01)
    log_a10 = math.log(1.0 - p11)
    log_a11 = math.log(p11)
    n = len(observations)
    f0 = [0.0] * n
    f1 = [0.0] * n
    f0[0] = math.log(1.0 - pi1) + math.log(normal_emission[observations[0]])
    f1[0] = math.log(pi1) + math.log(fault_emission[observations[0]])
    scale = logadd(f0[0], f1[0])
    f0[0] -= scale
    f1[0] -= scale
    log_likelihood = scale
    for t in range(1, n):
        obs = observations[t]
        next0 = math.log(normal_emission[obs]) + logadd(f0[t - 1] + log_a00, f1[t - 1] + log_a10)
        next1 = math.log(fault_emission[obs]) + logadd(f0[t - 1] + log_a01, f1[t - 1] + log_a11)
        scale = logadd(next0, next1)
        f0[t] = next0 - scale
        f1[t] = next1 - scale
        log_likelihood += scale
    if not return_gamma:
        return log_likelihood, []

    beta0 = 0.0
    beta1 = 0.0
    gamma = [0.0] * n
    for t in range(n - 1, -1, -1):
        g1 = f1[t] + beta1
        denom = logadd(f0[t] + beta0, g1)
        gamma[t] = math.exp(g1 - denom)
        if t > 0:
            obs = observations[t]
            next_beta0 = logadd(
                log_a00 + math.log(normal_emission[obs]) + beta0,
                log_a01 + math.log(fault_emission[obs]) + beta1,
            )
            next_beta1 = logadd(
                log_a10 + math.log(normal_emission[obs]) + beta0,
                log_a11 + math.log(fault_emission[obs]) + beta1,
            )
            scale = logadd(next_beta0, next_beta1)
            beta0 = next_beta0 - scale
            beta1 = next_beta1 - scale
    return log_likelihood, gamma


def channel_likelihood_scan(observations: list[int], matrix: list[list[float]], alpha: list[float]) -> tuple[list[float], int, float]:
    normal = [max(value, 1e-12) for value in matvec(matrix, alpha)]
    scores: list[float] = []
    candidates: list[int] = []
    for obs in observations:
        values = [math.log(max(matrix[obs][cand], 1e-12) / normal[obs]) for cand in range(len(alpha))]
        cand = max(range(len(values)), key=lambda index: values[index])
        candidates.append(cand)
        scores.append(values[cand])
    mode = max(set(candidates), key=candidates.count)
    positive_rate = sum(1 for score in scores if score > 0.0) / len(scores)
    return scores, mode, positive_rate


def mixture_infer(observations: list[int], matrix: list[list[float]], alpha: list[float]) -> tuple[list[float], int, float]:
    output_buckets = len(matrix)
    counts = [0.0] * output_buckets
    for obs in observations:
        counts[obs] += 1.0
    empirical = [count / len(observations) for count in counts]
    normal = [max(value, 1e-12) for value in matvec(matrix, alpha)]
    best: tuple[float, int, float, list[float]] | None = None
    for cand in range(len(alpha)):
        fault = [max(matrix[row][cand], 1e-12) for row in range(output_buckets)]
        for step in range(1, 71):
            rate = step / 100.0
            loss = sum((empirical[row] - ((1.0 - rate) * normal[row] + rate * fault[row])) ** 2 for row in range(output_buckets))
            if best is None or loss < best[0]:
                best = (loss, cand, rate, fault)
    assert best is not None
    _, cand, rate, fault = best
    scores = []
    for obs in observations:
        mixture = max((1.0 - rate) * normal[obs] + rate * fault[obs], 1e-12)
        scores.append(rate * fault[obs] / mixture)
    return scores, cand, rate


def hmm_infer(
    observations: list[int],
    matrix: list[list[float]],
    alpha: list[float],
    expected_rate: float,
    segment_length: int,
) -> tuple[list[float], int, float]:
    normal = [max(value, 1e-12) for value in matvec(matrix, alpha)]
    p11 = max(0.50, 1.0 - 1.0 / max(segment_length, 2))
    p01 = min(0.20, max(1e-4, expected_rate * (1.0 - p11) / max(1.0 - expected_rate, 1e-6)))
    best: tuple[float, int, list[float]] | None = None
    for cand in range(len(alpha)):
        fault = [max(matrix[row][cand], 1e-12) for row in range(len(matrix))]
        log_likelihood, _ = forward_backward_candidate(observations, normal, fault, p01, p11, expected_rate, False)
        if best is None or log_likelihood > best[0]:
            best = (log_likelihood, cand, fault)
    assert best is not None
    _, cand, fault = best
    _, gamma = forward_backward_candidate(observations, normal, fault, p01, p11, expected_rate, True)
    return gamma, cand, sum(gamma) / len(gamma)


def load_station(work_dir: Path) -> tuple[list[str], list[list[int]], dict[str, list[float]], list[list[float]]]:
    qc_text = (work_dir / f"{STATION_ID}_qc.cdl").read_text(encoding="utf-8")
    values_text = (work_dir / f"{STATION_ID}_values.cdl").read_text(encoding="utf-8")
    _, order_info = parse_summary_order(work_dir / "all_fails_summary_20251006.dat")
    order = [code for code, info in sorted(order_info.items(), key=lambda item: int(item[1]["summary_index"])) if int(info["summary_index"]) < 71]
    rows_match = re.search(r"\btime\s*=\s*(\d+)\s*;", qc_text)
    tests_match = re.search(r"\btest\s*=\s*(\d+)\s*;", qc_text)
    if not rows_match or not tests_match:
        raise ValueError("Could not parse HadISD dimensions.")
    qc = parse_qc_matrix(qc_text, int(rows_match.group(1)), int(tests_match.group(1)))
    arrays = {variable: parse_number_array(values_text, variable) for variable in VALUE_VARIABLES if variable != "flagged_obs"}
    flat_flagged = parse_number_array(values_text, "flagged_obs")
    flagged_rows = [flat_flagged[i : i + 19] for i in range(0, len(flat_flagged), 19)]
    return order, qc, arrays, flagged_rows


def row_value(arrays: dict[str, list[float]], flagged_rows: list[list[float]], variable: str, flag_col: int, index: int) -> float:
    flagged = flagged_rows[index][flag_col]
    return flagged if not math.isnan(flagged) else arrays[variable][index]


def valid_value(variable: str, value: float) -> bool:
    if math.isnan(value) or abs(value) > 1e20:
        return False
    if value in {-999.0, -888.0}:
        return False
    if variable == "winddirs" and not (0.0 <= value <= 360.0):
        return False
    return True


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


def run_case(
    code: str,
    values: list[float],
    labels: list[int],
    label: str,
    epsilon: float,
    seed: int,
    raw_buckets: int,
    output_buckets: int,
    segment_length: int,
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
                "panel": "hadisd_real_streak_pmldp",
                "station_id": STATION_ID,
                "label_code": code,
                "label": label,
                "epsilon": epsilon,
                "seed": seed,
                "method": method,
                "calibration": "all_nonflagged_nonlabel_rows",
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


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["label_code"]), str(row["method"])), []).append(row)
    out = []
    for (code, method), part in groups.items():
        out.append(
            {
                "panel": "hadisd_real_streak_pmldp",
                "station_id": STATION_ID,
                "label_code": code,
                "method": method,
                "runs": len(part),
                "epsilons": ",".join(str(item) for item in sorted({row["epsilon"] for row in part})),
                "seeds": ",".join(str(item) for item in sorted({row["seed"] for row in part})),
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
    out.sort(key=lambda row: (row["label_code"], -float(row["auprc_mean"])))
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small PM-LDP operator panel on HadISD straight-string labels.")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/privsaf_hadisd_audit"))
    parser.add_argument("--ncdump", default=None)
    parser.add_argument("--epsilons", default="2,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--raw-buckets", type=int, default=32)
    parser.add_argument("--output-buckets", type=int, default=32)
    parser.add_argument("--segment-length", type=int, default=8)
    parser.add_argument("--output-prefix", default="hadisd_streak")
    args = parser.parse_args()

    ncdump = find_ncdump(args.ncdump)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    build_outputs(args.work_dir, ncdump)
    order, qc, arrays, flagged_rows = load_station(args.work_dir)

    epsilons = [float(item) for item in args.epsilons.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    rows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for code in CASES:
        values, labels, label = build_case(code, order, qc, arrays, flagged_rows)
        inventory.append(
            {
                "station_id": STATION_ID,
                "label_code": code,
                "label": label,
                "n_rows": len(labels),
                "positive_rows": sum(labels),
                "fault_rate": sum(labels) / len(labels),
                "usable_value_rule": "stored variable unless HadISD flagged_obs preserves removed value",
            }
        )
        for epsilon in epsilons:
            for seed in seeds:
                rows.extend(
                    run_case(
                        code,
                        values,
                        labels,
                        label,
                        epsilon,
                        seed,
                        args.raw_buckets,
                        args.output_buckets,
                        args.segment_length,
                    )
                )

    summary = summarize(rows)
    source = {
        "status": "pass",
        "panel": "hadisd_real_streak_pmldp",
        "source": "HadISD v3.4.3.2025f",
        "station_id": STATION_ID,
        "station_url": STATION_URL,
        "cases": list(CASES),
        "epsilons": epsilons,
        "seeds": seeds,
        "methods": ["pm_column_likelihood_scan", "privsaf_mixture", "privsaf_hmm"],
        "interpretation": (
            "Single-station PM-LDP check. RSS wind-direction straight strings select PrivSAF-HMM "
            "strongly; TSS/DSS are too sparse for an acceptance-level real-fault claim."
        ),
    }

    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / f"{args.output_prefix}_pmldp_runs.csv", rows)
    write_csv(RESULTS / f"{args.output_prefix}_pmldp_summary.csv", summary)
    write_csv(RESULTS / f"{args.output_prefix}_pmldp_inventory.csv", inventory)
    (RESULTS / f"{args.output_prefix}_pmldp_source_metadata.json").write_text(
        json.dumps(source, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} HadISD PM-LDP rows.")
    print(RESULTS / f"{args.output_prefix}_pmldp_summary.csv")


if __name__ == "__main__":
    main()
