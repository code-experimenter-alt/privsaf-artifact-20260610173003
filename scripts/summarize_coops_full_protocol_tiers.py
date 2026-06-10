from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNS = RESULTS / "coops_verified_flat_full_protocol_runs.csv"
OUT = RESULTS / "coops_verified_flat_full_protocol_tier_summary.csv"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def bootstrap_ci(values: list[float], seed: int = 314159, rounds: int = 500) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    estimates = []
    for _ in range(rounds):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(mean(sample))
    estimates.sort()
    return estimates[int(0.025 * (rounds - 1))], estimates[int(0.975 * (rounds - 1))]


def tier_names(fault_rows: int) -> list[str]:
    out = ["all_ge_5"]
    if fault_rows >= 25:
        out.append("support_ge_25")
    else:
        out.append("support_5_to_24")
    return out


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["buckets"]) not in (0, 24):
            continue
        for tier in tier_names(int(float(row["fault_rows"]))):
            groups[(tier, row["method"], float(row["epsilon"]), int(row["buckets"]))].append(row)

    out: list[dict[str, object]] = []
    for (tier, method, epsilon, buckets), group in groups.items():
        by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in group:
            by_case[row["case"]].append(row)
        case_auroc = [mean([float(row["auroc"]) for row in vals]) for vals in by_case.values()]
        case_auprc = [mean([float(row["auprc"]) for row in vals]) for vals in by_case.values()]
        lo, hi = bootstrap_ci(case_auprc)
        case_exemplars = [vals[0] for vals in by_case.values()]
        total_test_rows = sum(int(float(row["n_test"])) for row in case_exemplars)
        total_fault_rows = sum(int(float(row["fault_rows"])) for row in case_exemplars)
        out.append(
            {
                "panel": "coops_verified_flat_full_protocol",
                "tier": tier,
                "method": method,
                "epsilon": epsilon,
                "buckets": buckets,
                "cases": len(by_case),
                "runs": len(group),
                "total_test_rows": total_test_rows,
                "total_fault_rows": total_fault_rows,
                "pooled_fault_rate": total_fault_rows / max(total_test_rows, 1),
                "case_mean_fault_rows": mean([float(row["fault_rows"]) for row in case_exemplars]),
                "case_mean_auroc": mean(case_auroc),
                "case_mean_auprc": mean(case_auprc),
                "case_mean_auprc_ci95_low": lo,
                "case_mean_auprc_ci95_high": hi,
                "mean_recall_at_5pct_fpr": mean([float(row["recall_at_5pct_fpr"]) for row in group]),
                "mean_precision_at_top_5pct": mean([float(row["precision_at_top_5pct"]) for row in group]),
            }
        )
    out.sort(key=lambda row: (row["tier"], float(row["epsilon"]), int(row["buckets"]), -float(row["case_mean_auprc"])))
    return out


def main() -> None:
    rows = list(csv.DictReader(RUNS.open(encoding="utf-8")))
    out = summarize(rows)
    with OUT.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {OUT} rows={len(out)}")


if __name__ == "__main__":
    main()
