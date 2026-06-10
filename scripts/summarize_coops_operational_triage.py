from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNS = RESULTS / "coops_verified_flat_full_protocol_runs.csv"
OUT = RESULTS / "coops_verified_flat_operational_triage_summary.csv"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def ci(values: list[float], seed: int = 271828, rounds: int = 500) -> tuple[float, float]:
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


def tiers(fault_rows: int) -> list[str]:
    return ["all_ge_5", "support_ge_25" if fault_rows >= 25 else "support_5_to_24"]


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, float, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["buckets"]) != 24 or float(row["epsilon"]) <= 0.0:
            continue
        for tier in tiers(int(float(row["fault_rows"]))):
            groups[(tier, row["method"], float(row["epsilon"]), int(row["buckets"]))].append(row)

    out = []
    for (tier, method, epsilon, buckets), group in groups.items():
        by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in group:
            by_case[row["case"]].append(row)
        exemplars = [vals[0] for vals in by_case.values()]
        total_test = sum(int(float(row["n_test"])) for row in exemplars)
        total_fault = sum(int(float(row["fault_rows"])) for row in exemplars)
        prevalence = total_fault / max(total_test, 1)
        case_top1 = [mean([float(row["precision_at_top_1pct"]) for row in vals]) for vals in by_case.values()]
        case_top5 = [mean([float(row["precision_at_top_5pct"]) for row in vals]) for vals in by_case.values()]
        top1_lo, top1_hi = ci(case_top1)
        top5_lo, top5_hi = ci(case_top5)
        mean_top1 = mean(case_top1)
        mean_top5 = mean(case_top5)
        out.append(
            {
                "panel": "coops_verified_flat_full_protocol",
                "tier": tier,
                "method": method,
                "epsilon": epsilon,
                "buckets": buckets,
                "cases": len(by_case),
                "runs": len(group),
                "total_test_rows": total_test,
                "total_fault_rows": total_fault,
                "pooled_prevalence": prevalence,
                "mean_precision_at_top_1pct": mean_top1,
                "precision_at_top_1pct_ci95_low": top1_lo,
                "precision_at_top_1pct_ci95_high": top1_hi,
                "lift_top_1pct_vs_prevalence": mean_top1 / prevalence if prevalence else float("nan"),
                "mean_precision_at_top_5pct": mean_top5,
                "precision_at_top_5pct_ci95_low": top5_lo,
                "precision_at_top_5pct_ci95_high": top5_hi,
                "lift_top_5pct_vs_prevalence": mean_top5 / prevalence if prevalence else float("nan"),
                "mean_recall_at_5pct_fpr": mean([float(row["recall_at_5pct_fpr"]) for row in group]),
            }
        )
    out.sort(key=lambda row: (row["tier"], float(row["epsilon"]), -float(row["mean_precision_at_top_1pct"])))
    return out


def main() -> None:
    rows = list(csv.DictReader(RUNS.open(encoding="utf-8")))
    out = summarize(rows)
    with OUT.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {OUT} rows={len(out)}")


if __name__ == "__main__":
    main()
