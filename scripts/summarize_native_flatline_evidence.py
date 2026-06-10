from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT_DIR = Path(os.environ.get("NATIVE_FLATLINE_RESULTS", str(RESULTS)))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fin:
        return list(csv.DictReader(fin))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_events() -> list[dict[str, object]]:
    candidates = read_csv(RESULTS / "icde_native_weaklabel_candidates.csv")
    negatives = read_csv(RESULTS / "icde_native_weaklabel_matched_negatives.csv")
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    neg_by_dataset: dict[str, int] = defaultdict(int)
    for row in candidates:
        by_dataset[row["dataset_id"]].append(row)
    for row in negatives:
        neg_by_dataset[row["dataset_id"]] += 1

    rows: list[dict[str, object]] = []
    for dataset_id, group in sorted(by_dataset.items()):
        lengths = [int(row["length"]) for row in group]
        ranges = [float(row["raw_range"]) for row in group]
        edge_flags = [row["flatline_bin"] in {"0", "31"} for row in group]
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset": group[0]["dataset"],
                "target": group[0]["target"],
                "positive_events": len(group),
                "matched_negative_windows": neg_by_dataset[dataset_id],
                "min_length": min(lengths),
                "median_length": median(lengths),
                "max_length": max(lengths),
                "exact_constant_events": sum(1 for value in ranges if value == 0.0),
                "small_range_events": sum(1 for value in ranges if value > 0.0),
                "edge_bucket_events": sum(1 for value in edge_flags if value),
                "interior_bucket_events": sum(1 for value in edge_flags if not value),
                "max_raw_range": max(ranges),
                "flatline_bins": ";".join(sorted({row["flatline_bin"] for row in group}, key=int)),
            }
        )
    all_lengths = [int(row["length"]) for row in candidates]
    all_ranges = [float(row["raw_range"]) for row in candidates]
    all_edge_flags = [row["flatline_bin"] in {"0", "31"} for row in candidates]
    rows.append(
        {
            "dataset_id": "all",
            "dataset": "All native flatline evidence",
            "target": "mixed scalar targets",
            "positive_events": len(candidates),
            "matched_negative_windows": len(negatives),
            "min_length": min(all_lengths),
            "median_length": median(all_lengths),
            "max_length": max(all_lengths),
            "exact_constant_events": sum(1 for value in all_ranges if value == 0.0),
            "small_range_events": sum(1 for value in all_ranges if value > 0.0),
            "edge_bucket_events": sum(1 for value in all_edge_flags if value),
            "interior_bucket_events": sum(1 for value in all_edge_flags if not value),
            "max_raw_range": max(all_ranges),
            "flatline_bins": ";".join(sorted({row["flatline_bin"] for row in candidates}, key=int)),
        }
    )
    return rows


def detail_events() -> list[dict[str, object]]:
    candidates = read_csv(RESULTS / "icde_native_weaklabel_candidates.csv")
    rows: list[dict[str, object]] = []
    for row in candidates:
        raw_range = float(row["raw_range"])
        edge_bucket = row["flatline_bin"] in {"0", "31"}
        rows.append(
            {
                "dataset_id": row["dataset_id"],
                "dataset": row["dataset"],
                "target": row["target"],
                "episode_id": row["episode_id"],
                "start": row["start"],
                "end": row["end"],
                "length": row["length"],
                "flatline_bin": row["flatline_bin"],
                "raw_mean": row["raw_mean"],
                "raw_std": row["raw_std"],
                "raw_range": row["raw_range"],
                "exact_constant": int(raw_range == 0.0),
                "small_range": int(raw_range > 0.0),
                "edge_bucket": int(edge_bucket),
                "interior_bucket": int(not edge_bucket),
                "evidence_grade": "edge_exact" if edge_bucket and raw_range == 0.0 else "edge_small_range" if edge_bucket else "interior",
                "clipping_risk": int(edge_bucket),
            }
        )
    return rows


def summarize_methods() -> list[dict[str, object]]:
    summary = read_csv(RESULTS / "icde_native_weaklabel_summary.csv")
    rows: list[dict[str, object]] = []
    for row in summary:
        rows.append(
            {
                "method": row["method"],
                "epsilon": row["epsilon"],
                "weak_positive_windows": row["weak_positive_windows"],
                "matched_negative_windows": row["matched_negative_windows"],
                "auroc": row["auroc"],
                "auprc": row["auprc"],
                "recall_at_5pct_fpr": row["recall_at_5pct_fpr"],
                "precision_at_k": row["precision_at_k"],
                "value_hit_rate_pm1": row["value_hit_rate_pm1"],
            }
        )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    event_rows = summarize_events()
    write_csv(
        OUTPUT_DIR / "native_flatline_event_audit.csv",
        event_rows,
        [
            "dataset_id",
            "dataset",
            "target",
            "positive_events",
            "matched_negative_windows",
            "min_length",
            "median_length",
            "max_length",
            "exact_constant_events",
            "small_range_events",
            "edge_bucket_events",
            "interior_bucket_events",
            "max_raw_range",
            "flatline_bins",
        ],
    )

    detail_rows = detail_events()
    write_csv(
        OUTPUT_DIR / "native_flatline_event_detail_audit.csv",
        detail_rows,
        [
            "dataset_id",
            "dataset",
            "target",
            "episode_id",
            "start",
            "end",
            "length",
            "flatline_bin",
            "raw_mean",
            "raw_std",
            "raw_range",
            "exact_constant",
            "small_range",
            "edge_bucket",
            "interior_bucket",
            "evidence_grade",
            "clipping_risk",
        ],
    )

    method_rows = summarize_methods()
    write_csv(
        OUTPUT_DIR / "native_flatline_method_audit.csv",
        method_rows,
        [
            "method",
            "epsilon",
            "weak_positive_windows",
            "matched_negative_windows",
            "auroc",
            "auprc",
            "recall_at_5pct_fpr",
            "precision_at_k",
            "value_hit_rate_pm1",
        ],
    )
    print(f"Wrote {len(event_rows)} event audit rows, {len(detail_rows)} event detail rows, and {len(method_rows)} method rows.")


if __name__ == "__main__":
    main()
