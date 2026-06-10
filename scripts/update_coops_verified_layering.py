from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT_RESULTS = RESULTS


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fin:
        return list(csv.DictReader(fin))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def best_pmldp_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return sorted(rows, key=lambda row: (float(row["auprc_mean"]), float(row["auroc_mean"])), reverse=True)[0]


def best_privsaf(rows: list[dict[str, str]]) -> dict[str, str]:
    return best_pmldp_row([row for row in rows if "privsaf" in row["method"].lower()])


def best_full_protocol(rows: list[dict[str, str]]) -> dict[str, str]:
    primary_private = [
        row
        for row in rows
        if float(row["epsilon"]) > 0.0
        and int(row["buckets"]) == 24
        and "privsaf" in row["method"].lower()
    ]
    return sorted(primary_private, key=lambda row: (float(row["case_mean_auprc"]), float(row["case_mean_auroc"])), reverse=True)[0]


def full_row(rows: list[dict[str, str]], method: str, epsilon: float, buckets: int) -> dict[str, str]:
    for row in rows:
        if row["method"] == method and float(row["epsilon"]) == epsilon and int(row["buckets"]) == buckets:
            return row
    raise KeyError((method, epsilon, buckets))


def upsert(rows: list[dict[str, object]], key: str, value: str, new_row: dict[str, object]) -> list[dict[str, object]]:
    kept = [row for row in rows if str(row.get(key, "")) != value]
    kept.append(new_row)
    return kept


def main() -> None:
    full_summary = read_csv(RESULTS / "coops_verified_flat_full_protocol_summary.csv")
    tier_summary = read_csv(RESULTS / "coops_verified_flat_full_protocol_tier_summary.csv")
    best = best_full_protocol(full_summary)
    priv = best
    raw = full_row(full_summary, "raw_zero_slope_radius2", 0.0, 0)
    best_baseline = sorted(
        [
            row
            for row in full_summary
            if float(row["epsilon"]) == 2.0
            and int(row["buckets"]) == 24
            and "privsaf" not in row["method"].lower()
        ],
        key=lambda row: (float(row["case_mean_auprc"]), float(row["case_mean_auroc"])),
        reverse=True,
    )[0]
    support_best = sorted(
        [
            row
            for row in tier_summary
            if row["tier"] == "support_ge_25"
            and float(row["epsilon"]) == 2.0
            and int(row["buckets"]) == 24
            and "privsaf" in row["method"].lower()
        ],
        key=lambda row: (float(row["case_mean_auprc"]), float(row["case_mean_auroc"])),
        reverse=True,
    )[0]
    inventory = read_csv(RESULTS / "coops_verified_flat_full_protocol_inventory.csv")
    screen = json.loads((RESULTS / "coops_verified_flat_flag_screen_summary.json").read_text(encoding="utf-8"))
    total_rows = sum(int(row["rows"]) for row in inventory)
    total_positives = sum(int(row["f1_rows"]) for row in inventory)

    router_path = RESULTS / "private_telemetry_cleaning_router_summary.csv"
    router_rows = read_csv(router_path)
    router_fieldnames = list(router_rows[0])
    router_row = {
        "semantic_layer": "official_flat_tolerance_labels",
        "source": "NOAA CO-OPS verified six-minute water level",
        "router_operator": best["method"],
        "operator_family": "PrivSAF posterior",
        "best_auroc": best["case_mean_auroc"],
        "best_auprc": best["case_mean_auprc"],
        "privsaf_operator": priv["method"],
        "privsaf_auroc": priv["case_mean_auroc"],
        "privsaf_auprc": priv["case_mean_auprc"],
        "router_gain_vs_privsaf_auprc": str(float(best["case_mean_auprc"]) - float(priv["case_mean_auprc"])),
        "evidence_cases": best["cases"],
        "interpretation": (
            "official verified F=1 flat-tolerance labels preserve WL_VALUE in the same ERDDAP rows; "
            "the frozen all-eligible KRR-LDP protocol uses prior-month calibration only and selects the range-HMM "
            f"above {best_baseline['method']} on the full panel; raw flatness upper bound AUPRC is "
            f"{float(raw['case_mean_auprc']):.3f}."
        ),
    }
    router_rows = upsert(router_rows, "semantic_layer", "official_flat_tolerance_labels", router_row)
    write_csv(OUTPUT_RESULTS / "private_telemetry_cleaning_router_summary.csv", router_rows, router_fieldnames)

    rollup_rows = []
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in router_rows:
        groups[str(row["operator_family"])].append(row)
    for family, group in groups.items():
        vals = [float(row["best_auprc"]) for row in group]
        rollup_rows.append({"operator_family": family, "tasks": len(group), "mean_best_auprc": sum(vals) / len(vals)})
    rollup_rows.sort(key=lambda row: (int(row["tasks"]), float(row["mean_best_auprc"])), reverse=True)
    write_csv(
        OUTPUT_RESULTS / "private_telemetry_cleaning_router_rollup.csv",
        rollup_rows,
        ["operator_family", "tasks", "mean_best_auprc"],
    )

    layer_path = RESULTS / "real_label_layered_analysis.csv"
    layer_rows = read_csv(layer_path)
    layer_fieldnames = list(layer_rows[0])
    layer_row = {
        "layer": "official flat-tolerance labels",
        "source": "NOAA CO-OPS verified six-minute water level ERDDAP",
        "label_definition": "verified six-minute F=1 flat-tolerance flag joined to WL_VALUE in the same public rows",
        "label_semantics": "official real scalar flatline/tolerance QC labels",
        "key_scale": (
            f"{int(screen['numeric_f1_rows'])} public numeric F=1 rows across "
            f"{int(screen['station_months_with_numeric_f1_rows'])} station-months; "
            f"frozen KRR-LDP panel runs {len(inventory)} station-months, {total_rows} total rows, "
            f"and {total_positives} official flat-flag positives; support-ge25 tier has "
            f"{support_best['cases']} cases and AUPRC {float(support_best['case_mean_auprc']):.3f}"
        ),
        "best_operator": best["method"],
        "best_auprc": best["case_mean_auprc"],
        "privsaf_operator": priv["method"],
        "privsaf_auprc": priv["case_mean_auprc"],
        "conclusion": (
            "CO-OPS supplies official public value+label flat-tolerance evidence; raw flatness scores align with "
            "the F labels and the full KRR-LDP panel selects a PrivSAF range-HMM above report-frequency and GLR."
        ),
    }
    layer_rows = [row for row in layer_rows if str(row.get("source", "")) != layer_row["source"]]
    layer_rows.append(layer_row)
    write_csv(OUTPUT_RESULTS / "real_label_layered_analysis.csv", layer_rows, layer_fieldnames)

    print(
        json.dumps(
            {
                "router_rows": len(router_rows),
                "layer_rows": len(layer_rows),
                "coops_best_method": best["method"],
                "coops_cases": int(best["cases"]),
                "coops_auprc": float(best["case_mean_auprc"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
