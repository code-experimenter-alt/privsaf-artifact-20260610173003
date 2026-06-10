from __future__ import annotations

import csv
import json
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


def f(value: str) -> float:
    return float(value) if value not in {"", "nan", "None"} else float("nan")


def boundary_class(row: dict[str, str]) -> tuple[str, str, str]:
    source = row["source"]
    best_operator = row["best_operator"]
    privsaf_operator = row["privsaf_operator"]
    if "dropout" in source.lower() or "availability" in row["layer"].lower():
        return (
            "availability_extension",
            "supports dropout emission, not stuck-at generality",
            "count separately from stuck-at/flatline evidence",
        )
    if "hadisd" in source.lower():
        return (
            "privsaf_compatible",
            "supports scalar repeated-value/stuck semantics",
            "RSS/WSS anchor high-AUPRC real-streak evidence; TSS/DSS add station-page coverage",
        )
    if "co-ops" in source.lower() or "coops" in source.lower():
        return (
            "privsaf_compatible",
            "supports official scalar flat-tolerance semantics with public values preserved",
            "direct value+label benchmark with official flat-tolerance labels and measured full-panel KRR-LDP range-HMM selection",
        )
    if best_operator == privsaf_operator or best_operator.startswith("identity_channel_hmm"):
        return (
            "privsaf_compatible",
            "supports scalar flatline/stuck semantics",
            "mined native flatline-like evidence with edge-bucket provenance recorded",
        )
    return (
        "semantic_mismatch_router_boundary",
        "supports router selection for label semantics outside single-bucket stuck-at",
        "validation selects local channel statistics for field-QC or row-level labels",
    )


def main() -> None:
    layered = read_csv(RESULTS / "real_label_layered_analysis.csv")
    native_detail = read_csv(RESULTS / "native_flatline_event_detail_audit.csv")
    edge_events = sum(int(row["edge_bucket"]) for row in native_detail)
    clipping_events = sum(int(row["clipping_risk"]) for row in native_detail)
    total_native_events = len(native_detail)

    rows: list[dict[str, object]] = []
    for row in layered:
        boundary, role, scope = boundary_class(row)
        best = f(row["best_auprc"])
        priv = f(row["privsaf_auprc"])
        rows.append(
            {
                "layer": row["layer"],
                "source": row["source"],
                "label_semantics": row["label_semantics"],
                "key_scale": row["key_scale"],
                "best_operator": row["best_operator"],
                "best_auprc": best,
                "privsaf_operator": row["privsaf_operator"],
                "privsaf_auprc": priv,
                "privsaf_gap_to_best_auprc": priv - best,
                "boundary_class": boundary,
                "evidence_role": role,
                "claim_scope": scope,
                "counts_as_privsaf_stuck_evidence": int(boundary == "privsaf_compatible"),
            }
        )

    stuck_rows = [row for row in rows if row["boundary_class"] != "availability_extension"]
    privsaf_rows = [row for row in stuck_rows if row["boundary_class"] == "privsaf_compatible"]
    mismatch_rows = [row for row in stuck_rows if row["boundary_class"] == "semantic_mismatch_router_boundary"]
    summary = {
        "real_or_hardware_adjacent_layers": len(rows),
        "stuck_or_flatline_layers": len(stuck_rows),
        "privsaf_compatible_stuck_layers": len(privsaf_rows),
        "semantic_mismatch_router_boundary_layers": len(mismatch_rows),
        "availability_extension_layers": len(rows) - len(stuck_rows),
        "native_flatline_events": total_native_events,
        "native_edge_bucket_events": edge_events,
        "native_clipping_risk_events": clipping_events,
        "interpretation": (
            "HadISD adds one direct PrivSAF-compatible real straight-string layer: the strongest RSS "
            "station selects PrivSAF-HMM, the compact screen favors HMM on RSS/WSS wind strings, "
            "and page-7/page-0 screens add 58 low-prevalence TSS/DSS cases where page 0 roughly "
            "doubles AUPRC over prevalence. "
            "NOAA CO-OPS verified six-minute rows join WL_VALUE with official F=1 flat-tolerance flags; "
            "the frozen all-eligible KRR-LDP panel uses prior-month calibration only and selects a PrivSAF "
            "range-HMM above report-frequency and GLR baselines. "
            "Native flatline evidence records edge-bucket provenance, while I-ORS stuck-QC and "
            "WSN prepared stuck labels demonstrate router selection for field-QC and row-level semantics."
        ),
    }

    write_csv(
        OUTPUT_RESULTS / "real_fault_privsaf_boundary_audit.csv",
        rows,
        [
            "layer",
            "source",
            "label_semantics",
            "key_scale",
            "best_operator",
            "best_auprc",
            "privsaf_operator",
            "privsaf_auprc",
            "privsaf_gap_to_best_auprc",
            "boundary_class",
            "evidence_role",
            "claim_scope",
            "counts_as_privsaf_stuck_evidence",
        ],
    )
    write_csv(
        OUTPUT_RESULTS / "real_fault_privsaf_boundary_summary.csv",
        [summary],
        [
            "real_or_hardware_adjacent_layers",
            "stuck_or_flatline_layers",
            "privsaf_compatible_stuck_layers",
            "semantic_mismatch_router_boundary_layers",
            "availability_extension_layers",
            "native_flatline_events",
            "native_edge_bucket_events",
            "native_clipping_risk_events",
            "interpretation",
        ],
    )
    (OUTPUT_RESULTS / "real_fault_privsaf_boundary_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
