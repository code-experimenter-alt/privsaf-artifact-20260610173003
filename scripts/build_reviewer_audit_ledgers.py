from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / path)


def policy_for_layer(layer: str, policy: pd.DataFrame) -> pd.Series:
    if layer in {"iid_stuck"}:
        key = "iid scalar stuck-at points"
    elif layer in {
        "segment_stuck",
        "template_stuck",
        "native_flatline_windows",
        "real_station_straight_string",
        "duchi_binary_segment_stuck",
    }:
        key = "persistent scalar stuck-at or flatline segments"
    elif layer == "official_flat_tolerance_labels":
        key = "official flat-tolerance labels with scalar values preserved"
    elif layer.startswith("field_qc"):
        key = "deployed field-QC stuck labels encoding local regime changes"
    elif layer == "prepared_multivariate_stuck_labels":
        key = "row-level prepared multivariate stuck labels"
    elif layer == "explicit_availability_dropout":
        key = "explicit availability or dropout labels"
    else:
        key = "persistent scalar stuck-at or flatline segments"
    rows = policy[policy["semantic_trigger"] == key]
    if rows.empty:
        raise KeyError(f"No router policy row for {layer}")
    return rows.iloc[0]


def build_router_ledger() -> pd.DataFrame:
    policy = read_csv("private_telemetry_router_decision_policy.csv")
    summary = read_csv("private_telemetry_cleaning_router_summary.csv")
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        prow = policy_for_layer(str(row["semantic_layer"]), policy)
        rows.append(
            {
                "semantic_layer": row["semantic_layer"],
                "allowed_methods_before_validation": prow["compatibility_set"],
                "validation_metric": prow["selection_metric"],
                "selected_method": row["router_operator"],
                "test_metric": f"AUROC={float(row['best_auroc']):.6f}; AUPRC={float(row['best_auprc']):.6f}",
                "evidence_file": prow["evidence_files"],
                "audit_note": prow["audit_rule"],
            }
        )
    return pd.DataFrame(rows)


def best_non_privsaf_native() -> tuple[str, float]:
    native = read_csv("icde_native_weaklabel_summary.csv")
    non_priv = native[~native["method"].str.contains("privsaf", case=False, na=False)]
    best = non_priv.sort_values("auprc", ascending=False).iloc[0]
    return str(best["method"]), float(best["auprc"])


def best_raw_coops() -> tuple[str, float]:
    coops = read_csv("coops_verified_flat_full_protocol_summary.csv")
    raw = coops[coops["method"].str.startswith("raw_", na=False)]
    best = raw.sort_values("case_mean_auprc", ascending=False).iloc[0]
    return str(best["method"]), float(best["case_mean_auprc"])


def build_router_ablation() -> pd.DataFrame:
    summary = read_csv("private_telemetry_cleaning_router_summary.csv")
    real_layers = {
        "native_flatline_windows": ("results/icde_native_weaklabel_summary.csv", best_non_privsaf_native),
        "field_qc_stuck_labels": ("results/iors_stuck_qc_pmldp_summary.csv", None),
        "field_qc_low_epsilon_0.5": ("results/iors_stuck_qc_loweps_pmldp_by_epsilon.csv", None),
        "field_qc_low_epsilon_1": ("results/iors_stuck_qc_loweps_pmldp_by_epsilon.csv", None),
        "explicit_availability_dropout": ("results/iors_dropout_qc_pmldp_summary.csv", None),
        "prepared_multivariate_stuck_labels": ("results/wsn_stuck_labeled_pmldp_summary.csv", None),
        "real_station_straight_string": ("results/hadisd_multistation_streak_pmldp_rollup.csv", None),
        "official_flat_tolerance_labels": ("results/coops_verified_flat_full_protocol_summary.csv", best_raw_coops),
    }
    rows: list[dict[str, object]] = []
    for layer, (evidence, oracle_fn) in real_layers.items():
        part = summary[summary["semantic_layer"] == layer]
        if part.empty:
            continue
        row = part.iloc[0]
        oracle_method = "not_reported"
        oracle_auprc = np.nan
        if oracle_fn is not None:
            oracle_method, oracle_auprc = oracle_fn()
        rows.append(
            {
                "semantic_layer": layer,
                "panel": row["source"],
                "privsaf_only_method": row["privsaf_operator"],
                "privsaf_only_AUPRC": float(row["privsaf_auprc"]),
                "router_selected_method": row["router_operator"],
                "router_selected_AUPRC": float(row["best_auprc"]),
                "best_incompatible_or_oracle_method": oracle_method,
                "best_incompatible_or_oracle_AUPRC": oracle_auprc,
                "evidence_file": evidence,
                "interpretation": row["interpretation"],
            }
        )
    return pd.DataFrame(rows)


def build_coops_operational() -> pd.DataFrame:
    triage = read_csv("coops_verified_flat_operational_triage_summary.csv")
    preferred = triage[
        (triage["method"] == "privsaf_range_hmm_r1_fixed_prior")
        & (triage["buckets"] == 24)
        & (triage["tier"] == "all_ge_5")
    ]
    if preferred.empty:
        preferred = triage.sort_values("mean_precision_at_top_1pct", ascending=False).head(1)
    row = preferred.iloc[0]
    rows: list[dict[str, object]] = []
    for frac, precision_col, lift_col in [
        (0.01, "mean_precision_at_top_1pct", "lift_top_1pct_vs_prevalence"),
        (0.05, "mean_precision_at_top_5pct", "lift_top_5pct_vs_prevalence"),
    ]:
        review_rows = float(row["total_test_rows"]) * frac
        expected_hits = review_rows * float(row[precision_col])
        recall = expected_hits / max(float(row["total_fault_rows"]), 1.0)
        rows.append(
            {
                "panel": row["panel"],
                "tier": row["tier"],
                "method": row["method"],
                "epsilon": float(row["epsilon"]),
                "buckets": int(row["buckets"]),
                "threshold": f"top_{int(frac * 100)}pct_rank",
                "precision": float(row[precision_col]),
                "recall": float(recall),
                "topk_precision": float(row[precision_col]),
                "prevalence_lift": float(row[lift_col]),
                "estimated_review_burden_rows": int(round(review_rows)),
                "total_test_rows": int(row["total_test_rows"]),
                "total_fault_rows": int(row["total_fault_rows"]),
                "pooled_prevalence": float(row["pooled_prevalence"]),
                "recall_source": "derived_from_topk_precision_and_total_fault_rows",
            }
        )
    rows.append(
        {
            "panel": row["panel"],
            "tier": row["tier"],
            "method": row["method"],
            "epsilon": float(row["epsilon"]),
            "buckets": int(row["buckets"]),
            "threshold": "5pct_fpr",
            "precision": np.nan,
            "recall": float(row["mean_recall_at_5pct_fpr"]),
            "topk_precision": np.nan,
            "prevalence_lift": np.nan,
            "estimated_review_burden_rows": np.nan,
            "total_test_rows": int(row["total_test_rows"]),
            "total_fault_rows": int(row["total_fault_rows"]),
            "pooled_prevalence": float(row["pooled_prevalence"]),
            "recall_source": "reported_mean_recall_at_5pct_fpr",
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reviewer-facing audit ledgers from existing result CSVs.")
    parser.add_argument("--output-dir", default=str(RESULTS))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    router_ledger = build_router_ledger()
    router_ablation = build_router_ablation()
    coops_operational = build_coops_operational()
    router_ledger.to_csv(out / "reviewer_router_ledger.csv", index=False)
    router_ablation.to_csv(out / "reviewer_router_ablation.csv", index=False)
    coops_operational.to_csv(out / "reviewer_coops_operational_metrics.csv", index=False)
    print(f"Wrote {len(router_ledger)} rows to {out / 'reviewer_router_ledger.csv'}")
    print(f"Wrote {len(router_ablation)} rows to {out / 'reviewer_router_ablation.csv'}")
    print(f"Wrote {len(coops_operational)} rows to {out / 'reviewer_coops_operational_metrics.csv'}")


if __name__ == "__main__":
    main()
