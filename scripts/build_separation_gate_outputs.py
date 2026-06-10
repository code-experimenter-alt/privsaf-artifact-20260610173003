from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_icde_revision_grid import (
    histogram_alpha,
    load_streams,
    pm_matrix,
    raw_bucket_index,
    robust_normalize,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def separation(m: np.ndarray, alpha: np.ndarray, stuck_bucket: int) -> tuple[float, float]:
    normal = np.clip(m @ alpha, 1e-12, None)
    normal /= normal.sum()
    stuck = np.clip(m[:, stuck_bucket], 1e-12, None)
    stuck /= stuck.sum()
    tv = 0.5 * float(np.sum(np.abs(stuck - normal)))
    kl = float(np.sum(stuck * (np.log(stuck) - np.log(normal))))
    return tv, kl


def stream_alpha_by_dataset(
    raw_buckets: int,
    output_buckets: int,
    eps_values: list[float],
    detection: pd.DataFrame,
) -> dict[tuple[str, float, float], tuple[float, float]]:
    out: dict[tuple[str, float, float], tuple[float, float]] = {}
    for stream in load_streams(include_optional=False):
        split = robust_normalize(stream.values, test_len=2400)
        alpha = histogram_alpha(split.train, raw_buckets)
        for eps in eps_values:
            m, _ = pm_matrix(eps, raw_buckets, output_buckets)
            part = detection[
                (detection["dataset_id"] == stream.dataset_id)
                & (detection["epsilon"] == eps)
                & (detection["fault_mode"] == "segment_stuck")
                & (detection["method"] == "privsaf_hmm")
                & detection["stuck_value"].notna()
            ]
            for stuck_value in sorted(part["stuck_value"].unique()):
                bucket = raw_bucket_index(float(stuck_value), raw_buckets)
                out[(stream.dataset_id, eps, float(stuck_value))] = separation(m, alpha, bucket)
    return out


def build_failure_table(raw_buckets: int, output_buckets: int) -> pd.DataFrame:
    detection = pd.read_csv(RESULTS / "icde_revision_detection_runs.csv")
    stress = pd.read_csv(RESULTS / "reviewer_stress_runs.csv")
    eps_values = sorted(float(x) for x in detection["epsilon"].dropna().unique())
    sep_lookup = stream_alpha_by_dataset(raw_buckets, output_buckets, eps_values, detection)

    tail = detection[
        (detection["fault_mode"] == "segment_stuck")
        & (detection["method"] == "privsaf_hmm")
        & (detection["method_category"] == "privacy_aware")
        & detection["stuck_value"].notna()
    ].copy()
    tail["rank"] = tail.groupby(["dataset_id", "epsilon"])["stuck_value"].rank(method="dense")
    max_rank = tail.groupby(["dataset_id", "epsilon"])["rank"].transform("max")
    tail["stuck_value_type"] = np.where(tail["rank"] == 1, "tail_q05", np.where(tail["rank"] == max_rank, "tail_q95", "tail_other"))
    tail["TV_sep"] = [
        sep_lookup.get((row.dataset_id, float(row.epsilon), float(row.stuck_value)), (np.nan, np.nan))[0]
        for row in tail.itertuples()
    ]
    tail["KL_sep"] = [
        sep_lookup.get((row.dataset_id, float(row.epsilon), float(row.stuck_value)), (np.nan, np.nan))[1]
        for row in tail.itertuples()
    ]
    tail_summary = (
        tail[tail["stuck_value_type"].isin(["tail_q05", "tail_q95"])]
        .groupby(["stuck_value_type", "epsilon"], as_index=False)
        .agg(
            cases=("auprc", "size"),
            TV_sep=("TV_sep", "mean"),
            KL_sep=("KL_sep", "mean"),
            AUPRC=("auprc", "mean"),
            bucket_hit1=("bucket_acc", "mean"),
            prevalence=("fault_rate", "mean"),
        )
    )

    central = stress[
        (stress["method"] == "privsaf_hmm")
        & (stress["calibration_condition"] == "matched_train")
        & (stress["fault_mode"] == "segment_stuck")
    ]
    central_summary = (
        central.groupby(["stuck_value_type", "epsilon"], as_index=False)
        .agg(
            cases=("AUPRC", "size"),
            TV_sep=("TV_sep", "mean"),
            KL_sep=("KL_sep", "mean"),
            AUPRC=("AUPRC", "mean"),
            bucket_hit1=("bucket_hit1", "mean"),
            prevalence=("fault_rate", "mean"),
        )
    )
    out = pd.concat([tail_summary, central_summary], ignore_index=True)
    order = {"tail_q05": 0, "tail_q95": 1, "median": 2, "mode_bucket": 3, "min_tv_column": 4}
    out["sort_key"] = out["stuck_value_type"].map(order).fillna(99)
    return out.sort_values(["sort_key", "epsilon"]).drop(columns=["sort_key"])


def tail_gate_runs(raw_buckets: int, output_buckets: int) -> pd.DataFrame:
    detection = pd.read_csv(RESULTS / "icde_revision_detection_runs.csv")
    eps_values = sorted(float(x) for x in detection["epsilon"].dropna().unique())
    sep_lookup = stream_alpha_by_dataset(raw_buckets, output_buckets, eps_values, detection)
    tail = detection[
        (detection["fault_mode"] == "segment_stuck")
        & (detection["method"].isin(["privsaf_hmm", "ldp_window_glr", "ldp_distribution_surprise"]))
        & detection["stuck_value"].notna()
    ].copy()
    tail["rank"] = tail.groupby(["dataset_id", "epsilon"])["stuck_value"].rank(method="dense")
    max_rank = tail.groupby(["dataset_id", "epsilon"])["rank"].transform("max")
    tail["stuck_value_type"] = np.where(tail["rank"] == 1, "tail_q05", np.where(tail["rank"] == max_rank, "tail_q95", "tail_other"))
    tail = tail[tail["stuck_value_type"].isin(["tail_q05", "tail_q95"])]
    rows = pd.DataFrame(
        {
            "semantic_layer": "controlled_tail_segment",
            "dataset": tail["dataset_id"],
            "epsilon": tail["epsilon"].astype(float),
            "stuck_value_type": tail["stuck_value_type"],
            "seed": tail["seed"].astype(int),
            "method": tail["method"],
            "AUPRC": tail["auprc"].astype(float),
            "prevalence": tail["fault_rate"].astype(float),
            "TV_sep": [
                sep_lookup.get((row.dataset_id, float(row.epsilon), float(row.stuck_value)), (np.nan, np.nan))[0]
                for row in tail.itertuples()
            ],
            "KL_sep": [
                sep_lookup.get((row.dataset_id, float(row.epsilon), float(row.stuck_value)), (np.nan, np.nan))[1]
                for row in tail.itertuples()
            ],
            "evidence_file": "results/icde_revision_detection_runs.csv",
        }
    )
    return rows


def stress_gate_runs() -> pd.DataFrame:
    runs = pd.read_csv(RESULTS / "reviewer_stress_runs.csv")
    runs = runs[
        (runs["calibration_condition"] == "matched_train")
        & (runs["fault_mode"] == "segment_stuck")
    ].copy()
    candidate_methods = ["privsaf_hmm", "ldp_window_glr", "ldp_distribution_surprise"]
    runs = runs[runs["method"].isin(candidate_methods)]
    return pd.DataFrame(
        {
            "semantic_layer": "controlled_central_segment",
            "dataset": runs["dataset"],
            "epsilon": runs["epsilon"].astype(float),
            "stuck_value_type": runs["stuck_value_type"],
            "seed": runs["seed"].astype(int),
            "method": runs["method"],
            "AUPRC": runs["AUPRC"].astype(float),
            "prevalence": runs["fault_rate"].astype(float),
            "TV_sep": runs["TV_sep"].astype(float),
            "KL_sep": runs["KL_sep"].astype(float),
            "evidence_file": "results/reviewer_stress_runs.csv",
        }
    )


def lift(value: float, prevalence: float) -> float:
    if prevalence <= 0 or np.isnan(prevalence):
        return np.nan
    return float(value / prevalence)


def select_by_validation(validation: pd.DataFrame, allowed_methods: list[str]) -> str:
    pool = validation[validation["method"].isin(allowed_methods)]
    if pool.empty:
        return "abstain"
    scores = pool.groupby("method")["AUPRC"].mean().sort_values(ascending=False)
    return str(scores.index[0])


def build_gate_ledger(
    tv_threshold: float,
    kl_threshold: float,
    raw_buckets: int,
    output_buckets: int,
    utility_lift_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs = pd.concat([tail_gate_runs(raw_buckets, output_buckets), stress_gate_runs()], ignore_index=True)
    rows: list[dict[str, object]] = []
    group_cols = ["semantic_layer", "dataset", "epsilon", "stuck_value_type"]
    route_methods = ["ldp_window_glr", "ldp_distribution_surprise"]
    validation_methods = ["privsaf_hmm", *route_methods]
    for key, group in runs.groupby(group_cols):
        semantic_layer, dataset, epsilon, stuck_value_type = key
        validation = group[group["seed"] == 0]
        test = group[group["seed"] != 0]
        tv = float(group["TV_sep"].mean())
        kl = float(group["KL_sep"].mean())
        prevalence = float(group["prevalence"].mean())
        separation_pass = bool(tv >= tv_threshold and kl >= kl_threshold)
        route_method = select_by_validation(validation, route_methods)
        if separation_pass:
            selected_method = select_by_validation(validation, validation_methods)
            if selected_method == "privsaf_hmm":
                gate_stage = "separation_pass_privsaf_validation_selected"
                selected_action = "use_privsaf"
            elif selected_method == "abstain":
                gate_stage = "separation_pass_validation_unavailable"
                selected_action = "abstain"
            else:
                gate_stage = "separation_pass_validation_route"
                selected_action = "route_non_privsaf"
        else:
            selected_method = route_method
            gate_stage = "separation_fail_route_or_abstain"
            selected_action = "abstain" if route_method == "abstain" else "route_non_privsaf"
        selected_validation = validation[validation["method"] == selected_method]["AUPRC"].mean()
        selected_test = test[test["method"] == selected_method]["AUPRC"].mean()
        ungated_validation = validation[validation["method"] == "privsaf_hmm"]["AUPRC"].mean()
        ungated_test = test[test["method"] == "privsaf_hmm"]["AUPRC"].mean()
        rows.append(
            {
                "semantic_layer": semantic_layer,
                "dataset": dataset,
                "epsilon": float(epsilon),
                "stuck_value_type": stuck_value_type,
                "TV_sep": tv,
                "KL_sep": kl,
                "selected_action": selected_action,
                "selected_method": selected_method,
                "validation_AUPRC": float(selected_validation),
                "validation_lift": lift(float(selected_validation), prevalence),
                "test_AUPRC": float(selected_test),
                "test_lift": lift(float(selected_test), prevalence),
                "prevalence": prevalence,
                "abstain_flag": 0 if separation_pass else 1,
                "separation_screen_pass": int(separation_pass),
                "gate_stage": gate_stage,
                "candidate_bucket_source": "synthetic_audit_bucket",
                "evidence_file": ";".join(sorted(group["evidence_file"].unique())),
                "ungated_privsaf_validation_AUPRC": float(ungated_validation),
                "ungated_privsaf_validation_lift": lift(float(ungated_validation), prevalence),
                "ungated_privsaf_test_AUPRC": float(ungated_test),
                "ungated_privsaf_test_lift": lift(float(ungated_test), prevalence),
                "tv_threshold": float(tv_threshold),
                "kl_threshold": float(kl_threshold),
                "utility_lift_threshold": float(utility_lift_threshold),
                "threshold_source": "fixed_diagnostic_thresholds_not_tuned_on_test_labels",
                "test_label_use": "final_reporting_and_false_accept_audit_only",
            }
        )
    ledger = pd.DataFrame(rows).sort_values(["epsilon", "stuck_value_type", "dataset"])
    accepted = ledger[ledger["separation_screen_pass"] == 1]
    rejected = ledger[ledger["separation_screen_pass"] == 0]
    summary = pd.DataFrame(
        [
            {
                "semantic_layer": "controlled_tail_and_central_segments",
                "groups": int(len(ledger)),
                "accepted_groups": int(len(accepted)),
                "rejected_groups": int(len(rejected)),
                "coverage_rate": float(len(accepted) / max(len(ledger), 1)),
                "gated_selected_test_AUPRC": float(ledger["test_AUPRC"].mean()),
                "ungated_privsaf_test_AUPRC": float(ledger["ungated_privsaf_test_AUPRC"].mean()),
                "accepted_selected_test_AUPRC": float(accepted["test_AUPRC"].mean()) if len(accepted) else np.nan,
                "accepted_ungated_privsaf_test_AUPRC": float(accepted["ungated_privsaf_test_AUPRC"].mean()) if len(accepted) else np.nan,
                "rejected_selected_test_AUPRC": float(rejected["test_AUPRC"].mean()) if len(rejected) else np.nan,
                "rejected_ungated_privsaf_test_AUPRC": float(rejected["ungated_privsaf_test_AUPRC"].mean()) if len(rejected) else np.nan,
                "tv_threshold": float(tv_threshold),
                "kl_threshold": float(kl_threshold),
                "utility_lift_threshold": float(utility_lift_threshold),
                "threshold_source": "fixed_diagnostic_thresholds_not_tuned_on_test_labels",
            }
        ]
    )
    audited = ledger.copy()
    audited["utility_good"] = audited["ungated_privsaf_test_lift"] >= utility_lift_threshold
    audited["screen_bucket"] = np.where(audited["separation_screen_pass"] == 1, "accepted", "rejected")
    audited["utility_bucket"] = np.where(audited["utility_good"], "good", "bad")
    audited["accounting_bucket"] = audited["screen_bucket"] + "-" + audited["utility_bucket"]
    false_rows = []
    for bucket in ["accepted-good", "accepted-bad", "rejected-good", "rejected-bad"]:
        part = audited[audited["accounting_bucket"] == bucket]
        false_rows.append(
            {
                "accounting_bucket": bucket,
                "count": int(len(part)),
                "mean_privsaf_AUPRC": float(part["ungated_privsaf_test_AUPRC"].mean()) if len(part) else np.nan,
                "mean_privsaf_prevalence_lift": float(part["ungated_privsaf_test_lift"].mean()) if len(part) else np.nan,
                "mean_selected_AUPRC": float(part["test_AUPRC"].mean()) if len(part) else np.nan,
                "mean_selected_prevalence_lift": float(part["test_lift"].mean()) if len(part) else np.nan,
                "tv_threshold": float(tv_threshold),
                "kl_threshold": float(kl_threshold),
                "utility_lift_threshold": float(utility_lift_threshold),
                "test_label_use": "post_hoc_false_accept_false_reject_accounting_only",
            }
        )
    false_accounting = pd.DataFrame(false_rows)
    return ledger, summary, false_accounting


def main() -> None:
    parser = argparse.ArgumentParser(description="Build separation-gated deployment ledgers from measured result CSVs.")
    parser.add_argument("--output-dir", default=str(RESULTS))
    parser.add_argument("--tv-threshold", type=float, default=0.10)
    parser.add_argument("--kl-threshold", type=float, default=0.03)
    parser.add_argument("--utility-lift-threshold", type=float, default=2.0)
    parser.add_argument("--raw-buckets", type=int, default=32)
    parser.add_argument("--output-buckets", type=int, default=32)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gate_ledger, gate_summary, false_accounting = build_gate_ledger(
        args.tv_threshold,
        args.kl_threshold,
        args.raw_buckets,
        args.output_buckets,
        args.utility_lift_threshold,
    )
    failure = build_failure_table(args.raw_buckets, args.output_buckets)
    gate_ledger.to_csv(out / "reviewer_separation_gate_ledger.csv", index=False)
    gate_summary.to_csv(out / "reviewer_separation_gate_summary.csv", index=False)
    false_accounting.to_csv(out / "reviewer_separation_gate_false_accounting.csv", index=False)
    failure.to_csv(out / "reviewer_separation_failure_table.csv", index=False)
    print(f"Wrote {len(gate_ledger)} rows to {out / 'reviewer_separation_gate_ledger.csv'}")
    print(f"Wrote {len(gate_summary)} rows to {out / 'reviewer_separation_gate_summary.csv'}")
    print(f"Wrote {len(false_accounting)} rows to {out / 'reviewer_separation_gate_false_accounting.csv'}")
    print(f"Wrote {len(failure)} rows to {out / 'reviewer_separation_failure_table.csv'}")


if __name__ == "__main__":
    main()
