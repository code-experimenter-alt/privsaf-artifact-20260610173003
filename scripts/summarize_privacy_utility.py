from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def summarize_runs(
    runs: pd.DataFrame,
    group_cols: list[str],
    privsaf_methods: set[str],
    benchmark: str,
) -> pd.DataFrame:
    grouped = (
        runs.groupby(group_cols + ["method"], as_index=False)
        .agg(
            cases=("auroc", "size"),
            auroc_mean=("auroc", "mean"),
            auprc_mean=("auprc", "mean"),
        )
        .sort_values(group_cols + ["auprc_mean", "auroc_mean"], ascending=[True] * len(group_cols) + [False, False])
    )
    rows: list[dict[str, object]] = []
    for key, part in grouped.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        best = part.iloc[0]
        priv = part[part["method"].isin(privsaf_methods)].sort_values(["auprc_mean", "auroc_mean"], ascending=False)
        priv_row = priv.iloc[0] if len(priv) else None
        row = {"benchmark": benchmark}
        row.update(dict(zip(group_cols, key)))
        row.update(
            {
                "best_method": best["method"],
                "best_auroc": float(best["auroc_mean"]),
                "best_auprc": float(best["auprc_mean"]),
                "best_cases": int(best["cases"]),
                "best_privsaf_method": priv_row["method"] if priv_row is not None else "",
                "best_privsaf_auroc": float(priv_row["auroc_mean"]) if priv_row is not None else np.nan,
                "best_privsaf_auprc": float(priv_row["auprc_mean"]) if priv_row is not None else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_privacy_utility_summary() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    stuck_paths = [
        RESULTS / "iors_stuck_qc_loweps_pmldp_runs.csv",
        RESULTS / "iors_stuck_qc_pmldp_runs.csv",
    ]
    stuck_runs = pd.concat([pd.read_csv(path) for path in stuck_paths if path.exists()], ignore_index=True)
    if len(stuck_runs):
        stuck_runs["task"] = "real_stuck_qc"
        frames.append(
            summarize_runs(
                stuck_runs,
                ["task", "epsilon"],
                {"privsaf_hmm_global", "privsaf_hmm_scan", "privsaf_mixture"},
                "I-ORS real stuck-QC PM-LDP",
            )
        )

    dropout_path = RESULTS / "iors_dropout_qc_pmldp_runs.csv"
    if dropout_path.exists():
        dropout_runs = pd.read_csv(dropout_path)
        dropout_runs["task"] = "real_dropout_qc"
        frames.append(
            summarize_runs(
                dropout_runs,
                ["task", "epsilon"],
                {"privsaf_dropout_hmm"},
                "I-ORS real dropout-QC PM-LDP",
            )
        )

    mechanism_path = RESULTS / "mechanism_fault_extension_runs.csv"
    if mechanism_path.exists():
        mechanism_runs = pd.read_csv(mechanism_path)
        mechanism_runs["task"] = mechanism_runs["mechanism"] + "_" + mechanism_runs["fault_mode"]
        frames.append(
            summarize_runs(
                mechanism_runs,
                ["task", "epsilon"],
                {"privsaf_hmm", "privsaf_mixture"},
                "Air Quality mechanism/fault extension",
            )
        )

    if not frames:
        raise RuntimeError("No privacy-utility inputs found.")
    out = pd.concat(frames, ignore_index=True)
    out["low_epsilon"] = out["epsilon"].astype(float) <= 1.0
    return out.sort_values(["benchmark", "task", "epsilon"])


def build_device_ledger() -> pd.DataFrame:
    epsilons = [0.5, 1.0, 2.0, 4.0]
    cadences = [
        ("I-ORS 10-minute telemetry", 10),
        ("hourly telemetry", 60),
        ("daily telemetry", 24 * 60),
    ]
    device_budgets = [1.0, 10.0]
    rows: list[dict[str, object]] = []
    for eps in epsilons:
        for cadence_name, minutes in cadences:
            reports_per_day = 24 * 60 / minutes
            base = {
                "epsilon_per_report": eps,
                "cadence": cadence_name,
                "minutes_per_report": minutes,
                "reports_per_day": reports_per_day,
                "device_epsilon_per_day_basic_composition": eps * reports_per_day,
                "device_epsilon_per_30_days_basic_composition": eps * reports_per_day * 30,
                "device_epsilon_per_year_basic_composition": eps * reports_per_day * 365,
            }
            for budget in device_budgets:
                max_reports = int(np.floor(budget / eps))
                days = max_reports / reports_per_day
                row = dict(base)
                row.update(
                    {
                        "device_budget": budget,
                        "max_reports_under_budget": max_reports,
                        "days_until_budget_exhausted": days,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    privacy_utility = build_privacy_utility_summary()
    device_ledger = build_device_ledger()
    privacy_utility.to_csv(RESULTS / "privacy_utility_loweps_summary.csv", index=False)
    device_ledger.to_csv(RESULTS / "device_level_privacy_ledger.csv", index=False)
    print(RESULTS / "privacy_utility_loweps_summary.csv")
    print(RESULTS / "device_level_privacy_ledger.csv")


if __name__ == "__main__":
    main()
