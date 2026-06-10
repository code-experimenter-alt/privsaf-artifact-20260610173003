from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def best_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(["auprc_mean", "auroc_mean"], ascending=[False, False]).iloc[0]


def best_privsaf_row(df: pd.DataFrame) -> pd.Series | None:
    sub = df[df["method"].str.contains("privsaf", case=False, na=False)]
    if sub.empty:
        return None
    return best_row(sub)


def add_selector_row(
    rows: list[dict[str, object]],
    *,
    benchmark: str,
    task: str,
    evidence_layer: str,
    df: pd.DataFrame,
    guidance: str,
    cases_col: str = "cases",
) -> None:
    best = best_row(df)
    priv = best_privsaf_row(df)
    priv_method = "" if priv is None else str(priv["method"])
    priv_auroc = "" if priv is None else float(priv["auroc_mean"])
    priv_auprc = "" if priv is None else float(priv["auprc_mean"])
    rows.append(
        {
            "benchmark": benchmark,
            "task": task,
            "evidence_layer": evidence_layer,
            "recommended_method": str(best["method"]),
            "recommended_family": recommended_family(str(best["method"])),
            "best_auroc": float(best["auroc_mean"]),
            "best_auprc": float(best["auprc_mean"]),
            "privsaf_best_method": priv_method,
            "privsaf_best_auroc": priv_auroc,
            "privsaf_best_auprc": priv_auprc,
            "suite_auprc_gain_vs_privsaf": "" if priv is None else float(best["auprc_mean"] - priv["auprc_mean"]),
            "cases": int(best[cases_col]) if cases_col in best and pd.notna(best[cases_col]) else "",
            "guidance": guidance,
        }
    )


def recommended_family(method: str) -> str:
    method_l = method.lower()
    if "privsaf" in method_l:
        return "PrivSAF posterior model"
    if "glr" in method_l or "cusum" in method_l or "likelihood" in method_l or "spike_score" in method_l:
        return "local channel statistic"
    if "iforest" in method_l or "lof" in method_l or "subseq" in method_l:
        return "PM-window anomaly baseline"
    if "missing_symbol" in method_l or "rolling_missing" in method_l:
        return "availability detector"
    if "generic_pm_hmm" in method_l:
        return "generic PM-HMM"
    return "other"


def main() -> None:
    rows: list[dict[str, object]] = []

    channel = pd.read_csv(RESULTS / "icde_channel_baseline_summary.csv")
    for fault_mode, guidance in {
        "iid_stuck": "Use the mixture posterior when faults are iid stuck-at points.",
        "segment_stuck": "Use the HMM posterior for persistent single-bucket scalar stuck-at segments.",
        "template_stuck": "Use the HMM posterior when stuck values and durations are template-like but still single-bucket.",
    }.items():
        add_selector_row(
            rows,
            benchmark="five public scalar streams",
            task=fault_mode,
            evidence_layer="controlled pre-PM stuck-at",
            df=channel[channel["fault_mode"] == fault_mode],
            guidance=guidance,
        )

    nab = pd.read_csv(RESULTS / "icde_issue_native_labeled_summary.csv")
    add_selector_row(
        rows,
        benchmark="NAB official system-failure labels",
        task="system_failure",
        evidence_layer="official real/weak labels",
        df=nab,
        guidance="Use PM-window anomaly baselines when labels denote broad system failures rather than stuck-at semantics.",
    )

    iors = pd.read_csv(RESULTS / "iors_stuck_qc_pmldp_summary.csv")
    add_selector_row(
        rows,
        benchmark="I-ORS deployed rangefinder",
        task="real_stuck_qc",
        evidence_layer="real deployment QC labels",
        df=iors,
        guidance="Use local GLR for field QC streams with multiple stuck levels or regime-local evidence.",
    )

    low = pd.read_csv(RESULTS / "iors_stuck_qc_loweps_pmldp_by_epsilon.csv")
    for eps in [0.5, 1.0]:
        sub = low[low["epsilon"].astype(float) == eps]
        add_selector_row(
            rows,
            benchmark="I-ORS deployed rangefinder",
            task=f"real_stuck_qc_eps_{eps:g}",
            evidence_layer="low-epsilon real deployment QC labels",
            df=sub,
            guidance="Use LLR-CUSUM first at low epsilon for local changes in real QC streams.",
        )

    dropout = pd.read_csv(RESULTS / "iors_dropout_qc_pmldp_summary.csv")
    add_selector_row(
        rows,
        benchmark="I-ORS deployed rangefinder",
        task="real_dropout_qc",
        evidence_layer="real availability labels",
        df=dropout,
        guidance="Use the explicit missing-symbol path when the availability bit is reported.",
    )

    wsn = pd.read_csv(RESULTS / "wsn_stuck_labeled_pmldp_summary.csv")
    add_selector_row(
        rows,
        benchmark="WSN prepared stuck-at labels",
        task="prepared_multivariate_stuck",
        evidence_layer="public prepared stuck labels",
        df=wsn,
        guidance="Use PM-window GLR for row-level prepared labels that mix several sensor channels.",
    )

    mech = pd.read_csv(RESULTS / "mechanism_fault_extension_summary.csv")
    for (mechanism, fault_mode), sub in mech.groupby(["mechanism", "fault_mode"]):
        add_selector_row(
            rows,
            benchmark=f"Air Quality {mechanism}",
            task=str(fault_mode),
            evidence_layer="mechanism/fault-family extension",
            df=sub,
            guidance=mechanism_fault_guidance(str(fault_mode)),
        )

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "channel_aware_detector_suite_summary.csv", index=False)

    rollup = (
        out.groupby(["recommended_family"], as_index=False)
        .agg(tasks=("task", "count"), mean_best_auprc=("best_auprc", "mean"))
        .sort_values(["tasks", "mean_best_auprc"], ascending=[False, False])
    )
    rollup.to_csv(RESULTS / "channel_aware_detector_suite_rollup.csv", index=False)
    print(f"Wrote {len(out)} selector rows.")
    print(RESULTS / "channel_aware_detector_suite_summary.csv")


def mechanism_fault_guidance(fault_mode: str) -> str:
    if fault_mode == "segment_stuck":
        return "Use the HMM posterior once epsilon is high enough; otherwise compare against local GLR."
    if fault_mode == "spike":
        return "Use lightweight local statistics for isolated spikes; reserve posterior stuck models for stuck-at rows."
    return "Use local channel-window statistics for smooth value-shift faults such as bias, scale, and drift."


if __name__ == "__main__":
    main()
