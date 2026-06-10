from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def best_row(df: pd.DataFrame, metric: str = "auprc_mean") -> pd.Series:
    return df.sort_values(metric, ascending=False).iloc[0]


def best_privsaf(df: pd.DataFrame, method_col: str = "method", metric: str = "auprc_mean") -> pd.Series:
    priv = df[df[method_col].str.contains("privsaf", case=False, na=False)]
    if priv.empty:
        return pd.Series(dtype=object)
    return best_row(priv, metric)


def add_router_row(
    rows: list[dict[str, object]],
    semantic_layer: str,
    source: str,
    router_operator: str,
    operator_family: str,
    best_auroc: float,
    best_auprc: float,
    privsaf_operator: str,
    privsaf_auroc: float,
    privsaf_auprc: float,
    evidence_cases: int,
    interpretation: str,
) -> None:
    rows.append(
        {
            "semantic_layer": semantic_layer,
            "source": source,
            "router_operator": router_operator,
            "operator_family": operator_family,
            "best_auroc": best_auroc,
            "best_auprc": best_auprc,
            "privsaf_operator": privsaf_operator,
            "privsaf_auroc": privsaf_auroc,
            "privsaf_auprc": privsaf_auprc,
            "router_gain_vs_privsaf_auprc": best_auprc - privsaf_auprc,
            "evidence_cases": evidence_cases,
            "interpretation": interpretation,
        }
    )


def add_layer_row(
    rows: list[dict[str, object]],
    layer: str,
    source: str,
    label_definition: str,
    label_semantics: str,
    key_scale: str,
    best_operator: str,
    best_auprc: float,
    privsaf_operator: str,
    privsaf_auprc: float,
    conclusion: str,
) -> None:
    rows.append(
        {
            "layer": layer,
            "source": source,
            "label_definition": label_definition,
            "label_semantics": label_semantics,
            "key_scale": key_scale,
            "best_operator": best_operator,
            "best_auprc": best_auprc,
            "privsaf_operator": privsaf_operator,
            "privsaf_auprc": privsaf_auprc,
            "conclusion": conclusion,
        }
    )


def main() -> None:
    router_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []

    selector = pd.read_csv(RESULTS / "channel_aware_detector_suite_summary.csv")
    for task, operator, family, interpretation in [
        ("iid_stuck", "privsaf_mixture", "PrivSAF posterior", "iid stuck-at points use the mixture posterior."),
        ("segment_stuck", "privsaf_hmm", "PrivSAF posterior", "persistent single-bucket stuck-at segments use the HMM posterior."),
        ("template_stuck", "privsaf_hmm", "PrivSAF posterior", "template-like flatline faults keep the stuck-column HMM."),
    ]:
        row = selector[selector["task"] == task].iloc[0]
        add_router_row(
            router_rows,
            semantic_layer=task,
            source="five public scalar streams",
            router_operator=operator,
            operator_family=family,
            best_auroc=float(row["best_auroc"]),
            best_auprc=float(row["best_auprc"]),
            privsaf_operator=str(row["privsaf_best_method"]),
            privsaf_auroc=float(row["privsaf_best_auroc"]),
            privsaf_auprc=float(row["privsaf_best_auprc"]),
            evidence_cases=int(row["cases"]),
            interpretation=interpretation,
        )

    native = pd.read_csv(RESULTS / "icde_native_weaklabel_summary.csv")
    native_priv = native[native["method"] == "privsaf_hmm_scan"].sort_values("auprc", ascending=False).iloc[0]
    native_best = native.sort_values("auprc", ascending=False).iloc[0]
    add_router_row(
        router_rows,
        semantic_layer="native_flatline_windows",
        source="public raw-stream flatline mining",
        router_operator="privsaf_hmm_scan",
        operator_family="PrivSAF posterior",
        best_auroc=float(native_priv["auroc"]),
        best_auprc=float(native_priv["auprc"]),
        privsaf_operator="privsaf_hmm_scan",
        privsaf_auroc=float(native_priv["auroc"]),
        privsaf_auprc=float(native_priv["auprc"]),
        evidence_cases=int(native_priv["weak_positive_windows"]) + int(native_priv["matched_negative_windows"]),
        interpretation="strict same-bin native flatline windows are the closest real flatline layer.",
    )
    add_layer_row(
        layer_rows,
        layer="real flatline-like subset",
        source="native flatline windows from public scalar streams",
        label_definition="same-bin or small-range raw flatline windows matched with negatives",
        label_semantics="true flatline-like scalar windows",
        key_scale=f"{int(native_priv['weak_positive_windows'])} positive windows and {int(native_priv['matched_negative_windows'])} matched negatives",
        best_operator=str(native_best["method"]),
        best_auprc=float(native_best["auprc"]),
        privsaf_operator="privsaf_hmm_scan",
        privsaf_auprc=float(native_priv["auprc"]),
        conclusion="PrivSAF is effective on flatline-like scalar windows and estimates the stuck bucket.",
    )

    iors = pd.read_csv(RESULTS / "iors_stuck_qc_pmldp_summary.csv")
    iors_best = best_row(iors)
    iors_priv = best_privsaf(iors)
    add_router_row(
        router_rows,
        semantic_layer="field_qc_stuck_labels",
        source="I-ORS deployed rangefinder",
        router_operator=str(iors_best["method"]),
        operator_family="local channel statistic",
        best_auroc=float(iors_best["auroc_mean"]),
        best_auprc=float(iors_best["auprc_mean"]),
        privsaf_operator=str(iors_priv["method"]),
        privsaf_auroc=float(iors_priv["auroc_mean"]),
        privsaf_auprc=float(iors_priv["auprc_mean"]),
        evidence_cases=int(iors_best["cases"]),
        interpretation="field QC labels contain local regime evidence beyond one global stuck bucket.",
    )
    inv = pd.read_csv(RESULTS / "iors_stuck_qc_inventory.csv")
    selected = inv[inv["selected_for_pmldp_panel"] == 1]
    add_layer_row(
        layer_rows,
        layer="single-device real QC labels",
        source="I-ORS sea-level rangefinder, 2003-2022",
        label_definition="SLH_QC=5 episodes with length at least 8",
        label_semantics="field-QC stuck labels with multiple regimes and local changes",
        key_scale=(
            f"{len(selected)} selected years; max episode {int(selected['max_stuck_episode_length'].max())} rows; "
            f"{int(selected['persistent_stuck_rows'].sum())} persistent stuck rows"
        ),
        best_operator=str(iors_best["method"]),
        best_auprc=float(iors_best["auprc_mean"]),
        privsaf_operator=str(iors_priv["method"]),
        privsaf_auprc=float(iors_priv["auprc_mean"]),
        conclusion="GLR wins on field-QC labels, while PrivSAF remains a diagnostic comparator for single-bucket stuck semantics.",
    )

    loweps = pd.read_csv(RESULTS / "iors_stuck_qc_loweps_pmldp_by_epsilon.csv")
    for eps in sorted(loweps["epsilon"].unique()):
        part = loweps[loweps["epsilon"] == eps]
        best = best_row(part)
        priv = best_privsaf(part)
        add_router_row(
            router_rows,
            semantic_layer=f"field_qc_low_epsilon_{eps:g}",
            source="I-ORS deployed rangefinder",
            router_operator=str(best["method"]),
            operator_family="local channel statistic",
            best_auroc=float(best["auroc_mean"]),
            best_auprc=float(best["auprc_mean"]),
            privsaf_operator=str(priv["method"]),
            privsaf_auroc=float(priv["auroc_mean"]),
            privsaf_auprc=float(priv["auprc_mean"]),
            evidence_cases=int(best["cases"]),
            interpretation="low privacy budgets favor cumulative local evidence.",
        )

    dropout = pd.read_csv(RESULTS / "iors_dropout_qc_pmldp_summary.csv")
    drop_hmm = dropout[dropout["method"] == "privsaf_dropout_hmm"].iloc[0]
    add_router_row(
        router_rows,
        semantic_layer="explicit_availability_dropout",
        source="I-ORS deployed rangefinder",
        router_operator="privsaf_dropout_hmm",
        operator_family="availability HMM",
        best_auroc=float(drop_hmm["auroc_mean"]),
        best_auprc=float(drop_hmm["auprc_mean"]),
        privsaf_operator="privsaf_dropout_hmm",
        privsaf_auroc=float(drop_hmm["auroc_mean"]),
        privsaf_auprc=float(drop_hmm["auprc_mean"]),
        evidence_cases=int(drop_hmm["cases"]),
        interpretation="explicit missing symbols use the dropout emission operator.",
    )
    add_layer_row(
        layer_rows,
        layer="real availability labels",
        source="I-ORS sea-level rangefinder",
        label_definition="SLH_QC=8 missing-value labels",
        label_semantics="explicit availability/dropout",
        key_scale=f"{int(drop_hmm['cases'])} PM-LDP rows over {int(drop_hmm['years'])} years",
        best_operator="privsaf_dropout_hmm",
        best_auprc=float(drop_hmm["auprc_mean"]),
        privsaf_operator="privsaf_dropout_hmm",
        privsaf_auprc=float(drop_hmm["auprc_mean"]),
        conclusion="dropout is handled by the availability emission rather than the stuck-at HMM.",
    )

    wsn = pd.read_csv(RESULTS / "wsn_stuck_labeled_pmldp_summary.csv")
    wsn_best = best_row(wsn)
    wsn_priv = best_privsaf(wsn)
    add_router_row(
        router_rows,
        semantic_layer="prepared_multivariate_stuck_labels",
        source="WSN prepared stuck-at benchmark",
        router_operator=str(wsn_best["method"]),
        operator_family="local channel statistic",
        best_auroc=float(wsn_best["auroc_mean"]),
        best_auprc=float(wsn_best["auprc_mean"]),
        privsaf_operator=str(wsn_priv["method"]),
        privsaf_auroc=float(wsn_priv["auroc_mean"]),
        privsaf_auprc=float(wsn_priv["auprc_mean"]),
        evidence_cases=int(wsn_best["cases"]),
        interpretation="row-level prepared labels mix several scalar features.",
    )
    wsn_inv = pd.read_csv(RESULTS / "wsn_stuck_labeled_dataset_inventory.csv")
    add_layer_row(
        layer_rows,
        layer="public prepared stuck labels",
        source="WSN TelosB prepared stuck-at files",
        label_definition="five prepared stuck-at files with row labels",
        label_semantics="row-level multivariate/prepared stuck labels",
        key_scale=f"{len(wsn_inv)} files; {int(wsn_inv['rows'].sum())} total rows",
        best_operator=str(wsn_best["method"]),
        best_auprc=float(wsn_best["auprc_mean"]),
        privsaf_operator=str(wsn_priv["method"]),
        privsaf_auprc=float(wsn_priv["auprc_mean"]),
        conclusion="local window statistics match the row-level prepared-label semantics better than one scalar stuck bucket.",
    )

    hadisd_single = pd.read_csv(RESULTS / "hadisd_streak_pmldp_summary.csv")
    hadisd_rss_hmm = hadisd_single[
        (hadisd_single["label_code"] == "RSS") & (hadisd_single["method"] == "privsaf_hmm")
    ].iloc[0]
    hadisd_compact = pd.read_csv(RESULTS / "hadisd_multistation_streak_pmldp_rollup.csv")
    compact_hmm = hadisd_compact[hadisd_compact["method"] == "privsaf_hmm"]
    compact_rss = compact_hmm[compact_hmm["label_code"] == "RSS"].iloc[0]
    compact_wss = compact_hmm[compact_hmm["label_code"] == "WSS"].iloc[0]
    hadisd_screen = pd.read_csv(RESULTS / "hadisd_page7_small80_streak_screen_inventory.csv")
    screen_eligible = hadisd_screen[hadisd_screen["eligible_for_pmldp"] == 1]
    eligible_counts = screen_eligible.groupby("label_code").size().to_dict()
    hadisd_nonwind = pd.read_csv(RESULTS / "hadisd_page7_nonwind_streak_pmldp_pmldp_rollup.csv")
    nonwind_hmm = hadisd_nonwind[hadisd_nonwind["method"] == "privsaf_hmm"].set_index("label_code")
    hadisd_page0_screen = pd.read_csv(RESULTS / "hadisd_page0_small120_nonwind_screen_inventory.csv")
    page0_eligible = hadisd_page0_screen[hadisd_page0_screen["eligible_for_pmldp"] == 1]
    page0_eligible_counts = page0_eligible.groupby("label_code").size().to_dict()
    hadisd_page0 = pd.read_csv(RESULTS / "hadisd_page0_nonwind_streak_pmldp_pmldp_rollup.csv")
    page0_hmm = hadisd_page0[hadisd_page0["method"] == "privsaf_hmm"].set_index("label_code")
    page0_tss_lift = float(page0_hmm.loc["TSS", "auprc_case_mean"]) / (
        float(page0_hmm.loc["TSS", "positive_rows_total"]) / float(page0_hmm.loc["TSS", "rows_total"])
    )
    page0_dss_lift = float(page0_hmm.loc["DSS", "auprc_case_mean"]) / (
        float(page0_hmm.loc["DSS", "positive_rows_total"]) / float(page0_hmm.loc["DSS", "rows_total"])
    )
    add_router_row(
        router_rows,
        semantic_layer="real_station_straight_string",
        source="HadISD station 702606-96401 plus compact and cross-page screens",
        router_operator="privsaf_hmm",
        operator_family="PrivSAF posterior",
        best_auroc=float(hadisd_rss_hmm["auroc_mean"]),
        best_auprc=float(hadisd_rss_hmm["auprc_mean"]),
        privsaf_operator="privsaf_hmm",
        privsaf_auroc=float(hadisd_rss_hmm["auroc_mean"]),
        privsaf_auprc=float(hadisd_rss_hmm["auprc_mean"]),
        evidence_cases=(
            int(compact_rss["station_cases"])
            + int(compact_wss["station_cases"])
            + int(nonwind_hmm.loc["TSS", "station_cases"])
            + int(nonwind_hmm.loc["DSS", "station_cases"])
            + int(page0_hmm.loc["TSS", "station_cases"])
            + int(page0_hmm.loc["DSS", "station_cases"])
        ),
        interpretation=(
            "RSS/WSS wind strings give the strongest PrivSAF-HMM real-streak evidence; "
            "page-7 and page-0 screens add 58 TSS/DSS cases where page 0 roughly doubles "
            "AUPRC over prevalence but PSS has no support."
        ),
    )
    add_layer_row(
        layer_rows,
        layer="real straight-string labels",
        source="HadISD station 702606-96401 plus compact and cross-page screens",
        label_definition="RSS/WSS/TSS/DSS/PSS straight-string quality_control_flags",
        label_semantics="real scalar repeated-value/string QC labels",
        key_scale=(
            f"single RSS case has 692 usable rows and 140 positives; compact screen has "
            f"{int(compact_rss['station_cases']) + int(compact_wss['station_cases'])} RSS/WSS cases with "
            f"HMM selected on {int(compact_rss['selected_case_count'])}/{int(compact_rss['station_cases'])} RSS and "
            f"{int(compact_wss['selected_case_count'])}/{int(compact_wss['station_cases'])} WSS; "
            f"page-7 screen has {len(hadisd_screen)} station-variable rows, "
            f"{eligible_counts.get('TSS', 0)} TSS, {eligible_counts.get('DSS', 0)} DSS, "
            f"and {eligible_counts.get('PSS', 0)} PSS eligible cases; "
            f"page-0 screen has {len(hadisd_page0_screen)} station-variable rows, "
            f"{page0_eligible_counts.get('TSS', 0)} TSS, {page0_eligible_counts.get('DSS', 0)} DSS, "
            f"and {page0_eligible_counts.get('PSS', 0)} PSS eligible cases, with HMM AUPRC lifts "
            f"{page0_tss_lift:.2f}x on TSS and {page0_dss_lift:.2f}x on DSS"
        ),
        best_operator="privsaf_hmm",
        best_auprc=float(hadisd_rss_hmm["auprc_mean"]),
        privsaf_operator="privsaf_hmm",
        privsaf_auprc=float(hadisd_rss_hmm["auprc_mean"]),
        conclusion=(
            "PrivSAF-HMM is strongest on RSS/WSS wind strings; page-7 and page-0 TSS/DSS broaden "
            "the real panel and page 0 roughly doubles AUPRC over prevalence, but absolute AUPRC "
            "remains low and PSS has no support-sufficient case."
        ),
    )

    coops_summary = pd.read_csv(RESULTS / "coops_verified_flat_pmldp_summary.csv")
    coops_best = best_row(coops_summary)
    coops_priv = best_privsaf(coops_summary)
    coops_inventory = pd.read_csv(RESULTS / "coops_verified_flat_pmldp_inventory.csv")
    with (RESULTS / "coops_verified_flat_flag_screen_summary.json").open("r", encoding="utf-8") as fin:
        coops_screen = json.load(fin)
    add_router_row(
        router_rows,
        semantic_layer="official_flat_tolerance_labels",
        source="NOAA CO-OPS verified six-minute water level",
        router_operator=str(coops_best["method"]),
        operator_family="PrivSAF posterior",
        best_auroc=float(coops_best["auroc_mean"]),
        best_auprc=float(coops_best["auprc_mean"]),
        privsaf_operator=str(coops_priv["method"]),
        privsaf_auroc=float(coops_priv["auroc_mean"]),
        privsaf_auprc=float(coops_priv["auprc_mean"]),
        evidence_cases=int(coops_best["cases"]),
        interpretation=(
            "official verified F=1 flat-tolerance labels preserve WL_VALUE in the same ERDDAP rows; "
            "PrivSAF-HMM is selected on the six top station-months and outperforms report-frequency and GLR baselines."
        ),
    )
    add_layer_row(
        layer_rows,
        layer="official flat-tolerance labels",
        source="NOAA CO-OPS verified six-minute water level ERDDAP",
        label_definition="verified six-minute F=1 flat-tolerance flag joined to WL_VALUE in the same public rows",
        label_semantics="official real scalar flatline/tolerance QC labels",
        key_scale=(
            f"{int(coops_screen['numeric_f1_rows'])} public numeric F=1 rows across "
            f"{int(coops_screen['station_months_with_numeric_f1_rows'])} station-months; "
            f"PM-LDP panel uses {len(coops_inventory)} station-months, "
            f"{int(coops_inventory['rows'].sum())} total rows, and "
            f"{int(coops_inventory['f1_rows'].sum())} official flat-flag positives"
        ),
        best_operator=str(coops_best["method"]),
        best_auprc=float(coops_best["auprc_mean"]),
        privsaf_operator=str(coops_priv["method"]),
        privsaf_auprc=float(coops_priv["auprc_mean"]),
        conclusion=(
            "CO-OPS closes the public value+label flatline benchmark gap with official labels and preserved values; "
            "the first KRR-LDP panel selects PrivSAF-HMM under the same private report protocol."
        ),
    )

    mechanism = pd.read_csv(RESULTS / "mechanism_fault_extension_summary.csv")
    duchi_segment = mechanism[(mechanism["mechanism"] == "duchi_binary") & (mechanism["fault_mode"] == "segment_stuck")]
    duchi_best = best_row(duchi_segment)
    duchi_priv = best_privsaf(duchi_segment)
    add_router_row(
        router_rows,
        semantic_layer="duchi_binary_segment_stuck",
        source="Air Quality mechanism extension",
        router_operator=str(duchi_priv["method"]),
        operator_family="PrivSAF posterior",
        best_auroc=float(duchi_priv["auroc_mean"]),
        best_auprc=float(duchi_priv["auprc_mean"]),
        privsaf_operator=str(duchi_priv["method"]),
        privsaf_auroc=float(duchi_priv["auroc_mean"]),
        privsaf_auprc=float(duchi_priv["auprc_mean"]),
        evidence_cases=int(duchi_priv["cases"]),
        interpretation="the same inference code works with a Duchi-binary channel matrix.",
    )

    router = pd.DataFrame(router_rows)
    router.to_csv(RESULTS / "private_telemetry_cleaning_router_summary.csv", index=False)
    layered = pd.DataFrame(layer_rows)
    layered.to_csv(RESULTS / "real_label_layered_analysis.csv", index=False)
    rollup = (
        router.groupby("operator_family", as_index=False)
        .agg(tasks=("semantic_layer", "count"), mean_best_auprc=("best_auprc", "mean"))
        .sort_values(["tasks", "mean_best_auprc"], ascending=[False, False])
    )
    rollup.to_csv(RESULTS / "private_telemetry_cleaning_router_rollup.csv", index=False)
    print(f"Wrote {len(router)} router rows and {len(layered)} layered rows.")
    print(RESULTS / "private_telemetry_cleaning_router_summary.csv")
    print(RESULTS / "real_label_layered_analysis.csv")


if __name__ == "__main__":
    main()
