from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


DATASETS = [
    {
        "dataset_id": "air_quality",
        "name": "UCI Air Quality",
        "source_page": "https://archive.ics.uci.edu/dataset/360/air+quality",
        "source_download": "https://archive.ics.uci.edu/static/public/360/air+quality.zip",
        "local_status": "run",
        "target": "C6H6(GT), optional CO(GT), NO2(GT)",
        "time_step": "hourly",
        "rows_or_streams": 9358,
        "role": "verified anchor",
        "must_run": "yes",
        "preprocessing": "drop -200 target values, chronological split, train MinMax to [-1,1]",
        "split": "60/20/20 chronological, current reproduced core",
        "fault_modes": "iid stuck, segment stuck, bias, drift, dropout, saturation",
        "expected_result": "HMM wins on segment stuck faults; mixture wins on iid faults; low epsilon and median stuck values are hard.",
    },
    {
        "dataset_id": "household_power",
        "name": "UCI Individual Household Electric Power Consumption",
        "source_page": "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption",
        "source_download": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
        "local_status": "downloaded",
        "target": "Global_active_power, Voltage",
        "time_step": "1 min raw, resample 15 min or hourly",
        "rows_or_streams": 2075259,
        "role": "dense energy stream",
        "must_run": "yes",
        "preprocessing": "parse ? as missing, resample, train quantile normalization",
        "split": "chronological 60/20/20 by timestamp",
        "fault_modes": "segment stuck, bias, gradual drift, dropout",
        "expected_result": "Long smooth traces should reward temporal smoothing and HMM on persistent faults; dropout stress tests count-aware PM aggregation.",
    },
    {
        "dataset_id": "bike_sharing",
        "name": "UCI Bike Sharing",
        "source_page": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
        "source_download": "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        "local_status": "downloaded",
        "target": "hour.csv: cnt, temp, hum",
        "time_step": "hourly",
        "rows_or_streams": 17379,
        "role": "seasonal count stream",
        "must_run": "yes",
        "preprocessing": "use hour.csv, avoid leakage from casual/registered, train quantile normalization",
        "split": "chronological 60/20/20 by date",
        "fault_modes": "iid stuck, segment stuck, saturation, dropout",
        "expected_result": "Seasonality makes naive residual methods fragile; PM-selected should improve persistent fault detection without claiming real anomaly labels.",
    },
    {
        "dataset_id": "beijing_air",
        "name": "UCI Beijing Multi-Site Air Quality",
        "source_page": "https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data",
        "source_download": "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip",
        "local_status": "downloaded",
        "target": "PM2.5, TEMP, WSPM across 12 stations",
        "time_step": "hourly",
        "rows_or_streams": 420768,
        "role": "multi-station environmental stream",
        "must_run": "yes",
        "preprocessing": "station-day windows, short-gap imputation, per-station or global train normalization",
        "split": "2013-2015 train, 2016 validation, 2017 test",
        "fault_modes": "all six modes; strongest for bias, drift, dropout",
        "expected_result": "Station heterogeneity tests normalization assumptions; drift and bias failures should be documented rather than hidden.",
    },
    {
        "dataset_id": "nab",
        "name": "Numenta Anomaly Benchmark subset",
        "source_page": "https://github.com/numenta/NAB",
        "source_download": "GitHub raw CSV subset in data/nab/raw",
        "local_status": "downloaded",
        "target": "machine temperature, ambient temperature, occupancy",
        "time_step": "native timestamp",
        "rows_or_streams": 32342,
        "role": "benchmark scalar time series",
        "must_run": "yes",
        "preprocessing": "parse timestamp/value, remove or flag native anomaly windows, train quantile normalization",
        "split": "per-file chronological 60/20/20",
        "fault_modes": "iid stuck, segment stuck, dropout, saturation",
        "expected_result": "Shows method is not tied to environmental chemistry; native anomaly windows become an optional stress analysis.",
    },
    {
        "dataset_id": "gas_drift",
        "name": "UCI Gas Sensor Array Drift",
        "source_page": "https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset",
        "source_download": "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
        "local_status": "downloaded",
        "target": "sensor channels R1-R16 over batches",
        "time_step": "batch/order index",
        "rows_or_streams": 13910,
        "role": "optional drift-heavy stress dataset",
        "must_run": "optional",
        "preprocessing": "parse .dat sparse features, choose one or more sensor channels, batch-aware split",
        "split": "early batches train, middle validation, late batches test",
        "fault_modes": "gradual drift, additive bias, saturation",
        "expected_result": "Hard negative case: natural drift can violate clean-stationarity assumptions and expose limits of stuck-at-specific modeling.",
    },
]

GAPS = [
    "dataset breadth",
    "fault taxonomy",
    "baselines",
    "model selection",
    "budget estimation",
    "repair/recovery",
    "scaling",
]

GAP_SCORES = {
    "air_quality": [2, 2, 1, 2, 1, 1, 1],
    "household_power": [2, 2, 2, 2, 2, 2, 2],
    "bike_sharing": [2, 1, 2, 2, 1, 1, 1],
    "beijing_air": [2, 2, 2, 2, 2, 2, 2],
    "nab": [2, 1, 2, 2, 1, 1, 1],
    "gas_drift": [1, 2, 1, 1, 1, 1, 1],
}

FAULTS = [
    {
        "fault_mode": "iid_stuck",
        "definition": "independent faulty points are replaced by a fixed bucket value",
        "sweep": "rate={0.05,0.10,0.20}; value={q05,q50,q95,-1,+1}",
        "expected": "mixture should be competitive or better than HMM",
    },
    {
        "fault_mode": "segment_stuck",
        "definition": "contiguous segments are replaced by a fixed bucket value",
        "sweep": "rate={0.05,0.10,0.20}; length={4,8,16,32,64}",
        "expected": "HMM should win as segment length increases",
    },
    {
        "fault_mode": "additive_bias",
        "definition": "faulty values shift by a constant offset before PM-LDP",
        "sweep": "bias={-0.5,-0.25,-0.1,0.1,0.25,0.5}",
        "expected": "harder than stuck-at; robust and temporal baselines may be competitive",
    },
    {
        "fault_mode": "gradual_drift",
        "definition": "bias ramps across a segment until a maximum drift",
        "sweep": "max_drift={0.1,0.25,0.5}; length={32,64,128}",
        "expected": "tests theory/family mismatch; failure is acceptable if explained",
    },
    {
        "fault_mode": "dropout",
        "definition": "reports are absent or encoded as a sentinel/last value",
        "sweep": "rate={0.05,0.10,0.20}; length={4,16,64}",
        "expected": "count-aware PM aggregation should beat methods ignoring missingness",
    },
    {
        "fault_mode": "saturation",
        "definition": "values clip to rails or a small set of saturation levels",
        "sweep": "rails={2,3,5}; rails chosen from {-1,+1} or quantiles",
        "expected": "two-state stuck model may underfit multi-rail failures",
    },
]

WORKPLAN = [
    ("manifest and loaders", 0, 1.0, "all datasets"),
    ("fault taxonomy injection", 1.0, 1.0, "six modes"),
    ("baseline suite", 2.0, 1.5, "PM-no-clean, robust, mixture, HMM"),
    ("model selection rule", 3.5, 1.0, "privatized validation likelihood"),
    ("budget estimator", 4.5, 1.5, "support + likelihood + bootstrap"),
    ("expanded grid run", 6.0, 2.0, "must-run datasets"),
    ("figures and tables", 8.0, 1.0, "generated from CSV"),
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_manifest() -> pd.DataFrame:
    write_csv(RESULTS / "dataset_manifest_plan.csv", DATASETS)
    return pd.DataFrame(DATASETS)


def save_faults() -> pd.DataFrame:
    write_csv(RESULTS / "fault_taxonomy_plan.csv", FAULTS)
    return pd.DataFrame(FAULTS)


def save_workplan() -> pd.DataFrame:
    rows = [
        {
            "work_package": name,
            "start_day": start,
            "duration_days": duration,
            "scope": scope,
        }
        for name, start, duration, scope in WORKPLAN
    ]
    write_csv(RESULTS / "experiment_workplan.csv", rows)
    return pd.DataFrame(rows)


def expected_trend_rows() -> pd.DataFrame:
    eps = np.array([0.25, 0.5, 1.0, 2.0, 4.0])
    rows = []
    for fault in ["iid_stuck", "segment_stuck", "bias_drift"]:
        for method in ["PM-NoClean", "PM-Robust", "PM-Mixture", "PM-HMM", "PM-Selected"]:
            if fault == "iid_stuck":
                base = {"PM-NoClean": 0.35, "PM-Robust": 0.48, "PM-Mixture": 0.68, "PM-HMM": 0.55, "PM-Selected": 0.68}[method]
            elif fault == "segment_stuck":
                base = {"PM-NoClean": 0.35, "PM-Robust": 0.52, "PM-Mixture": 0.62, "PM-HMM": 0.78, "PM-Selected": 0.78}[method]
            else:
                base = {"PM-NoClean": 0.30, "PM-Robust": 0.45, "PM-Mixture": 0.48, "PM-HMM": 0.52, "PM-Selected": 0.54}[method]
            for e in eps:
                score = min(0.96, base + 0.09 * math.log2(float(e) + 1.0))
                rows.append(
                    {
                        "status": "expected_not_measured",
                        "fault_mode": fault,
                        "method": method,
                        "epsilon": float(e),
                        "expected_relative_score": round(score, 4),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "expected_metric_trends.csv", index=False)
    return df


def model_selection_rows() -> pd.DataFrame:
    rows = []
    for eps in [0.5, 1.0, 2.0, 4.0]:
        for length in [1, 2, 4, 8, 16, 32, 64]:
            delta = -0.15 + 0.18 * math.log2(length) + 0.04 * math.log2(eps + 1)
            rows.append(
                {
                    "status": "expected_not_measured",
                    "epsilon": eps,
                    "segment_length": length,
                    "delta_score_mix_minus_hmm": round(delta, 4),
                    "expected_selection": "HMM" if delta > 0.05 else "mixture",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "model_selection_expectation.csv", index=False)
    return df


def budget_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    profiles = []
    for dataset_id in ["air_quality", "household_power", "beijing_air", "nab"]:
        quality = {"air_quality": 0.16, "household_power": 0.12, "beijing_air": 0.10, "nab": 0.22}[dataset_id]
        for e in [0.5, 1.0, 2.0, 4.0]:
            center = e * (1.0 + quality * (0.8 - min(e, 3.0) / 3.0))
            width = max(0.15, quality * e * 1.6)
            rows.append(
                {
                    "status": "expected_not_measured",
                    "dataset_id": dataset_id,
                    "epsilon_true": e,
                    "epsilon_hat": round(center, 3),
                    "ci_low": round(max(0.1, center - width), 3),
                    "ci_high": round(center + width, 3),
                    "stable_flag": width / e < 0.35,
                }
            )
            for cand in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]:
                nll = (math.log(cand + 0.2) - math.log(e + 0.2)) ** 2 + quality
                profiles.append(
                    {
                        "status": "expected_not_measured",
                        "dataset_id": dataset_id,
                        "epsilon_true": e,
                        "epsilon_candidate": cand,
                        "relative_validation_nll": round(nll, 4),
                    }
                )
    est = pd.DataFrame(rows)
    prof = pd.DataFrame(profiles)
    est.to_csv(RESULTS / "budget_estimation_plan.csv", index=False)
    prof.to_csv(RESULTS / "budget_profile_plan.csv", index=False)
    return est, prof


def plot_dataset_heatmap() -> None:
    data = np.array([GAP_SCORES[d["dataset_id"]] for d in DATASETS], dtype=float)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=2)
    ax.set_xticks(range(len(GAPS)), GAPS, rotation=35, ha="right")
    ax.set_yticks(range(len(DATASETS)), [d["dataset_id"] for d in DATASETS])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, int(data[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Planned Dataset Coverage for Review Gaps")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([0, 1, 2])
    cbar.set_ticklabels(["none", "partial", "strong"])
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_dataset_gap_coverage.png", dpi=220)
    plt.close(fig)


def plot_dataset_scale(manifest: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ordered = manifest.sort_values("rows_or_streams")
    colors = ["#2a9d8f" if s == "run" else "#457b9d" if m == "yes" else "#8d99ae" for s, m in zip(ordered["local_status"], ordered["must_run"])]
    ax.barh(ordered["dataset_id"], ordered["rows_or_streams"], color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("rows or scalar stream records (log scale)")
    ax.set_title("Downloaded and Planned Dataset Scale")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_dataset_scale.png", dpi=220)
    plt.close(fig)


def plot_fault_taxonomy() -> None:
    x = np.arange(160)
    clean = 0.42 * np.sin(2 * np.pi * x / 48) + 0.18 * np.sin(2 * np.pi * x / 13)
    traces = {
        "clean": clean,
        "iid stuck": clean.copy(),
        "segment stuck": clean.copy(),
        "bias": clean.copy(),
        "drift": clean.copy(),
        "dropout": clean.copy(),
        "saturation": clean.copy(),
    }
    traces["iid stuck"][np.arange(20, 150, 17)] = 0.82
    traces["segment stuck"][55:92] = 0.82
    traces["bias"][45:112] = np.clip(traces["bias"][45:112] + 0.35, -1, 1)
    traces["drift"][40:125] = np.clip(traces["drift"][40:125] + np.linspace(0, 0.55, 85), -1, 1)
    traces["dropout"][68:102] = np.nan
    traces["saturation"] = np.clip(traces["saturation"], -0.45, 0.55)
    fig, axes = plt.subplots(4, 2, figsize=(9.0, 7.0), sharex=True, sharey=True)
    for ax, (name, y) in zip(axes.ravel(), traces.items()):
        ax.plot(x, clean, color="#b7b7b7", linewidth=1.0, label="clean")
        ax.plot(x, y, color="#1d3557", linewidth=1.4, label="faulted")
        ax.set_title(name)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.2)
    axes.ravel()[-1].axis("off")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Fault Taxonomy Examples for PM-LDP Scalar Streams")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_fault_taxonomy_examples.png", dpi=220)
    plt.close(fig)


def plot_expected_trends(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, fault in zip(axes, ["iid_stuck", "segment_stuck", "bias_drift"]):
        part = df[df["fault_mode"] == fault]
        for method, group in part.groupby("method"):
            ax.plot(group["epsilon"], group["expected_relative_score"], marker="o", linewidth=1.4, label=method)
        ax.set_title(fault)
        ax.set_xlabel("epsilon per report")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("expected relative score")
    axes[-1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Expected Metric Trends for Planned Extensions")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_expected_metric_trends.png", dpi=220)
    plt.close(fig)


def plot_model_selection(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for eps, group in df.groupby("epsilon"):
        ax.plot(group["segment_length"], group["delta_score_mix_minus_hmm"], marker="o", label=f"eps={eps}")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axhline(0.05, color="#9d0208", linewidth=1.0, linestyle="--", label="HMM threshold")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("fault segment length (1 = iid-like)")
    ax.set_ylabel("expected score(mixture) - score(HMM)")
    ax.set_title("Expected HMM-vs-Mixture Selection Boundary")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_model_selection_expectation.png", dpi=220)
    plt.close(fig)


def plot_budget(est: pd.DataFrame, prof: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for dataset_id, group in est.groupby("dataset_id"):
        axes[0].errorbar(
            group["epsilon_true"],
            group["epsilon_hat"],
            yerr=[group["epsilon_hat"] - group["ci_low"], group["ci_high"] - group["epsilon_hat"]],
            marker="o",
            capsize=3,
            label=dataset_id,
        )
    axes[0].plot([0.4, 4.2], [0.4, 4.2], color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("true epsilon")
    axes[0].set_ylabel("estimated epsilon")
    axes[0].set_title("Blind Budget Estimation Plan")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)

    part = prof[(prof["dataset_id"].isin(["household_power", "nab"])) & (prof["epsilon_true"].isin([1.0, 4.0]))]
    for (dataset_id, eps), group in part.groupby(["dataset_id", "epsilon_true"]):
        axes[1].plot(group["epsilon_candidate"], group["relative_validation_nll"], marker="o", label=f"{dataset_id}, eps={eps}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("candidate epsilon")
    axes[1].set_ylabel("relative validation NLL")
    axes[1].set_title("Expected NLL Profiles")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_budget_estimation_plan.png", dpi=220)
    plt.close(fig)


def plot_workplan(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    y = np.arange(len(df))
    ax.barh(y, df["duration_days"], left=df["start_day"], color="#52796f")
    ax.set_yticks(y, df["work_package"])
    ax.invert_yaxis()
    ax.set_xlabel("estimated work day")
    ax.set_title("Experiment Expansion Work Packages")
    for i, row in df.iterrows():
        ax.text(row["start_day"] + row["duration_days"] + 0.08, i, row["scope"], va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_experiment_workplan.png", dpi=220)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    manifest = save_manifest()
    save_faults()
    workplan = save_workplan()
    trend = expected_trend_rows()
    selection = model_selection_rows()
    budget_est, budget_prof = budget_rows()

    plot_dataset_heatmap()
    plot_dataset_scale(manifest)
    plot_fault_taxonomy()
    plot_expected_trends(trend)
    plot_model_selection(selection)
    plot_budget(budget_est, budget_prof)
    plot_workplan(workplan)

    print("Wrote experiment plan CSVs and figures to results/")


if __name__ == "__main__":
    main()
