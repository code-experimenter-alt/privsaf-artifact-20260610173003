from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

METHODS = ["random", "mixture", "hmm"]
METHOD_LABELS = {"random": "random", "mixture": "mixture", "hmm": "HMM"}
COLORS = {"random": "#8d99ae", "mixture": "#2a9d8f", "hmm": "#d95f02"}
STYLES = {0.03125: ("--", "o", "a=0.03125"), 0.84375: ("-", "s", "a=0.84375")}


def load_summary() -> pd.DataFrame:
    path = RESULTS / "air_quality_summary.csv"
    df = pd.read_csv(path)
    df["stuck_value"] = df["stuck_value"].astype(float)
    return df.sort_values(["mode", "stuck_value", "method", "epsilon"])


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_detection_sweep(df: pd.DataFrame) -> None:
    metrics = [("auroc_mean", "auroc_std", "AUROC"), ("auprc_mean", "auprc_std", "AUPRC")]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.4), sharex=True)
    for col, mode in enumerate(["iid", "segment"]):
        mode_df = df[df["mode"] == mode]
        for row, (mean_col, std_col, label) in enumerate(metrics):
            ax = axes[row, col]
            for method in METHODS:
                for stuck, (linestyle, marker, stuck_label) in STYLES.items():
                    part = mode_df[(mode_df["method"] == method) & (mode_df["stuck_value"] == stuck)]
                    if part.empty:
                        continue
                    x = part["epsilon"].to_numpy()
                    y = part[mean_col].to_numpy()
                    err = part[std_col].to_numpy()
                    ax.plot(
                        x,
                        y,
                        color=COLORS[method],
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=1.8,
                        markersize=4,
                        label=f"{METHOD_LABELS[method]}, {stuck_label}",
                    )
                    ax.fill_between(x, y - err, y + err, color=COLORS[method], alpha=0.08)
            ax.set_title(f"{mode} faults: {label}")
            ax.set_ylabel(label)
            ax.set_ylim(0.25 if label == "AUPRC" else 0.45, 1.02)
            ax.grid(True, axis="y", alpha=0.25)
            if row == 1:
                ax.set_xlabel(r"privacy budget $\varepsilon$")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Air Quality detection improves with privacy budget and model match")
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    fig.savefig(RESULTS / "fig_air_quality_detection_sweep.png", dpi=240)
    plt.close(fig)


def plot_hmm_minus_mixture(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.0), sharex=True)
    metrics = [("auroc_mean", "HMM - mixture AUROC"), ("auprc_mean", "HMM - mixture AUPRC")]
    for col, mode in enumerate(["iid", "segment"]):
        mode_df = df[df["mode"] == mode]
        for row, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            for stuck, (linestyle, marker, stuck_label) in STYLES.items():
                part = mode_df[mode_df["stuck_value"] == stuck]
                pivot = part.pivot_table(index="epsilon", columns="method", values=metric, aggfunc="mean")
                if {"hmm", "mixture"}.issubset(pivot.columns):
                    delta = pivot["hmm"] - pivot["mixture"]
                    ax.plot(
                        delta.index,
                        delta.to_numpy(),
                        color="#33415c" if stuck == 0.03125 else "#d95f02",
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=2.0,
                        label=stuck_label,
                    )
            ax.axhline(0.0, color="#333333", linewidth=0.9)
            ax.set_title(f"{mode} faults")
            ax.set_ylabel(label)
            ax.grid(True, axis="y", alpha=0.25)
            if row == 1:
                ax.set_xlabel(r"privacy budget $\varepsilon$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Positive deltas identify the persistent-fault regime where HMM is preferred")
    fig.tight_layout(rect=(0, 0.09, 1, 0.94))
    fig.savefig(RESULTS / "fig_air_quality_hmm_minus_mixture.png", dpi=240)
    plt.close(fig)


def plot_identification(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.2), sharex=True)
    metrics = [("r_error_mean", r"$|\hat r-r|$", False), ("bucket_acc_mean", "bucket accuracy", True)]
    for col, mode in enumerate(["iid", "segment"]):
        mode_df = df[df["mode"] == mode]
        for row, (metric, label, higher_is_better) in enumerate(metrics):
            ax = axes[row, col]
            for method in ["mixture", "hmm"]:
                for stuck, (linestyle, marker, stuck_label) in STYLES.items():
                    part = mode_df[(mode_df["method"] == method) & (mode_df["stuck_value"] == stuck)]
                    ax.plot(
                        part["epsilon"],
                        part[metric],
                        color=COLORS[method],
                        linestyle=linestyle,
                        marker=marker,
                        linewidth=1.9,
                        markersize=4,
                        label=f"{METHOD_LABELS[method]}, {stuck_label}",
                    )
            ax.set_title(f"{mode} faults")
            ax.set_ylabel(label)
            ax.grid(True, axis="y", alpha=0.25)
            ax.set_ylim(-0.02, 1.05 if higher_is_better else 0.22)
            if row == 1:
                ax.set_xlabel(r"privacy budget $\varepsilon$")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Air Quality identification metrics from released run summaries")
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(RESULTS / "fig_air_quality_identification.png", dpi=240)
    plt.close(fig)


def plot_runtime(df: pd.DataFrame) -> None:
    part = df[df["method"].isin(["mixture", "hmm"])].copy()
    agg = (
        part.groupby(["method", "epsilon"], as_index=False)["runtime_sec_mean"]
        .agg(["mean", "std"])
        .reset_index()
        .fillna(0.0)
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for method in ["mixture", "hmm"]:
        g = agg[agg["method"] == method]
        x = g["epsilon"].to_numpy()
        y = g["mean"].to_numpy()
        err = g["std"].to_numpy()
        ax.plot(x, y, color=COLORS[method], marker="o", linewidth=2.0, label=METHOD_LABELS[method])
        ax.fill_between(x, np.maximum(0, y - err), y + err, color=COLORS[method], alpha=0.12)
    ax.set_xlabel(r"privacy budget $\varepsilon$")
    ax.set_ylabel("runtime per run (seconds)")
    ax.set_title("Runtime remains sub-second to near-second for the Air Quality test stream")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_air_quality_runtime.png", dpi=240)
    plt.close(fig)


def main() -> None:
    setup_style()
    df = load_summary()
    plot_detection_sweep(df)
    plot_hmm_minus_mixture(df)
    plot_identification(df)
    plot_runtime(df)


if __name__ == "__main__":
    main()
