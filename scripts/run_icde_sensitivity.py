from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_icde_revision_grid import (
    RESULTS,
    discretize_output,
    histogram_alpha,
    hmm_infer,
    inject_fault,
    load_air_quality,
    ldp_distribution_score,
    load_streams,
    pm_matrix,
    pm_sample,
    robust_normalize,
    safe_detection_metrics,
    window_glr_score,
)


def run_sensitivity() -> pd.DataFrame:
    stream = load_air_quality()
    split = robust_normalize(stream.values, test_len=1800)
    alpha = histogram_alpha(split.train, 32)
    stuck_value = float(np.quantile(split.train, 0.95))
    eps = 2.0
    m, out_edges = pm_matrix(eps, 32, 32)

    rows: list[dict[str, object]] = []
    for axis_name, values in {
        "segment_length": [12, 24, 48, 96],
        "fault_rate": [0.05, 0.10, 0.20, 0.30],
    }.items():
        for value in values:
            for seed in [0, 1, 2]:
                rate = float(value) if axis_name == "fault_rate" else 0.20
                seg_len = int(value) if axis_name == "segment_length" else 48
                rng = np.random.default_rng(seed)
                faulty, labels, _ = inject_fault(split.test, "segment_stuck", rate, stuck_value, seg_len, rng)
                privatized = pm_sample(faulty, eps, rng)
                obs = discretize_output(privatized, out_edges)

                method_specs = []
                t0 = time.perf_counter()
                scores, _, _ = window_glr_score(obs, m, alpha)
                method_specs.append(("ldp_window_glr", scores, time.perf_counter() - t0))

                t0 = time.perf_counter()
                scores, _, _ = hmm_infer(obs, m, alpha, rate, seg_len)
                method_specs.append(("privsaf_hmm", scores, time.perf_counter() - t0))

                t0 = time.perf_counter()
                scores = ldp_distribution_score(obs, m, alpha)
                method_specs.append(("ldp_distribution_surprise", scores, time.perf_counter() - t0))

                for method, scores, runtime in method_specs:
                    auroc, auprc = safe_detection_metrics(labels, scores)
                    rows.append(
                        {
                            "dataset_id": stream.dataset_id,
                            "axis": axis_name,
                            "axis_value": float(value),
                            "epsilon": eps,
                            "seed": seed,
                            "fault_rate": float(np.mean(labels)),
                            "segment_length": seg_len,
                            "method": method,
                            "auroc": auroc,
                            "auprc": auprc,
                            "runtime_sec": runtime,
                        }
                    )
    return pd.DataFrame(rows)


def plot_sensitivity(df: pd.DataFrame, out: Path) -> None:
    labels = {
        "ldp_distribution_surprise": "LDP surprise",
        "ldp_window_glr": "Window GLR",
        "privsaf_hmm": "PrivSAF-HMM",
    }
    colors = {
        "ldp_distribution_surprise": "#b279a2",
        "ldp_window_glr": "#7f7f7f",
        "privsaf_hmm": "#54a24b",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    for ax, axis_name, xlabel in [
        (axes[0], "segment_length", "segment length"),
        (axes[1], "fault_rate", "fault rate"),
    ]:
        part = df[df["axis"] == axis_name]
        summary = part.groupby(["axis_value", "method"], as_index=False).agg(auprc=("auprc", "mean"))
        for method in ["ldp_distribution_surprise", "ldp_window_glr", "privsaf_hmm"]:
            mpart = summary[summary["method"] == method]
            ax.plot(mpart["axis_value"], mpart["auprc"], marker="o", lw=1.7, label=labels[method], color=colors[method])
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("AUPRC")
    axes[0].set_title("Persistence Sensitivity")
    axes[1].set_title("Fault-Rate Sensitivity")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = run_sensitivity()
    df.to_csv(RESULTS / "icde_revision_sensitivity_runs.csv", index=False)
    summary = (
        df.groupby(["axis", "axis_value", "method"], as_index=False)
        .agg(auroc_mean=("auroc", "mean"), auprc_mean=("auprc", "mean"), runtime_sec_mean=("runtime_sec", "mean"))
        .sort_values(["axis", "axis_value", "method"])
    )
    summary.to_csv(RESULTS / "icde_revision_sensitivity_summary.csv", index=False)
    plot_sensitivity(df, RESULTS / "fig_icde_sensitivity.png")
    print(f"Wrote {RESULTS / 'icde_revision_sensitivity_runs.csv'}")
    print(f"Wrote {RESULTS / 'icde_revision_sensitivity_summary.csv'}")
    print(f"Wrote {RESULTS / 'fig_icde_sensitivity.png'}")


if __name__ == "__main__":
    main()
