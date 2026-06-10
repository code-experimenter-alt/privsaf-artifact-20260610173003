from __future__ import annotations

import sqlite3
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
    pm_matrix,
    pm_sample,
    raw_bucket_index,
    robust_normalize,
    safe_detection_metrics,
)


def _mean_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            cases=("auprc", "size"),
            auroc_mean=("auroc", "mean"),
            auprc_mean=("auprc", "mean"),
            bucket_hit_mean=("bucket_hit", "mean"),
            runtime_ms_per_frame=("runtime_ms_per_frame", "mean"),
        )
        .sort_values(group_cols)
    )


def run_inference_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    stream = load_air_quality()
    split = robust_normalize(stream.values, test_len=1800)
    seeds = [0, 1, 2]
    rows: list[dict[str, object]] = []

    eps = 2.0
    fault_rate = 0.20
    segment_length = 48
    stuck_value = float(np.quantile(split.train, 0.95))

    # Calibration contamination: the normal calibration histogram is mixed with
    # the true stuck bucket, making the normal emission closer to the fault column.
    d = 32
    m, out_edges = pm_matrix(eps, d, d)
    alpha_clean = histogram_alpha(split.train, d)
    true_bucket = raw_bucket_index(stuck_value, d)
    onehot = np.zeros(d, dtype=float)
    onehot[true_bucket] = 1.0
    for contamination in [0.0, 0.02, 0.05, 0.10, 0.20, 0.35]:
        alpha = (1.0 - contamination) * alpha_clean + contamination * onehot
        alpha = alpha / alpha.sum()
        for seed in seeds:
            rng = np.random.default_rng(3000 + seed)
            faulty, labels, _ = inject_fault(split.test, "segment_stuck", fault_rate, stuck_value, segment_length, rng)
            reports = pm_sample(faulty, eps, rng)
            obs = discretize_output(reports, out_edges)
            t0 = time.perf_counter()
            scores, cand, _ = hmm_infer(obs, m, alpha, fault_rate, segment_length)
            runtime = time.perf_counter() - t0
            auroc, auprc = safe_detection_metrics(labels, scores)
            rows.append(
                {
                    "panel": "calibration_contamination",
                    "x": contamination,
                    "x_label": "contamination",
                    "epsilon": eps,
                    "raw_buckets": d,
                    "devices": 1,
                    "trace_length": len(labels),
                    "seed": seed,
                    "auroc": auroc,
                    "auprc": auprc,
                    "bucket_hit": float(cand == true_bucket),
                    "runtime_ms_per_frame": 1000.0 * runtime / len(labels),
                }
            )

    # Bucket count: raw and output discretization use the same count.
    for buckets in [8, 16, 32, 64]:
        m_b, out_edges_b = pm_matrix(eps, buckets, buckets)
        alpha_b = histogram_alpha(split.train, buckets)
        true_bucket_b = raw_bucket_index(stuck_value, buckets)
        for seed in seeds:
            rng = np.random.default_rng(4000 + seed)
            faulty, labels, _ = inject_fault(split.test, "segment_stuck", fault_rate, stuck_value, segment_length, rng)
            reports = pm_sample(faulty, eps, rng)
            obs = discretize_output(reports, out_edges_b)
            t0 = time.perf_counter()
            scores, cand, _ = hmm_infer(obs, m_b, alpha_b, fault_rate, segment_length)
            runtime = time.perf_counter() - t0
            auroc, auprc = safe_detection_metrics(labels, scores)
            rows.append(
                {
                    "panel": "bucket_count",
                    "x": float(buckets),
                    "x_label": "buckets",
                    "epsilon": eps,
                    "raw_buckets": buckets,
                    "devices": 1,
                    "trace_length": len(labels),
                    "seed": seed,
                    "auroc": auroc,
                    "auprc": auprc,
                    "bucket_hit": float(cand == true_bucket_b),
                    "runtime_ms_per_frame": 1000.0 * runtime / len(labels),
                }
            )

    # Assumed epsilon: reports are generated at eps=2, while inference changes M.
    true_eps = 2.0
    m_true, out_edges_true = pm_matrix(true_eps, d, d)
    _ = m_true
    for assumed_eps in [0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
        m_assumed, _ = pm_matrix(assumed_eps, d, d)
        for seed in seeds:
            rng = np.random.default_rng(5000 + seed)
            faulty, labels, _ = inject_fault(split.test, "segment_stuck", fault_rate, stuck_value, segment_length, rng)
            reports = pm_sample(faulty, true_eps, rng)
            obs = discretize_output(reports, out_edges_true)
            t0 = time.perf_counter()
            scores, cand, _ = hmm_infer(obs, m_assumed, alpha_clean, fault_rate, segment_length)
            runtime = time.perf_counter() - t0
            auroc, auprc = safe_detection_metrics(labels, scores)
            rows.append(
                {
                    "panel": "epsilon_mismatch",
                    "x": assumed_eps,
                    "x_label": "assumed_epsilon",
                    "epsilon": true_eps,
                    "raw_buckets": d,
                    "devices": 1,
                    "trace_length": len(labels),
                    "seed": seed,
                    "auroc": auroc,
                    "auprc": auprc,
                    "bucket_hit": float(cand == true_bucket),
                    "runtime_ms_per_frame": 1000.0 * runtime / len(labels),
                }
            )

    # Multi-device short trajectories: independent devices contribute short
    # privatized traces; metrics are computed after concatenating the private scores.
    full = np.concatenate([split.validation, split.test])
    m, out_edges = pm_matrix(eps, d, d)
    for trace_len in [64, 128, 256]:
        seg_len = max(8, min(segment_length, trace_len // 4))
        for devices in [4, 16, 32]:
            for seed in seeds:
                rng = np.random.default_rng(6000 + seed + trace_len + devices)
                all_scores: list[np.ndarray] = []
                all_labels: list[np.ndarray] = []
                t0 = time.perf_counter()
                for _device in range(devices):
                    start = int(rng.integers(0, max(1, len(full) - trace_len)))
                    raw = np.array(full[start : start + trace_len], dtype=float)
                    if len(raw) < trace_len:
                        raw = np.resize(raw, trace_len)
                    faulty, labels, _ = inject_fault(raw, "segment_stuck", fault_rate, stuck_value, seg_len, rng)
                    reports = pm_sample(faulty, eps, rng)
                    obs = discretize_output(reports, out_edges)
                    scores, _, _ = hmm_infer(obs, m, alpha_clean, fault_rate, seg_len)
                    all_scores.append(scores)
                    all_labels.append(labels)
                runtime = time.perf_counter() - t0
                labels_all = np.concatenate(all_labels)
                scores_all = np.concatenate(all_scores)
                auroc, auprc = safe_detection_metrics(labels_all, scores_all)
                rows.append(
                    {
                        "panel": "multi_device_short_traces",
                        "x": float(trace_len),
                        "x_label": "trace_length",
                        "epsilon": eps,
                        "raw_buckets": d,
                        "devices": devices,
                        "trace_length": trace_len,
                        "seed": seed,
                        "auroc": auroc,
                        "auprc": auprc,
                        "bucket_hit": np.nan,
                        "runtime_ms_per_frame": 1000.0 * runtime / len(labels_all),
                    }
                )

    runs = pd.DataFrame(rows)
    summary = _mean_summary(runs, ["panel", "x", "x_label", "devices"])
    return runs, summary


def run_system_benchmarks() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7000)
    rows: list[dict[str, object]] = []

    for devices in [100, 500, 1000, 2500]:
        reports_per_device = 64
        buckets = 32
        reports = rng.integers(0, buckets, size=(devices, reports_per_device), dtype=np.int16)
        counts = np.zeros((devices, buckets), dtype=np.int32)
        t0 = time.perf_counter()
        for device in range(devices):
            for bucket in reports[device]:
                counts[device, int(bucket)] += 1
        elapsed = time.perf_counter() - t0
        total = devices * reports_per_device
        rows.append(
            {
                "panel": "stream_operator",
                "scale": devices,
                "scale_label": "devices",
                "rows": total,
                "elapsed_sec": elapsed,
                "throughput_rows_per_sec": total / max(elapsed, 1e-12),
                "memory_mb": counts.nbytes / 1_000_000.0,
            }
        )

    for total_rows in [50_000, 100_000, 200_000, 400_000]:
        devices = max(100, total_rows // 64)
        device_id = rng.integers(0, devices, size=total_rows, dtype=np.int32)
        ts = np.arange(total_rows, dtype=np.int32)
        eps = rng.choice([1, 2, 4], size=total_rows).astype(np.int16)
        bucket = rng.integers(0, 32, size=total_rows, dtype=np.int16)
        records = list(zip(device_id.tolist(), ts.tolist(), eps.tolist(), bucket.tolist()))
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        t0 = time.perf_counter()
        cur.execute("create table reports(device_id integer, ts integer, epsilon integer, report_bucket integer)")
        cur.executemany("insert into reports values (?, ?, ?, ?)", records)
        cur.execute("create index idx_reports_channel on reports(epsilon, report_bucket)")
        cur.execute(
            "create table mv_counts as "
            "select epsilon, report_bucket, count(*) as n "
            "from reports group by epsilon, report_bucket"
        )
        cur.execute("select sum(n) from mv_counts")
        cur.fetchone()
        conn.commit()
        elapsed = time.perf_counter() - t0
        conn.close()
        rows.append(
            {
                "panel": "materialized_view",
                "scale": total_rows,
                "scale_label": "rows",
                "rows": total_rows,
                "elapsed_sec": elapsed,
                "throughput_rows_per_sec": total_rows / max(elapsed, 1e-12),
                "memory_mb": np.nan,
            }
        )

    for devices in [100, 1000, 5000, 10000]:
        reports_per_device = 32
        total = devices * reports_per_device
        device_id = np.repeat(np.arange(devices, dtype=np.int32), reports_per_device)
        bucket = rng.integers(0, 32, size=total, dtype=np.int16)
        counts = np.zeros((devices, 32), dtype=np.int16)
        t0 = time.perf_counter()
        np.add.at(counts, (device_id, bucket), 1)
        active = (counts.sum(axis=1) > 0).sum()
        elapsed = time.perf_counter() - t0
        rows.append(
            {
                "panel": "multi_device_groupby",
                "scale": devices,
                "scale_label": "devices",
                "rows": total,
                "elapsed_sec": elapsed,
                "throughput_rows_per_sec": total / max(elapsed, 1e-12),
                "memory_mb": counts.nbytes / 1_000_000.0,
                "active_groups": int(active),
            }
        )

    runs = pd.DataFrame(rows)
    summary = runs.sort_values(["panel", "scale"]).copy()
    return runs, summary


def plot_sensitivity(summary: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharey=False)
    panels = [
        ("calibration_contamination", "Calibration", "contamination"),
        ("bucket_count", "Buckets", "raw/output buckets"),
        ("epsilon_mismatch", "Epsilon", "assumed epsilon"),
        ("multi_device_short_traces", "Short Traces", "trace length"),
    ]
    for ax, (panel, title, xlabel) in zip(axes.ravel(), panels):
        part = summary[summary["panel"] == panel]
        if panel == "multi_device_short_traces":
            for devices, group in part.groupby("devices"):
                ax.plot(group["x"], group["auprc_mean"], marker="o", lw=1.5, label=f"{int(devices)} devices")
            ax.legend(frameon=False, fontsize=7)
        else:
            ax.plot(part["x"], part["auprc_mean"], marker="o", lw=1.7, color="#3b6ea8")
            if panel == "epsilon_mismatch":
                ax.axvline(2.0, color="#7f7f7f", ls="--", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("AUPRC")
        ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def plot_systems(summary: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    panels = [
        ("stream_operator", "Stream", "devices"),
        ("materialized_view", "View", "rows"),
        ("multi_device_groupby", "Groupby", "devices"),
    ]
    colors = ["#54a24b", "#4c78a8", "#f58518"]
    for ax, (panel, title, xlabel), color in zip(axes, panels, colors):
        part = summary[summary["panel"] == panel]
        ax.plot(part["scale"], part["throughput_rows_per_sec"] / 1000.0, marker="o", lw=1.7, color=color)
        if panel == "materialized_view":
            ticks = part["scale"].iloc[[0, 2, 3]].to_list()
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{int(x / 1000)}k" for x in ticks], fontsize=7)
        else:
            ax.set_xscale("log")
            ax.set_xticks(part["scale"])
            ax.set_xticklabels([str(int(x)) for x in part["scale"]], fontsize=7, rotation=30)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("k rows/s")
        ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(out, dpi=240)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    sens_runs, sens_summary = run_inference_sensitivity()
    system_runs, system_summary = run_system_benchmarks()
    sens_runs.to_csv(RESULTS / "private_stuck_cleaning_sensitivity_runs.csv", index=False)
    sens_summary.to_csv(RESULTS / "private_stuck_cleaning_sensitivity_summary.csv", index=False)
    system_runs.to_csv(RESULTS / "private_stuck_cleaning_system_runs.csv", index=False)
    system_summary.to_csv(RESULTS / "private_stuck_cleaning_system_summary.csv", index=False)
    plot_sensitivity(sens_summary, RESULTS / "fig_private_stuck_sensitivity.png")
    plot_systems(system_summary, RESULTS / "fig_private_stuck_systems.png")
    print(f"Wrote {len(sens_runs)} sensitivity rows and {len(system_runs)} system rows.")


if __name__ == "__main__":
    main()
