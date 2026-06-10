from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from run_icde_revision_grid import (
    RESULTS,
    centers,
    discretize_output,
    histogram_alpha,
    hmm_infer,
    inject_fault,
    ldp_distribution_score,
    load_streams,
    mixture_infer,
    pm_matrix,
    pm_sample,
    raw_bucket_index,
    robust_normalize,
    safe_detection_metrics,
    window_glr_score,
)


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def safe_kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p.astype(float), 1e-12, None)
    q = np.clip(q.astype(float), 1e-12, None)
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * (np.log(p) - np.log(q))))


def separation_diagnostics(m: np.ndarray, alpha: np.ndarray, stuck_bucket: int) -> tuple[float, float]:
    normal = np.clip(m @ alpha, 1e-12, None)
    normal /= normal.sum()
    stuck = np.clip(m[:, stuck_bucket], 1e-12, None)
    stuck /= stuck.sum()
    tv = 0.5 * float(np.sum(np.abs(stuck - normal)))
    return tv, safe_kl(stuck, normal)


def central_stuck_values(train: np.ndarray, m: np.ndarray, d: int) -> list[tuple[str, float, int]]:
    alpha = histogram_alpha(train, d)
    raw_centers = centers(d)
    normal = np.clip(m @ alpha, 1e-12, None)
    normal /= normal.sum()
    tv_by_col = np.array([0.5 * np.sum(np.abs(m[:, idx] - normal)) for idx in range(d)])
    specs = [
        ("median", float(np.nanmedian(train)), raw_bucket_index(float(np.nanmedian(train)), d)),
        ("mode_bucket", float(raw_centers[int(np.argmax(alpha))]), int(np.argmax(alpha))),
        ("min_tv_column", float(raw_centers[int(np.argmin(tv_by_col))]), int(np.argmin(tv_by_col))),
    ]
    return [(name, float(np.clip(value, -1.0, 1.0)), int(bucket)) for name, value, bucket in specs]


def method_scores(
    method: str,
    obs: np.ndarray,
    m: np.ndarray,
    alpha: np.ndarray,
    expected_rate: float,
    segment_length: int,
) -> tuple[np.ndarray, int | None, float | None]:
    if method == "privsaf_hmm":
        scores, cand, rhat = hmm_infer(obs, m, alpha, expected_rate, segment_length)
        return scores, cand, rhat
    if method == "privsaf_mixture":
        scores, cand, rhat = mixture_infer(obs, m, alpha)
        return scores, cand, rhat
    if method == "ldp_window_glr":
        scores, cand, rhat = window_glr_score(obs, m, alpha, windows=(8, 16, 32, 64))
        return scores, cand, rhat
    if method == "ldp_distribution_surprise":
        scores = ldp_distribution_score(obs, m, alpha)
        return scores, None, None
    raise ValueError(f"Unknown method: {method}")


def append_rows(
    rows: list[dict[str, object]],
    *,
    dataset: str,
    stream: str,
    clean: np.ndarray,
    train_alpha: np.ndarray,
    infer_alpha: np.ndarray,
    calibration_condition: str,
    stuck_value_type: str,
    stuck_value: float,
    true_bucket: int,
    epsilon: float,
    seed: int,
    fault_rate: float,
    segment_length: int,
    d: int,
    out_d: int,
    methods: Iterable[str],
) -> None:
    rng = np.random.default_rng(seed)
    m, out_edges = pm_matrix(epsilon, d, out_d)
    faulty, labels, _ = inject_fault(clean, "segment_stuck", fault_rate, stuck_value, segment_length, rng)
    reports = pm_sample(faulty, epsilon, rng)
    obs = discretize_output(reports, out_edges)
    tv, kl = separation_diagnostics(m, infer_alpha, true_bucket)
    actual_rate = float(np.mean(labels))
    fault_mode = "segment_stuck" if calibration_condition == "matched_train" else "segment_stuck_distribution_shift"
    for method in methods:
        scores, cand, rhat = method_scores(method, obs, m, infer_alpha, actual_rate, segment_length)
        auroc, auprc = safe_detection_metrics(labels, scores)
        rows.append(
            {
                "dataset": dataset,
                "stream": stream,
                "fault_mode": fault_mode,
                "stuck_value_type": stuck_value_type,
                "epsilon": float(epsilon),
                "seed": int(seed),
                "method": method,
                "AUROC": auroc,
                "AUPRC": auprc,
                "TV_sep": tv,
                "KL_sep": kl,
                "ratio_MAE": float("nan") if rhat is None else abs(float(rhat) - actual_rate),
                "bucket_hit1": float("nan") if cand is None else float(int(cand) == int(true_bucket)),
                "calibration_condition": calibration_condition,
                "fault_rate": actual_rate,
                "target_fault_rate": float(fault_rate),
                "segment_length": int(segment_length),
                "stuck_value": float(stuck_value),
                "true_bucket": int(true_bucket),
                "calibration_train_TV_sep": separation_diagnostics(m, train_alpha, true_bucket)[0],
            }
        )


def summarize_runs(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        runs.groupby(
            [
                "dataset",
                "stream",
                "fault_mode",
                "stuck_value_type",
                "epsilon",
                "method",
                "calibration_condition",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(
            rows=("AUPRC", "size"),
            AUROC_mean=("AUROC", "mean"),
            AUROC_std=("AUROC", "std"),
            AUPRC_mean=("AUPRC", "mean"),
            AUPRC_std=("AUPRC", "std"),
            TV_sep_mean=("TV_sep", "mean"),
            KL_sep_mean=("KL_sep", "mean"),
            ratio_MAE_mean=("ratio_MAE", "mean"),
            bucket_hit1_mean=("bucket_hit1", "mean"),
            fault_rate_mean=("fault_rate", "mean"),
            segment_length_min=("segment_length", "min"),
            segment_length_max=("segment_length", "max"),
        )
        .sort_values(["fault_mode", "stuck_value_type", "epsilon", "dataset", "method"])
    )
    corr_rows: list[dict[str, object]] = []
    for method, group in runs.groupby("method"):
        valid = group[["AUPRC", "TV_sep", "KL_sep"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) >= 3:
            corr_rows.append(
                {
                    "method": method,
                    "rows": int(len(valid)),
                    "corr_AUPRC_TV_sep": float(valid["AUPRC"].corr(valid["TV_sep"])),
                    "corr_AUPRC_KL_sep": float(valid["AUPRC"].corr(valid["KL_sep"])),
                }
            )
        else:
            corr_rows.append(
                {
                    "method": method,
                    "rows": int(len(valid)),
                    "corr_AUPRC_TV_sep": float("nan"),
                    "corr_AUPRC_KL_sep": float("nan"),
                }
            )
    return summary, pd.DataFrame(corr_rows).sort_values("method")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reviewer-requested central stuck and distribution-shift stress tests.")
    parser.add_argument("--epsilons", default="0.5,1.0,2.0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--fault-rates", default="0.05,0.10")
    parser.add_argument("--segment-lengths", default="12,24")
    parser.add_argument("--raw-buckets", type=int, default=32)
    parser.add_argument("--output-buckets", type=int, default=32)
    parser.add_argument("--test-len", type=int, default=1200)
    parser.add_argument("--methods", default="privsaf_hmm,privsaf_mixture,ldp_window_glr,ldp_distribution_surprise")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--output-dir", default=str(RESULTS))
    args = parser.parse_args()

    epsilons = parse_floats(args.epsilons)
    seeds = parse_ints(args.seeds)
    fault_rates = parse_floats(args.fault_rates)
    segment_lengths = parse_ints(args.segment_lengths)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    d = args.raw_buckets
    out_d = args.output_buckets

    output_dir = RESULTS if args.output_dir == str(RESULTS) else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for stream in load_streams(include_optional=args.include_optional):
        split = robust_normalize(stream.values, test_len=args.test_len)
        train_alpha = histogram_alpha(split.train, d)
        shifted_alpha = histogram_alpha(split.validation, d)
        for epsilon in epsilons:
            m, _ = pm_matrix(epsilon, d, out_d)
            stuck_specs = central_stuck_values(split.train, m, d)
            for stuck_value_type, stuck_value, true_bucket in stuck_specs:
                for fault_rate in fault_rates:
                    for segment_length in segment_lengths:
                        for seed in seeds:
                            append_rows(
                                rows,
                                dataset=stream.dataset_id,
                                stream=stream.target,
                                clean=split.test,
                                train_alpha=train_alpha,
                                infer_alpha=train_alpha,
                                calibration_condition="matched_train",
                                stuck_value_type=stuck_value_type,
                                stuck_value=stuck_value,
                                true_bucket=true_bucket,
                                epsilon=epsilon,
                                seed=seed,
                                fault_rate=fault_rate,
                                segment_length=segment_length,
                                d=d,
                                out_d=out_d,
                                methods=methods,
                            )
                            append_rows(
                                rows,
                                dataset=stream.dataset_id,
                                stream=stream.target,
                                clean=split.test,
                                train_alpha=train_alpha,
                                infer_alpha=shifted_alpha,
                                calibration_condition="shifted_validation",
                                stuck_value_type=stuck_value_type,
                                stuck_value=stuck_value,
                                true_bucket=true_bucket,
                                epsilon=epsilon,
                                seed=seed,
                                fault_rate=fault_rate,
                                segment_length=segment_length,
                                d=d,
                                out_d=out_d,
                                methods=methods,
                            )

    runs = pd.DataFrame(rows)
    required = [
        "dataset",
        "stream",
        "fault_mode",
        "stuck_value_type",
        "epsilon",
        "seed",
        "method",
        "AUROC",
        "AUPRC",
        "TV_sep",
        "KL_sep",
        "ratio_MAE",
        "bucket_hit1",
    ]
    runs = runs[required + [col for col in runs.columns if col not in required]]
    summary, corr = summarize_runs(runs)
    runs.to_csv(output_dir / "reviewer_stress_runs.csv", index=False)
    summary.to_csv(output_dir / "reviewer_stress_summary.csv", index=False)
    corr.to_csv(output_dir / "reviewer_stress_separation_correlation.csv", index=False)
    print(f"Wrote {len(runs)} stress rows to {output_dir / 'reviewer_stress_runs.csv'}")
    print(f"Wrote {len(summary)} summary rows to {output_dir / 'reviewer_stress_summary.csv'}")
    print(f"Wrote {len(corr)} separation-correlation rows to {output_dir / 'reviewer_stress_separation_correlation.csv'}")


if __name__ == "__main__":
    main()
