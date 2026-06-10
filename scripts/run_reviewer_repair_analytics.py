from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_icde_revision_grid import (
    RESULTS,
    apply_masked_interpolation,
    discretize_output,
    histogram_alpha,
    hmm_infer,
    inject_fault,
    load_streams,
    mixture_infer,
    pm_matrix,
    pm_sample,
    posterior_clean_mean,
    robust_normalize,
    topk_mask,
)


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def block_mean_error(clean: np.ndarray, repaired: np.ndarray, block: int) -> float:
    n = min(len(clean), len(repaired))
    if n == 0:
        return float("nan")
    usable = (n // block) * block
    if usable == 0:
        return float(abs(np.mean(repaired[:n]) - np.mean(clean[:n])))
    clean_blocks = clean[:usable].reshape(-1, block).mean(axis=1)
    repair_blocks = repaired[:usable].reshape(-1, block).mean(axis=1)
    return float(np.mean(np.abs(repair_blocks - clean_blocks)))


def aggregate_metrics(clean: np.ndarray, repaired: np.ndarray, labels: np.ndarray, repaired_mask: np.ndarray, block: int) -> dict[str, float]:
    err = np.asarray(repaired, dtype=float) - np.asarray(clean, dtype=float)
    abs_err = np.abs(err)
    fault = labels.astype(bool)
    clean_points = ~fault
    threshold = float(np.quantile(clean, 0.90))
    clean_exceed = int(np.sum(clean > threshold))
    repaired_exceed = int(np.sum(repaired > threshold))
    return {
        "mae": float(np.mean(abs_err)),
        "rmse": float(math.sqrt(np.mean(err**2))),
        "fault_point_mae": float(np.mean(abs_err[fault])) if np.any(fault) else float("nan"),
        "clean_point_mae": float(np.mean(abs_err[clean_points])) if np.any(clean_points) else float("nan"),
        "false_clean_repair_fraction": float(np.mean(repaired_mask.astype(bool) & clean_points)) if len(clean_points) else float("nan"),
        "downstream_mean_abs_error": float(abs(np.mean(repaired) - np.mean(clean))),
        "downstream_hourly_mean_abs_error": block_mean_error(clean, repaired, block),
        "downstream_p95_abs_error": float(abs(np.quantile(repaired, 0.95) - np.quantile(clean, 0.95))),
        "downstream_exceedance_count_abs_error": float(abs(repaired_exceed - clean_exceed)),
        "downstream_exceedance_rate_abs_error": float(abs(repaired_exceed - clean_exceed) / max(len(clean), 1)),
        "exceedance_threshold_q90": threshold,
    }


def repair_rows_for_case(
    *,
    dataset: str,
    stream: str,
    clean: np.ndarray,
    train: np.ndarray,
    epsilon: float,
    seed: int,
    fault_mode: str,
    fault_rate: float,
    segment_length: int,
    d: int,
    out_d: int,
    block: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    stuck_value = float(np.quantile(train, 0.95))
    faulty, labels, _ = inject_fault(clean, fault_mode, fault_rate, stuck_value, segment_length, rng)
    alpha = histogram_alpha(train, d)
    m, out_edges = pm_matrix(epsilon, d, out_d)
    obs = discretize_output(pm_sample(faulty, epsilon, rng), out_edges)
    postmean = posterior_clean_mean(obs, m, alpha, np.linspace(-1.0 + 1.0 / d, 1.0 - 1.0 / d, d))
    if fault_mode == "iid_stuck":
        scores, _, _ = mixture_infer(obs, m, alpha)
    else:
        scores, _, _ = hmm_infer(obs, m, alpha, float(np.mean(labels)), segment_length)
    budget = int(labels.sum())
    privsaf_mask = topk_mask(scores, budget)
    oracle_mask = labels.astype(bool)
    repairs = [
        ("no_repair", "none", faulty, np.zeros(len(clean), dtype=bool)),
        ("ldp_smoothing", "all_points_ldp_postmean", postmean, np.ones(len(clean), dtype=bool)),
        ("privsaf_repair", "privsaf_topk_mask_linear", apply_masked_interpolation(faulty, privsaf_mask, "linear_interpolation", block), privsaf_mask),
        ("oracle_mask_interpolation", "oracle_fault_mask_linear", apply_masked_interpolation(faulty, oracle_mask, "linear_interpolation", block), oracle_mask),
    ]
    rows: list[dict[str, object]] = []
    for method, mask_source, repaired, mask in repairs:
        row: dict[str, object] = {
            "dataset": dataset,
            "stream": stream,
            "epsilon": float(epsilon),
            "seed": int(seed),
            "fault_mode": fault_mode,
            "fault_rate": float(np.mean(labels)),
            "segment_length": int(segment_length),
            "repair_method": method,
            "mask_source": mask_source,
            "review_budget_points": int(budget),
            "block_size_for_hourly_mean": int(block),
        }
        row.update(aggregate_metrics(clean, repaired, labels, mask, block))
        rows.append(row)
    return rows


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    return (
        runs.groupby(["repair_method", "mask_source", "fault_mode"], as_index=False)
        .agg(
            cases=("mae", "size"),
            mae_mean=("mae", "mean"),
            fault_point_mae_mean=("fault_point_mae", "mean"),
            clean_point_mae_mean=("clean_point_mae", "mean"),
            false_clean_repair_fraction_mean=("false_clean_repair_fraction", "mean"),
            downstream_mean_abs_error_mean=("downstream_mean_abs_error", "mean"),
            downstream_hourly_mean_abs_error_mean=("downstream_hourly_mean_abs_error", "mean"),
            downstream_p95_abs_error_mean=("downstream_p95_abs_error", "mean"),
            downstream_exceedance_count_abs_error_mean=("downstream_exceedance_count_abs_error", "mean"),
            downstream_exceedance_rate_abs_error_mean=("downstream_exceedance_rate_abs_error", "mean"),
        )
        .sort_values(["fault_mode", "repair_method"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reviewer-facing downstream repair analytics.")
    parser.add_argument("--output-dir", default=str(RESULTS))
    parser.add_argument("--epsilons", default="2.0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--fault-rate", type=float, default=0.20)
    parser.add_argument("--segment-length", type=int, default=48)
    parser.add_argument("--raw-buckets", type=int, default=32)
    parser.add_argument("--output-buckets", type=int, default=32)
    parser.add_argument("--test-len", type=int, default=1200)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for stream in load_streams(include_optional=args.include_optional):
        split = robust_normalize(stream.values, test_len=args.test_len)
        for epsilon in parse_floats(args.epsilons):
            for seed in parse_ints(args.seeds):
                for fault_mode in ["iid_stuck", "segment_stuck"]:
                    rows.extend(
                        repair_rows_for_case(
                            dataset=stream.dataset_id,
                            stream=stream.target,
                            clean=split.test,
                            train=split.train,
                            epsilon=epsilon,
                            seed=seed,
                            fault_mode=fault_mode,
                            fault_rate=args.fault_rate,
                            segment_length=args.segment_length,
                            d=args.raw_buckets,
                            out_d=args.output_buckets,
                            block=args.block_size,
                        )
                    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    summary = summarize(runs)
    runs.to_csv(out / "reviewer_repair_analytics_runs.csv", index=False)
    summary.to_csv(out / "reviewer_repair_analytics_summary.csv", index=False)
    print(f"Wrote {len(runs)} rows to {out / 'reviewer_repair_analytics_runs.csv'}")
    print(f"Wrote {len(summary)} rows to {out / 'reviewer_repair_analytics_summary.csv'}")


if __name__ == "__main__":
    main()
