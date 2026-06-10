from __future__ import annotations

import argparse
import math
import time

import numpy as np
import pandas as pd

from run_icde_revision_grid import (
    RESULTS,
    discretize_output,
    histogram_alpha,
    hmm_infer,
    iid_mask,
    inject_fault,
    load_air_quality,
    mixture_infer,
    pm_matrix,
    pm_sample,
    posterior_clean_mean,
    robust_normalize,
    safe_detection_metrics,
    window_glr_score,
)


def duchi_binary_matrix(eps: float, raw_buckets: int) -> np.ndarray:
    raw_centers = np.linspace(-1.0 + 1.0 / raw_buckets, 1.0 - 1.0 / raw_buckets, raw_buckets)
    slope = (math.exp(eps) - 1.0) / (2.0 * (math.exp(eps) + 1.0))
    p_plus = np.clip(0.5 + raw_centers * slope, 1e-12, 1.0 - 1e-12)
    return np.vstack([1.0 - p_plus, p_plus])


def sample_duchi_binary(values: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    slope = (math.exp(eps) - 1.0) / (2.0 * (math.exp(eps) + 1.0))
    p_plus = np.clip(0.5 + values * slope, 1e-12, 1.0 - 1e-12)
    return (rng.random(len(values)) < p_plus).astype(int)


def channel_spike_score(obs: np.ndarray, m: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, int, float]:
    raw_buckets = m.shape[1]
    raw_centers = np.linspace(-1.0 + 1.0 / raw_buckets, 1.0 - 1.0 / raw_buckets, raw_buckets)
    post = posterior_clean_mean(obs, m, alpha, raw_centers)
    local = pd.Series(post).rolling(window=9, center=True, min_periods=1).median().to_numpy(dtype=float)
    residual = np.abs(post - local)
    scale = max(1.4826 * float(np.median(np.abs(residual - np.median(residual)))), 1e-6)
    return residual / scale, 0, float(np.mean(residual))


def observations(
    values: np.ndarray,
    mechanism: str,
    eps: float,
    raw_buckets: int,
    output_buckets: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, str]:
    if mechanism == "pm":
        m, out_edges = pm_matrix(eps, raw_buckets, output_buckets)
        obs = discretize_output(pm_sample(values, eps, rng), out_edges)
        return obs, m, "Piecewise Mechanism"
    if mechanism == "duchi_binary":
        m = duchi_binary_matrix(eps, raw_buckets)
        obs = sample_duchi_binary(values, eps, rng)
        return obs, m, "Duchi binary real-valued mechanism"
    raise ValueError(f"Unknown mechanism: {mechanism}")


def inject_scale_fault(
    clean: np.ndarray,
    rate: float,
    seg_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    labels = np.zeros(len(clean), dtype=int)
    target = max(1, int(round(rate * len(clean))))
    # Reuse the segment generator through a neutral stuck call, then replace values with scale corruption.
    _, labels, _ = inject_fault(clean, "segment_stuck", rate, float(np.quantile(clean, 0.95)), seg_len, rng)
    if int(labels.sum()) > target:
        ones = np.flatnonzero(labels)
        labels[ones[target:]] = 0
    faulty = clean.copy()
    center = float(np.median(clean))
    faulty[labels == 1] = np.clip(center + 1.7 * (faulty[labels == 1] - center), -1.0, 1.0)
    return faulty, labels, {"scale_factor": 1.7, "segment_length": int(seg_len)}


def inject_spike_fault(
    clean: np.ndarray,
    rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    labels = np.zeros(len(clean), dtype=int)
    target = max(1, int(round(rate * len(clean))))
    labels = iid_mask(len(clean), target, rng)
    faulty = clean.copy()
    signs = rng.choice(np.array([-1.0, 1.0]), size=int(labels.sum()))
    faulty[labels == 1] = np.clip(faulty[labels == 1] + signs * 0.85, -1.0, 1.0)
    return faulty, labels, {"spike_magnitude": 0.85, "segment_length": None}


def make_fault(
    clean: np.ndarray,
    fault_mode: str,
    rate: float,
    stuck_value: float,
    seg_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if fault_mode == "scale":
        return inject_scale_fault(clean, rate, seg_len, rng)
    if fault_mode == "spike":
        return inject_spike_fault(clean, rate, rng)
    return inject_fault(clean, fault_mode, rate, stuck_value, seg_len, rng)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stream = load_air_quality()
    split = robust_normalize(stream.values, test_len=args.test_len)
    alpha = histogram_alpha(split.train, args.raw_buckets)
    fault_modes = [item.strip() for item in args.fault_modes.split(",") if item.strip()]
    mechanisms = [item.strip() for item in args.mechanisms.split(",") if item.strip()]
    epsilons = [float(item) for item in args.epsilons.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    stuck_value = float(np.quantile(split.train, 0.95))

    rows: list[dict[str, object]] = []
    for fault_mode in fault_modes:
        for eps in epsilons:
            for seed in seeds:
                fault_rng = np.random.default_rng(seed)
                faulty, labels, params = make_fault(
                    split.test,
                    fault_mode,
                    args.fault_rate,
                    stuck_value,
                    args.segment_length,
                    fault_rng,
                )
                expected_rate = float(np.clip(np.mean(labels), 0.005, 0.70))
                for mechanism in mechanisms:
                    obs_seed = seed + 1000 * int(round(eps * 10)) + (0 if mechanism == "pm" else 100000)
                    obs_rng = np.random.default_rng(obs_seed)
                    obs, m, mechanism_label = observations(
                        faulty,
                        mechanism,
                        eps,
                        args.raw_buckets,
                        args.output_buckets,
                        obs_rng,
                    )
                    methods = {
                        "channel_window_glr": lambda: window_glr_score(obs, m, alpha),
                        "channel_spike_score": lambda: channel_spike_score(obs, m, alpha),
                        "privsaf_mixture": lambda: mixture_infer(obs, m, alpha),
                        "privsaf_hmm": lambda: hmm_infer(
                            obs,
                            m,
                            alpha,
                            expected_rate,
                            args.segment_length,
                        ),
                    }
                    for method, fn in methods.items():
                        start = time.perf_counter()
                        scores, cand, rhat = fn()
                        runtime = time.perf_counter() - start
                        auroc, auprc = safe_detection_metrics(labels, scores)
                        row = {
                            "panel": "mechanism_fault_extension",
                            "dataset_id": stream.dataset_id,
                            "dataset": stream.dataset_name,
                            "mechanism": mechanism,
                            "mechanism_label": mechanism_label,
                            "fault_mode": fault_mode,
                            "epsilon": eps,
                            "seed": seed,
                            "method": method,
                            "n_test": int(len(labels)),
                            "fault_rows": int(labels.sum()),
                            "fault_rate": float(np.mean(labels)),
                            "auroc": auroc,
                            "auprc": auprc,
                            "estimated_bucket": int(cand),
                            "estimated_fault_rate": float(rhat) if np.isfinite(rhat) else "",
                            "runtime_sec": float(runtime),
                        }
                        row.update({f"param_{key}": value for key, value in params.items()})
                        rows.append(row)

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["mechanism", "fault_mode", "method"], as_index=False)
        .agg(
            cases=("auroc", "size"),
            auroc_mean=("auroc", "mean"),
            auprc_mean=("auprc", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
        )
        .sort_values(["mechanism", "fault_mode", "auprc_mean"], ascending=[True, True, False])
    )
    by_epsilon = (
        runs.groupby(["mechanism", "fault_mode", "epsilon", "method"], as_index=False)
        .agg(
            cases=("auroc", "size"),
            auroc_mean=("auroc", "mean"),
            auprc_mean=("auprc", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
        )
        .sort_values(["mechanism", "fault_mode", "epsilon", "auprc_mean"], ascending=[True, True, True, False])
    )
    return runs, summary, by_epsilon


def main() -> None:
    parser = argparse.ArgumentParser(description="Known-channel mechanism and fault-family extension.")
    parser.add_argument("--mechanisms", default="pm,duchi_binary")
    parser.add_argument("--fault-modes", default="segment_stuck,bias,scale,drift,spike")
    parser.add_argument("--epsilons", default="0.5,1,2")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--raw-buckets", type=int, default=32)
    parser.add_argument("--output-buckets", type=int, default=32)
    parser.add_argument("--test-len", type=int, default=1200)
    parser.add_argument("--fault-rate", type=float, default=0.20)
    parser.add_argument("--segment-length", type=int, default=48)
    parser.add_argument("--output-prefix", default="mechanism_fault_extension")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    runs, summary, by_epsilon = run(args)
    runs.to_csv(RESULTS / f"{args.output_prefix}_runs.csv", index=False)
    summary.to_csv(RESULTS / f"{args.output_prefix}_summary.csv", index=False)
    by_epsilon.to_csv(RESULTS / f"{args.output_prefix}_by_epsilon.csv", index=False)
    print(f"Wrote {len(runs)} mechanism/fault extension rows.")
    print(RESULTS / f"{args.output_prefix}_summary.csv")


if __name__ == "__main__":
    main()
