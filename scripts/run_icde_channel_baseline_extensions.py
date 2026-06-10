from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from run_icde_revision_grid import (
    RESULTS,
    discretize_output,
    histogram_alpha,
    hmm_infer,
    inject_fault,
    inject_template_stuck,
    load_streams,
    mine_flatline_templates,
    mixture_infer,
    pm_matrix,
    pm_sample,
    raw_bucket_index,
    robust_normalize,
    safe_detection_metrics,
    window_glr_score,
)


def channel_likelihood_scan(obs: np.ndarray, m: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Pointwise PM-column likelihood scan without temporal posterior inference."""
    p0 = np.clip(m @ alpha, 1e-12, None)
    log_ratio = np.log(np.clip(m, 1e-12, None)) - np.log(p0)[:, None]
    per_frame = log_ratio[obs]
    cand_by_frame = np.argmax(per_frame, axis=1)
    scores = np.max(per_frame, axis=1)
    positive = scores > np.quantile(scores, 0.80) if len(scores) else np.array([], dtype=bool)
    cand = int(pd.Series(cand_by_frame).mode().iloc[0]) if len(cand_by_frame) else 0
    return scores, cand, float(np.mean(positive)) if len(scores) else 0.0


def channel_llr_cusum(obs: np.ndarray, m: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, int, float]:
    """One-sided CUSUM over known-channel stuck-column log-likelihood ratios."""
    p0 = np.clip(m @ alpha, 1e-12, None)
    log_ratio = np.log(np.clip(m, 1e-12, None)) - np.log(p0)[:, None]
    mean0 = p0 @ log_ratio
    var0 = p0 @ ((log_ratio - mean0[None, :]) ** 2)
    std0 = np.sqrt(np.maximum(var0, 1e-8))
    centered = (log_ratio[obs] - mean0[None, :]) / std0[None, :]
    state = np.zeros(m.shape[1], dtype=float)
    scores = np.zeros(len(obs), dtype=float)
    cand_by_frame = np.zeros(len(obs), dtype=int)
    for t, row in enumerate(centered):
        state = np.maximum(0.0, state + row - 0.25)
        cand = int(np.argmax(state))
        scores[t] = float(state[cand])
        cand_by_frame[t] = cand
    positive = scores > np.quantile(scores, 0.80) if len(scores) else np.array([], dtype=bool)
    cand = int(pd.Series(cand_by_frame).mode().iloc[0]) if len(cand_by_frame) else 0
    return scores, cand, float(np.mean(positive)) if len(scores) else 0.0


def generic_pm_hmm(
    obs: np.ndarray,
    m: np.ndarray,
    alpha: np.ndarray,
    expected_rate: float,
    seg_len: int,
    iterations: int = 20,
) -> tuple[np.ndarray, int, float]:
    """Two-state PM-domain HMM with a free fault emission, not a stuck-column model."""
    out_d = m.shape[0]
    b0 = np.clip(m @ alpha, 1e-12, None)
    counts = np.bincount(obs, minlength=out_d).astype(float) + 1e-3
    empirical = counts / counts.sum()
    surplus = np.maximum(empirical - b0, 0.0) + 1e-3
    b1 = surplus / surplus.sum()
    p11 = max(0.50, 1.0 - 1.0 / max(seg_len, 2))
    p01 = min(0.20, max(1e-4, expected_rate * (1.0 - p11) / max(1.0 - expected_rate, 1e-6)))
    pi1 = min(max(expected_rate, 1e-4), 0.95)
    onehot = np.eye(out_d, dtype=float)[obs]
    gamma = np.full(len(obs), pi1, dtype=float)
    for _ in range(iterations):
        _, gamma = _forward_backward_generic(obs, b0, b1, p01, p11, pi1)
        weights = gamma[:, None] * onehot + 1e-3
        b1 = weights.sum(axis=0)
        b1 /= b1.sum()
        pi1 = float(np.clip(gamma[0], 1e-4, 0.95))
    cand = int(np.argmin(np.linalg.norm(m - b1[:, None], axis=0)))
    return gamma, cand, float(np.mean(gamma))


def _forward_backward_generic(
    obs: np.ndarray,
    b0: np.ndarray,
    b1: np.ndarray,
    p01: float,
    p11: float,
    pi1: float,
) -> tuple[float, np.ndarray]:
    n = len(obs)
    log_b = np.vstack([np.log(np.clip(b0[obs], 1e-15, None)), np.log(np.clip(b1[obs], 1e-15, None))]).T
    log_a = np.log(np.array([[1.0 - p01, p01], [1.0 - p11, p11]], dtype=float))
    log_alpha = np.zeros((n, 2), dtype=float)
    scales = np.zeros(n, dtype=float)
    log_alpha[0] = np.log(np.array([1.0 - pi1, pi1], dtype=float)) + log_b[0]
    scales[0] = np.logaddexp.reduce(log_alpha[0])
    log_alpha[0] -= scales[0]
    for t in range(1, n):
        for s in range(2):
            log_alpha[t, s] = log_b[t, s] + np.logaddexp.reduce(log_alpha[t - 1] + log_a[:, s])
        scales[t] = np.logaddexp.reduce(log_alpha[t])
        log_alpha[t] -= scales[t]
    log_beta = np.zeros((n, 2), dtype=float)
    for t in range(n - 2, -1, -1):
        for s in range(2):
            log_beta[t, s] = np.logaddexp.reduce(log_a[s, :] + log_b[t + 1, :] + log_beta[t + 1, :])
        log_beta[t] -= np.logaddexp.reduce(log_beta[t])
    log_gamma = log_alpha + log_beta
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    return float(scales.sum()), np.exp(log_gamma[:, 1])


def run_channel_baselines(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    d = args.raw_buckets
    out_d = args.output_buckets
    eps_values = [float(x) for x in args.epsilons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    streams = load_streams(include_optional=False)
    for stream in streams:
        split = robust_normalize(stream.values, test_len=args.test_len)
        alpha = histogram_alpha(split.train, d)
        templates = mine_flatline_templates(np.concatenate([split.train, split.validation, split.test]))
        stuck_values = [float(x) for x in np.quantile(split.train, [0.05, 0.95])]
        for eps in eps_values:
            m, out_edges = pm_matrix(eps, d, out_d)
            for seed in seeds:
                for fault_mode in ["iid_stuck", "segment_stuck", "template_stuck"]:
                    mode_values = [float("nan")] if fault_mode == "template_stuck" else stuck_values
                    for stuck_value in mode_values:
                        rng = np.random.default_rng(seed)
                        if fault_mode == "template_stuck":
                            faulty, labels, params = inject_template_stuck(split.test, args.fault_rate, templates, rng)
                        else:
                            faulty, labels, params = inject_fault(
                                split.test, fault_mode, args.fault_rate, stuck_value, args.segment_length, rng
                            )
                        obs = discretize_output(pm_sample(faulty, eps, rng), out_edges)
                        stuck_param = params.get("stuck_value")
                        true_cand = raw_bucket_index(float(stuck_param), d) if stuck_param is not None else -1
                        base = {
                            "panel": "channel_baseline_ablation",
                            "dataset_id": stream.dataset_id,
                            "dataset": stream.dataset_name,
                            "epsilon": eps,
                            "seed": seed,
                            "fault_mode": fault_mode,
                            "n_test": int(len(labels)),
                            "fault_rate": float(np.mean(labels)),
                            "true_bucket": true_cand,
                        }
                        method_fns = [
                            ("channel_likelihood_scan", lambda: channel_likelihood_scan(obs, m, alpha)),
                            ("channel_llr_cusum", lambda: channel_llr_cusum(obs, m, alpha)),
                            ("channel_window_glr_scan", lambda: window_glr_score(obs, m, alpha)),
                            (
                                "generic_pm_hmm",
                                lambda: generic_pm_hmm(obs, m, alpha, args.fault_rate, args.segment_length),
                            ),
                        ]
                        if fault_mode == "iid_stuck":
                            method_fns.append(("privsaf_matched_mixture", lambda: mixture_infer(obs, m, alpha)))
                        else:
                            method_fns.append(
                                ("privsaf_matched_hmm", lambda: hmm_infer(obs, m, alpha, args.fault_rate, args.segment_length))
                            )
                        for method, fn in method_fns:
                            t0 = time.perf_counter()
                            scores, cand, rhat = fn()
                            runtime = time.perf_counter() - t0
                            auroc, auprc = safe_detection_metrics(labels, scores)
                            row = dict(base)
                            row.update(
                                {
                                    "method": method,
                                    "auroc": auroc,
                                    "auprc": auprc,
                                    "runtime_sec": runtime,
                                    "estimated_bucket": cand,
                                    "bucket_hit": float(cand == true_cand) if true_cand >= 0 else float("nan"),
                                    "estimated_fault_rate": rhat,
                                }
                            )
                            rows.append(row)
    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["panel", "method", "fault_mode"], as_index=False)
        .agg(
            cases=("auroc", "size"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            runtime_sec_mean=("runtime_sec", "mean"),
            bucket_hit_mean=("bucket_hit", "mean"),
        )
        .sort_values(["fault_mode", "auprc_mean"], ascending=[True, False])
    )
    return runs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct channel-aware baseline extensions.")
    parser.add_argument("--epsilons", default="2,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--raw-buckets", type=int, default=32)
    parser.add_argument("--output-buckets", type=int, default=32)
    parser.add_argument("--test-len", type=int, default=2400)
    parser.add_argument("--fault-rate", type=float, default=0.20)
    parser.add_argument("--segment-length", type=int, default=48)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    runs, summary = run_channel_baselines(args)
    runs.to_csv(RESULTS / "icde_channel_baseline_runs.csv", index=False)
    summary.to_csv(RESULTS / "icde_channel_baseline_summary.csv", index=False)
    print(f"Wrote {len(runs)} channel-baseline rows.")
    print(RESULTS / "icde_channel_baseline_summary.csv")


if __name__ == "__main__":
    main()
