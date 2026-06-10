from __future__ import annotations

import math
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
RAW_CSV = ROOT / "data" / "air_quality" / "raw" / "AirQualityUCI.csv"
RESULTS = ROOT / "results"


def load_air_quality() -> np.ndarray:
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Missing {RAW_CSV}; run scripts/download_air_quality.py first.")
    df = pd.read_csv(RAW_CSV, sep=";", decimal=",")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S", errors="coerce")
    df = df.sort_values("timestamp")
    target = pd.to_numeric(df["C6H6(GT)"], errors="coerce")
    target = target[(target.notna()) & (target != -200)]
    return target.to_numpy(dtype=float)


def split_and_normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    n = len(values)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train = values[:n_train]
    val = values[n_train : n_train + n_val]
    test = values[n_train + n_val : n_train + n_val + 2000]
    lo, hi = float(np.min(train)), float(np.max(train))
    scale = max(hi - lo, 1e-12)

    def norm(x: np.ndarray) -> np.ndarray:
        return np.clip(2.0 * (x - lo) / scale - 1.0, -1.0, 1.0)

    meta = {
        "n_after_missing_value_filter": int(n),
        "n_train": int(len(train)),
        "n_validation": int(len(val)),
        "n_test": int(len(test)),
        "train_min": lo,
        "train_max": hi,
    }
    return norm(train), norm(val), norm(test), meta


def centers(d: int) -> np.ndarray:
    return np.linspace(-1.0 + 1.0 / d, 1.0 - 1.0 / d, d)


def raw_bucket_index(value: float, d: int) -> int:
    return int(np.argmin(np.abs(centers(d) - value)))


def pm_sample(v: np.ndarray, eps: float, rng: np.random.Generator) -> np.ndarray:
    s = math.exp(eps / 2.0)
    c = (s + 1.0) / (s - 1.0)
    left = (s * v - 1.0) / (s - 1.0)
    right = (s * v + 1.0) / (s - 1.0)
    p_high = s / (s + 1.0)
    high = rng.random(len(v)) < p_high
    x = np.empty_like(v)
    x[high] = rng.uniform(left[high], right[high])
    low_idx = np.where(~high)[0]
    for idx in low_idx:
        left_len = max(left[idx] + c, 0.0)
        right_len = max(c - right[idx], 0.0)
        if rng.random() < left_len / max(left_len + right_len, 1e-12):
            x[idx] = rng.uniform(-c, left[idx])
        else:
            x[idx] = rng.uniform(right[idx], c)
    return x


def pm_matrix(eps: float, d: int, out_d: int) -> tuple[np.ndarray, np.ndarray]:
    s = math.exp(eps / 2.0)
    c = (s + 1.0) / (s - 1.0)
    raw_centers = centers(d)
    out_edges = np.linspace(-c, c, out_d + 1)
    m = np.zeros((out_d, d), dtype=float)
    f_in = s / (s + 1.0) / (c - 1.0)
    f_out = 1.0 / (s + 1.0) / (c + 1.0)
    for i, v in enumerate(raw_centers):
        left = (s * v - 1.0) / (s - 1.0)
        right = (s * v + 1.0) / (s - 1.0)
        for j in range(out_d):
            a, b = out_edges[j], out_edges[j + 1]
            overlap = max(0.0, min(b, right) - max(a, left))
            width = b - a
            m[j, i] = f_in * overlap + f_out * (width - overlap)
    m /= m.sum(axis=0, keepdims=True)
    return m, out_edges


def histogram_alpha(v: np.ndarray, d: int) -> np.ndarray:
    bins = np.linspace(-1.0, 1.0, d + 1)
    counts, _ = np.histogram(v, bins=bins)
    alpha = counts.astype(float) + 1e-3
    return alpha / alpha.sum()


def inject_faults(v: np.ndarray, rate: float, stuck_value: float, mode: str, seg_len: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = len(v)
    labels = np.zeros(n, dtype=int)
    target = max(1, int(round(rate * n)))
    if mode == "iid":
        idx = rng.choice(n, size=target, replace=False)
        labels[idx] = 1
    else:
        attempts = 0
        while labels.sum() < target and attempts < 10000:
            start = int(rng.integers(0, max(1, n - seg_len + 1)))
            end = min(n, start + seg_len)
            if labels[start:end].sum() == 0:
                labels[start:end] = 1
            attempts += 1
        if labels.sum() > target:
            ones = np.where(labels == 1)[0]
            labels[ones[target:]] = 0
    faulty = v.copy()
    faulty[labels == 1] = stuck_value
    return faulty, labels


def discretize_output(x: np.ndarray, out_edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(out_edges, x, side="right") - 1, 0, len(out_edges) - 2)


def mixture_infer(jobs: np.ndarray, m: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, int, float]:
    out_d, d = m.shape
    q = np.bincount(jobs, minlength=out_d).astype(float)
    q /= q.sum()
    b0 = np.clip(m @ alpha, 1e-12, None)
    best = None
    for cand in range(d):
        b1 = np.clip(m[:, cand], 1e-12, None)
        for r in np.linspace(0.01, 0.80, 160):
            mix = (1.0 - r) * b0 + r * b1
            loss = float(np.sum((q - mix) ** 2))
            if best is None or loss < best[0]:
                best = (loss, cand, r, mix, b1)
    _, cand, r, mix, b1 = best
    gamma = r * b1[jobs] / np.clip(mix[jobs], 1e-12, None)
    return gamma, cand, float(r)


def forward_backward(obs: np.ndarray, b0: np.ndarray, b1: np.ndarray, p01: float, p11: float, pi1: float) -> tuple[float, np.ndarray]:
    n = len(obs)
    log_b = np.vstack([np.log(np.clip(b0[obs], 1e-15, None)), np.log(np.clip(b1[obs], 1e-15, None))]).T
    log_a = np.log(np.array([[1 - p01, p01], [1 - p11, p11]], dtype=float))
    log_alpha = np.zeros((n, 2), dtype=float)
    scale = np.zeros(n, dtype=float)
    init = np.log(np.array([1 - pi1, pi1]))
    log_alpha[0] = init + log_b[0]
    scale[0] = np.logaddexp.reduce(log_alpha[0])
    log_alpha[0] -= scale[0]
    for t in range(1, n):
        for s in range(2):
            log_alpha[t, s] = log_b[t, s] + np.logaddexp.reduce(log_alpha[t - 1] + log_a[:, s])
        scale[t] = np.logaddexp.reduce(log_alpha[t])
        log_alpha[t] -= scale[t]
    log_beta = np.zeros((n, 2), dtype=float)
    for t in range(n - 2, -1, -1):
        for s in range(2):
            log_beta[t, s] = np.logaddexp.reduce(log_a[s, :] + log_b[t + 1, :] + log_beta[t + 1, :])
        log_beta[t] -= np.logaddexp.reduce(log_beta[t])
    log_gamma = log_alpha + log_beta
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    gamma = np.exp(log_gamma[:, 1])
    return float(scale.sum()), gamma


def hmm_infer(obs: np.ndarray, m: np.ndarray, alpha: np.ndarray, expected_rate: float, seg_len: int) -> tuple[np.ndarray, int, float]:
    b0 = np.clip(m @ alpha, 1e-12, None)
    p11 = max(0.50, 1.0 - 1.0 / max(seg_len, 2))
    p01 = min(0.20, max(1e-4, expected_rate * (1.0 - p11) / max(1.0 - expected_rate, 1e-6)))
    best = None
    for cand in range(m.shape[1]):
        b1 = np.clip(m[:, cand], 1e-12, None)
        ll, gamma = forward_backward(obs, b0, b1, p01, p11, expected_rate)
        if best is None or ll > best[0]:
            best = (ll, cand, gamma)
    _, cand, gamma = best
    return gamma, int(cand), float(np.mean(gamma))


def eval_run(labels: np.ndarray, scores: np.ndarray, cand: int, true_cand: int, rhat: float, rate: float, alpha_hat: np.ndarray, alpha_ref: np.ndarray, raw_centers: np.ndarray, runtime: float) -> dict[str, float]:
    return {
        "auroc": roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else np.nan,
        "auprc": average_precision_score(labels, scores),
        "r_error": abs(rhat - rate),
        "bucket_error": abs(cand - true_cand),
        "bucket_acc": 1.0 if cand == true_cand else 0.0,
        "js": float(jensenshannon(alpha_ref, alpha_hat, base=2.0) ** 2),
        "wasserstein": float(wasserstein_distance(raw_centers, raw_centers, u_weights=alpha_ref, v_weights=alpha_hat)),
        "runtime_sec": runtime,
    }


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    values = load_air_quality()
    train, val, test, split_meta = split_and_normalize(values)
    d = out_d = 32
    eps_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    seeds = [0, 1, 2, 3, 4]
    stuck_values = [0.03125, 0.84375]
    modes = ["segment", "iid"]
    rate = 0.30
    seg_len = 24
    raw_centers = centers(d)
    alpha_train = histogram_alpha(train, d)
    alpha_ref = histogram_alpha(test, d)
    rows: list[dict[str, object]] = []
    for eps in eps_values:
        m, out_edges = pm_matrix(eps, d, out_d)
        for stuck in stuck_values:
            true_cand = raw_bucket_index(stuck, d)
            for mode in modes:
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    faulty, labels = inject_faults(test, rate, stuck, mode, seg_len, rng)
                    privatized = pm_sample(faulty, eps, rng)
                    obs = discretize_output(privatized, out_edges)
                    for method in ["mixture", "hmm"]:
                        t0 = time.perf_counter()
                        if method == "mixture":
                            scores, cand, rhat = mixture_infer(obs, m, alpha_train)
                        else:
                            scores, cand, rhat = hmm_infer(obs, m, alpha_train, rate, seg_len)
                        runtime = time.perf_counter() - t0
                        alpha_hat = alpha_train.copy()
                        row = {
                            "dataset": "UCI Air Quality",
                            "target": "C6H6(GT)",
                            "epsilon": eps,
                            "seed": seed,
                            "stuck_value": stuck,
                            "true_bucket": true_cand,
                            "fault_rate": rate,
                            "mode": mode,
                            "method": method,
                            "n_test": len(test),
                        }
                        row.update(eval_run(labels, scores, cand, true_cand, rhat, rate, alpha_hat, alpha_ref, raw_centers, runtime))
                        rows.append(row)
                    rng_scores = np.random.default_rng(seed + 1000).random(len(labels))
                    row = {
                        "dataset": "UCI Air Quality",
                        "target": "C6H6(GT)",
                        "epsilon": eps,
                        "seed": seed,
                        "stuck_value": stuck,
                        "true_bucket": true_cand,
                        "fault_rate": rate,
                        "mode": mode,
                        "method": "random",
                        "n_test": len(test),
                    }
                    row.update(eval_run(labels, rng_scores, -1, true_cand, 0.0, rate, alpha_train, alpha_ref, raw_centers, 0.0))
                    rows.append(row)

    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS / "air_quality_runs.csv", index=False)
    summary = (
        runs.groupby(["method", "mode", "stuck_value", "epsilon"], as_index=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            r_error_mean=("r_error", "mean"),
            bucket_acc_mean=("bucket_acc", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
        )
    )
    summary.to_csv(RESULTS / "air_quality_summary.csv", index=False)

    metadata = {
        "dataset": "UCI Air Quality",
        "source_url": DATA_URL,
        "raw_csv": str(RAW_CSV.relative_to(ROOT)),
        "target": "C6H6(GT)",
        "missing_value_marker": -200,
        "split": split_meta,
        "normalization": "train MinMax to [-1, 1], validation/test clipped",
        "privacy_mechanism": "piecewise mechanism, independent per report",
        "epsilon_values": eps_values,
        "seeds": seeds,
        "raw_buckets": d,
        "output_buckets": out_d,
        "fault_rate": rate,
        "segment_length": seg_len,
        "stuck_values": stuck_values,
        "injection_modes": modes,
        "methods": ["random", "mixture", "hmm"],
        "outputs": [
            "results/air_quality_runs.csv",
            "results/air_quality_summary.csv",
            "results/fig_air_quality_repro.png",
        ],
    }
    (RESULTS / "air_quality_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0), sharex=True)
    plot_data = summary[(summary["mode"] == "segment") & (summary["stuck_value"] == 0.84375)]
    for method, group in plot_data.groupby("method"):
        group = group.sort_values("epsilon")
        axes[0].plot(group["epsilon"], group["auroc_mean"], marker="o", label=method)
        axes[1].plot(group["epsilon"], group["auprc_mean"], marker="o", label=method)
    axes[0].set_title("AUROC")
    axes[1].set_title("AUPRC")
    for ax in axes:
        ax.set_xlabel("epsilon")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("score")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_air_quality_repro.png", dpi=200)
    print(f"Wrote {RESULTS / 'air_quality_runs.csv'}")
    print(f"Wrote {RESULTS / 'air_quality_summary.csv'}")
    print(f"Wrote {RESULTS / 'air_quality_metadata.json'}")
    print(f"Wrote {RESULTS / 'fig_air_quality_repro.png'}")


if __name__ == "__main__":
    main()
