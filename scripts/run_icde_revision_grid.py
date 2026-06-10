from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class StreamSpec:
    dataset_id: str
    dataset_name: str
    target: str
    values: np.ndarray
    source_file: Path


@dataclass(frozen=True)
class SplitStream:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    metadata: dict[str, object]


def finite_values(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    return out[np.isfinite(out)]


def robust_normalize(values: np.ndarray, train_fraction: float = 0.6, val_fraction: float = 0.2, test_len: int = 1200) -> SplitStream:
    clean = finite_values(values)
    if len(clean) < 300:
        raise ValueError("Need at least 300 finite points for chronological split.")

    n_train = int(train_fraction * len(clean))
    n_val = int(val_fraction * len(clean))
    train_raw = clean[:n_train]
    val_raw = clean[n_train : n_train + n_val]
    test_raw = clean[n_train + n_val : n_train + n_val + test_len]
    if len(test_raw) < min(200, test_len // 2):
        test_raw = clean[-test_len:]

    lo, hi = np.nanquantile(train_raw, [0.01, 0.99])
    scale = max(float(hi - lo), 1e-12)

    def norm(x: np.ndarray) -> np.ndarray:
        return np.clip(2.0 * (x - lo) / scale - 1.0, -1.0, 1.0)

    metadata = {
        "raw_rows": int(len(values)),
        "finite_rows": int(len(clean)),
        "train_rows": int(len(train_raw)),
        "validation_rows": int(len(val_raw)),
        "test_rows": int(len(test_raw)),
        "normalization": "train p01/p99 mapped to [-1,1] with clipping",
        "train_p01": float(lo),
        "train_p99": float(hi),
    }
    return SplitStream(norm(train_raw), norm(val_raw), norm(test_raw), metadata)


def load_air_quality() -> StreamSpec:
    path = ROOT / "data" / "air_quality" / "raw" / "AirQualityUCI.csv"
    df = pd.read_csv(path, sep=";", decimal=",", usecols=["Date", "Time", "C6H6(GT)"])
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H.%M.%S", errors="coerce")
    df = df.sort_values("timestamp")
    values = pd.to_numeric(df["C6H6(GT)"], errors="coerce").replace(-200, np.nan).to_numpy(dtype=float)
    return StreamSpec("air_quality", "UCI Air Quality", "C6H6(GT)", values, path)


def load_household_power() -> StreamSpec:
    path = ROOT / "data" / "household_power" / "raw" / "household_power_consumption.txt"
    df = pd.read_csv(path, sep=";", usecols=["Global_active_power"], na_values=["?"], low_memory=False)
    values = pd.to_numeric(df["Global_active_power"], errors="coerce").to_numpy(dtype=float)
    # Use hourly sampling to keep the review grid small while preserving chronology.
    values = values[::60]
    return StreamSpec("household_power", "UCI Household Power", "Global_active_power hourly sample", values, path)


def load_bike_sharing() -> StreamSpec:
    path = ROOT / "data" / "bike_sharing" / "raw" / "hour.csv"
    df = pd.read_csv(path, usecols=["dteday", "hr", "cnt"])
    df["timestamp"] = pd.to_datetime(df["dteday"]) + pd.to_timedelta(df["hr"], unit="h")
    df = df.sort_values("timestamp")
    values = pd.to_numeric(df["cnt"], errors="coerce").to_numpy(dtype=float)
    return StreamSpec("bike_sharing", "UCI Bike Sharing", "cnt", values, path)


def load_beijing_air() -> StreamSpec:
    path = (
        ROOT
        / "data"
        / "beijing_air"
        / "raw"
        / "PRSA2017_Data_20130301-20170228"
        / "PRSA_Data_20130301-20170228"
        / "PRSA_Data_Aotizhongxin_20130301-20170228.csv"
    )
    df = pd.read_csv(path, usecols=["year", "month", "day", "hour", "PM2.5"])
    df["timestamp"] = pd.to_datetime(df[["year", "month", "day", "hour"]], errors="coerce")
    df = df.sort_values("timestamp")
    values = pd.to_numeric(df["PM2.5"], errors="coerce").to_numpy(dtype=float)
    return StreamSpec("beijing_air", "UCI Beijing Multi-Site Air", "Aotizhongxin PM2.5", values, path)


def load_nab() -> StreamSpec:
    path = ROOT / "data" / "nab" / "raw" / "realKnownCause" / "machine_temperature_system_failure.csv"
    df = pd.read_csv(path, usecols=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")
    values = pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype=float)
    return StreamSpec("nab", "NAB Machine Temperature", "value", values, path)


def load_gas_drift() -> StreamSpec:
    root = ROOT / "data" / "gas_drift" / "raw" / "Dataset"
    values: list[float] = []
    for path in sorted(root.glob("batch*.dat"), key=lambda p: int(p.stem.replace("batch", ""))):
        with path.open("r", encoding="utf-8", errors="replace") as fin:
            for line in fin:
                parts = line.strip().split()
                for item in parts[1:]:
                    if item.startswith("1:"):
                        values.append(float(item.split(":", 1)[1]))
                        break
    return StreamSpec("gas_drift", "UCI Gas Sensor Array Drift", "feature 1", np.asarray(values), root / "batch*.dat")


def load_streams(include_optional: bool) -> list[StreamSpec]:
    loaders = [load_air_quality, load_household_power, load_bike_sharing, load_beijing_air, load_nab]
    if include_optional:
        loaders.append(load_gas_drift)

    streams: list[StreamSpec] = []
    for loader in loaders:
        try:
            streams.append(loader())
        except FileNotFoundError as exc:
            print(f"Skipping missing dataset: {exc}")
    return streams


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
    m /= np.clip(m.sum(axis=0, keepdims=True), 1e-12, None)
    return m, out_edges


def discretize_output(x: np.ndarray, out_edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(out_edges, x, side="right") - 1, 0, len(out_edges) - 2)


def histogram_alpha(v: np.ndarray, d: int) -> np.ndarray:
    bins = np.linspace(-1.0, 1.0, d + 1)
    counts, _ = np.histogram(v[np.isfinite(v)], bins=bins)
    alpha = counts.astype(float) + 1e-3
    return alpha / alpha.sum()


def estimate_alpha_from_ldp(obs: np.ndarray, m: np.ndarray) -> np.ndarray:
    q = np.bincount(obs, minlength=m.shape[0]).astype(float)
    q /= max(float(q.sum()), 1.0)
    alpha, *_ = np.linalg.lstsq(m, q, rcond=None)
    alpha = np.clip(alpha, 1e-9, None)
    return alpha / alpha.sum()


def iid_mask(n: int, target: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, size=min(target, n), replace=False)] = 1
    return labels


def segment_mask(n: int, target: int, seg_len: int, rng: np.random.Generator) -> np.ndarray:
    labels = np.zeros(n, dtype=int)
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
    if labels.sum() < target:
        zeros = np.where(labels == 0)[0]
        fill = rng.choice(zeros, size=target - int(labels.sum()), replace=False)
        labels[fill] = 1
    return labels


def mine_flatline_templates(values: np.ndarray, min_len: int = 8, max_templates: int = 24) -> list[dict[str, float]]:
    series = finite_values(values)
    if len(series) < min_len + 4:
        return []
    diffs = np.abs(np.diff(series))
    nonzero = diffs[diffs > 1e-12]
    tol = max(1e-3, float(np.quantile(nonzero, 0.01)) if len(nonzero) else 1e-3)
    candidates: list[dict[str, float]] = []
    start = 0
    for idx in range(1, len(series)):
        if abs(series[idx] - series[idx - 1]) > 2.0 * tol:
            end = idx
            if end - start >= min_len:
                segment = series[start:end]
                left = series[max(0, start - min_len) : start]
                right = series[end : min(len(series), end + min_len)]
                flank = np.concatenate([left, right]) if len(left) + len(right) else np.array([])
                flank_delta = float(abs(np.mean(flank) - np.mean(segment))) if len(flank) else 0.0
                candidates.append(
                    {
                        "start": float(start),
                        "end": float(end),
                        "length": float(end - start),
                        "value": float(np.mean(segment)),
                        "range": float(np.max(segment) - np.min(segment)),
                        "flank_delta": flank_delta,
                        "score": flank_delta * math.sqrt(float(end - start)) / max(float(np.max(segment) - np.min(segment)), tol),
                    }
                )
            start = idx
    candidates.sort(key=lambda row: row["score"], reverse=True)
    return candidates[:max_templates]


def inject_template_stuck(
    clean: np.ndarray,
    rate: float,
    templates: list[dict[str, float]],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if not templates:
        q95 = float(np.quantile(clean, 0.95))
        return inject_fault(clean, "segment_stuck", rate, q95, 48, rng)
    template = templates[int(rng.integers(0, len(templates)))]
    seg_len = int(np.clip(round(template["length"]), 8, max(8, min(96, len(clean) // 3))))
    stuck_value = float(np.clip(template["value"], -1.0, 1.0))
    faulty, labels, _ = inject_fault(clean, "segment_stuck", rate, stuck_value, seg_len, rng)
    params = {
        "stuck_value": stuck_value,
        "segment_length": seg_len,
        "bias": None,
        "template_source": "native_flatline",
        "template_range": float(template["range"]),
        "template_flank_delta": float(template["flank_delta"]),
    }
    return faulty, labels, params


def inject_fault(
    clean: np.ndarray,
    fault_mode: str,
    rate: float,
    stuck_value: float,
    seg_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    n = len(clean)
    target = max(1, int(round(rate * n)))
    faulty = clean.copy()
    if fault_mode == "iid_stuck":
        labels = iid_mask(n, target, rng)
        faulty[labels == 1] = stuck_value
        params = {"stuck_value": float(stuck_value), "segment_length": None, "bias": None}
    elif fault_mode == "segment_stuck":
        labels = segment_mask(n, target, seg_len, rng)
        faulty[labels == 1] = stuck_value
        params = {"stuck_value": float(stuck_value), "segment_length": int(seg_len), "bias": None}
    elif fault_mode == "bias":
        labels = segment_mask(n, target, seg_len, rng)
        direction = 1.0 if stuck_value >= 0 else -1.0
        bias = 0.35 * direction
        faulty[labels == 1] = np.clip(faulty[labels == 1] + bias, -1.0, 1.0)
        params = {"stuck_value": None, "segment_length": int(seg_len), "bias": float(bias)}
    elif fault_mode == "drift":
        labels = segment_mask(n, target, seg_len, rng)
        idx = np.where(labels == 1)[0]
        drift = np.linspace(0.0, 0.45, len(idx)) if len(idx) else np.array([])
        faulty[idx] = np.clip(faulty[idx] + drift, -1.0, 1.0)
        params = {"stuck_value": None, "segment_length": int(seg_len), "bias": "ramp_to_0.45"}
    else:
        raise ValueError(f"Unknown fault mode: {fault_mode}")
    return faulty, labels, params


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=max(3, window // 4))
        .median()
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )


def rolling_mad(values: np.ndarray, window: int) -> np.ndarray:
    series = pd.Series(values)
    med = series.rolling(window=window, center=True, min_periods=max(3, window // 4)).median()
    mad = (series - med).abs().rolling(window=window, center=True, min_periods=max(3, window // 4)).median()
    return mad.bfill().ffill().to_numpy(dtype=float)


def robust_zscore(values: np.ndarray, center: float | None = None, scale: float | None = None) -> np.ndarray:
    if center is None:
        center = float(np.nanmedian(values))
    if scale is None:
        scale = 1.4826 * float(np.nanmedian(np.abs(values - center)))
    scale = max(scale, 1e-6)
    return np.abs(values - center) / scale


def raw_rolling_score(values: np.ndarray, window: int) -> np.ndarray:
    med = rolling_median(values, window)
    mad = np.maximum(1.4826 * rolling_mad(values, window), 1e-6)
    return np.abs(values - med) / mad


def raw_hampel_score(values: np.ndarray, window: int) -> np.ndarray:
    return np.clip(raw_rolling_score(values, window), 0.0, 20.0)


def raw_cusum_score(values: np.ndarray, train: np.ndarray, drift: float = 0.25) -> np.ndarray:
    mu = float(np.nanmedian(train))
    sigma = max(1.4826 * float(np.nanmedian(np.abs(train - mu))), 1e-6)
    return cusum_score_with_reference(values, mu, sigma, drift)


def cusum_score_with_reference(values: np.ndarray, mu: float, sigma: float, drift: float = 0.25) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    pos = np.zeros(len(values), dtype=float)
    neg = np.zeros(len(values), dtype=float)
    for i, value in enumerate(values):
        prev_pos = pos[i - 1] if i else 0.0
        prev_neg = neg[i - 1] if i else 0.0
        z = (value - mu) / sigma
        pos[i] = max(0.0, prev_pos + z - drift)
        neg[i] = max(0.0, prev_neg - z - drift)
    return np.maximum(pos, neg)


def raw_bocpd_score(values: np.ndarray, train: np.ndarray, max_run: int = 48, hazard: float = 1.0 / 48.0) -> np.ndarray:
    base_mu = float(np.nanmean(train))
    base_var = max(float(np.nanvar(train)), 1e-4)
    return bocpd_score_with_reference(values, base_mu, base_var, max_run, hazard)


def bocpd_score_with_reference(
    values: np.ndarray,
    base_mu: float,
    base_var: float,
    max_run: int = 48,
    hazard: float = 1.0 / 48.0,
) -> np.ndarray:
    base_var = max(float(base_var), 1e-4)
    run_probs = np.array([1.0])
    scores = np.zeros(len(values), dtype=float)
    history: list[float] = []
    for t, value in enumerate(values):
        preds = []
        for run_len in range(len(run_probs)):
            if run_len < 3:
                mu, var = base_mu, base_var
            else:
                suffix = np.asarray(history[-run_len:], dtype=float)
                mu = float(np.mean(suffix))
                var = max(float(np.var(suffix)), 1e-4)
            pred = math.exp(-0.5 * (value - mu) ** 2 / var) / math.sqrt(2.0 * math.pi * var)
            preds.append(max(pred, 1e-300))
        preds_arr = np.asarray(preds)
        growth = run_probs * preds_arr * (1.0 - hazard)
        change = float(np.sum(run_probs * preds_arr * hazard))
        new_probs = np.concatenate([[change], growth])
        if len(new_probs) > max_run:
            new_probs = new_probs[:max_run]
        evidence = max(float(new_probs.sum()), 1e-300)
        new_probs /= evidence
        scores[t] = new_probs[0]
        run_probs = new_probs
        history.append(float(value))
    return scores


def forward_backward(obs: np.ndarray, b0: np.ndarray, b1: np.ndarray, p01: float, p11: float, pi1: float) -> tuple[float, np.ndarray]:
    n = len(obs)
    log_b = np.vstack([np.log(np.clip(b0[obs], 1e-15, None)), np.log(np.clip(b1[obs], 1e-15, None))]).T
    log_a = np.log(np.array([[1.0 - p01, p01], [1.0 - p11, p11]], dtype=float))
    log_alpha = np.zeros((n, 2), dtype=float)
    scale = np.zeros(n, dtype=float)
    init = np.log(np.array([1.0 - pi1, pi1], dtype=float))
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
    return float(scale.sum()), np.exp(log_gamma[:, 1])


def mixture_infer(obs: np.ndarray, m: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, int, float]:
    out_d, d = m.shape
    q = np.bincount(obs, minlength=out_d).astype(float)
    q /= q.sum()
    b0 = np.clip(m @ alpha, 1e-12, None)
    best = None
    for cand in range(d):
        b1 = np.clip(m[:, cand], 1e-12, None)
        for r in np.linspace(0.01, 0.70, 100):
            mix = (1.0 - r) * b0 + r * b1
            loss = float(np.sum((q - mix) ** 2))
            if best is None or loss < best[0]:
                best = (loss, cand, r, mix, b1)
    _, cand, r, mix, b1 = best
    gamma = r * b1[obs] / np.clip(mix[obs], 1e-12, None)
    return gamma, int(cand), float(r)


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


def raw_hmm_stuck_score(values: np.ndarray, train: np.ndarray, expected_rate: float, seg_len: int, d: int) -> tuple[np.ndarray, int, float]:
    bins = np.linspace(-1.0, 1.0, d + 1)
    obs = np.clip(np.searchsorted(bins, values, side="right") - 1, 0, d - 1)
    alpha = histogram_alpha(train, d)
    smooth = 1e-4
    m = np.eye(d) * (1.0 - smooth * (d - 1)) + (1.0 - np.eye(d)) * smooth
    return hmm_infer(obs, m, alpha, expected_rate, seg_len)


def ldp_distribution_score(obs: np.ndarray, m: np.ndarray, alpha_hat: np.ndarray) -> np.ndarray:
    pred = np.clip(m @ alpha_hat, 1e-12, None)
    return -np.log(pred[obs])


def posterior_clean_mean(obs: np.ndarray, m: np.ndarray, alpha: np.ndarray, raw_centers: np.ndarray) -> np.ndarray:
    weights = alpha[None, :] * m[obs, :]
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
    return weights @ raw_centers


def pm_reference_moments(m: np.ndarray, alpha: np.ndarray, out_edges: np.ndarray) -> tuple[np.ndarray, float, float]:
    out_centers = 0.5 * (out_edges[:-1] + out_edges[1:])
    p0 = np.clip(m @ alpha, 1e-12, None)
    p0 /= p0.sum()
    mu = float(np.sum(out_centers * p0))
    var = float(np.sum(((out_centers - mu) ** 2) * p0))
    return out_centers, mu, max(var, 1e-4)


def pm_cusum_score(obs: np.ndarray, m: np.ndarray, alpha: np.ndarray, out_edges: np.ndarray) -> np.ndarray:
    out_centers, mu, var = pm_reference_moments(m, alpha, out_edges)
    return cusum_score_with_reference(out_centers[obs], mu, math.sqrt(var))


def pm_bocpd_score(obs: np.ndarray, m: np.ndarray, alpha: np.ndarray, out_edges: np.ndarray) -> np.ndarray:
    out_centers, mu, var = pm_reference_moments(m, alpha, out_edges)
    return bocpd_score_with_reference(out_centers[obs], mu, var)


def window_kl_score(obs: np.ndarray, m: np.ndarray, alpha: np.ndarray, windows: tuple[int, ...] = (8, 16, 32, 64)) -> np.ndarray:
    p0 = np.clip(m @ alpha, 1e-12, None)
    p0 /= p0.sum()
    log_p0 = np.log(p0)
    out_d = m.shape[0]
    onehot = np.eye(out_d, dtype=float)[obs]
    prefix = np.vstack([np.zeros((1, out_d), dtype=float), np.cumsum(onehot, axis=0)])
    scores = np.zeros(len(obs), dtype=float)
    for window in windows:
        if window > len(obs):
            continue
        window_scores = np.zeros(len(obs), dtype=float)
        for start in range(0, len(obs) - window + 1):
            counts = prefix[start + window] - prefix[start]
            empirical = counts / max(float(window), 1.0)
            nonzero = empirical > 0
            kl = float(window * np.sum(empirical[nonzero] * (np.log(empirical[nonzero]) - log_p0[nonzero])))
            center = start + window // 2
            window_scores[center] = max(window_scores[center], kl)
        expanded = (
            pd.Series(window_scores)
            .rolling(window=window, center=True, min_periods=1)
            .max()
            .to_numpy(dtype=float)
        )
        scores = np.maximum(scores, expanded)
    return scores


def window_glr_score(
    obs: np.ndarray,
    m: np.ndarray,
    alpha: np.ndarray,
    windows: tuple[int, ...] = (8, 16, 32, 64),
) -> tuple[np.ndarray, int, float]:
    p0 = np.clip(m @ alpha, 1e-12, None)
    p0 /= p0.sum()
    log_p0 = np.log(p0)
    log_m = np.log(np.clip(m, 1e-12, None))
    out_d = m.shape[0]
    onehot = np.eye(out_d, dtype=float)[obs]
    prefix = np.vstack([np.zeros((1, out_d), dtype=float), np.cumsum(onehot, axis=0)])
    scores = np.zeros(len(obs), dtype=float)
    best_score = -float("inf")
    best_cand = 0
    for window in windows:
        if window > len(obs):
            continue
        window_scores = np.zeros(len(obs), dtype=float)
        for start in range(0, len(obs) - window + 1):
            counts = prefix[start + window] - prefix[start]
            null_ll = float(counts @ log_p0)
            alt_ll_by_cand = counts @ log_m
            cand = int(np.argmax(alt_ll_by_cand))
            glr = float(alt_ll_by_cand[cand] - null_ll)
            if glr > best_score:
                best_score = glr
                best_cand = cand
            center = start + window // 2
            window_scores[center] = max(window_scores[center], glr)
        expanded = (
            pd.Series(window_scores)
            .rolling(window=window, center=True, min_periods=1)
            .max()
            .to_numpy(dtype=float)
        )
        scores = np.maximum(scores, expanded)
    positive = scores > np.quantile(scores, 0.80) if len(scores) else np.array([], dtype=bool)
    return scores, best_cand, float(np.mean(positive)) if len(scores) else 0.0


def safe_detection_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def precision_recall_at_fault_budget(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    k = int(labels.sum())
    if k <= 0:
        return float("nan"), float("nan")
    mask = topk_mask(scores, k)
    hits = float(np.sum(mask & labels.astype(bool)))
    return hits / max(float(k), 1.0), hits / max(float(labels.sum()), 1.0)


def method_profile(method: str, category: str) -> dict[str, object]:
    raw = category == "raw_domain_upper_bound"
    pm_report = category in {"privatized_domain", "channel_aware_baseline", "privacy_aware", "channel_ablation"}
    uses_channel = method.startswith("ldp_") or method.startswith("privsaf") or "wrong_epsilon" in method
    temporal = any(token in method for token in ["hmm", "cusum", "bocpd", "rolling", "window"])
    diagnoses = method in {"privsaf_mixture", "privsaf_hmm", "ldp_window_glr"} or "wrong_epsilon" in method
    return {
        "access_regime": "raw_upper_bound" if raw else "pm_ldp_reports",
        "uses_raw_values": bool(raw),
        "uses_pm_reports": bool(pm_report),
        "uses_pm_channel": bool(uses_channel),
        "uses_temporal_model": bool(temporal),
        "diagnostic_outputs": bool(diagnoses),
        "privacy_comparable": bool(not raw),
    }


def detection_row(
    base: dict[str, object],
    method: str,
    category: str,
    scores: np.ndarray,
    runtime: float,
    labels: np.ndarray,
    cand: int | None,
    true_cand: int | None,
    rhat: float | None,
) -> dict[str, object]:
    auroc, auprc = safe_detection_metrics(labels, scores)
    precision_at_budget, recall_at_budget = precision_recall_at_fault_budget(labels, scores)
    row = dict(base)
    row.update(
        {
            "method": method,
            "method_category": category,
            "auroc": auroc,
            "auprc": auprc,
            "runtime_sec": float(runtime),
            "precision_at_fault_budget": precision_at_budget,
            "recall_at_fault_budget": recall_at_budget,
            "estimated_bucket": "" if cand is None else int(cand),
            "true_bucket": "" if true_cand is None else int(true_cand),
            "bucket_error": "" if cand is None or true_cand is None else abs(int(cand) - int(true_cand)),
            "bucket_acc": "" if cand is None or true_cand is None else float(int(cand) == int(true_cand)),
            "estimated_fault_rate": "" if rhat is None else float(rhat),
            "fault_rate_error": "" if rhat is None else abs(float(rhat) - float(np.mean(labels))),
        }
    )
    row.update(method_profile(method, category))
    return row


def posterior_calibration(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> tuple[float, float]:
    probs = np.clip(scores.astype(float), 0.0, 1.0)
    y = labels.astype(float)
    brier = float(np.mean((probs - y) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        in_bin = (probs >= left) & (probs < right if right < 1.0 else probs <= right)
        if not np.any(in_bin):
            continue
        ece += float(np.mean(in_bin)) * abs(float(np.mean(probs[in_bin])) - float(np.mean(y[in_bin])))
    return brier, float(ece)


def diagnostic_output_row(
    base: dict[str, object],
    method: str,
    scores: np.ndarray,
    labels: np.ndarray,
    cand: int | None,
    true_cand: int | None,
    rhat: float | None,
) -> dict[str, object]:
    brier, ece = posterior_calibration(labels, scores)
    row = dict(base)
    row.update(
        {
            "method": method,
            "fault_ratio_true": float(np.mean(labels)),
            "fault_ratio_estimated": "" if rhat is None else float(rhat),
            "fault_ratio_abs_error": "" if rhat is None else abs(float(rhat) - float(np.mean(labels))),
            "stuck_bucket_true": "" if true_cand is None else int(true_cand),
            "stuck_bucket_estimated": "" if cand is None else int(cand),
            "stuck_bucket_top1": "" if cand is None or true_cand is None else float(int(cand) == int(true_cand)),
            "posterior_brier": brier,
            "posterior_ece": ece,
        }
    )
    return row


def topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    if k <= 0:
        return mask
    idx = np.argsort(scores)[-min(k, len(scores)) :]
    mask[idx] = True
    return mask


def apply_masked_interpolation(values: np.ndarray, mask: np.ndarray, method: str, window: int) -> np.ndarray:
    series = pd.Series(values.copy())
    if method == "linear_interpolation":
        series[mask] = np.nan
        return series.interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)
    if method == "locf":
        series[mask] = np.nan
        return series.ffill().bfill().to_numpy(dtype=float)
    if method == "rolling_median":
        med = pd.Series(values).rolling(window=window, center=True, min_periods=max(3, window // 4)).median().bfill().ffill()
        out = np.array(series.to_numpy(dtype=float), copy=True)
        out[mask] = med.to_numpy(dtype=float)[mask]
        return out
    raise ValueError(method)


def repair_metrics(clean: np.ndarray, repaired: np.ndarray, d: int, labels: np.ndarray | None = None, mask: np.ndarray | None = None) -> dict[str, float]:
    err = repaired - clean
    clean_hist = histogram_alpha(clean, d)
    repaired_hist = histogram_alpha(repaired, d)
    metrics = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(math.sqrt(np.mean(err**2))),
        "js": float(jensenshannon(clean_hist, repaired_hist, base=2.0) ** 2),
        "wasserstein": float(wasserstein_distance(clean, repaired)),
        "downstream_mean_abs_error": float(abs(np.mean(repaired) - np.mean(clean))),
        "downstream_std_abs_error": float(abs(np.std(repaired) - np.std(clean))),
        "downstream_p95_abs_error": float(abs(np.quantile(repaired, 0.95) - np.quantile(clean, 0.95))),
    }
    if labels is not None:
        fault = labels.astype(bool)
        clean_points = ~fault
        abs_err = np.abs(err)
        metrics.update(
            {
                "mae_fault_points": float(np.mean(abs_err[fault])) if np.any(fault) else float("nan"),
                "mae_clean_points": float(np.mean(abs_err[clean_points])) if np.any(clean_points) else float("nan"),
                "false_repaired_clean_fraction": float(np.mean(mask.astype(bool) & clean_points)) if mask is not None else 0.0,
            }
        )
    return metrics


def theoretical_diagnostics(
    dataset_id: str,
    dataset_name: str,
    target: str,
    eps: float,
    m: np.ndarray,
    alpha_train: np.ndarray,
    stuck_values: list[float],
    d: int,
) -> list[dict[str, object]]:
    b0 = np.clip(m @ alpha_train, 1e-12, None)
    singular_values = np.linalg.svd(m, compute_uv=False)
    min_col_tv = float("inf")
    for i in range(m.shape[1] - 1):
        tv = 0.5 * float(np.sum(np.abs(m[:, i] - m[:, i + 1])))
        min_col_tv = min(min_col_tv, tv)
    rows: list[dict[str, object]] = []
    for stuck in stuck_values:
        cand = raw_bucket_index(stuck, d)
        b1 = np.clip(m[:, cand], 1e-12, None)
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset": dataset_name,
                "target": target,
                "epsilon": eps,
                "stuck_value": float(stuck),
                "stuck_bucket": int(cand),
                "normal_fault_l1_separation": float(np.sum(np.abs(b0 - b1))),
                "normal_fault_l2_separation": float(np.linalg.norm(b0 - b1)),
                "min_adjacent_pm_column_tv": min_col_tv,
                "pm_condition_number": float(singular_values[0] / max(singular_values[-1], 1e-12)),
                "normal_entropy_bits": float(-np.sum(alpha_train * np.log2(np.clip(alpha_train, 1e-12, None)))),
            }
        )
    return rows


def make_repair_rows(
    base: dict[str, object],
    clean: np.ndarray,
    faulty: np.ndarray,
    labels: np.ndarray,
    privsaf_scores: np.ndarray,
    privsaf_clean_mean: np.ndarray,
    baseline_scores: dict[str, np.ndarray],
    d: int,
    window: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    oracle_mask = labels.astype(bool)
    empty_mask = np.zeros(len(labels), dtype=bool)
    repair_defs = [
        ("no_repair", "none", faulty, empty_mask),
        ("linear_interpolation", "oracle_fault_mask", apply_masked_interpolation(faulty, oracle_mask, "linear_interpolation", window), oracle_mask),
        ("locf", "oracle_fault_mask", apply_masked_interpolation(faulty, oracle_mask, "locf", window), oracle_mask),
        ("rolling_median", "oracle_fault_mask", apply_masked_interpolation(faulty, oracle_mask, "rolling_median", window), oracle_mask),
    ]
    k = int(labels.sum())
    repair_budget = k
    for name, scores in baseline_scores.items():
        mask = topk_mask(scores, repair_budget)
        repaired = apply_masked_interpolation(faulty, mask, "linear_interpolation", window)
        repair_defs.append((f"{name}_repair", f"{name}_topk_mask_linear", repaired, mask))

    predicted_mask = topk_mask(privsaf_scores, repair_budget)
    # PrivSAF supplies the private posterior mask; the repair operator uses the
    # same local interpolation primitive as the oracle-mask control, without
    # access to ground-truth labels.
    privsaf_repaired = apply_masked_interpolation(faulty, predicted_mask, "linear_interpolation", window)
    repair_defs.append(("privsaf_repair", "privsaf_topk_mask_linear", privsaf_repaired, predicted_mask))

    private_interp = apply_masked_interpolation(privsaf_clean_mean, predicted_mask, "linear_interpolation", window)
    soft_weight = np.clip(privsaf_scores, 0.0, 1.0)
    soft_repaired = (1.0 - soft_weight) * privsaf_clean_mean + soft_weight * private_interp
    repair_defs.append(("privsaf_soft_private_repair", "privsaf_posterior_soft_private", soft_repaired, predicted_mask))

    for repair_method, mask_source, repaired, mask in repair_defs:
        row = dict(base)
        row.update(
            {
                "repair_method": repair_method,
                "mask_source": mask_source,
                "repair_budget_points": int(repair_budget) if mask_source != "none" else 0,
            }
        )
        row.update(repair_metrics(clean, repaired, d, labels, mask))
        rows.append(row)
    return rows


def run_grid(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    streams = load_streams(args.include_optional)
    if args.datasets:
        wanted = set(args.datasets.split(","))
        streams = [stream for stream in streams if stream.dataset_id in wanted]

    d = args.raw_buckets
    out_d = args.output_buckets
    raw_centers = centers(d)
    eps_values = [float(x) for x in args.epsilons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    fault_modes = [x.strip() for x in args.fault_modes.split(",") if x.strip()]
    detection_rows: list[dict[str, object]] = []
    repair_rows: list[dict[str, object]] = []
    theory_diagnostic_rows: list[dict[str, object]] = []
    diagnosis_rows: list[dict[str, object]] = []
    native_flatline_rows: list[dict[str, object]] = []
    metadata: dict[str, object] = {
        "status": "measured",
        "script": "scripts/run_icde_revision_grid.py",
        "epsilons": eps_values,
        "seeds": seeds,
        "fault_modes": fault_modes,
        "raw_buckets": d,
        "output_buckets": out_d,
        "fault_rate": args.fault_rate,
        "segment_length": args.segment_length,
        "test_len": args.test_len,
        "datasets": [],
        "baseline_categories": {
            "raw_domain_upper_bound": ["raw_nonprivate_hmm", "raw_hampel", "raw_cusum", "raw_bocpd", "raw_rolling_median"],
            "privatized_domain": ["privatized_zscore", "privatized_rolling_median", "privatized_cusum", "privatized_bocpd"],
            "channel_aware_baseline": ["ldp_distribution_surprise", "ldp_postmean_hampel", "ldp_postmean_cusum", "ldp_postmean_rolling_median", "ldp_window_kl", "ldp_window_glr"],
            "privacy_aware": ["privsaf_mixture", "privsaf_hmm"],
            "channel_ablation": ["privsaf_hmm_wrong_epsilon_half", "privsaf_hmm_wrong_epsilon_double"],
        },
        "repair_methods": [
            "no_repair",
            "linear_interpolation",
            "locf",
            "rolling_median",
            "privatized_zscore_repair",
            "ldp_postmean_hampel_repair",
            "ldp_window_glr_repair",
            "privsaf_repair",
            "privsaf_soft_private_repair",
        ],
        "repair_note": "Interpolation/LOCF/rolling-median use oracle masks as raw-domain repair-operator controls; PrivSAF repair uses a top-k mask from privatized posterior scores and local linear interpolation without oracle labels.",
    }

    for stream in streams:
        split = robust_normalize(stream.values, test_len=args.test_len)
        alpha_train = histogram_alpha(split.train, d)
        flatline_templates = mine_flatline_templates(np.concatenate([split.train, split.validation, split.test]))
        for rank, item in enumerate(flatline_templates, start=1):
            native_flatline_rows.append(
                {
                    "dataset_id": stream.dataset_id,
                    "dataset": stream.dataset_name,
                    "target": stream.target,
                    "rank": rank,
                    "start": int(item["start"]),
                    "end": int(item["end"]),
                    "length": int(item["length"]),
                    "value": float(item["value"]),
                    "range": float(item["range"]),
                    "flank_delta": float(item["flank_delta"]),
                    "score": float(item["score"]),
                    "label_type": "weak_native_flatline_candidate",
                }
            )
        q05, q95 = np.quantile(split.train, [0.05, 0.95])
        stuck_values = [float(q05), float(q95)]
        metadata["datasets"].append(
            {
                "dataset_id": stream.dataset_id,
                "dataset_name": stream.dataset_name,
                "target": stream.target,
                "source_file": stream.source_file.relative_to(ROOT).as_posix() if stream.source_file.exists() else stream.source_file.as_posix(),
                **split.metadata,
                "implemented_fault_modes": fault_modes,
            }
        )

        for eps in eps_values:
            m, out_edges = pm_matrix(eps, d, out_d)
            theory_diagnostic_rows.extend(theoretical_diagnostics(stream.dataset_id, stream.dataset_name, stream.target, eps, m, alpha_train, stuck_values, d))
            for fault_mode in fault_modes:
                if fault_mode == "template_stuck":
                    mode_stuck_values = [float("nan")]
                else:
                    mode_stuck_values = stuck_values if fault_mode.endswith("stuck") else [float(q95)]
                for stuck_value in mode_stuck_values:
                    for seed in seeds:
                        rng = np.random.default_rng(seed)
                        if fault_mode == "template_stuck":
                            faulty, labels, fault_params = inject_template_stuck(split.test, args.fault_rate, flatline_templates, rng)
                        else:
                            faulty, labels, fault_params = inject_fault(
                                split.test,
                                fault_mode,
                                args.fault_rate,
                                stuck_value,
                                args.segment_length,
                                rng,
                            )
                        true_cand = raw_bucket_index(float(fault_params["stuck_value"]), d) if fault_params.get("stuck_value") is not None else None
                        privatized = pm_sample(faulty, eps, rng)
                        obs = discretize_output(privatized, out_edges)
                        post_mean = posterior_clean_mean(obs, m, alpha_train, raw_centers)
                        base = {
                            "dataset_id": stream.dataset_id,
                            "dataset": stream.dataset_name,
                            "target": stream.target,
                            "epsilon": eps,
                            "seed": seed,
                            "fault_mode": fault_mode,
                            "fault_rate": float(np.mean(labels)),
                            "n_test": int(len(split.test)),
                            "evidence_type": "template_based_semireal_injection" if fault_mode == "template_stuck" else "random_synthetic_injection",
                            **fault_params,
                        }

                        method_scores: dict[str, np.ndarray] = {}
                        privsaf_hmm_scores: np.ndarray | None = None

                        raw_methods = [
                            ("raw_hampel", "raw_domain_upper_bound"),
                            ("raw_cusum", "raw_domain_upper_bound"),
                            ("raw_bocpd", "raw_domain_upper_bound"),
                            ("raw_rolling_median", "raw_domain_upper_bound"),
                            ("raw_nonprivate_hmm", "raw_domain_upper_bound"),
                        ]
                        for method, category in raw_methods:
                            t0 = time.perf_counter()
                            cand: int | None = None
                            rhat: float | None = None
                            if method == "raw_hampel":
                                scores = raw_hampel_score(faulty, args.repair_window)
                            elif method == "raw_cusum":
                                scores = raw_cusum_score(faulty, split.train)
                            elif method == "raw_bocpd":
                                scores = raw_bocpd_score(faulty, split.train)
                            elif method == "raw_rolling_median":
                                scores = raw_rolling_score(faulty, args.repair_window)
                            else:
                                scores, cand, rhat = raw_hmm_stuck_score(faulty, split.train, args.fault_rate, args.segment_length, d)
                            runtime = time.perf_counter() - t0
                            method_scores[method] = scores
                            detection_rows.append(detection_row(base, method, category, scores, runtime, labels, cand, true_cand, rhat))

                        priv_methods = [
                            ("privatized_zscore", "privatized_domain"),
                            ("privatized_rolling_median", "privatized_domain"),
                            ("privatized_cusum", "privatized_domain"),
                            ("privatized_bocpd", "privatized_domain"),
                        ]
                        for method, category in priv_methods:
                            t0 = time.perf_counter()
                            if method == "privatized_zscore":
                                scores = robust_zscore(privatized)
                            elif method == "privatized_rolling_median":
                                scores = raw_rolling_score(privatized, args.repair_window)
                            elif method == "privatized_cusum":
                                scores = raw_cusum_score(privatized, privatized[: max(50, min(200, len(privatized)))])
                            else:
                                scores = raw_bocpd_score(privatized, privatized[: max(50, min(200, len(privatized)))])
                            runtime = time.perf_counter() - t0
                            method_scores[method] = scores
                            detection_rows.append(detection_row(base, method, category, scores, runtime, labels, None, true_cand, None))

                        t0 = time.perf_counter()
                        alpha_hat = estimate_alpha_from_ldp(obs, m)
                        scores = ldp_distribution_score(obs, m, alpha_hat)
                        runtime = time.perf_counter() - t0
                        detection_rows.append(
                            detection_row(base, "ldp_distribution_surprise", "channel_aware_baseline", scores, runtime, labels, None, true_cand, None)
                        )
                        method_scores["ldp_distribution_surprise"] = scores

                        channel_methods: list[tuple[str, np.ndarray, int | None, float | None, float]] = []

                        t0 = time.perf_counter()
                        scores = raw_hampel_score(post_mean, args.repair_window)
                        channel_methods.append(("ldp_postmean_hampel", scores, None, None, time.perf_counter() - t0))

                        t0 = time.perf_counter()
                        scores = raw_cusum_score(post_mean, split.train)
                        channel_methods.append(("ldp_postmean_cusum", scores, None, None, time.perf_counter() - t0))

                        t0 = time.perf_counter()
                        scores = raw_rolling_score(post_mean, args.repair_window)
                        channel_methods.append(("ldp_postmean_rolling_median", scores, None, None, time.perf_counter() - t0))

                        t0 = time.perf_counter()
                        scores = window_kl_score(obs, m, alpha_train)
                        channel_methods.append(("ldp_window_kl", scores, None, None, time.perf_counter() - t0))

                        t0 = time.perf_counter()
                        scores, cand, rhat = window_glr_score(obs, m, alpha_train)
                        channel_methods.append(("ldp_window_glr", scores, cand, rhat, time.perf_counter() - t0))

                        for method, scores, cand, rhat, runtime in channel_methods:
                            method_scores[method] = scores
                            detection_rows.append(detection_row(base, method, "channel_aware_baseline", scores, runtime, labels, cand, true_cand, rhat))

                        t0 = time.perf_counter()
                        scores, cand, rhat = mixture_infer(obs, m, alpha_train)
                        runtime = time.perf_counter() - t0
                        detection_rows.append(detection_row(base, "privsaf_mixture", "privacy_aware", scores, runtime, labels, cand, true_cand, rhat))
                        mixture_scores = scores
                        diagnosis_rows.append(diagnostic_output_row(base, "privsaf_mixture", scores, labels, cand, true_cand, rhat))

                        t0 = time.perf_counter()
                        scores, cand, rhat = hmm_infer(obs, m, alpha_train, args.fault_rate, args.segment_length)
                        runtime = time.perf_counter() - t0
                        detection_rows.append(detection_row(base, "privsaf_hmm", "privacy_aware", scores, runtime, labels, cand, true_cand, rhat))
                        privsaf_hmm_scores = scores
                        diagnosis_rows.append(diagnostic_output_row(base, "privsaf_hmm", scores, labels, cand, true_cand, rhat))

                        for factor, suffix in [(0.5, "half"), (2.0, "double")]:
                            t0 = time.perf_counter()
                            wrong_m, _ = pm_matrix(max(0.05, eps * factor), d, out_d)
                            scores, cand, rhat = hmm_infer(obs, wrong_m, alpha_train, args.fault_rate, args.segment_length)
                            runtime = time.perf_counter() - t0
                            method = f"privsaf_hmm_wrong_epsilon_{suffix}"
                            detection_rows.append(detection_row(base, method, "channel_ablation", scores, runtime, labels, cand, true_cand, rhat))

                        if fault_mode == "iid_stuck":
                            selected_scores = mixture_scores
                        else:
                            selected_scores = privsaf_hmm_scores
                        repair_base = dict(base)
                        repair_base.update({"privsaf_repair_variant": "mixture" if fault_mode == "iid_stuck" else "hmm"})
                        repair_baseline_scores = {
                            "privatized_zscore": method_scores["privatized_zscore"],
                            "ldp_postmean_hampel": method_scores["ldp_postmean_hampel"],
                            "ldp_window_glr": method_scores["ldp_window_glr"],
                        }
                        repair_rows.extend(
                            make_repair_rows(
                                repair_base,
                                split.test,
                                faulty,
                                labels,
                                selected_scores,
                                post_mean,
                                repair_baseline_scores,
                                d,
                                args.repair_window,
                            )
                        )

    metadata["theory_diagnostic_rows"] = theory_diagnostic_rows
    metadata["diagnosis_rows"] = diagnosis_rows
    metadata["native_flatline_rows"] = native_flatline_rows
    return pd.DataFrame(detection_rows), pd.DataFrame(repair_rows), metadata


def summarize_detection(runs: pd.DataFrame) -> pd.DataFrame:
    return (
        runs.groupby(["method_category", "method", "dataset_id", "fault_mode", "epsilon"], as_index=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            runtime_sec_mean=("runtime_sec", "mean"),
            precision_at_fault_budget_mean=("precision_at_fault_budget", "mean"),
            recall_at_fault_budget_mean=("recall_at_fault_budget", "mean"),
            fault_rate_error_mean=("fault_rate_error", lambda x: pd.to_numeric(x, errors="coerce").mean()),
            bucket_acc_mean=("bucket_acc", lambda x: pd.to_numeric(x, errors="coerce").mean()),
        )
        .sort_values(["method_category", "method", "dataset_id", "fault_mode", "epsilon"])
    )


def summarize_repair(repairs: pd.DataFrame) -> pd.DataFrame:
    summary = (
        repairs.groupby(["repair_method", "mask_source", "dataset_id", "fault_mode", "epsilon"], as_index=False)
        .agg(
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            js_mean=("js", "mean"),
            wasserstein_mean=("wasserstein", "mean"),
            mae_fault_points_mean=("mae_fault_points", "mean"),
            mae_clean_points_mean=("mae_clean_points", "mean"),
            false_repaired_clean_fraction_mean=("false_repaired_clean_fraction", "mean"),
            downstream_mean_abs_error_mean=("downstream_mean_abs_error", "mean"),
            downstream_std_abs_error_mean=("downstream_std_abs_error", "mean"),
            downstream_p95_abs_error_mean=("downstream_p95_abs_error", "mean"),
        )
        .sort_values(["repair_method", "dataset_id", "fault_mode", "epsilon"])
    )
    summary["oracle_gap_closed_mean"] = np.nan
    for (dataset_id, fault_mode, epsilon), idx in summary.groupby(["dataset_id", "fault_mode", "epsilon"]).groups.items():
        part = summary.loc[list(idx)]
        no = part[part["repair_method"] == "no_repair"]
        oracle = part[part["repair_method"] == "linear_interpolation"]
        if no.empty or oracle.empty:
            continue
        no_mae = float(no["mae_mean"].iloc[0])
        oracle_mae = float(oracle["mae_mean"].iloc[0])
        denom = no_mae - oracle_mae
        if abs(denom) < 1e-12:
            continue
        summary.loc[list(idx), "oracle_gap_closed_mean"] = (no_mae - summary.loc[list(idx), "mae_mean"]) / denom
    return summary


def plot_detection(summary: pd.DataFrame, out: Path) -> None:
    plot_df = summary[summary["fault_mode"].isin(["iid_stuck", "segment_stuck"])]
    plot_df = plot_df.groupby(["method_category", "method", "fault_mode"], as_index=False).agg(auprc=("auprc_mean", "mean"))
    selected = [
        ("raw_domain_upper_bound", "raw_nonprivate_hmm", "Raw HMM"),
        ("privatized_domain", "privatized_rolling_median", "PM median"),
        ("privatized_domain", "privatized_cusum", "PM CUSUM"),
        ("channel_aware_baseline", "ldp_window_glr", "Window GLR"),
        ("channel_aware_baseline", "ldp_distribution_surprise", "Dist. surprise"),
        ("privacy_aware", "privsaf_mixture", "PrivSAF mix"),
        ("privacy_aware", "privsaf_hmm", "PrivSAF HMM"),
    ]
    colors = {
        "raw_domain_upper_bound": "#4c78a8",
        "privatized_domain": "#f58518",
        "channel_aware_baseline": "#b279a2",
        "privacy_aware": "#54a24b",
        "channel_ablation": "#9d755d",
    }
    faults = [("iid_stuck", "iid stuck-at"), ("segment_stuck", "segment stuck-at")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.8), sharex=True)
    for ax, (fault, title) in zip(axes, faults):
        labels = []
        values = []
        bar_colors = []
        for category, method, label in selected:
            part = plot_df[(plot_df["method_category"] == category) & (plot_df["method"] == method) & (plot_df["fault_mode"] == fault)]
            if part.empty:
                continue
            labels.append(label)
            values.append(float(part["auprc"].iloc[0]))
            bar_colors.append(colors[category])
        y = np.arange(len(labels), dtype=float)
        ax.barh(y, values, color=bar_colors)
        ax.set_yticks(y, labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("AUPRC")
        ax.set_xlim(0.0, 1.0)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("representative method")
    fig.suptitle("Model-Matched Private Stuck-at Detection", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_repair(summary: pd.DataFrame, out: Path) -> None:
    plot_df = summary[summary["fault_mode"].isin(["iid_stuck", "segment_stuck"])]
    plot_df = plot_df.groupby(["repair_method", "fault_mode"], as_index=False).agg(mae=("mae_mean", "mean"))
    methods = ["no_repair", "linear_interpolation", "ldp_window_glr_repair", "privsaf_repair", "privsaf_soft_private_repair"]
    faults = list(plot_df["fault_mode"].drop_duplicates())
    x = np.arange(len(faults))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for i, method in enumerate(methods):
        values = []
        for fault in faults:
            part = plot_df[(plot_df["repair_method"] == method) & (plot_df["fault_mode"] == fault)]
            values.append(float(part["mae"].iloc[0]) if not part.empty else np.nan)
        ax.bar(x + (i - 2) * width, values, width=width, label=method)
    ax.set_xticks(x, faults)
    ax.set_ylabel("MAE to clean sequence")
    ax.set_title("Repair Operators on Stuck-at Faults")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_access_regime_table(runs: pd.DataFrame, out: Path) -> None:
    cols = [
        "method_category",
        "method",
        "access_regime",
        "uses_raw_values",
        "uses_pm_reports",
        "uses_pm_channel",
        "uses_temporal_model",
        "diagnostic_outputs",
        "privacy_comparable",
    ]
    table = runs[cols].drop_duplicates().sort_values(["method_category", "method"])
    table.to_csv(out, index=False)


def plot_channel_ablation(runs: pd.DataFrame, out: Path) -> None:
    selected = [
        ("ldp_window_glr", "Window GLR"),
        ("privsaf_hmm_wrong_epsilon_half", "HMM, eps/2"),
        ("privsaf_hmm", "HMM, calibrated"),
        ("privsaf_hmm_wrong_epsilon_double", "HMM, 2eps"),
    ]
    plot_df = runs[runs["fault_mode"].isin(["segment_stuck", "template_stuck"])]
    plot_df = (
        plot_df[plot_df["method"].isin([m for m, _ in selected])]
        .groupby(["fault_mode", "method"], as_index=False)
        .agg(auprc=("auprc", "mean"), precision_at_budget=("precision_at_fault_budget", "mean"))
    )
    faults = ["segment_stuck", "template_stuck"]
    x = np.arange(len(faults), dtype=float)
    width = 0.18
    colors = ["#7f7f7f", "#9d755d", "#54a24b", "#d37295"]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, (method, label) in enumerate(selected):
        values = []
        for fault in faults:
            part = plot_df[(plot_df["fault_mode"] == fault) & (plot_df["method"] == method)]
            values.append(float(part["auprc"].iloc[0]) if not part.empty else np.nan)
        ax.bar(x + (i - 1.5) * width, values, width=width, label=label, color=colors[i])
    ax.set_xticks(x, ["segment stuck-at", "template stuck-at"])
    ax.set_ylabel("AUPRC")
    ax.set_ylim(0.0, 0.9)
    ax.set_title("Channel Calibration and Persistent Fault Detection")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_representative_timeline(out: Path, d: int, out_d: int, eps: float, seed: int, fault_rate: float, seg_len: int) -> None:
    stream = load_air_quality()
    split = robust_normalize(stream.values, test_len=900)
    alpha = histogram_alpha(split.train, d)
    raw_centers = centers(d)
    stuck_value = float(np.quantile(split.train, 0.95))
    rng = np.random.default_rng(seed)
    faulty, labels, _ = inject_fault(split.test, "segment_stuck", fault_rate, stuck_value, seg_len, rng)
    privatized = pm_sample(faulty, eps, rng)
    m, out_edges = pm_matrix(eps, d, out_d)
    obs = discretize_output(privatized, out_edges)
    scores, _, _ = hmm_infer(obs, m, alpha, fault_rate, seg_len)
    post_mean = posterior_clean_mean(obs, m, alpha, raw_centers)
    mask = topk_mask(scores, int(labels.sum()))
    repaired = apply_masked_interpolation(post_mean, mask, "linear_interpolation", 25)

    if labels.sum() > 0:
        idx = np.where(labels == 1)[0]
        start = max(0, int(idx[0]) - 40)
        end = min(len(labels), int(idx[-1]) + 41)
    else:
        start, end = 0, min(180, len(labels))
    t = np.arange(start, end)
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.1, 0.8, 0.9]})
    axes[0].plot(t, split.test[start:end], color="#4c78a8", lw=1.4, label="clean")
    axes[0].plot(t, faulty[start:end], color="#e45756", lw=1.2, label="faulty")
    axes[0].plot(t, repaired[start:end], color="#54a24b", lw=1.1, label="repaired")
    axes[0].set_ylabel("raw scale")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].scatter(t, privatized[start:end], s=8, color="#9d755d", alpha=0.65)
    axes[1].set_ylabel("PM report")
    axes[1].grid(alpha=0.2)

    axes[2].plot(t, scores[start:end], color="#54a24b", lw=1.5, label="posterior")
    axes[2].fill_between(t, 0, labels[start:end], color="#e45756", alpha=0.18, step="mid", label="true fault")
    axes[2].set_ylabel("fault prob.")
    axes[2].set_xlabel("test index")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    axes[2].grid(alpha=0.2)
    fig.suptitle("Representative PM-LDP Segment Fault Diagnosis")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-dataset PrivSAF baseline and repair evaluation.")
    parser.add_argument("--datasets", default="", help="Comma-separated dataset ids. Default: all must-run datasets.")
    parser.add_argument("--include-optional", action="store_true", help="Include optional gas_drift dataset.")
    parser.add_argument("--epsilons", default="2.0,4.0", help="Comma-separated epsilon values.")
    parser.add_argument("--seeds", default="0,1,2", help="Comma-separated random seeds.")
    parser.add_argument("--fault-modes", default="iid_stuck,segment_stuck,template_stuck,bias", help="Comma-separated fault modes.")
    parser.add_argument("--fault-rate", type=float, default=0.20)
    parser.add_argument("--segment-length", type=int, default=48)
    parser.add_argument("--test-len", type=int, default=2400)
    parser.add_argument("--raw-buckets", type=int, default=32)
    parser.add_argument("--output-buckets", type=int, default=32)
    parser.add_argument("--repair-window", type=int, default=25)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    runs, repairs, metadata = run_grid(args)
    detection_summary = summarize_detection(runs)
    repair_summary = summarize_repair(repairs)
    diagnostics = pd.DataFrame(metadata.pop("theory_diagnostic_rows"))
    diagnosis_outputs = pd.DataFrame(metadata.pop("diagnosis_rows"))
    native_flatlines = pd.DataFrame(metadata.pop("native_flatline_rows"))

    runs.to_csv(RESULTS / "icde_revision_detection_runs.csv", index=False)
    repairs.to_csv(RESULTS / "icde_revision_repair_runs.csv", index=False)
    detection_summary.to_csv(RESULTS / "icde_revision_detection_summary.csv", index=False)
    repair_summary.to_csv(RESULTS / "icde_revision_repair_summary.csv", index=False)
    diagnostics.to_csv(RESULTS / "icde_revision_theory_diagnostics.csv", index=False)
    diagnosis_outputs.to_csv(RESULTS / "icde_revision_diagnostic_outputs.csv", index=False)
    native_flatlines.to_csv(RESULTS / "icde_revision_native_flatline_candidates.csv", index=False)
    save_access_regime_table(runs, RESULTS / "icde_revision_access_regimes.csv")
    (RESULTS / "icde_revision_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    plot_detection(detection_summary, RESULTS / "fig_icde_revision_baselines.png")
    plot_repair(repair_summary, RESULTS / "fig_icde_revision_repair.png")
    plot_channel_ablation(runs, RESULTS / "fig_icde_channel_ablation.png")
    plot_representative_timeline(
        RESULTS / "fig_icde_representative_timeline.png",
        args.raw_buckets,
        args.output_buckets,
        2.0,
        0,
        args.fault_rate,
        args.segment_length,
    )

    print(f"Wrote {RESULTS / 'icde_revision_detection_runs.csv'}")
    print(f"Wrote {RESULTS / 'icde_revision_repair_runs.csv'}")
    print(f"Wrote {RESULTS / 'icde_revision_detection_summary.csv'}")
    print(f"Wrote {RESULTS / 'icde_revision_repair_summary.csv'}")
    print(f"Wrote {RESULTS / 'icde_revision_theory_diagnostics.csv'}")
    print(f"Wrote {RESULTS / 'icde_revision_diagnostic_outputs.csv'}")
    print(f"Wrote {RESULTS / 'icde_revision_native_flatline_candidates.csv'}")
    print(f"Wrote {RESULTS / 'icde_revision_access_regimes.csv'}")
    print(f"Wrote {RESULTS / 'icde_revision_metadata.json'}")
    print(f"Wrote {RESULTS / 'fig_icde_revision_baselines.png'}")
    print(f"Wrote {RESULTS / 'fig_icde_revision_repair.png'}")
    print(f"Wrote {RESULTS / 'fig_icde_channel_ablation.png'}")
    print(f"Wrote {RESULTS / 'fig_icde_representative_timeline.png'}")


if __name__ == "__main__":
    main()
