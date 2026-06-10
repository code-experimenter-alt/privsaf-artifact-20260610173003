from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from run_icde_acceptability_extensions import precision_at_k, recall_at_fpr
from run_icde_channel_baseline_extensions import (
    channel_likelihood_scan,
    channel_llr_cusum,
    generic_pm_hmm,
)
from run_icde_revision_grid import (
    RESULTS,
    ROOT,
    discretize_output,
    forward_backward,
    histogram_alpha,
    mixture_infer,
    pm_matrix,
    pm_sample,
    safe_detection_metrics,
    window_glr_score,
)


DATA_PATH = ROOT / "data" / "iors_slh" / "I-ORS_2003_2022_D_SLH.nc"
GOOD_FLAG = 1
STUCK_FLAG = 5
FILL_VALUE = -99999.0


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        item = value.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace")
        return str(item)
    return str(value)


def load_iors(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Download it from the KIOST DOI bitstream before running this panel."
        )
    with h5py.File(path, "r") as handle:
        slh = handle["SLH"][:].astype(float)
        qc = handle["SLH_QC"][:].astype(float)
        time_days = handle["TIME"][:].astype(float)
        attrs = {key: decode_attr(value) for key, value in handle.attrs.items()}
    base = np.datetime64("1950-01-01T00:00")
    timestamps = base + (time_days * 24 * 60).astype("timedelta64[m]")
    years = np.asarray([str(item)[:4] for item in timestamps], dtype=object)
    return slh, qc, years, attrs


def persistent_stuck_mask(stuck: np.ndarray, min_episode_length: int) -> np.ndarray:
    out = np.zeros_like(stuck, dtype=bool)
    indices = np.flatnonzero(stuck)
    if len(indices) == 0:
        return out
    breaks = np.where(np.diff(indices) > 1)[0]
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    for start, end in zip(starts, ends):
        if end - start + 1 >= min_episode_length:
            out[start : end + 1] = True
    return out


def episode_lengths(mask: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return np.asarray([], dtype=int)
    breaks = np.where(np.diff(indices) > 1)[0]
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    return (ends - starts + 1).astype(int)


def normalize_with_reference(reference: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    ref = reference[np.isfinite(reference)]
    lo, hi = np.nanquantile(ref, [0.01, 0.99])
    scale = max(float(hi - lo), 1e-12)

    def norm(x: np.ndarray) -> np.ndarray:
        return np.clip(2.0 * (x - lo) / scale - 1.0, -1.0, 1.0)

    return norm(ref), norm(values), float(lo), float(hi)


def fast_hmm_infer(
    obs: np.ndarray,
    m: np.ndarray,
    alpha: np.ndarray,
    expected_rate: float,
    seg_len: int,
) -> tuple[np.ndarray, int, float]:
    """Exact full-candidate PrivSAF HMM search with vectorized candidate scoring."""
    b0 = np.clip(m @ alpha, 1e-12, None)
    p11 = max(0.50, 1.0 - 1.0 / max(seg_len, 2))
    p01 = min(0.20, max(1e-4, expected_rate * (1.0 - p11) / max(1.0 - expected_rate, 1e-6)))
    log_a00 = np.log(1.0 - p01)
    log_a01 = np.log(p01)
    log_a10 = np.log(1.0 - p11)
    log_a11 = np.log(p11)
    log_b0 = np.log(np.clip(b0[obs], 1e-15, None))
    log_b1 = np.log(np.clip(m[obs, :], 1e-15, None))

    d = m.shape[1]
    log_alpha0 = np.full(d, np.log(max(1.0 - expected_rate, 1e-15))) + log_b0[0]
    log_alpha1 = np.full(d, np.log(max(expected_rate, 1e-15))) + log_b1[0]
    scale = np.logaddexp(log_alpha0, log_alpha1)
    log_alpha0 -= scale
    log_alpha1 -= scale
    log_likelihood = scale.copy()

    for t in range(1, len(obs)):
        next0 = log_b0[t] + np.logaddexp(log_alpha0 + log_a00, log_alpha1 + log_a10)
        next1 = log_b1[t] + np.logaddexp(log_alpha0 + log_a01, log_alpha1 + log_a11)
        scale = np.logaddexp(next0, next1)
        log_alpha0 = next0 - scale
        log_alpha1 = next1 - scale
        log_likelihood += scale

    cand = int(np.argmax(log_likelihood))
    _, gamma = forward_backward(obs, b0, np.clip(m[:, cand], 1e-12, None), p01, p11, expected_rate)
    return gamma, cand, float(np.mean(gamma))


def fast_hmm_scan_scores(
    obs: np.ndarray,
    m: np.ndarray,
    alpha: np.ndarray,
    expected_rate: float,
    seg_len: int,
) -> tuple[np.ndarray, int, float]:
    """Per-frame best-candidate HMM posterior scan for streams with multiple stuck levels."""
    n = len(obs)
    d = m.shape[1]
    b0 = np.clip(m @ alpha, 1e-12, None)
    p11 = max(0.50, 1.0 - 1.0 / max(seg_len, 2))
    p01 = min(0.20, max(1e-4, expected_rate * (1.0 - p11) / max(1.0 - expected_rate, 1e-6)))
    log_a00 = np.log(1.0 - p01)
    log_a01 = np.log(p01)
    log_a10 = np.log(1.0 - p11)
    log_a11 = np.log(p11)
    log_b0 = np.log(np.clip(b0[obs], 1e-15, None))
    log_b1 = np.log(np.clip(m[obs, :], 1e-15, None))

    f0 = np.empty((n, d), dtype=float)
    f1 = np.empty((n, d), dtype=float)
    f0[0, :] = np.log(max(1.0 - expected_rate, 1e-15)) + log_b0[0]
    f1[0, :] = np.log(max(expected_rate, 1e-15)) + log_b1[0]
    scale = np.logaddexp(f0[0], f1[0])
    f0[0] -= scale
    f1[0] -= scale

    for t in range(1, n):
        next0 = log_b0[t] + np.logaddexp(f0[t - 1] + log_a00, f1[t - 1] + log_a10)
        next1 = log_b1[t] + np.logaddexp(f0[t - 1] + log_a01, f1[t - 1] + log_a11)
        scale = np.logaddexp(next0, next1)
        f0[t] = next0 - scale
        f1[t] = next1 - scale

    beta0 = np.zeros(d, dtype=float)
    beta1 = np.zeros(d, dtype=float)
    scores = np.zeros(n, dtype=float)
    candidates = np.zeros(n, dtype=int)
    for t in range(n - 1, -1, -1):
        gamma_log = f1[t] + beta1
        denom = np.logaddexp(f0[t] + beta0, gamma_log)
        gamma = np.exp(gamma_log - denom)
        cand = int(np.argmax(gamma))
        scores[t] = float(gamma[cand])
        candidates[t] = cand
        if t > 0:
            next_beta0 = np.logaddexp(log_a00 + log_b0[t] + beta0, log_a01 + log_b1[t] + beta1)
            next_beta1 = np.logaddexp(log_a10 + log_b0[t] + beta0, log_a11 + log_b1[t] + beta1)
            scale = np.logaddexp(next_beta0, next_beta1)
            beta0 = next_beta0 - scale
            beta1 = next_beta1 - scale

    mode_cand = int(np.bincount(candidates, minlength=d).argmax()) if len(candidates) else 0
    return scores, mode_cand, float(np.mean(scores)) if len(scores) else 0.0


def selected_years(
    years: np.ndarray,
    good: np.ndarray,
    persistent_stuck: np.ndarray,
    min_positive_rows: int,
    min_calibration_rows: int,
) -> list[str]:
    out: list[str] = []
    for year in sorted(str(item) for item in np.unique(years)):
        test_positive = int(np.sum(persistent_stuck & (years == year)))
        calibration_rows = int(np.sum(good & (years < year)))
        if test_positive >= min_positive_rows and calibration_rows >= min_calibration_rows:
            out.append(year)
    return out


def run_year(
    year: str,
    slh: np.ndarray,
    years: np.ndarray,
    good: np.ndarray,
    persistent_stuck: np.ndarray,
    eps: float,
    seed: int,
    raw_buckets: int,
    output_buckets: int,
    calibration_rows: int,
    segment_length: int,
    generic_iterations: int,
) -> list[dict[str, object]]:
    test_mask = (years == year) & (good | persistent_stuck)
    calibration_values = slh[good & (years < year)][-calibration_rows:]
    test_values = slh[test_mask]
    labels = persistent_stuck[test_mask].astype(int)
    train_norm, test_norm, lo, hi = normalize_with_reference(calibration_values, test_values)
    expected_rate = float(np.clip(np.mean(labels), 0.005, 0.70))
    m, out_edges = pm_matrix(eps, raw_buckets, output_buckets)
    alpha = histogram_alpha(train_norm, raw_buckets)
    rng = np.random.default_rng(seed)
    obs = discretize_output(pm_sample(test_norm, eps, rng), out_edges)

    methods = {
        "pm_column_likelihood_scan": lambda: channel_likelihood_scan(obs, m, alpha),
        "pm_llr_cusum": lambda: channel_llr_cusum(obs, m, alpha),
        "pm_window_glr": lambda: window_glr_score(obs, m, alpha),
        "generic_pm_hmm": lambda: generic_pm_hmm(
            obs, m, alpha, expected_rate, segment_length, iterations=generic_iterations
        ),
        "privsaf_mixture": lambda: mixture_infer(obs, m, alpha),
        "privsaf_hmm_global": lambda: fast_hmm_infer(obs, m, alpha, expected_rate, segment_length),
        "privsaf_hmm_scan": lambda: fast_hmm_scan_scores(obs, m, alpha, expected_rate, segment_length),
    }

    rows: list[dict[str, object]] = []
    for method, fn in methods.items():
        start = time.perf_counter()
        scores, cand, rhat = fn()
        runtime = time.perf_counter() - start
        auroc, auprc = safe_detection_metrics(labels, scores)
        rows.append(
            {
                "panel": "iors_real_stuck_qc_pmldp",
                "dataset_id": "iors_slh_qc",
                "dataset": "I-ORS sea level height QC",
                "case": year,
                "epsilon": eps,
                "seed": seed,
                "method": method,
                "calibration": "previous_year_good_qc_rows",
                "label_rule": f"SLH_QC_5_episode_length_ge_{segment_length}",
                "n_calibration": int(len(calibration_values)),
                "n_test": int(len(labels)),
                "fault_rows": int(labels.sum()),
                "fault_rate": float(np.mean(labels)),
                "normalization_p01": lo,
                "normalization_p99": hi,
                "auroc": auroc,
                "auprc": auprc,
                "recall_at_5pct_fpr": recall_at_fpr(labels, scores, 0.05),
                "precision_at_k": precision_at_k(labels, scores, int(np.sum(labels))),
                "estimated_bucket": int(cand),
                "estimated_fault_rate": float(rhat),
                "runtime_sec": float(runtime),
            }
        )
    return rows


def build_inventory(
    path: Path,
    slh: np.ndarray,
    qc: np.ndarray,
    years: np.ndarray,
    good: np.ndarray,
    stuck: np.ndarray,
    persistent_stuck: np.ndarray,
    selected: set[str],
    min_episode_length: int,
    attrs: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory_rows: list[dict[str, object]] = []
    for year in sorted(str(item) for item in np.unique(years)):
        year_mask = years == year
        year_stuck = stuck & year_mask
        year_persistent = persistent_stuck & year_mask
        lengths = episode_lengths(year_stuck)
        inventory_rows.append(
            {
                "year": year,
                "selected_for_pmldp_panel": int(year in selected),
                "good_rows": int(np.sum(good & year_mask)),
                "all_stuck_rows": int(np.sum(year_stuck)),
                "persistent_stuck_rows": int(np.sum(year_persistent)),
                "short_stuck_rows_excluded": int(np.sum(year_stuck) - np.sum(year_persistent)),
                "persistent_min_episode_length": int(min_episode_length),
                "stuck_episodes": int(len(lengths)),
                "median_stuck_episode_length": float(np.median(lengths)) if len(lengths) else float("nan"),
                "max_stuck_episode_length": int(lengths.max()) if len(lengths) else 0,
            }
        )
    source_rows = [
        {
            "source_file": str(path.relative_to(ROOT)),
            "md5": md5sum(path),
            "rows": int(len(slh)),
            "finite_slh_rows": int(np.sum(np.isfinite(slh) & (slh != FILL_VALUE))),
            "good_rows": int(np.sum(good)),
            "all_stuck_rows": int(np.sum(stuck)),
            "persistent_stuck_rows": int(np.sum(persistent_stuck)),
            "persistent_min_episode_length": int(min_episode_length),
            "title": attrs.get("title", ""),
            "citation": attrs.get("citation", ""),
            "time_coverage_start": attrs.get("time_coverage_start", ""),
            "time_coverage_end": attrs.get("time_coverage_end", ""),
            "slh_qc_flag_values": "1,2,4,5,7,8",
            "slh_qc_flag_meanings": "good,range,spike,stuck,metadata,missing_value",
        }
    ]
    return pd.DataFrame(inventory_rows), pd.DataFrame(source_rows)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slh, qc, years, attrs = load_iors(args.data_path)
    base_valid = np.isfinite(slh) & np.isfinite(qc) & (slh != FILL_VALUE)
    good = base_valid & (qc == GOOD_FLAG)
    stuck = base_valid & (qc == STUCK_FLAG)
    persistent_stuck = persistent_stuck_mask(stuck, args.min_episode_length)

    if args.years.strip().lower() == "auto":
        case_years = selected_years(
            years, good, persistent_stuck, args.min_positive_rows, args.min_calibration_rows
        )
    else:
        case_years = [item.strip() for item in args.years.split(",") if item.strip()]
    if not case_years:
        raise RuntimeError("No I-ORS years satisfy the selection rules.")

    epsilons = [float(item) for item in args.epsilons.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    rows: list[dict[str, object]] = []
    for year in case_years:
        for eps in epsilons:
            for seed in seeds:
                rows.extend(
                    run_year(
                        year,
                        slh,
                        years,
                        good,
                        persistent_stuck,
                        eps,
                        seed,
                        args.raw_buckets,
                        args.output_buckets,
                        args.calibration_rows,
                        args.min_episode_length,
                        args.generic_iterations,
                    )
                )

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["panel", "method"], as_index=False)
        .agg(
            cases=("auroc", "size"),
            years=("case", "nunique"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            recall_at_5pct_fpr_mean=("recall_at_5pct_fpr", "mean"),
            precision_at_k_mean=("precision_at_k", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
        )
        .sort_values(["auprc_mean", "auroc_mean"], ascending=[False, False])
    )
    summary_by_epsilon = (
        runs.groupby(["panel", "epsilon", "method"], as_index=False)
        .agg(
            cases=("auroc", "size"),
            years=("case", "nunique"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            recall_at_5pct_fpr_mean=("recall_at_5pct_fpr", "mean"),
            precision_at_k_mean=("precision_at_k", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
        )
        .sort_values(["epsilon", "auprc_mean", "auroc_mean"], ascending=[True, False, False])
    )
    inventory, source = build_inventory(
        args.data_path,
        slh,
        qc,
        years,
        good,
        stuck,
        persistent_stuck,
        set(case_years),
        args.min_episode_length,
        attrs,
    )
    return runs, summary, summary_by_epsilon, inventory, source


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PM-LDP detectors on I-ORS real stuck QC labels.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--epsilons", default="2,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--years", default="auto")
    parser.add_argument("--raw-buckets", type=int, default=32)
    parser.add_argument("--output-buckets", type=int, default=32)
    parser.add_argument("--min-episode-length", type=int, default=8)
    parser.add_argument("--min-positive-rows", type=int, default=500)
    parser.add_argument("--min-calibration-rows", type=int, default=1000)
    parser.add_argument("--calibration-rows", type=int, default=50000)
    parser.add_argument("--generic-iterations", type=int, default=8)
    parser.add_argument("--output-prefix", default="iors_stuck_qc")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    runs, summary, summary_by_epsilon, inventory, source = run(args)
    runs.to_csv(RESULTS / f"{args.output_prefix}_pmldp_runs.csv", index=False)
    summary.to_csv(RESULTS / f"{args.output_prefix}_pmldp_summary.csv", index=False)
    summary_by_epsilon.to_csv(RESULTS / f"{args.output_prefix}_pmldp_by_epsilon.csv", index=False)
    inventory.to_csv(RESULTS / f"{args.output_prefix}_inventory.csv", index=False)
    source.to_csv(RESULTS / f"{args.output_prefix}_source_metadata.csv", index=False)
    print(f"Wrote {len(runs)} I-ORS stuck-QC PM-LDP rows.")
    print(RESULTS / f"{args.output_prefix}_pmldp_summary.csv")


if __name__ == "__main__":
    main()
