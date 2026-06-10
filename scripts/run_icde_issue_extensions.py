from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from run_icde_revision_grid import (
    RESULTS,
    centers,
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
    posterior_clean_mean,
    raw_bucket_index,
    robust_normalize,
)


ROOT = Path(__file__).resolve().parents[1]


def safe_metrics(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = labels.astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def rolling_features(values: np.ndarray, windows: tuple[int, ...] = (8, 16, 32, 64)) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    feats = [series.to_numpy(dtype=float)]
    feats.append(series.diff().fillna(0.0).abs().to_numpy(dtype=float))
    for window in windows:
        roll = series.rolling(window=window, min_periods=max(3, window // 4))
        feats.append(roll.mean().bfill().ffill().to_numpy(dtype=float))
        feats.append(roll.std().fillna(0.0).bfill().ffill().to_numpy(dtype=float))
        feats.append((series - roll.mean()).abs().bfill().ffill().to_numpy(dtype=float))
    x = np.vstack(feats).T
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def subsequence_features(values: np.ndarray, window: int = 32) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return rolling_features(values, windows=(min(8, max(3, len(values))),))
    rows = []
    for end in range(1, len(values) + 1):
        start = max(0, end - window)
        seg = values[start:end]
        if len(seg) < window:
            seg = np.pad(seg, (window - len(seg), 0), mode="edge")
        seg = seg.astype(float)
        mu = float(np.mean(seg))
        sd = max(float(np.std(seg)), 1e-6)
        rows.append((seg - mu) / sd)
    return np.asarray(rows, dtype=float)


def iforest_scores(train_values: np.ndarray, test_values: np.ndarray, seed: int) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(rolling_features(train_values))
    x_test = scaler.transform(rolling_features(test_values))
    model = IsolationForest(n_estimators=128, contamination=0.08, random_state=seed)
    model.fit(x_train)
    return -model.score_samples(x_test)


def lof_scores(train_values: np.ndarray, test_values: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(rolling_features(train_values))
    x_test = scaler.transform(rolling_features(test_values))
    n_neighbors = min(35, max(5, len(x_train) // 20))
    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=0.08)
    model.fit(x_train)
    return -model.score_samples(x_test)


def subseq_nn_scores(train_values: np.ndarray, test_values: np.ndarray, window: int = 32) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(subsequence_features(train_values, window))
    x_test = scaler.transform(subsequence_features(test_values, window))
    if len(x_train) > 512:
        # Deterministic downsample keeps this matrix-profile-style baseline fast.
        idx = np.linspace(0, len(x_train) - 1, 512).astype(int)
        x_train = x_train[idx]
    model = NearestNeighbors(n_neighbors=min(5, len(x_train)), metric="euclidean")
    model.fit(x_train)
    dist, _ = model.kneighbors(x_test)
    return dist.mean(axis=1)


def modern_baseline_scores(
    train_pm: np.ndarray,
    test_pm: np.ndarray,
    train_postmean: np.ndarray,
    test_postmean: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    return {
        "pm_window_iforest": iforest_scores(train_pm, test_pm, seed),
        "pm_window_lof": lof_scores(train_pm, test_pm),
        "pm_subseq_nn": subseq_nn_scores(train_pm, test_pm),
        "ldp_postmean_iforest": iforest_scores(train_postmean, test_postmean, seed),
        "ldp_postmean_lof": lof_scores(train_postmean, test_postmean),
        "ldp_postmean_subseq_nn": subseq_nn_scores(train_postmean, test_postmean),
    }


def summarize_runs(runs: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        runs.groupby(group_cols, as_index=False)
        .agg(
            cases=("auroc", "size"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            runtime_sec_mean=("runtime_sec", "mean"),
        )
        .sort_values(group_cols)
    )


def streaming_hmm_shortlist(
    obs: np.ndarray,
    m: np.ndarray,
    alpha: np.ndarray,
    expected_rate: float,
    seg_len: int,
    shortlist: int,
    block: int,
) -> tuple[np.ndarray, int]:
    b0 = np.clip(m @ alpha, 1e-12, None)
    out_d, d = m.shape
    p11 = max(0.50, 1.0 - 1.0 / max(seg_len, 2))
    p01 = min(0.20, max(1e-4, expected_rate * (1.0 - p11) / max(1.0 - expected_rate, 1e-6)))
    trans = np.array([[1.0 - p01, p01], [1.0 - p11, p11]], dtype=float)
    phi_by_cand: dict[int, np.ndarray] = {}
    counts = np.zeros(out_d, dtype=float)
    active = list(range(min(shortlist, d)))
    scores = np.zeros(len(obs), dtype=float)
    best_cands = np.zeros(len(obs), dtype=int)

    def rank_candidates() -> list[int]:
        q = counts / max(float(counts.sum()), 1.0)
        best: list[tuple[float, int]] = []
        for cand in range(d):
            b1 = np.clip(m[:, cand], 1e-12, None)
            # Closed-form one-dimensional mixture projection on a coarse grid.
            losses = []
            for r in np.linspace(0.02, 0.65, 32):
                mix = (1.0 - r) * b0 + r * b1
                losses.append(float(np.sum((q - mix) ** 2)))
            best.append((min(losses), cand))
        return [cand for _, cand in sorted(best)[:shortlist]]

    for t, bucket in enumerate(obs):
        counts[bucket] += 1.0
        if t % block == 0 and t > 0:
            active = rank_candidates()
            for cand in active:
                phi_by_cand.setdefault(cand, np.array([1.0 - expected_rate, expected_rate], dtype=float))
        best_gamma = 0.0
        best_cand = active[0]
        for cand in active:
            phi = phi_by_cand.setdefault(cand, np.array([1.0 - expected_rate, expected_rate], dtype=float))
            b1 = np.clip(m[:, cand], 1e-12, None)
            emit = np.array([b0[bucket], b1[bucket]], dtype=float)
            phi = (phi @ trans) * emit
            phi /= max(float(phi.sum()), 1e-12)
            phi_by_cand[cand] = phi
            if phi[1] > best_gamma:
                best_gamma = float(phi[1])
                best_cand = cand
        scores[t] = best_gamma
        best_cands[t] = best_cand
    return scores, int(pd.Series(best_cands).mode().iloc[0])


def run_modern_baselines(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    streams = load_streams(include_optional=False)
    d = args.raw_buckets
    out_d = args.output_buckets
    raw_centers = centers(d)
    eps_values = [float(x) for x in args.epsilons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    fault_modes = ["iid_stuck", "segment_stuck", "template_stuck"]

    for stream in streams:
        split = robust_normalize(stream.values, test_len=args.test_len)
        alpha_train = histogram_alpha(split.train, d)
        templates = mine_flatline_templates(np.concatenate([split.train, split.validation, split.test]))
        stuck_values = [float(x) for x in np.quantile(split.train, [0.05, 0.95])]
        for eps in eps_values:
            m, out_edges = pm_matrix(eps, d, out_d)
            for seed in seeds:
                rng_train = np.random.default_rng(100000 + seed)
                train_pm = pm_sample(split.train, eps, rng_train)
                train_obs = discretize_output(train_pm, out_edges)
                train_postmean = posterior_clean_mean(train_obs, m, alpha_train, raw_centers)
                for fault_mode in fault_modes:
                    mode_values = [float("nan")] if fault_mode == "template_stuck" else stuck_values
                    for stuck_value in mode_values:
                        rng = np.random.default_rng(seed)
                        if fault_mode == "template_stuck":
                            faulty, labels, params = inject_template_stuck(split.test, args.fault_rate, templates, rng)
                        else:
                            faulty, labels, params = inject_fault(split.test, fault_mode, args.fault_rate, stuck_value, args.segment_length, rng)
                        test_pm = pm_sample(faulty, eps, rng)
                        obs = discretize_output(test_pm, out_edges)
                        test_postmean = posterior_clean_mean(obs, m, alpha_train, raw_centers)
                        base = {
                            "panel": "controlled_pre_pm_faults",
                            "dataset_id": stream.dataset_id,
                            "dataset": stream.dataset_name,
                            "epsilon": eps,
                            "seed": seed,
                            "fault_mode": fault_mode,
                            "evidence_type": "template_based_injection" if fault_mode == "template_stuck" else "controlled_injection",
                            "n_test": int(len(labels)),
                            "fault_rate": float(np.mean(labels)),
                            "stuck_value": params.get("stuck_value", ""),
                        }
                        score_map = modern_baseline_scores(train_pm, test_pm, train_postmean, test_postmean, seed)
                        t0 = time.perf_counter()
                        if fault_mode == "iid_stuck":
                            priv_scores, _, _ = mixture_infer(obs, m, alpha_train)
                            priv_method = "privsaf_matched_mixture"
                        else:
                            priv_scores, _, _ = hmm_infer(obs, m, alpha_train, args.fault_rate, args.segment_length)
                            priv_method = "privsaf_matched_hmm"
                        score_map[priv_method] = priv_scores
                        priv_runtime = time.perf_counter() - t0
                        for method, scores in score_map.items():
                            t0 = time.perf_counter()
                            # Scores are already computed; keep a measured zero-plus overhead for non-PrivSAF methods.
                            runtime = priv_runtime if method.startswith("privsaf") else time.perf_counter() - t0
                            auroc, auprc = safe_metrics(labels, scores)
                            row = dict(base)
                            row.update({"method": method, "auroc": auroc, "auprc": auprc, "runtime_sec": runtime})
                            rows.append(row)
    runs = pd.DataFrame(rows)
    return runs, summarize_runs(runs, ["panel", "method", "fault_mode"])


def load_nab_labeled_series(path: Path, windows: list[list[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
    labels = np.zeros(len(df), dtype=int)
    parsed_windows = []
    for start_s, end_s in windows:
        start = pd.to_datetime(start_s)
        end = pd.to_datetime(end_s)
        parsed_windows.append((start, end))
        labels[((df["timestamp"] >= start) & (df["timestamp"] <= end)).to_numpy()] = 1
    first_anom = min(start for start, _ in parsed_windows)
    train_end = int(np.searchsorted(df["timestamp"].to_numpy(), np.datetime64(first_anom), side="left"))
    train_end = max(200, min(train_end, len(df) // 2))
    values = df["value"].to_numpy(dtype=float)
    lo, hi = np.nanquantile(values[:train_end], [0.01, 0.99])
    scale = max(float(hi - lo), 1e-12)
    norm = np.clip(2.0 * (values - lo) / scale - 1.0, -1.0, 1.0)
    return norm[:train_end], norm[train_end:], labels[train_end:], train_end


def run_native_labeled_nab(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_path = ROOT / "data" / "nab" / "raw" / "combined_windows.json"
    with label_path.open("r", encoding="utf-8") as fin:
        windows_by_file = json.load(fin)
    files = [
        "realKnownCause/machine_temperature_system_failure.csv",
        "realKnownCause/ambient_temperature_system_failure.csv",
    ]
    d = args.raw_buckets
    out_d = args.output_buckets
    raw_centers = centers(d)
    eps_values = [float(x) for x in args.epsilons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    rows: list[dict[str, object]] = []
    for rel in files:
        path = ROOT / "data" / "nab" / "raw" / rel
        train, test, labels, train_rows = load_nab_labeled_series(path, windows_by_file[rel])
        alpha = histogram_alpha(train, d)
        event_count = int(len(windows_by_file[rel]))
        seg_len = max(12, int(np.mean([len(test) * max(float(np.mean(labels)), 1e-4) / max(event_count, 1)])))
        expected_rate = max(float(np.mean(labels)), 1e-4)
        for eps in eps_values:
            m, out_edges = pm_matrix(eps, d, out_d)
            for seed in seeds:
                rng_train = np.random.default_rng(200000 + seed)
                rng_test = np.random.default_rng(300000 + seed)
                train_pm = pm_sample(train, eps, rng_train)
                test_pm = pm_sample(test, eps, rng_test)
                train_obs = discretize_output(train_pm, out_edges)
                test_obs = discretize_output(test_pm, out_edges)
                train_postmean = posterior_clean_mean(train_obs, m, alpha, raw_centers)
                test_postmean = posterior_clean_mean(test_obs, m, alpha, raw_centers)
                score_map = modern_baseline_scores(train_pm, test_pm, train_postmean, test_postmean, seed)
                t0 = time.perf_counter()
                priv_scores, _, _ = hmm_infer(test_obs, m, alpha, expected_rate, seg_len)
                priv_runtime = time.perf_counter() - t0
                score_map["privsaf_hmm_scan"] = priv_scores
                for method, scores in score_map.items():
                    auroc, auprc = safe_metrics(labels, scores)
                    rows.append(
                        {
                            "panel": "nab_native_labeled_system_failure",
                            "dataset_id": Path(rel).stem,
                            "dataset": rel,
                            "epsilon": eps,
                            "seed": seed,
                            "method": method,
                            "auroc": auroc,
                            "auprc": auprc,
                            "runtime_sec": priv_runtime if method == "privsaf_hmm_scan" else 0.0,
                            "train_rows": train_rows,
                            "test_rows": int(len(test)),
                            "positive_frames": int(labels.sum()),
                            "official_windows": event_count,
                            "label_source": "NAB combined_windows.json",
                        }
                    )
    runs = pd.DataFrame(rows)
    return runs, summarize_runs(runs, ["panel", "method"])


def run_streaming(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    d = args.raw_buckets
    out_d = args.output_buckets
    eps_values = [float(x) for x in args.epsilons.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    streams = load_streams(include_optional=False)
    for stream in streams:
        split = robust_normalize(stream.values, test_len=args.test_len)
        alpha = histogram_alpha(split.train, d)
        stuck_values = [float(x) for x in np.quantile(split.train, [0.05, 0.95])]
        for eps in eps_values:
            m, out_edges = pm_matrix(eps, d, out_d)
            for seed in seeds:
                for stuck_value in stuck_values:
                    rng = np.random.default_rng(seed)
                    faulty, labels, _ = inject_fault(split.test, "segment_stuck", args.fault_rate, stuck_value, args.segment_length, rng)
                    obs = discretize_output(pm_sample(faulty, eps, rng), out_edges)
                    t0 = time.perf_counter()
                    full_scores, full_cand, _ = hmm_infer(obs, m, alpha, args.fault_rate, args.segment_length)
                    full_runtime = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    stream_scores, stream_cand = streaming_hmm_shortlist(obs, m, alpha, args.fault_rate, args.segment_length, args.shortlist, args.block)
                    stream_runtime = time.perf_counter() - t0
                    for method, scores, cand, runtime in [
                        ("privsaf_full_hmm", full_scores, full_cand, full_runtime),
                        (f"privsaf_streaming_top{args.shortlist}", stream_scores, stream_cand, stream_runtime),
                    ]:
                        auroc, auprc = safe_metrics(labels, scores)
                        rows.append(
                            {
                                "panel": "streaming_segment_stuck",
                                "dataset_id": stream.dataset_id,
                                "dataset": stream.dataset_name,
                                "epsilon": eps,
                                "seed": seed,
                                "method": method,
                                "auroc": auroc,
                                "auprc": auprc,
                                "runtime_sec": runtime,
                                "runtime_ms_per_frame": 1000.0 * runtime / len(obs),
                                "estimated_bucket": cand,
                                "n_test": int(len(labels)),
                            }
                        )
    runs = pd.DataFrame(rows)
    summary = summarize_runs(runs, ["panel", "method"])
    per_frame = runs.groupby(["panel", "method"], as_index=False).agg(runtime_ms_per_frame_mean=("runtime_ms_per_frame", "mean"))
    return runs, summary.merge(per_frame, on=["panel", "method"], how="left")


def repair_downstream_summary() -> pd.DataFrame:
    path = RESULTS / "icde_revision_repair_runs.csv"
    repairs = pd.read_csv(path)
    repairs = repairs[repairs["fault_mode"].isin(["iid_stuck", "segment_stuck"])]
    keep = ["no_repair", "linear_interpolation", "ldp_window_glr_repair", "privsaf_repair"]
    repairs = repairs[repairs["repair_method"].isin(keep)]
    return (
        repairs.groupby("repair_method", as_index=False)
        .agg(
            cases=("mae", "size"),
            mae=("mae", "mean"),
            fault_mae=("mae_fault_points", "mean"),
            clean_mae=("mae_clean_points", "mean"),
            false_clean_repair=("false_repaired_clean_fraction", "mean"),
            downstream_mean_error=("downstream_mean_abs_error", "mean"),
            downstream_std_error=("downstream_std_abs_error", "mean"),
            downstream_p95_error=("downstream_p95_abs_error", "mean"),
        )
        .sort_values("repair_method")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted ICDE issue-response extensions.")
    parser.add_argument("--epsilons", default="2,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--raw-buckets", type=int, default=32)
    parser.add_argument("--output-buckets", type=int, default=32)
    parser.add_argument("--test-len", type=int, default=2400)
    parser.add_argument("--fault-rate", type=float, default=0.20)
    parser.add_argument("--segment-length", type=int, default=48)
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--block", type=int, default=64)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    modern_runs, modern_summary = run_modern_baselines(args)
    native_runs, native_summary = run_native_labeled_nab(args)
    streaming_runs, streaming_summary = run_streaming(args)
    repair_summary = repair_downstream_summary()

    modern_runs.to_csv(RESULTS / "icde_issue_modern_baselines_runs.csv", index=False)
    modern_summary.to_csv(RESULTS / "icde_issue_modern_baselines_summary.csv", index=False)
    native_runs.to_csv(RESULTS / "icde_issue_native_labeled_runs.csv", index=False)
    native_summary.to_csv(RESULTS / "icde_issue_native_labeled_summary.csv", index=False)
    streaming_runs.to_csv(RESULTS / "icde_issue_streaming_runs.csv", index=False)
    streaming_summary.to_csv(RESULTS / "icde_issue_streaming_summary.csv", index=False)
    repair_summary.to_csv(RESULTS / "icde_issue_repair_downstream_summary.csv", index=False)

    print(f"Wrote {len(modern_runs)} modern-baseline rows.")
    print(f"Wrote {len(native_runs)} native labeled NAB rows.")
    print(f"Wrote {len(streaming_runs)} streaming rows.")
    print(f"Wrote {RESULTS / 'icde_issue_repair_downstream_summary.csv'}.")


if __name__ == "__main__":
    main()
