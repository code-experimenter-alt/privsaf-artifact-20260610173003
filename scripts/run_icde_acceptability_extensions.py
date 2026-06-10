from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from run_icde_revision_grid import (
    RESULTS,
    centers,
    discretize_output,
    forward_backward,
    histogram_alpha,
    load_streams,
    mine_flatline_templates,
    pm_matrix,
    pm_sample,
    raw_bucket_index,
    robust_normalize,
    safe_detection_metrics,
    segment_mask,
)


def recall_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr: float) -> float:
    labels = labels.astype(int)
    negatives = scores[labels == 0]
    if len(negatives) == 0 or np.sum(labels == 1) == 0:
        return float("nan")
    threshold = float(np.quantile(negatives, 1.0 - fpr))
    return float(np.mean(scores[labels == 1] >= threshold))


def precision_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0:
        return float("nan")
    order = np.argsort(scores)[-min(k, len(scores)) :]
    return float(np.mean(labels[order])) if len(order) else float("nan")


def contiguous_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(labels.astype(int)):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(labels)))
    return segments


def event_metrics(labels: np.ndarray, scores: np.ndarray, fpr: float) -> tuple[float, float, float]:
    negatives = scores[labels == 0]
    if len(negatives) == 0:
        return float("nan"), float("nan"), float("nan")
    threshold = float(np.quantile(negatives, 1.0 - fpr))
    pred = scores >= threshold
    segments = contiguous_segments(labels)
    if not segments:
        return float("nan"), float("nan"), float("nan")
    recalls: list[float] = []
    delays: list[float] = []
    for start, end in segments:
        hits = np.where(pred[start:end])[0]
        recalls.append(float(len(hits) > 0))
        delays.append(float(hits[0]) if len(hits) else float(end - start))
    union = np.sum(pred | labels.astype(bool))
    iou = float(np.sum(pred & labels.astype(bool)) / union) if union else float("nan")
    return float(np.mean(recalls)), float(np.median(delays)), iou


def hmm_scan_scores(
    obs: np.ndarray,
    m: np.ndarray,
    alpha: np.ndarray,
    expected_rate: float,
    seg_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    b0 = np.clip(m @ alpha, 1e-12, None)
    p11 = max(0.50, 1.0 - 1.0 / max(seg_len, 2))
    p01 = min(0.20, max(1e-4, expected_rate * (1.0 - p11) / max(1.0 - expected_rate, 1e-6)))
    best_scores = np.zeros(len(obs), dtype=float)
    best_cands = np.zeros(len(obs), dtype=int)
    for cand in range(m.shape[1]):
        b1 = np.clip(m[:, cand], 1e-12, None)
        _, gamma = forward_backward(obs, b0, b1, p01, p11, expected_rate)
        update = gamma > best_scores
        best_scores[update] = gamma[update]
        best_cands[update] = cand
    return best_scores, best_cands


def bin_for_raw_value(value: float, out_edges: np.ndarray) -> int:
    return int(np.clip(np.searchsorted(out_edges, value, side="right") - 1, 0, len(out_edges) - 2))


def run_channel_advantage(
    epsilons: list[float],
    segment_lengths: list[int],
    d: int,
    out_d: int,
    sim_windows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    streams = load_streams(include_optional=False)
    candidate_bins = sorted(set([0, 1, d // 4, d // 2, 3 * d // 4, d - 2, d - 1]))

    for stream in streams:
        split = robust_normalize(stream.values, test_len=2400)
        alpha = histogram_alpha(split.train, d)
        for eps in epsilons:
            m, out_edges = pm_matrix(eps, d, out_d)
            normal = np.clip(m @ alpha, 1e-12, None)
            normal /= normal.sum()
            for cand in candidate_bins:
                fault = np.clip(m[:, cand], 1e-12, None)
                fault /= fault.sum()
                raw_value = float(centers(d)[cand])
                out_bin = bin_for_raw_value(raw_value, out_edges)
                log_ratio = np.log(fault) - np.log(normal)
                kl = float(np.sum(fault * log_ratio))
                chernoff = float(-math.log(np.sum(np.sqrt(fault * normal))))
                bucket_delta = float(fault[out_bin] - normal[out_bin])
                for seg_len in segment_lengths:
                    normal_obs = rng.choice(out_d, size=(sim_windows, seg_len), p=normal)
                    fault_obs = rng.choice(out_d, size=(sim_windows, seg_len), p=fault)
                    normal_llr = log_ratio[normal_obs].sum(axis=1)
                    fault_llr = log_ratio[fault_obs].sum(axis=1)
                    normal_bucket = (normal_obs == out_bin).sum(axis=1)
                    fault_bucket = (fault_obs == out_bin).sum(axis=1)
                    labels = np.concatenate([np.zeros(sim_windows, dtype=int), np.ones(sim_windows, dtype=int)])
                    llr_scores = np.concatenate([normal_llr, fault_llr])
                    bucket_scores = np.concatenate([normal_bucket, fault_bucket])
                    bucket_auc = float(roc_auc_score(labels, bucket_scores))
                    llr_auc = float(roc_auc_score(labels, llr_scores))
                    llr_auprc = float(average_precision_score(labels, llr_scores))
                    rows.append(
                        {
                            "dataset_id": stream.dataset_id,
                            "dataset": stream.dataset_name,
                            "epsilon": eps,
                            "segment_len": seg_len,
                            "stuck_bin": cand,
                            "stuck_value": raw_value,
                            "output_bin_for_value": out_bin,
                            "kl_qs_r": kl,
                            "chernoff_qs_r": chernoff,
                            "bucket_p_under_fault": float(fault[out_bin]),
                            "bucket_p_under_normal": float(normal[out_bin]),
                            "bucket_delta": bucket_delta,
                            "bucket_auc": bucket_auc,
                            "llr_auc": llr_auc,
                            "llr_auprc": llr_auprc,
                            "auroc_gap": llr_auc - bucket_auc,
                            "hard_case_flag": float(bucket_auc <= 0.55 and llr_auc >= 0.70),
                        }
                    )

    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby("epsilon", as_index=False)
        .agg(
            settings=("hard_case_flag", "size"),
            weak_bucket_case_fraction=("hard_case_flag", "mean"),
            median_bucket_auc=("bucket_auc", "median"),
            median_llr_auc=("llr_auc", "median"),
            median_llr_auprc=("llr_auprc", "median"),
            median_kl=("kl_qs_r", "median"),
            median_auroc_gap=("auroc_gap", "median"),
        )
        .sort_values("epsilon")
    )
    return runs, summary


def build_native_windows(series: np.ndarray, d: int, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    templates = mine_flatline_templates(series, min_len=12, max_templates=12)
    positives: list[dict[str, object]] = []
    negatives: list[dict[str, object]] = []
    occupied = np.zeros(len(series), dtype=bool)
    bin_width = 2.0 / d

    for episode_id, item in enumerate(templates):
        start = int(item["start"])
        end = int(item["end"])
        if end <= start or end > len(series):
            continue
        flat_bin = raw_bucket_index(float(item["value"]), d)
        positives.append(
            {
                "episode_id": episode_id,
                "start": start,
                "end": end,
                "length": end - start,
                "flatline_bin": flat_bin,
                "raw_mean": float(np.mean(series[start:end])),
                "raw_std": float(np.std(series[start:end])),
                "raw_range": float(np.max(series[start:end]) - np.min(series[start:end])),
                "source_rule": "native_same-bin_or_small-range_run",
            }
        )
        occupied[start:end] = True

    for pos in positives:
        length = int(pos["length"])
        candidates: list[int] = []
        if length <= 0 or length >= len(series):
            continue
        for start in range(0, len(series) - length + 1):
            end = start + length
            if occupied[start:end].any():
                continue
            segment = series[start:end]
            if float(np.max(segment) - np.min(segment)) <= bin_width:
                continue
            candidates.append(start)
        if len(candidates) < 5:
            candidates = [
                start
                for start in range(0, len(series) - length + 1)
                if not occupied[start : start + length].any()
            ]
        if not candidates:
            continue
        selected = rng.choice(candidates, size=min(5, len(candidates)), replace=False)
        for neg_rank, start in enumerate(selected):
            end = int(start) + length
            negatives.append(
                {
                    "episode_id": int(pos["episode_id"]),
                    "negative_id": neg_rank,
                    "start": int(start),
                    "end": end,
                    "length": length,
                    "flatline_bin": int(pos["flatline_bin"]),
                    "source_rule": "matched_nonflatline_same_length_window",
                }
            )
    return positives, negatives


def run_native_weaklabel(
    epsilons: list[float],
    seeds: list[int],
    d: int,
    out_d: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    negative_rows: list[dict[str, object]] = []
    streams = load_streams(include_optional=False)
    for stream in streams:
        split = robust_normalize(stream.values, test_len=2400)
        series = np.concatenate([split.train, split.validation, split.test])
        alpha = histogram_alpha(split.train, d)
        positives, negatives = build_native_windows(series, d, seed=1729)
        for row in positives:
            enriched = {"dataset_id": stream.dataset_id, "dataset": stream.dataset_name, "target": stream.target, **row}
            candidate_rows.append(enriched)
        for row in negatives:
            negative_rows.append({"dataset_id": stream.dataset_id, "dataset": stream.dataset_name, "target": stream.target, **row})
        if not positives or not negatives:
            continue
        windows = [(1, pos) for pos in positives] + [(0, neg) for neg in negatives]
        for eps in epsilons:
            m, out_edges = pm_matrix(eps, d, out_d)
            identity = np.eye(out_d, d, dtype=float)
            for seed in seeds:
                rng = np.random.default_rng(seed)
                obs = discretize_output(pm_sample(series, eps, rng), out_edges)
                scan_scores, scan_cands = hmm_scan_scores(obs, m, alpha, expected_rate=0.05, seg_len=32)
                wrong_scores, _ = hmm_scan_scores(obs, identity, alpha, expected_rate=0.05, seg_len=32)
                rows_by_method: dict[str, list[dict[str, object]]] = {
                    "privsaf_hmm_scan": [],
                    "privatized_bucket_count": [],
                    "identity_channel_hmm_scan": [],
                }
                for label, window in windows:
                    start = int(window["start"])
                    end = int(window["end"])
                    flat_bin = int(window["flatline_bin"])
                    raw_value = float(centers(d)[flat_bin])
                    out_bin = bin_for_raw_value(raw_value, out_edges)
                    common_cand = int(np.bincount(scan_cands[start:end], minlength=d).argmax())
                    base = {
                        "dataset_id": stream.dataset_id,
                        "dataset": stream.dataset_name,
                        "target": stream.target,
                        "epsilon": eps,
                        "seed": seed,
                        "episode_id": int(window["episode_id"]),
                        "label": int(label),
                        "start": start,
                        "end": end,
                        "length": int(window["length"]),
                        "flatline_bin": flat_bin,
                    }
                    rows_by_method["privsaf_hmm_scan"].append(
                        {
                            **base,
                            "method": "privsaf_hmm_scan",
                            "score": float(np.mean(scan_scores[start:end])),
                            "predicted_bucket": common_cand,
                            "value_hit": float(abs(common_cand - flat_bin) <= 1) if label else float("nan"),
                        }
                    )
                    rows_by_method["privatized_bucket_count"].append(
                        {
                            **base,
                            "method": "privatized_bucket_count",
                            "score": float(np.mean(obs[start:end] == out_bin)),
                            "predicted_bucket": "",
                            "value_hit": float("nan"),
                        }
                    )
                    rows_by_method["identity_channel_hmm_scan"].append(
                        {
                            **base,
                            "method": "identity_channel_hmm_scan",
                            "score": float(np.mean(wrong_scores[start:end])),
                            "predicted_bucket": "",
                            "value_hit": float("nan"),
                        }
                    )
                for rows in rows_by_method.values():
                    event_rows.extend(rows)

    runs = pd.DataFrame(event_rows)
    if runs.empty:
        return runs, pd.DataFrame(), pd.DataFrame(candidate_rows), pd.DataFrame(negative_rows)

    summary_rows: list[dict[str, object]] = []
    for (method, eps), group in runs.groupby(["method", "epsilon"]):
        labels = group["label"].to_numpy(dtype=int)
        scores = group["score"].to_numpy(dtype=float)
        auroc, auprc = safe_detection_metrics(labels, scores)
        positives = group[group["label"] == 1]
        value_hit = (
            pd.to_numeric(positives["value_hit"], errors="coerce").mean()
            if method == "privsaf_hmm_scan" and not positives.empty
            else float("nan")
        )
        summary_rows.append(
            {
                "method": method,
                "epsilon": eps,
                "weak_positive_windows": int(np.sum(labels == 1)),
                "matched_negative_windows": int(np.sum(labels == 0)),
                "median_positive_length": float(positives["length"].median()) if not positives.empty else float("nan"),
                "auroc": auroc,
                "auprc": auprc,
                "recall_at_5pct_fpr": recall_at_fpr(labels, scores, 0.05),
                "precision_at_k": precision_at_k(labels, scores, int(np.sum(labels == 1))),
                "value_hit_rate_pm1": float(value_hit),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["epsilon", "method"])
    return runs, summary, pd.DataFrame(candidate_rows), pd.DataFrame(negative_rows)


def dropout_hmm_scores(
    obs: np.ndarray,
    normal: np.ndarray,
    expected_rate: float,
    seg_len: int,
    rho_n: float,
    rho_candidates: list[float],
) -> tuple[np.ndarray, float]:
    out_d = len(normal)
    p11 = max(0.50, 1.0 - 1.0 / max(seg_len, 2))
    p01 = min(0.20, max(1e-4, expected_rate * (1.0 - p11) / max(1.0 - expected_rate, 1e-6)))
    b0 = np.concatenate([(1.0 - rho_n) * normal, [rho_n]])
    best_ll = -float("inf")
    best_gamma = np.zeros(len(obs), dtype=float)
    best_rho = float("nan")
    for rho_d in rho_candidates:
        if rho_d <= rho_n:
            continue
        b1 = np.concatenate([(1.0 - rho_d) * normal, [rho_d]])
        ll, gamma = forward_backward(obs, b0, b1, p01, p11, expected_rate)
        if ll > best_ll:
            best_ll = ll
            best_gamma = gamma
            best_rho = rho_d
    return best_gamma, best_rho


def rolling_missing_score(obs: np.ndarray, missing_symbol: int, window: int) -> np.ndarray:
    missing = (obs == missing_symbol).astype(float)
    return (
        pd.Series(missing)
        .rolling(window=window, center=True, min_periods=max(2, window // 4))
        .mean()
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )


def run_dropout(
    epsilons: list[float],
    seeds: list[int],
    lengths: list[int],
    rho_ds: list[float],
    d: int,
    out_d: int,
    fault_rate: float,
    rho_n: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    streams = load_streams(include_optional=False)
    for stream in streams:
        split = robust_normalize(stream.values, test_len=2400)
        alpha = histogram_alpha(split.train, d)
        for eps in epsilons:
            m, out_edges = pm_matrix(eps, d, out_d)
            normal = np.clip(m @ alpha, 1e-12, None)
            normal /= normal.sum()
            for seg_len in lengths:
                for rho_d in rho_ds:
                    for seed in seeds:
                        rng = np.random.default_rng(seed)
                        labels = segment_mask(len(split.test), max(1, int(round(fault_rate * len(split.test)))), seg_len, rng)
                        privatized = pm_sample(split.test, eps, rng)
                        obs = discretize_output(privatized, out_edges)
                        missing_probs = np.where(labels == 1, rho_d, rho_n)
                        missing = rng.random(len(obs)) < missing_probs
                        obs_with_missing = obs.copy()
                        obs_with_missing[missing] = out_d
                        method_scores = {
                            "privsaf_dropout_hmm": dropout_hmm_scores(
                                obs_with_missing,
                                normal,
                                expected_rate=float(np.mean(labels)),
                                seg_len=seg_len,
                                rho_n=rho_n,
                                rho_candidates=[0.15, 0.25, 0.50, 0.75, 0.90],
                            )[0],
                            "rolling_missing_fraction": rolling_missing_score(obs_with_missing, out_d, seg_len),
                        }
                        for method, scores in method_scores.items():
                            auroc, auprc = safe_detection_metrics(labels, scores)
                            recall, delay, iou = event_metrics(labels, scores, 0.01)
                            rows.append(
                                {
                                    "dataset_id": stream.dataset_id,
                                    "dataset": stream.dataset_name,
                                    "target": stream.target,
                                    "epsilon": eps,
                                    "seed": seed,
                                    "fault_mode": "dropout",
                                    "dropout_length": seg_len,
                                    "rho_normal": rho_n,
                                    "rho_dropout": rho_d,
                                    "fault_rate": float(np.mean(labels)),
                                    "method": method,
                                    "auroc": auroc,
                                    "auprc": auprc,
                                    "recall_at_1pct_fpr": recall,
                                    "median_detection_delay": delay,
                                    "segment_iou_at_1pct_fpr": iou,
                                }
                            )
    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["method", "rho_dropout", "dropout_length"], as_index=False)
        .agg(
            auroc_mean=("auroc", "mean"),
            auprc_mean=("auprc", "mean"),
            event_recall_mean=("recall_at_1pct_fpr", "mean"),
            median_delay_mean=("median_detection_delay", "mean"),
            segment_iou_mean=("segment_iou_at_1pct_fpr", "mean"),
        )
        .sort_values(["rho_dropout", "dropout_length", "method"])
    )
    return runs, summary


def plot_channel_advantage(runs: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    hard = runs["hard_case_flag"] > 0
    ax.scatter(runs.loc[~hard, "bucket_auc"], runs.loc[~hard, "llr_auc"], s=18, alpha=0.45, color="#4c78a8", label="settings")
    ax.scatter(runs.loc[hard, "bucket_auc"], runs.loc[hard, "llr_auc"], s=26, alpha=0.80, color="#e45756", label="weak bucket cases")
    ax.plot([0.0, 1.0], [0.0, 1.0], color="#444444", linewidth=1.0, linestyle="--")
    ax.axvspan(0.0, 0.55, ymin=0.70, ymax=1.0, color="#f58518", alpha=0.12)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("privatized-bucket detector AUROC")
    ax.set_ylabel("known-channel LLR AUROC")
    ax.set_title("Channel Advantage Diagnostic")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_native_weaklabel(summary: pd.DataFrame, out: Path) -> None:
    plot_df = summary.groupby("method", as_index=False).agg(auprc=("auprc", "mean"), recall=("recall_at_5pct_fpr", "mean"))
    order = ["privatized_bucket_count", "identity_channel_hmm_scan", "privsaf_hmm_scan"]
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("method")
    labels = ["PM bucket", "identity HMM", "PrivSAF scan"]
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(x - 0.18, plot_df["auprc"], width=0.36, color="#72b7b2", label="AUPRC")
    ax.bar(x + 0.18, plot_df["recall"], width=0.36, color="#54a24b", label="Recall@5%FPR")
    ax.set_xticks(x, labels[: len(plot_df)], rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("event-level metric")
    ax.set_title("Native-Source Weak Labels")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_dropout(summary: pd.DataFrame, out: Path) -> None:
    plot_df = summary.groupby(["method", "rho_dropout"], as_index=False).agg(auprc=("auprc_mean", "mean"), recall=("event_recall_mean", "mean"))
    methods = ["rolling_missing_fraction", "privsaf_dropout_hmm"]
    labels = ["Rolling missing", "Dropout HMM"]
    rhos = sorted(plot_df["rho_dropout"].unique())
    x = np.arange(len(rhos))
    width = 0.36
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for idx, method in enumerate(methods):
        vals = []
        for rho in rhos:
            part = plot_df[(plot_df["method"] == method) & (plot_df["rho_dropout"] == rho)]
            vals.append(float(part["auprc"].iloc[0]) if not part.empty else float("nan"))
        ax.bar(x + (idx - 0.5) * width, vals, width=width, label=labels[idx])
    ax.set_xticks(x, [f"{rho:.2f}" for rho in rhos])
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("dropout missing probability in fault state")
    ax.set_ylabel("mean AUPRC")
    ax.set_title("PM-LDP Dropout Fault Extension")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ICDE acceptability extension experiments for PrivSAF.")
    parser.add_argument("--raw-buckets", type=int, default=32)
    parser.add_argument("--output-buckets", type=int, default=32)
    parser.add_argument("--sim-windows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    epsilons = [0.5, 1.0, 2.0]
    seeds = [0, 1, 2]

    channel_runs, channel_summary = run_channel_advantage(
        epsilons=epsilons,
        segment_lengths=[8, 16, 32, 64],
        d=args.raw_buckets,
        out_d=args.output_buckets,
        sim_windows=args.sim_windows,
        seed=args.seed,
    )
    channel_runs.to_csv(RESULTS / "icde_channel_advantage_runs.csv", index=False)
    channel_summary.to_csv(RESULTS / "icde_channel_advantage_summary.csv", index=False)
    plot_channel_advantage(channel_runs, RESULTS / "fig_icde_channel_advantage.png")

    native_runs, native_summary, native_candidates, native_negatives = run_native_weaklabel(
        epsilons=[1.0, 2.0],
        seeds=seeds,
        d=args.raw_buckets,
        out_d=args.output_buckets,
    )
    native_runs.to_csv(RESULTS / "icde_native_weaklabel_runs.csv", index=False)
    native_summary.to_csv(RESULTS / "icde_native_weaklabel_summary.csv", index=False)
    native_candidates.to_csv(RESULTS / "icde_native_weaklabel_candidates.csv", index=False)
    native_negatives.to_csv(RESULTS / "icde_native_weaklabel_matched_negatives.csv", index=False)
    if not native_summary.empty:
        plot_native_weaklabel(native_summary, RESULTS / "fig_icde_native_weaklabel.png")

    dropout_runs, dropout_summary = run_dropout(
        epsilons=epsilons,
        seeds=seeds,
        lengths=[16, 64],
        rho_ds=[0.25, 0.50, 0.75],
        d=args.raw_buckets,
        out_d=args.output_buckets,
        fault_rate=0.15,
        rho_n=0.01,
    )
    dropout_runs.to_csv(RESULTS / "icde_dropout_runs.csv", index=False)
    dropout_summary.to_csv(RESULTS / "icde_dropout_summary.csv", index=False)
    plot_dropout(dropout_summary, RESULTS / "fig_icde_dropout.png")

    print(f"Wrote {RESULTS / 'icde_channel_advantage_runs.csv'} ({len(channel_runs)} rows)")
    print(f"Wrote {RESULTS / 'icde_native_weaklabel_runs.csv'} ({len(native_runs)} rows)")
    print(f"Wrote {RESULTS / 'icde_dropout_runs.csv'} ({len(dropout_runs)} rows)")
    print(f"Wrote {RESULTS / 'fig_icde_channel_advantage.png'}")
    print(f"Wrote {RESULTS / 'fig_icde_native_weaklabel.png'}")
    print(f"Wrote {RESULTS / 'fig_icde_dropout.png'}")


if __name__ == "__main__":
    main()
