from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_icde_acceptability_extensions import precision_at_k, recall_at_fpr
from run_icde_revision_grid import (
    RESULTS,
    ROOT,
    discretize_output,
    forward_backward,
    histogram_alpha,
    pm_matrix,
    pm_sample,
    safe_detection_metrics,
)
from run_iors_stuck_qc_pmldp import (
    DATA_PATH,
    FILL_VALUE,
    GOOD_FLAG,
    load_iors,
    md5sum,
    normalize_with_reference,
)


MISSING_FLAG = 8


def selected_years(
    years: np.ndarray,
    good: np.ndarray,
    missing: np.ndarray,
    min_positive_rows: int,
    min_calibration_rows: int,
    max_years: int,
) -> list[str]:
    candidates: list[tuple[int, str]] = []
    for year in sorted(str(item) for item in np.unique(years)):
        positives = int(np.sum(missing & (years == year)))
        calibration_rows = int(np.sum(good & (years < year)))
        if positives >= min_positive_rows and calibration_rows >= min_calibration_rows:
            candidates.append((positives, year))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return sorted(year for _, year in candidates[:max_years])


def dropout_hmm_scores(
    obs: np.ndarray,
    m: np.ndarray,
    alpha: np.ndarray,
    expected_rate: float,
    segment_length: int,
    rho_normal: float,
    rho_dropout: float,
) -> tuple[np.ndarray, int, float]:
    normal_pm = np.clip(m @ alpha, 1e-12, None)
    normal_pm /= normal_pm.sum()
    b0 = np.r_[normal_pm * (1.0 - rho_normal), rho_normal]
    b1 = np.r_[normal_pm * (1.0 - rho_dropout), rho_dropout]
    b0 = np.clip(b0, 1e-12, None)
    b1 = np.clip(b1, 1e-12, None)
    b0 /= b0.sum()
    b1 /= b1.sum()
    p11 = max(0.50, 1.0 - 1.0 / max(segment_length, 2))
    p01 = min(0.20, max(1e-4, expected_rate * (1.0 - p11) / max(1.0 - expected_rate, 1e-6)))
    _, gamma = forward_backward(obs, b0, b1, p01, p11, expected_rate)
    return gamma, int(m.shape[0]), float(np.mean(gamma))


def rolling_missing_fraction(obs: np.ndarray, bot_symbol: int, window: int) -> np.ndarray:
    missing = (obs == bot_symbol).astype(float)
    return (
        pd.Series(missing)
        .rolling(window=window, center=True, min_periods=max(3, window // 4))
        .mean()
        .bfill()
        .ffill()
        .to_numpy(dtype=float)
    )


def run_year(
    year: str,
    slh: np.ndarray,
    years: np.ndarray,
    good: np.ndarray,
    missing: np.ndarray,
    eps: float,
    seed: int,
    raw_buckets: int,
    output_buckets: int,
    calibration_rows: int,
    segment_length: int,
    normal_missing_rate: float,
    dropout_missing_rate: float,
) -> list[dict[str, object]]:
    test_mask = (years == year) & (good | missing)
    labels = missing[test_mask].astype(int)
    calibration_values = slh[good & (years < year)][-calibration_rows:]
    test_values = slh[test_mask]
    _, test_norm, lo, hi = normalize_with_reference(calibration_values, test_values)
    m, out_edges = pm_matrix(eps, raw_buckets, output_buckets)
    alpha = histogram_alpha(normalize_with_reference(calibration_values, calibration_values)[0], raw_buckets)
    bot = output_buckets
    obs = np.full(len(labels), bot, dtype=int)
    good_positions = labels == 0
    rng = np.random.default_rng(seed)
    obs[good_positions] = discretize_output(pm_sample(test_norm[good_positions], eps, rng), out_edges)
    expected_rate = float(np.clip(np.mean(labels), 0.005, 0.90))

    methods = {
        "missing_symbol": lambda: ((obs == bot).astype(float), bot, float(np.mean(obs == bot))),
        "rolling_missing_fraction": lambda: (
            rolling_missing_fraction(obs, bot, segment_length),
            bot,
            float(np.mean(obs == bot)),
        ),
        "privsaf_dropout_hmm": lambda: dropout_hmm_scores(
            obs,
            m,
            alpha,
            expected_rate,
            segment_length,
            normal_missing_rate,
            dropout_missing_rate,
        ),
    }

    rows: list[dict[str, object]] = []
    for method, fn in methods.items():
        start = time.perf_counter()
        scores, symbol, rhat = fn()
        runtime = time.perf_counter() - start
        auroc, auprc = safe_detection_metrics(labels, scores)
        rows.append(
            {
                "panel": "iors_real_dropout_qc_pmldp",
                "dataset_id": "iors_slh_qc",
                "dataset": "I-ORS sea level height QC",
                "case": year,
                "epsilon": eps,
                "seed": seed,
                "method": method,
                "calibration": "previous_year_good_qc_rows",
                "label_rule": "SLH_QC_8_missing_value",
                "n_calibration": int(len(calibration_values)),
                "n_test": int(len(labels)),
                "dropout_rows": int(labels.sum()),
                "dropout_rate": float(np.mean(labels)),
                "normalization_p01": lo,
                "normalization_p99": hi,
                "auroc": auroc,
                "auprc": auprc,
                "recall_at_5pct_fpr": recall_at_fpr(labels, scores, 0.05),
                "precision_at_k": precision_at_k(labels, scores, int(np.sum(labels))),
                "estimated_symbol": int(symbol),
                "estimated_dropout_rate": float(rhat),
                "runtime_sec": float(runtime),
            }
        )
    return rows


def summarize(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    by_epsilon = (
        runs.groupby(["panel", "epsilon", "method"], as_index=False)
        .agg(
            cases=("auroc", "size"),
            years=("case", "nunique"),
            auroc_mean=("auroc", "mean"),
            auprc_mean=("auprc", "mean"),
            recall_at_5pct_fpr_mean=("recall_at_5pct_fpr", "mean"),
            precision_at_k_mean=("precision_at_k", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
        )
        .sort_values(["epsilon", "auprc_mean"], ascending=[True, False])
    )
    return summary, by_epsilon


def build_inventory(
    path: Path,
    slh: np.ndarray,
    qc: np.ndarray,
    years: np.ndarray,
    good: np.ndarray,
    missing: np.ndarray,
    selected: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inventory_rows: list[dict[str, object]] = []
    for year in sorted(str(item) for item in np.unique(years)):
        year_mask = years == year
        inventory_rows.append(
            {
                "year": year,
                "selected_for_pmldp_panel": int(year in selected),
                "good_rows": int(np.sum(good & year_mask)),
                "missing_value_rows": int(np.sum(missing & year_mask)),
            }
        )
    source_rows = [
        {
            "source_file": str(path.relative_to(ROOT)),
            "md5": md5sum(path),
            "rows": int(len(slh)),
            "good_rows": int(np.sum(good)),
            "missing_value_rows": int(np.sum(missing)),
            "slh_qc_flag_values": "1,2,4,5,7,8",
            "slh_qc_flag_meanings": "good,range,spike,stuck,metadata,missing_value",
            "dropout_interpretation": "SLH_QC=8 is treated as a real missing-value/dropout QC label; if availability is sensitive, the missing symbol needs its own LDP accounting or cover traffic.",
        }
    ]
    return pd.DataFrame(inventory_rows), pd.DataFrame(source_rows)


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slh, qc, years, _ = load_iors(args.data_path)
    good = np.isfinite(slh) & np.isfinite(qc) & (slh != FILL_VALUE) & (qc == GOOD_FLAG)
    missing = np.isfinite(qc) & (qc == MISSING_FLAG)
    if args.years.strip().lower() == "auto":
        case_years = selected_years(
            years,
            good,
            missing,
            args.min_positive_rows,
            args.min_calibration_rows,
            args.max_years,
        )
    else:
        case_years = [item.strip() for item in args.years.split(",") if item.strip()]
    if not case_years:
        raise RuntimeError("No I-ORS years satisfy the dropout selection rules.")

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
                        missing,
                        eps,
                        seed,
                        args.raw_buckets,
                        args.output_buckets,
                        args.calibration_rows,
                        args.segment_length,
                        args.normal_missing_rate,
                        args.dropout_missing_rate,
                    )
                )
    runs = pd.DataFrame(rows)
    summary, by_epsilon = summarize(runs)
    inventory, source = build_inventory(args.data_path, slh, qc, years, good, missing, set(case_years))
    return runs, summary, by_epsilon, inventory, source


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PM-LDP detectors on I-ORS real missing-value QC labels.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--epsilons", default="0.5,1,2,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--years", default="auto")
    parser.add_argument("--max-years", type=int, default=8)
    parser.add_argument("--raw-buckets", type=int, default=32)
    parser.add_argument("--output-buckets", type=int, default=32)
    parser.add_argument("--min-positive-rows", type=int, default=2000)
    parser.add_argument("--min-calibration-rows", type=int, default=1000)
    parser.add_argument("--calibration-rows", type=int, default=50000)
    parser.add_argument("--segment-length", type=int, default=24)
    parser.add_argument("--normal-missing-rate", type=float, default=0.001)
    parser.add_argument("--dropout-missing-rate", type=float, default=0.98)
    parser.add_argument("--output-prefix", default="iors_dropout_qc")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    runs, summary, by_epsilon, inventory, source = run(args)
    runs.to_csv(RESULTS / f"{args.output_prefix}_pmldp_runs.csv", index=False)
    summary.to_csv(RESULTS / f"{args.output_prefix}_pmldp_summary.csv", index=False)
    by_epsilon.to_csv(RESULTS / f"{args.output_prefix}_pmldp_by_epsilon.csv", index=False)
    inventory.to_csv(RESULTS / f"{args.output_prefix}_inventory.csv", index=False)
    source.to_csv(RESULTS / f"{args.output_prefix}_source_metadata.csv", index=False)
    print(f"Wrote {len(runs)} I-ORS dropout-QC PM-LDP rows.")
    print(RESULTS / f"{args.output_prefix}_pmldp_summary.csv")


if __name__ == "__main__":
    main()
