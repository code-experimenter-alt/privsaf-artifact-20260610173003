from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import time
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

from run_icde_channel_baseline_extensions import (
    channel_likelihood_scan,
    channel_llr_cusum,
    generic_pm_hmm,
)
from run_icde_revision_grid import (
    RESULTS,
    ROOT,
    discretize_output,
    histogram_alpha,
    hmm_infer,
    mixture_infer,
    pm_matrix,
    pm_sample,
    safe_detection_metrics,
    window_glr_score,
)


DATA_URL = "https://raw.githubusercontent.com/tmoulahi/Dataset-for-WSN-fault-detection/master/Stuck-at.rar"
DATA_DIR = ROOT / "data" / "wsn_fault_detection" / "raw"
EXTRACTED_DIR = DATA_DIR / "Stuck-at"
RAR_PATH = DATA_DIR / "Stuck-at.rar"
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def ensure_dataset() -> None:
    if sorted(EXTRACTED_DIR.glob("stuck*.xlsx")) and sorted(EXTRACTED_DIR.glob("ystuck*.xlsx")):
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAR_PATH.exists():
        urllib.request.urlretrieve(DATA_URL, RAR_PATH)
    tar = shutil.which("tar")
    if tar is None:
        raise RuntimeError("Need tar on PATH to extract the downloaded RAR archive.")
    subprocess.run([tar, "-xf", str(RAR_PATH), "-C", str(DATA_DIR)], check=True)


def column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref)
    if letters is None:
        return 0
    out = 0
    for char in letters.group(0):
        out = out * 26 + ord(char) - ord("A") + 1
    return out - 1


def shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root.findall("x:si", NS):
        values.append("".join(text.text or "" for text in item.findall(".//x:t", NS)))
    return values


def read_xlsx_numeric(path: Path) -> np.ndarray:
    with zipfile.ZipFile(path) as zf:
        strings = shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    rows: list[list[float]] = []
    for row in root.findall(".//x:sheetData/x:row", NS):
        cells: dict[int, float] = {}
        for cell in row.findall("x:c", NS):
            ref = cell.attrib.get("r", "A1")
            value_node = cell.find("x:v", NS)
            if value_node is None or value_node.text is None:
                value = float("nan")
            elif cell.attrib.get("t") == "s":
                try:
                    text = strings[int(value_node.text)]
                    value = float(text)
                except (ValueError, IndexError):
                    value = float("nan")
            else:
                try:
                    value = float(value_node.text)
                except ValueError:
                    value = float("nan")
            cells[column_index(ref)] = value
        width = max(cells.keys(), default=-1) + 1
        rows.append([cells.get(idx, float("nan")) for idx in range(width)])
    max_width = max(len(row) for row in rows)
    return np.array([row + [float("nan")] * (max_width - len(row)) for row in rows], dtype=float)


def load_case(feature_path: Path) -> tuple[np.ndarray, np.ndarray]:
    label_path = feature_path.with_name("y" + feature_path.name)
    x_raw = read_xlsx_numeric(feature_path)
    y_raw = read_xlsx_numeric(label_path).reshape(-1)
    x = x_raw[np.all(np.isfinite(x_raw), axis=1)]
    y = y_raw[np.isfinite(y_raw)]
    n = min(len(x), len(y))
    if n < 1000:
        raise ValueError(f"{feature_path} has too few aligned rows after parsing.")
    labels = (y[:n] < 0).astype(int)
    return x[:n], labels


def normalize_with_reference(reference: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = reference[np.isfinite(reference)]
    lo, hi = np.nanquantile(ref, [0.01, 0.99])
    scale = max(float(hi - lo), 1e-12)

    def norm(x: np.ndarray) -> np.ndarray:
        return np.clip(2.0 * (x - lo) / scale - 1.0, -1.0, 1.0)

    return norm(ref), norm(values)


def rank_scale(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    if len(values) == 0:
        return values
    if not np.any(np.isfinite(values)) or np.nanmax(values) <= np.nanmin(values):
        return np.zeros(len(values), dtype=float)
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def run_case(
    case_name: str,
    x: np.ndarray,
    labels: np.ndarray,
    eps: float,
    seed: int,
    raw_buckets: int,
    output_buckets: int,
    split_fraction: float,
    segment_length: int,
    generic_iterations: int,
    feature_slice: str,
) -> list[dict[str, object]]:
    split = int(split_fraction * len(labels))
    train_x, train_labels = x[:split], labels[:split]
    test_x, test_labels = x[split:], labels[split:]
    expected_rate = float(np.clip(np.mean(test_labels), 0.01, 0.70))
    m, out_edges = pm_matrix(eps, raw_buckets, output_buckets)
    per_method_scores: dict[str, list[np.ndarray]] = defaultdict(list)
    per_method_runtime: dict[str, float] = defaultdict(float)

    if feature_slice == "all":
        feature_indices = list(range(x.shape[1]))
        aggregation_label = f"max_rank_over_{x.shape[1]}_scalar_features"
    else:
        feature_indices = list(range(max(0, x.shape[1] - 4), x.shape[1]))
        aggregation_label = "max_rank_over_final_timestep_4_scalar_features"

    for feature_idx in feature_indices:
        clean_reference = train_x[train_labels == 0, feature_idx]
        if len(clean_reference) < 50:
            clean_reference = train_x[:, feature_idx]
        train_norm, test_norm = normalize_with_reference(clean_reference, test_x[:, feature_idx])
        alpha = histogram_alpha(train_norm, raw_buckets)
        rng = np.random.default_rng(seed * 1000 + feature_idx)
        obs = discretize_output(pm_sample(test_norm, eps, rng), out_edges)
        methods = {
            "pm_column_likelihood_scan": lambda: channel_likelihood_scan(obs, m, alpha),
            "pm_llr_cusum": lambda: channel_llr_cusum(obs, m, alpha),
            "pm_window_glr": lambda: window_glr_score(obs, m, alpha),
            "generic_pm_hmm": lambda: generic_pm_hmm(
                obs, m, alpha, expected_rate, segment_length, iterations=generic_iterations
            ),
            "privsaf_mixture": lambda: mixture_infer(obs, m, alpha),
            "privsaf_hmm": lambda: hmm_infer(obs, m, alpha, expected_rate, segment_length),
        }
        for method, fn in methods.items():
            t0 = time.perf_counter()
            scores, _, _ = fn()
            per_method_runtime[method] += time.perf_counter() - t0
            per_method_scores[method].append(rank_scale(scores))

    rows: list[dict[str, object]] = []
    for method, feature_scores in sorted(per_method_scores.items()):
        aggregate = np.max(np.vstack(feature_scores), axis=0)
        auroc, auprc = safe_detection_metrics(test_labels, aggregate)
        rows.append(
            {
                "panel": "wsn_labeled_stuck_pmldp",
                "dataset_id": "wsn_stuck_at",
                "dataset": "Prepared WSN stuck-at labels",
                "case": case_name,
                "epsilon": eps,
                "seed": seed,
                "method": method,
                "feature_aggregation": aggregation_label,
                "calibration": "normal_label_rows_in_first_half",
                "n_rows": int(len(labels)),
                "n_test": int(len(test_labels)),
                "fault_rate": float(np.mean(test_labels)),
                "auroc": auroc,
                "auprc": auprc,
                "runtime_sec": float(per_method_runtime[method]),
            }
        )
    return rows


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dataset()
    feature_files = sorted(EXTRACTED_DIR.glob("stuck*.xlsx"))
    epsilons = [float(item) for item in args.epsilons.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []
    for feature_file in feature_files:
        x, labels = load_case(feature_file)
        inventory_rows.append(
            {
                "case": feature_file.stem,
                "source_file": str(feature_file.relative_to(ROOT)),
                "label_file": str(feature_file.with_name("y" + feature_file.name).relative_to(ROOT)),
                "rows": int(len(labels)),
                "features": int(x.shape[1]),
                "fault_rows": int(labels.sum()),
                "fault_rate": float(np.mean(labels)),
            }
        )
        for eps in epsilons:
            for seed in seeds:
                rows.extend(
                    run_case(
                        feature_file.stem,
                        x,
                        labels,
                        eps,
                        seed,
                        args.raw_buckets,
                        args.output_buckets,
                        args.split_fraction,
                        args.segment_length,
                        args.generic_iterations,
                        args.feature_slice,
                    )
                )
    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["panel", "method"], as_index=False)
        .agg(
            cases=("auroc", "size"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            runtime_sec_mean=("runtime_sec", "mean"),
        )
        .sort_values(["auprc_mean", "auroc_mean"], ascending=[False, False])
    )
    inventory = pd.DataFrame(inventory_rows)
    return runs, summary, inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PM-LDP detector panel on public WSN stuck-at labels.")
    parser.add_argument("--epsilons", default="2,4")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--raw-buckets", type=int, default=32)
    parser.add_argument("--output-buckets", type=int, default=32)
    parser.add_argument("--split-fraction", type=float, default=0.5)
    parser.add_argument("--segment-length", type=int, default=8)
    parser.add_argument("--generic-iterations", type=int, default=8)
    parser.add_argument("--feature-slice", choices=["final", "all"], default="final")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    runs, summary, inventory = run(args)
    runs.to_csv(RESULTS / "wsn_stuck_labeled_pmldp_runs.csv", index=False)
    summary.to_csv(RESULTS / "wsn_stuck_labeled_pmldp_summary.csv", index=False)
    inventory.to_csv(RESULTS / "wsn_stuck_labeled_dataset_inventory.csv", index=False)
    print(f"Wrote {len(runs)} WSN stuck-at PM-LDP rows.")
    print(RESULTS / "wsn_stuck_labeled_pmldp_summary.csv")


if __name__ == "__main__":
    main()
