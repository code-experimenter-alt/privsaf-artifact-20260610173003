# PrivSAF Student Experiment Manual

This manual turns the execution checklist into a project-local runbook. It is written for students who need to reproduce the ICDE-style evidence without first understanding every HMM detail.

## Mental Model

PrivSAF studies private telemetry cleaning after local differential privacy (LDP):

```text
clean scalar stream
  -> inject or observe stuck-at / flatline / availability labels
  -> client-side PM-LDP or declared scalar channel
  -> server sees only private report buckets + channel matrix M
  -> operator emits posterior scores, stuck bucket, action, repair mask, privacy/provenance records
```

The server-side operator must not use raw test values except inside benchmark drivers for injection and metrics.

## Common Protocol

Use chronological splits, not random splits:

```text
train: first 60%
validation: next 20%
test: final 20%
```

Use train-only calibration:

```text
1. Fit clipping/normalization bounds on train.
2. Normalize train/validation/test to [-1, 1] using those bounds.
3. Discretize raw values and private reports, usually d = d_tilde = 32.
4. Estimate alpha from train raw buckets or from the separately accounted calibration source.
5. Build the declared channel matrix M.
6. For controlled runs, inject faults only into validation/test copies before LDP randomization.
7. Run private methods only on report buckets, M, alpha, epsilon, and public metadata.
```

Every run should record:

```text
dataset, split, epsilon, seed, fault_mode, stuck_value_type,
method, AUROC, AUPRC, prevalence, ratio_MAE, bucket_hit1,
TV, KL, selected_action, selected_operator, evidence_file
```

## E1. Air Quality Anchor

Purpose: run the full pipeline on one understandable dataset.

Data: UCI Air Quality, target `C6H6(GT)`, removing `-200` rows.

Run:

```powershell
py scripts\download_air_quality.py
py scripts\run_air_quality.py
```

Outputs:

- `results/air_quality_runs.csv`
- `results/air_quality_summary.csv`
- `results/air_quality_metadata.json`
- `results/fig_air_quality_repro.png`

Expected pattern: persistent segment faults favor the persistent segment scorer; iid faults favor the mixture backend.

## E2. Channel Advantage

Purpose: show that raw flatline logic does not survive PM-LDP reports, and the declared channel matrix is necessary.

Run:

```powershell
py scripts\run_icde_acceptability_extensions.py
```

Key outputs:

- `results/icde_channel_advantage_runs.csv`
- `results/icde_channel_advantage_summary.csv`
- `results/fig_icde_channel_advantage.png`

Expected pattern: known-channel likelihood improves over bucket-frequency tests, especially when private reports do not repeat exactly.

## E3. Multi-Dataset Controlled Detection

Purpose: compare model-matched operators across Air Quality, Household Power, Bike Sharing, Beijing Air, and NAB machine temperature.

Run:

```powershell
py scripts\run_icde_revision_grid.py
```

Key outputs:

- `results/icde_revision_detection_runs.csv`
- `results/icde_revision_detection_summary.csv`
- `results/icde_revision_diagnostic_outputs.csv`
- `results/icde_revision_repair_runs.csv`
- `results/icde_revision_repair_summary.csv`

Expected pattern: mixture is selected for iid stuck-at points; the segment backend is selected for persistent flatlines; generic private anomaly baselines are reported on the same PM-LDP ledgers.

## E4. Failure Boundary and Stress Tests

Purpose: prove where PrivSAF should abstain or route rather than claim success.

Run:

```powershell
python scripts\run_reviewer_stress_tests.py
python scripts\build_reviewer_audit_ledgers.py
```

Key outputs:

- `results/reviewer_stress_runs.csv`
- `results/reviewer_stress_summary.csv`
- `results/reviewer_stress_separation_correlation.csv`
- `results/reviewer_separation_gate_ledger.csv`
- `results/reviewer_separation_gate_false_accounting.csv`

Stress settings to check:

- median stuck value
- mode bucket
- minimum-TV bucket
- low epsilon
- short segments
- low fault rates
- validation-shifted calibration

Expected pattern: TV/KL are admission diagnostics only. Validation lift is required before automatic cleaning.

## E5. Real-Label Boundary and Semantic Routing

Purpose: treat real labels as semantic layers, not as one universal "stuck" class.

Run relevant panels:

```powershell
python scripts\summarize_private_cleaning_router.py
python scripts\summarize_real_fault_boundary_audit.py
python scripts\run_iors_stuck_qc_pmldp.py
python scripts\run_wsn_stuck_labeled_pmldp.py
python scripts\run_coops_verified_flat_pmldp.py
python scripts\run_coops_verified_flat_full_protocol.py
```

Key outputs:

- `results/private_telemetry_cleaning_router_summary.csv`
- `results/private_telemetry_router_decision_policy.csv`
- `results/reviewer_router_ledger.csv`
- `results/real_fault_privsaf_boundary_audit.csv`
- `results/coops_verified_flat_full_protocol_summary.csv`
- `results/iors_stuck_qc_pmldp_summary.csv`
- `results/wsn_stuck_labeled_pmldp_summary.csv`

Expected routing:

- HadISD straight-string labels: repeated-value semantics, PrivSAF-compatible.
- NOAA CO-OPS flat-tolerance labels: compatible but weak private lift, triage/audit.
- I-ORS field QC labels: local-regime statistics such as GLR/CUSUM.
- WSN prepared row labels: PM-window GLR boundary case.

## E6. Repair-Aware SQL Analytics

Purpose: evaluate cleaning as a downstream SQL analytics aid, not as perfect raw reconstruction.

Run:

```powershell
python scripts\run_reviewer_repair_analytics.py
```

Key outputs:

- `results/reviewer_repair_analytics_runs.csv`
- `results/reviewer_repair_analytics_summary.csv`
- `results/icde_revision_repair_runs.csv`
- `results/icde_revision_repair_summary.csv`

Metrics to report:

- repaired-sequence MAE
- fault-point MAE
- clean-point damage
- downstream mean error
- threshold exceedance-count error

Expected pattern: posterior-mask repair should improve aggregate error conservatively, while oracle-mask interpolation remains an upper bound.

## E7. Database Operator and System Artifact

Purpose: demonstrate the ICDE database contribution: immutable ledgers, materialized state, provenance, replay/correction, and repair-aware views.

Run:

```powershell
python scripts\run_private_telemetry_benchmark_from_config.py
python scripts\run_private_telemetry_pipeline_queries.py
python scripts\run_private_telemetry_physical_design_ablation.py
python scripts\run_private_telemetry_sqlite_operational_stress.py
python scripts\run_private_telemetry_end_to_end_workflow.py
python scripts\run_duckdb_private_telemetry_pipeline.py
python scripts\verify_private_telemetry_benchmark_config.py
python scripts\verify_private_telemetry_sqlite_artifact.py --db results\private_telemetry_pipeline_200k.sqlite --out results\private_telemetry_sqlite_verification.csv
```

Optional PostgreSQL path:

```powershell
$env:PRIVSAF_POSTGRES_DSN="dbname=privsaf_pg_benchmark user=codex host=/var/run/postgresql"
python scripts\run_postgres_private_telemetry_pipeline.py
```

Required database surfaces:

- `LDP_REPORTS`
- `LDP_CHANNEL`
- `CALIBRATION`
- `ROUTER_POLICY`
- `PRIVSAF_CLEANING`
- `PRIVACY_LEDGER`
- `PROVENANCE_EDGE`
- `LDP_CHANNEL_LIKELIHOOD`
- `PRIVSAF_POSTERIOR_STREAM`
- `REPAIR_AWARE_AGG`

Key validation commands used in the 2026-05-29 audit:

```powershell
python3 scripts\verify_private_telemetry_benchmark_config.py
python3 scripts\verify_private_telemetry_sqlite_artifact.py --db results\private_telemetry_pipeline_200k.sqlite --out results\private_telemetry_sqlite_verification.csv
```

Current expected verifier results:

- benchmark config: 76/76 checks pass
- SQLite artifact: 14/14 checks pass

## Paper Build

Active manuscript:

```text
main.tex
```

Synced single-file copies:

```text
PrivSAF_ICDE_revised_operator_contract_final.tex
GPT_web_edit_current_all_latex.tex
PrivSAF.tex
```

Build command used locally:

```bash
/home/fu/.local/bin/tectonic main.tex
```

`latexmk` is not installed in the current environment. The current PDF builds successfully with `tectonic`; remaining warnings are underfull box/font-substitution warnings.
