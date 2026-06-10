# PrivSAF Private Telemetry Cleaning Operator Project

This project contains the ICDE-style manuscript for PrivSAF, an auditable database operator for known-channel scalar LDP stuck-at and flatline cleaning, plus reproducible scalar-telemetry experiments, SQL/operator artifacts, privacy/provenance ledgers, and deployment-policy analyses over controlled and real-label boundary panels.

Current status:

- `main.tex` is the active operator-contract manuscript. It frames HMM/mixture inference as standard scoring backends inside a database operator with router policy, privacy/provenance ledgers, replay/correction lineage, and repair-aware SQL aggregates.
- `main.tex` is the authoritative manuscript source for the packaged artifact.
- The paper reports the measured five-dataset same-protocol detection/diagnosis/repair grid, with Air Quality retained as one workload in the shared protocol rather than as a separate appendix-style anchor section.
- Multi-dataset run logs are stored under `results/icde_revision_*.csv` and `results/icde_revision_metadata.json`.
- The current grid includes privatized-domain baselines, direct channel-aware likelihood/CUSUM/generic-HMM baselines, wrong-channel ablations, semi-real template faults, and native flatline candidates.
- Reviewer stress outputs in `results/reviewer_stress_*.csv` add median, mode-bucket, and minimum-TV stuck values, `epsilon={0.5,1,2}`, short segments, low fault rates, matched-train calibration, validation-shifted calibration, and TV/KL separation diagnostics. The run-level file has 4,320 rows with the schema requested for review.
- `results/reviewer_separation_gate_*.csv` adds the separation-aware deployment ledger. The fixed TV/KL screen is not tuned on test labels; passing the screen only makes PrivSAF eligible for validation-utility selection. `results/reviewer_separation_gate_false_accounting.csv` records screen-passing false accepts and screen-rejected false rejects using held-out test labels only for final audit.
- The manuscript now centers on a private telemetry database-operator contract. PrivSAF uses standard iid/HMM machinery only as optional scoring backends and constrains emissions as `M alpha` for normal behavior and `M[:,s]` for a stuck bucket; the deployment policy combines semantic compatibility, separation gating, validation lift, abstention, and routing.
- The real-label section stratifies I-ORS deployed rangefinder QC labels, WSN TelosB prepared stuck-at labels, dropout labels, and native-flatline evidence.
- A new HadISD audit records a direct real repeated-value layer: station `702606-96401` has official straight-string QC flags `TSS=13`, `DSS=13`, and `RSS=140` over 37,103 timesteps. The single-station PM-LDP panel selects PrivSAF-HMM on the RSS wind-direction straight-string case with mean AUPRC 0.896. A nine-station compact replication screen selects PrivSAF-HMM on both RSS cases and two of three WSS cases. Cross-page nonwind screens add 58 TSS/DSS eligible station-cases: page 7 has 7 TSS and 17 DSS, while page 0 has 20 TSS and 18 DSS. The page-0 PM-LDP panel roughly doubles AUPRC over prevalence on TSS/DSS.
- The latest real-fault source search added a PROMICE/GC-Net manual QA/QC flag audit. The repository-level pass covers 63 station CSV files, 1,606 active manual flag rows, 20 explicit scalar constant/flat/too-many-zero intervals, and 46 sensor-failure or suspicious intervals. A follow-up value-preservation audit found that public PROMICE L2 products mostly filter those flagged intervals to `NaN` (`128/132` station-variable samples all-NaN), so PROMICE remains a candidate source rather than a performance table.
- NOAA CO-OPS verified six-minute ERDDAP supplies public value+label flatline evidence for this revision: `WL_VALUE` and the official `F=1` flat-tolerance flag are present in the same rows. The corrected screen finds 4,687 public numeric `F=1` rows across 108 station-months. The frozen all-eligible KRR-LDP protocol runs 40 station-months, 291,283 rows, and 4,556 official flat positives with prior-month-only calibration; raw flatness scores reach case-mean AUPRC 0.874, but the `epsilon=2`, 24-bucket private panel is weak: `privsaf_range_hmm_r1_fixed_prior` reaches AUROC/AUPRC `0.539/0.0246`, above window GLR `0.0176`, report-frequency `0.0165`, and point GLR `0.0161` AUPRC. Treat this as small measurable private lift, not high-accuracy operational validation. The operational triage file records top-1% precision `0.0246`, recall `0.0157`, `1.57x` lift, and about 2,913 reviewed rows; the support-ge25 tier has seven station-months and selects PrivSAF range-HMM r2 at AUPRC 0.102.
- The method section now includes a private telemetry ingestion interface: report schema, public mechanism lookup, standalone SQL schema/query files, SQL/streaming operator state, batch materialized counts, cleaning provenance, privacy ledgers, verification checks, and analytics views.
- The private telemetry benchmark is packaged as an auditable deployment artifact with a machine-readable query/update mix, physical-design configuration, end-to-end workflow checks, optional PostgreSQL server-DBMS integration, and config-driven manifest runner in `configs/private_telemetry_benchmark_config.json` and `scripts/run_private_telemetry_benchmark_from_config.py`; the latest verifier records 76 passing checks, and the include-PostgreSQL manifest records eight stages.
- The latest PDF build target is `main.pdf`.
- Former supplementary material has been folded into the submission manuscript; no separate supplementary TeX or PDF is distributed.

## Paper and Artifact Scope

Paper entry point: `main.tex`.

This artifact is organized around reproducibility rather than revision history. It includes the active manuscript source, BibTeX file, paper figures, experiment scripts, SQL/configuration files, lightweight result ledgers, and selected reports. Experiment reproduction does not require LaTeX compilation; rebuilding the paper requires a local LaTeX distribution with BibTeX.

## Reproduce Air Quality Experiment

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Download the UCI data:

```powershell
py scripts\download_air_quality.py
```

Run the local reproducible experiment:

```powershell
py scripts\run_air_quality.py
```

Outputs:

- `results/air_quality_runs.csv`
- `results/air_quality_summary.csv`
- `results/air_quality_metadata.json`
- `results/fig_air_quality_repro.png`

The audio-derived CHiME/DESED/SINS scalar-stream protocol remains an optional extension pending local manifests and reproduced CSV logs.

## Reproduce Multi-Dataset Evaluation

The multi-dataset evaluation runs five scalar datasets with iid stuck-at, segment stuck-at, and template-based semi-real stuck-at faults. The mechanism extension adds PM/Duchi rows for segment-stuck routing. The evaluation reports five method families:

- raw-domain upper bounds: non-private HMM, Hampel, CUSUM, BOCPD, rolling median;
- privatized-domain baselines: anomaly scores directly on PM reports, including robust z-score, rolling median, CUSUM, and BOCPD;
- channel-aware baselines: LDP distribution surprise, posterior-mean scores, window KL/window GLR, PM-column likelihood scan, LLR-CUSUM, and generic PM-HMM;
- PrivSAF mixture and PrivSAF HMM;
- wrong-channel HMM ablations.

It also evaluates repair with no repair, LDP posterior smoothing, oracle-mask interpolation controls, baseline-mask interpolation, PrivSAF posterior-mask repair, oracle-gap closure, and reviewer-facing downstream aggregate errors.

Run:

```powershell
py scripts\run_icde_revision_grid.py
```

Outputs:

- `results/icde_revision_detection_runs.csv`
- `results/icde_revision_detection_summary.csv`
- `results/icde_revision_repair_runs.csv`
- `results/icde_revision_repair_summary.csv`
- `results/reviewer_stress_runs.csv`
- `results/reviewer_stress_summary.csv`
- `results/reviewer_stress_separation_correlation.csv`
- `results/reviewer_separation_failure_table.csv`
- `results/reviewer_separation_gate_ledger.csv`
- `results/reviewer_separation_gate_summary.csv`
- `results/reviewer_separation_gate_false_accounting.csv`
- `results/reviewer_router_ledger.csv`
- `results/reviewer_router_ablation.csv`
- `results/reviewer_coops_operational_metrics.csv`
- `results/reviewer_repair_analytics_runs.csv`
- `results/reviewer_repair_analytics_summary.csv`
- `results/icde_revision_theory_diagnostics.csv`
- `results/icde_revision_diagnostic_outputs.csv`
- `results/icde_revision_native_flatline_candidates.csv`
- `results/icde_revision_access_regimes.csv`
- `results/icde_revision_sensitivity_runs.csv`
- `results/icde_revision_sensitivity_summary.csv`
- `results/icde_revision_metadata.json`

To rebuild only the separation-gate ledgers from existing result CSVs:

```powershell
py scripts\build_separation_gate_outputs.py
```

Outputs:

- `results/reviewer_separation_failure_table.csv`
- `results/reviewer_separation_gate_ledger.csv`
- `results/reviewer_separation_gate_summary.csv`
- `results/reviewer_separation_gate_false_accounting.csv`
- `results/icde_channel_baseline_runs.csv`
- `results/icde_channel_baseline_summary.csv`
- `results/iors_stuck_qc_pmldp_runs.csv`
- `results/iors_stuck_qc_pmldp_summary.csv`
- `results/iors_stuck_qc_inventory.csv`
- `results/iors_stuck_qc_source_metadata.csv`
- `results/wsn_stuck_labeled_pmldp_runs.csv`
- `results/wsn_stuck_labeled_pmldp_summary.csv`
- `results/wsn_stuck_labeled_dataset_inventory.csv`
- `results/real_fault_source_audit.csv`
- `results/hadisd_real_streak_flag_audit.csv`
- `results/hadisd_real_streak_event_detail.csv`
- `results/hadisd_real_streak_summary.json`
- `results/hadisd_streak_pmldp_runs.csv`
- `results/hadisd_streak_pmldp_summary.csv`
- `results/hadisd_streak_pmldp_inventory.csv`
- `results/hadisd_streak_pmldp_source_metadata.json`
- `results/hadisd_multistation_streak_inventory.csv`
- `results/hadisd_multistation_streak_station_metadata.csv`
- `results/hadisd_multistation_streak_pmldp_runs.csv`
- `results/hadisd_multistation_streak_pmldp_summary.csv`
- `results/hadisd_multistation_streak_pmldp_rollup.csv`
- `results/hadisd_multistation_streak_pmldp_source_metadata.json`
- `results/hadisd_page7_small80_streak_screen_inventory.csv`
- `results/hadisd_page7_small80_streak_screen_station_metadata.csv`
- `results/hadisd_page7_small80_streak_screen_pmldp_source_metadata.json`
- `results/hadisd_page7_nonwind_streak_pmldp_inventory.csv`
- `results/hadisd_page7_nonwind_streak_pmldp_pmldp_runs.csv`
- `results/hadisd_page7_nonwind_streak_pmldp_pmldp_summary.csv`
- `results/hadisd_page7_nonwind_streak_pmldp_pmldp_rollup.csv`
- `results/hadisd_page7_nonwind_streak_pmldp_pmldp_source_metadata.json`
- `results/hadisd_page7_nonwind_streak_pmldp_station_metadata.csv`
- `results/hadisd_page0_small120_nonwind_screen_inventory.csv`
- `results/hadisd_page0_small120_nonwind_screen_station_metadata.csv`
- `results/hadisd_page0_small120_nonwind_screen_pmldp_source_metadata.json`
- `results/hadisd_page0_nonwind_streak_pmldp_inventory.csv`
- `results/hadisd_page0_nonwind_streak_pmldp_pmldp_runs.csv`
- `results/hadisd_page0_nonwind_streak_pmldp_pmldp_summary.csv`
- `results/hadisd_page0_nonwind_streak_pmldp_pmldp_rollup.csv`
- `results/hadisd_page0_nonwind_streak_pmldp_pmldp_source_metadata.json`
- `results/hadisd_page0_nonwind_streak_pmldp_station_metadata.csv`
- `results/promice_manual_flatline_flag_inventory.csv`
- `results/promice_manual_flatline_flag_summary.json`
- `results/promice_manual_flatline_flag_summary.csv`
- `results/promice_manual_flatline_flag_sources.csv`
- `results/promice_l2_flag_value_availability.csv`
- `results/promice_l2_flag_value_availability_summary.json`
- `results/coops_recent_flat_flag_screen.csv`
- `results/coops_recent_flat_flag_screen_summary.json`
- `results/coops_recent_flat_flag_value_availability.csv`
- `results/coops_recent_flat_flag_event_rows.csv`
- `results/coops_recent_flat_flag_value_availability_summary.json`
- `results/coops_verified_flat_flag_screen.csv`
- `results/coops_verified_flat_flag_events.csv`
- `results/coops_verified_flat_flag_screen_summary.json`
- `results/coops_verified_flat_pmldp_runs.csv`
- `results/coops_verified_flat_pmldp_summary.csv`
- `results/coops_verified_flat_pmldp_inventory.csv`
- `results/coops_verified_flat_pmldp_source_metadata.json`
- `results/coops_verified_flat_full_protocol_runs.csv`
- `results/coops_verified_flat_full_protocol_summary.csv`
- `results/coops_verified_flat_full_protocol_case_summary.csv`
- `results/coops_verified_flat_full_protocol_inventory.csv`
- `results/coops_verified_flat_full_protocol_tier_summary.csv`
- `results/coops_verified_flat_full_protocol_metadata.json`
- `results/coops_verified_flat_operational_triage_summary.csv`
- `results/private_telemetry_router_decision_policy.csv`
- `results/private_telemetry_cleaning_router_summary.csv`
- `results/private_telemetry_cleaning_router_rollup.csv`
- `results/real_label_layered_analysis.csv`
- `results/real_fault_privsaf_boundary_audit.csv`
- `results/real_fault_privsaf_boundary_summary.csv`
- `results/real_fault_privsaf_boundary_summary.json`
- `results/private_stuck_cleaning_sensitivity_runs.csv`
- `results/private_stuck_cleaning_sensitivity_summary.csv`
- `results/private_stuck_cleaning_system_runs.csv`
- `results/private_stuck_cleaning_system_summary.csv`
- `results/private_telemetry_pipeline_query_benchmark.csv`
- `results/private_telemetry_pipeline_query_benchmark.json`
- `results/private_telemetry_pipeline_query_plans.csv`
- `results/private_telemetry_pipeline_200k.sqlite`
- `results/private_telemetry_sql_query_suite.csv`
- `results/private_telemetry_sql_query_suite_plans.csv`
- `results/private_telemetry_sql_query_suite_summary.csv`
- `results/private_telemetry_sqlite_verification.csv`
- `results/private_telemetry_physical_design_build.csv`
- `results/private_telemetry_physical_design_build_summary.csv`
- `results/private_telemetry_physical_design_queries.csv`
- `results/private_telemetry_physical_design_summary.csv`
- `results/private_telemetry_physical_design_plans.csv`
- `results/private_telemetry_physical_design_checks.csv`
- `results/private_telemetry_physical_design_summary.json`
- `results/private_telemetry_benchmark_config_check.csv`
- `results/private_telemetry_benchmark_config_check.json`
- `results/private_telemetry_benchmark_config_manifest.csv`
- `results/private_telemetry_benchmark_config_manifest.json`
- `results/private_telemetry_sqlite_operational_stress.csv`
- `results/private_telemetry_sqlite_operational_stress_detail.csv`
- `results/private_telemetry_sqlite_operational_stress.json`
- `results/private_telemetry_end_to_end_workflow.sqlite`
- `results/private_telemetry_end_to_end_workflow_batches.csv`
- `results/private_telemetry_end_to_end_workflow_queries.csv`
- `results/private_telemetry_end_to_end_workflow_plans.csv`
- `results/private_telemetry_end_to_end_workflow_checks.csv`
- `results/private_telemetry_end_to_end_workflow_summary.json`
- `results/private_telemetry_duckdb_environment_audit.csv`
- `results/private_telemetry_duckdb_build.csv`
- `results/private_telemetry_duckdb_queries.csv`
- `results/private_telemetry_duckdb_plans.csv`
- `results/private_telemetry_duckdb_verification.csv`
- `results/private_telemetry_duckdb_summary.json`
- `results/private_telemetry_duckdb_pipeline.duckdb`
- `results/private_telemetry_postgres_environment_audit.csv`
- `results/private_telemetry_postgres_integration_summary.json`
- `results/private_telemetry_postgres_query_benchmark.csv`
- `results/private_telemetry_postgres_queries.csv`
- `results/private_telemetry_postgres_plans.csv`
- `results/private_telemetry_postgres_verification.csv`
- `results/native_flatline_event_audit.csv`
- `results/native_flatline_event_detail_audit.csv`
- `results/native_flatline_method_audit.csv`
- `results/fig_icde_revision_baselines.png`
- `results/fig_icde_revision_repair.png`
- `results/fig_icde_channel_ablation.png`
- `results/fig_icde_representative_timeline.png`
- `results/fig_icde_sensitivity.png`
- `results/fig_private_stuck_sensitivity.png`
- `results/fig_private_stuck_systems.png`

Default coverage is 3,420 controlled stuck-at detection rows, 1,620 repair rows, 360 diagnostic-output rows, 19 access-regime rows, 72 sensitivity rows, 41 native flatline candidates, and 20 theory-diagnostic rows over Air Quality, Household Power, Bike Sharing, Beijing Air, and NAB machine temperature. The direct channel-aware extension adds 750 rows for PM-column likelihood scan, LLR-CUSUM, window GLR, generic PM-HMM, and matched PrivSAF checks. The reviewer stress grid adds 4,320 rows for central stuck values, low epsilon, short segments, low fault rates, validation-shifted calibration, and TV/KL separation diagnostics. The native-flatline audit summarizes 17 original-stream flatline-like events, 85 matched negatives, and method performance under PM-LDP; the detail audit records edge-bucket and clipping-risk provenance.

The real-label evidence now includes I-ORS stuck-QC and dropout-QC panels, HadISD straight-string QC flags, WSN prepared stuck-at labels, and NOAA CO-OPS verified flat-tolerance labels. CO-OPS contributes 4,687 numeric official `F=1` rows across 108 station-months; the frozen full protocol runs 40 station-months with 291,283 rows and shows weak but measurable private lift for PrivSAF range-HMM r1 above report-frequency and GLR baselines under `epsilon=2`. The CO-OPS operational triage summary records top-1% precision `0.0246`, recall `0.0157`, a `1.57x` lift over pooled prevalence, and about 2,913 reviewed rows; top-5% precision is `0.0172` with recall `0.0549`. HadISD remains the strongest high-AUPRC real repeated-value layer: RSS wind-direction straight strings select PrivSAF-HMM with mean AUPRC 0.896, and compact station replication selects HMM on both RSS cases and two of three WSS cases.

The private-cleaning deployment policy writes 12 semantic-layer recommendations to `results/private_telemetry_cleaning_router_summary.csv`, and `results/private_telemetry_router_decision_policy.csv` fixes the semantic trigger, compatible operator set, selected operator, evidence files, and audit rule for each policy class. `results/reviewer_router_ledger.csv` restates this in the reviewer-requested schema, and `results/reviewer_router_ablation.csv` reports PrivSAF-only versus router-selected real-label panels. The real-fault boundary audit records six real or hardware-adjacent layers, of which five are stuck/flatline layers: three PrivSAF-compatible layers including HadISD straight-string and CO-OPS flat-tolerance labels, plus two semantic-mismatch boundary layers. The Duchi-binary mechanism extension verifies the same stuck-at HMM interface under a second scalar LDP channel.

The private stuck-cleaning extension adds 75 sensitivity rows and 12 database/system rows. These DBMS files should be read as deployment/reproducibility evidence rather than a standalone systems benchmark. The SQLite pipeline query workload adds 64 benchmark rows, 52 query-plan rows, 27 SQL query-suite timings, 9 p50 query summaries, standalone SQL schema/query files, and 14 SQLite artifact verification checks over in-memory and file-backed ingestion, report/mechanism indexes, materialized counts, candidate likelihood joins, cleaning records, provenance, privacy ledgers, windowed budget traces, event windows, repair-aware aggregates, drilldown, and hourly analytics views. The physical-design ablation adds 48 query timings, 16 p50 summaries, 79 query-plan rows, and 17 passing invariants for heap, indexed, batch-materialized, and trigger-maintained incremental designs. The end-to-end SQLite workflow adds 72k requested microbatch events, 60k accepted reports, 14k budget rejections, 10k replay duplicates, 120 late corrections, 24 dashboard/drilldown timings, 22 plan rows, and 13 passing invariants over idempotent ingestion, budget guards, provenance lineage, privacy traces, event windows, and repair-uncertainty analytics. The DuckDB integration adds a real DuckDB v1.5.3 database over the same 200k-report pipeline, 15 query timings, 374 plan rows, and 6 passing checks. The PostgreSQL integration adds an executed PostgreSQL 18.4 server-DBMS pipeline over the same 200k-report ledger, 12 build stages, 21 query timings, 59 `EXPLAIN` plan rows, and 12 passing checks over reports, cleaning records, provenance, privacy ledgers, budget traces, event windows, repair analytics, and server version. The config verifier now records 76 passing checks and the include-PostgreSQL manifest records eight stages.

Direct channel-aware extension:

```powershell
py scripts\run_icde_channel_baseline_extensions.py
```

Real I-ORS stuck-QC PM-LDP panel:

```powershell
python scripts\run_iors_stuck_qc_pmldp.py
python scripts\run_iors_stuck_qc_pmldp.py --epsilons 0.5,1 --output-prefix iors_stuck_qc_loweps
```

Real I-ORS dropout-QC PM-LDP panel:

```powershell
python scripts\run_iors_dropout_qc_pmldp.py
```

HadISD real straight-string QC audit:

```powershell
python scripts\audit_hadisd_streak_flags.py --ncdump path\to\ncdump
python scripts\run_hadisd_streak_pmldp_panel.py --ncdump path\to\ncdump
python scripts\run_hadisd_multistation_streak_pmldp_panel.py --ncdump path\to\ncdump --min-positive 50 --max-case-rows 0
python scripts\run_hadisd_multistation_streak_pmldp_panel.py --ncdump path\to\ncdump --case-codes TSS,DSS,PSS --min-positive 50 --max-case-rows 50000 --output-prefix hadisd_page7_nonwind_streak_pmldp
python scripts\run_hadisd_multistation_streak_pmldp_panel.py --ncdump path\to\ncdump --station-page-url https://hadleyserver.metoffice.gov.uk/hadobs/hadisd/v343_2025f/station_download_0.html --station-ids all --station-sort size --station-limit 120 --max-station-size-mb 4 --case-codes TSS,DSS,PSS --min-positive 50 --min-negative 100 --max-fault-rate 0.70 --max-case-rows 50000 --max-station-cases 40 --output-prefix hadisd_page0_nonwind_streak_pmldp
```

NOAA CO-OPS verified flat-tolerance value+label panel:

```powershell
python scripts\screen_coops_verified_flat_flags.py
python scripts\run_coops_verified_flat_pmldp.py
python scripts\run_coops_verified_flat_full_protocol.py
python scripts\summarize_coops_full_protocol_tiers.py
python scripts\summarize_coops_operational_triage.py
python scripts\write_router_decision_policy.py
python scripts\update_coops_verified_layering.py
python scripts\summarize_real_fault_boundary_audit.py
```

Mechanism and privacy-utility summaries:

```powershell
python scripts\run_mechanism_fault_extension.py
python scripts\summarize_privacy_utility.py
```

External WSN labeled-stuck PM-LDP stress panel:

```powershell
python scripts\run_wsn_stuck_labeled_pmldp.py
```

Private telemetry cleaning router and real-label layering:

```powershell
python scripts\summarize_private_cleaning_router.py
python scripts\summarize_real_fault_boundary_audit.py
```

Private stuck-cleaning sensitivity and system panels:

```powershell
python scripts\run_private_stuck_cleaning_extensions.py
```

Reviewer stress, router, CO-OPS triage, and repair audit ledgers:

```powershell
python scripts\run_reviewer_stress_tests.py
python scripts\build_reviewer_audit_ledgers.py
python scripts\run_reviewer_repair_analytics.py
```

Private telemetry SQL pipeline query workload:

```powershell
python scripts\run_private_telemetry_benchmark_from_config.py
python scripts\run_private_telemetry_pipeline_queries.py
python scripts\run_private_telemetry_physical_design_ablation.py
python scripts\run_private_telemetry_sqlite_operational_stress.py
python scripts\run_private_telemetry_end_to_end_workflow.py
python scripts\run_duckdb_private_telemetry_pipeline.py
python scripts\verify_private_telemetry_benchmark_config.py
$env:PRIVSAF_POSTGRES_DSN="dbname=privsaf_pg_benchmark user=codex host=/var/run/postgresql"
python scripts\run_postgres_private_telemetry_pipeline.py
python scripts\verify_private_telemetry_sqlite_artifact.py --db results\private_telemetry_pipeline_200k.sqlite --out results\private_telemetry_sqlite_verification.csv
```

Standalone SQL artifacts:

- `configs/private_telemetry_benchmark_config.json`
- `scripts/run_private_telemetry_benchmark_from_config.py`
- `sql/private_telemetry_schema.sql`
- `sql/private_telemetry_query_suite.sql`
- `sql/private_telemetry_physical_design.sql`
- `sql/private_telemetry_end_to_end_workflow.sql`
- `sql/private_telemetry_duckdb_schema.sql`
- `sql/private_telemetry_postgres_schema.sql`

Native flatline event audit:

```powershell
python scripts\summarize_native_flatline_evidence.py
```

## Reproduce ICDE Acceptability Extensions

The acceptability extension supplies channel-advantage diagnostics, native flatline evidence, dropout rows, low-epsilon real-label rows, and mechanism rows used by the cleaning router.

Run:

```powershell
py scripts\run_icde_acceptability_extensions.py
```

Outputs:

- `results/icde_channel_advantage_runs.csv`
- `results/icde_channel_advantage_summary.csv`
- `results/icde_native_weaklabel_runs.csv`
- `results/icde_native_weaklabel_summary.csv`
- `results/icde_native_weaklabel_candidates.csv`
- `results/icde_native_weaklabel_matched_negatives.csv`
- `results/icde_dropout_runs.csv`
- `results/icde_dropout_summary.csv`
- `results/fig_icde_channel_advantage.png`
- `results/fig_icde_native_weaklabel.png`
- `results/fig_icde_dropout.png`
- `results/iors_stuck_qc_loweps_pmldp_runs.csv`
- `results/iors_stuck_qc_loweps_pmldp_by_epsilon.csv`
- `results/iors_dropout_qc_pmldp_runs.csv`
- `results/iors_dropout_qc_pmldp_summary.csv`
- `results/mechanism_fault_extension_runs.csv`
- `results/mechanism_fault_extension_summary.csv`
- `results/private_telemetry_cleaning_router_summary.csv`
- `results/private_telemetry_cleaning_router_rollup.csv`
- `results/real_label_layered_analysis.csv`
- `results/private_stuck_cleaning_sensitivity_summary.csv`
- `results/private_stuck_cleaning_system_summary.csv`
- `results/privacy_utility_loweps_summary.csv`
- `results/device_level_privacy_ledger.csv`

Default coverage is 420 channel-advantage settings, 1,836 native weak-label method-event rows, 540 dropout method rows, 360 mechanism rows, 10 router rows, 75 sensitivity rows, and 12 system rows.

## Round-2 Experiment Expansion

The experiment expansion plan and must-run configuration are saved in:

- `reports/experiment_expansion_plan.md`
- `configs/expanded_must_run.yaml`

Download the additional public datasets:

```powershell
py scripts\download_expansion_datasets.py
```

Generate the planning CSVs and visualization figures:

```powershell
py scripts\build_experiment_visualizations.py
py scripts\profile_downloaded_streams.py
```

Key generated figures:

- `results/fig_dataset_gap_coverage.png`
- `results/fig_dataset_scale.png`
- `results/fig_downloaded_stream_previews.png`
- `results/fig_fault_taxonomy_examples.png`
- `results/fig_expected_metric_trends.png`
- `results/fig_model_selection_expectation.png`
- `results/fig_budget_estimation_plan.png`
- `results/fig_experiment_workplan.png`
