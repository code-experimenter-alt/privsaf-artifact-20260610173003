# ICDE Pro/Poe Revision Completion Audit

Date: 2026-05-23

Updated: 2026-05-27 for the narrowed private stuck-at/flatline cleaning framing, router/hybrid summary, real-label layering, Duchi-binary mechanism interface, sensitivity panels, database/system panels, standalone SQL artifacts, SQLite verification checks, physical-design ablation, and stricter native-flatline audit detail.

Objective audited: address the Pro/Poe experiment-focused recommendations and self-check the manuscript against an ICDE main-conference review standard, with the paper positioned as channel-aware private stuck-at/flatline cleaning for scalar LDP telemetry.

## Evidence Sources

- Manuscript: `main.tex`
- Experiment runner: `scripts/run_icde_revision_grid.py`
- SQLite artifact verifier: `scripts/verify_private_telemetry_sqlite_artifact.py`
- Main PDF: `main.pdf`
- Logs: `main.log`
- SQL artifacts:
  - `configs/private_telemetry_benchmark_config.json`
  - `sql/private_telemetry_schema.sql`
  - `sql/private_telemetry_query_suite.sql`
  - `sql/private_telemetry_physical_design.sql`
  - `sql/private_telemetry_postgres_schema.sql`
- Generated results:
  - `results/icde_revision_detection_runs.csv`
  - `results/icde_revision_detection_summary.csv`
  - `results/icde_revision_repair_runs.csv`
  - `results/icde_revision_repair_summary.csv`
  - `results/icde_revision_diagnostic_outputs.csv`
  - `results/icde_revision_native_flatline_candidates.csv`
  - `results/icde_revision_access_regimes.csv`
  - `results/icde_revision_theory_diagnostics.csv`
  - `results/icde_channel_advantage_runs.csv`
  - `results/icde_channel_advantage_summary.csv`
  - `results/icde_native_weaklabel_runs.csv`
  - `results/icde_native_weaklabel_summary.csv`
  - `results/icde_dropout_runs.csv`
  - `results/icde_dropout_summary.csv`
  - `results/icde_channel_baseline_runs.csv`
  - `results/icde_channel_baseline_summary.csv`
  - `results/iors_stuck_qc_pmldp_runs.csv`
  - `results/iors_stuck_qc_pmldp_summary.csv`
  - `results/iors_stuck_qc_inventory.csv`
  - `results/iors_stuck_qc_source_metadata.csv`
  - `results/wsn_stuck_labeled_pmldp_runs.csv`
  - `results/wsn_stuck_labeled_pmldp_summary.csv`
  - `results/wsn_stuck_labeled_dataset_inventory.csv`
  - `results/private_telemetry_cleaning_router_summary.csv`
  - `results/private_telemetry_cleaning_router_rollup.csv`
  - `results/real_label_layered_analysis.csv`
  - `results/real_fault_privsaf_boundary_audit.csv`
  - `results/real_fault_privsaf_boundary_summary.csv`
  - `results/real_fault_privsaf_boundary_summary.json`
  - `results/private_stuck_cleaning_sensitivity_summary.csv`
  - `results/private_stuck_cleaning_system_summary.csv`
  - `results/private_telemetry_pipeline_query_benchmark.csv`
  - `results/private_telemetry_pipeline_query_plans.csv`
  - `results/private_telemetry_pipeline_200k.sqlite`
  - `results/private_telemetry_sql_query_suite.csv`
  - `results/private_telemetry_sql_query_suite_summary.csv`
  - `results/private_telemetry_sql_query_suite_plans.csv`
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
  - `results/private_telemetry_duckdb_build.csv`
  - `results/private_telemetry_duckdb_queries.csv`
  - `results/private_telemetry_duckdb_plans.csv`
  - `results/private_telemetry_duckdb_verification.csv`
  - `results/private_telemetry_duckdb_summary.json`
  - `results/private_telemetry_postgres_environment_audit.csv`
  - `results/private_telemetry_postgres_integration_summary.json`
  - `results/native_flatline_event_audit.csv`
  - `results/native_flatline_event_detail_audit.csv`
  - `results/native_flatline_method_audit.csv`
  - `results/fig_private_stuck_sensitivity.png`
  - `results/fig_private_stuck_systems.png`
  - `results/real_fault_source_audit.csv`
  - `results/fig_icde_channel_ablation.png`
  - `results/fig_icde_channel_advantage.png`
  - `results/fig_icde_native_weaklabel.png`
  - `results/fig_icde_dropout.png`
  - `results/fig_icde_representative_timeline.png`
  - `results/fig_icde_sensitivity.png`
  - `results/fig_icde_revision_baselines.png`
  - `results/fig_icde_revision_repair.png`

## Checklist Status

| Recommendation | Status | Evidence |
|---|---:|---|
| Reposition novelty around scalar LDP stuck-at/flatline cleaning | Done | Title, abstract, contributions, method overview, router map, and conclusion in `main.tex` |
| Add channel-advantage diagnostic for privatized-domain scorers | Done | Proposition `Known-channel likelihood diagnostic under PM-LDP`; `fig_icde_channel_advantage.png`; 420 diagnostic rows |
| Add same-protocol private / privatized baselines | Done | PM CUSUM/BOCPD, LDP posterior-mean scores, window KL/GLR in `scripts/run_icde_revision_grid.py`; 3,420 detection rows |
| Add direct channel-aware likelihood / change-point / generic-HMM baselines | Done | `scripts/run_icde_channel_baseline_extensions.py`; `results/icde_channel_baseline_summary.csv`; manuscript table includes PM-column likelihood scan, LLR-CUSUM, window GLR, and generic PM-HMM |
| Add access-regime taxonomy separating raw upper bounds from privacy-comparable methods | Done | `tab:access_regime` in `main.tex`; `results/icde_revision_access_regimes.csv` with 19 rows |
| Add channel-awareness / wrong-channel ablation | Done | wrong-epsilon HMM runs in CSV; `results/fig_icde_channel_ablation.png`; manuscript reports correct vs wrong-channel numbers |
| Add semi-real, weak-natural, and external real-stuck source evidence | Done as real-label layering | Template-based semi-real injection; native-source weak-label evaluation with 17 stricter original-location episodes and 85 matched negatives; `results/iors_stuck_qc_pmldp_summary.csv` adds a 336-row real stuck-QC PM-LDP panel from a deployed long-running rangefinder; `results/real_fault_source_audit.csv` records I-ORS real scalar stuck-QC labels, MiFaD real induced stuck labels, AHU semi-real fault-label datasets, restricted automotive HIL sensor stuck-at labels, and public prepared WSN stuck-at labels; `results/wsn_stuck_labeled_pmldp_summary.csv` adds a 180-row PM-LDP panel |
| Add at least one non-stuck-at fault family | Done | Dropout emission extension with explicit missing symbol; 540 dropout rows and `fig_icde_dropout.png` |
| Add router/hybrid and real-label layered analysis | Done | `scripts/summarize_private_cleaning_router.py`; `results/private_telemetry_cleaning_router_summary.csv`; `results/real_label_layered_analysis.csv`; manuscript routes PrivSAF, GLR, LLR-CUSUM, and dropout HMM by scalar stuck/flatline label semantics |
| Add sensitivity and system experiments | Done | `scripts/run_private_stuck_cleaning_extensions.py`; sensitivity CSV/figure; system CSV/figure |
| Add relational cleaning pipeline workload | Done | `scripts/run_private_telemetry_pipeline_queries.py`; `scripts/run_private_telemetry_physical_design_ablation.py`; `scripts/run_private_telemetry_benchmark_from_config.py`; `scripts/run_private_telemetry_sqlite_operational_stress.py`; `scripts/run_duckdb_private_telemetry_pipeline.py`; `configs/private_telemetry_benchmark_config.json`; standalone SQL schema/query files; 64 SQLite benchmark rows; 52 `EXPLAIN QUERY PLAN` rows; 61.9 MB file-backed SQLite database; 27 SQL query-suite timings; 9 p50 query summaries; 48 physical-design timings; 79 physical-design query-plan rows; 31 total passing SQLite/physical-design checks; 41 passing benchmark-configuration/orchestration checks; 27 passing bounded SQLite operational-stress metrics; DuckDB v1.5.3 integration with 15 query timings, 374 plan rows, and 6 passing checks |
| Add stronger repair analysis and oracle gap | Done | baseline-mask repair, clean/fault MAE, false-repair fraction, oracle-gap closure in repair CSV and `tab:icde_repair_summary` |
| Reposition repair as secondary | Done | Abstract, contributions, repair table text, operational-use section |
| Add diagnostic-output evidence beyond AUROC/AUPRC | Done | `results/icde_revision_diagnostic_outputs.csv`; `tab:diagnostic_outputs` reports ratio MAE, bucket top-1, Brier |
| Add qualitative timeline figure | Done | `results/fig_icde_representative_timeline.png` included as `fig:icde_timeline` |
| Add sensitivity/runtime support | Done | Air Quality epsilon/stuck-value sweep, PM separation diagnostics, and segment-length/fault-rate sweep in `results/icde_revision_sensitivity_*.csv` and `fig_icde_sensitivity.png` |
| Keep comparisons on the same private-telemetry ledger | Done | Related Work and experiment protocol compare methods generated under shared PM-LDP reports, seeds, splits, and budgets |
| De-emphasize theory as diagnostics | Done | Theory subsection renamed `Separation Diagnostics`; operator-selection rows are reported through the router map |
| Keep optional audio from dominating | Done | Optional audio compressed to a short protocol paragraph |
| Rebuild and verify paper | Done | `tectonic --keep-logs main.tex`; 16-page `main.pdf`; no fatal errors, undefined refs/citations, or overfull boxes; remaining warnings are underfull/font-package warnings |

## Current Main Evidence

- Segment stuck-at: PrivSAF-HMM reaches AUROC/AUPRC `0.930/0.833`; LDP window GLR reaches `0.843/0.571`.
- Iid stuck-at: PrivSAF-mixture reaches `0.761/0.493`.
- Template-based semi-real stuck-at: PrivSAF-HMM reaches `0.851/0.613`; LDP window GLR reaches `0.687/0.380`.
- Main framing: the manuscript presents channel-aware private stuck-at and flatline cleaning for scalar LDP telemetry, with the declared LDP mechanism treated as an observation channel for server-side post-processing.
- Operational use: `main.tex` now has an `Operational Use` section that routes PrivSAF-HMM to persistent scalar stuck-at, PrivSAF-mixture to iid stuck-at, dropout HMM to explicit availability labels, and GLR/CUSUM to field-QC regime evidence.
- Direct channel-aware baselines: generic PM-HMM reaches `0.840/0.704` on segment faults and `0.834/0.583` on template faults, below PrivSAF-HMM; PM-column likelihood scan reaches `0.625/0.315` on iid faults.
- Real/semi-real source audit: I-ORS provides public real scalar sea-level rangefinder stuck-QC labels and reports instrument relocation/replacement context; MiFaD provides public real induced `stuck` labels for MEMS microphones; Scientific Data AHU files provide labeled real-building/HIL/simulated system faults including stuck/leaking dampers and valves plus sensor-bias faults; Scientific Reports automotive HIL data include APP/RPM sensor-related stuck-at labels but require data access by request; the prepared WSN benchmark provides public stuck-at Excel files over real TelosB base measurements; the hydraulic PS1 source is model-matched scalar stuck-at but explicitly injected on real rig data, so it is audited as non-ground-truth evidence rather than incorporated as a real-fault win.
- I-ORS real stuck-QC PM-LDP panel: over eight annual slices, two privacy budgets, and three seeds, PM-window GLR reaches AUROC/AUPRC `0.940/0.716`, generic PM-HMM reaches `0.745/0.370`, and global PrivSAF-HMM reaches `0.713/0.372`.
- I-ORS low-epsilon stuck-QC PM-LDP panel: `results/iors_stuck_qc_loweps_pmldp_runs.csv` adds 336 rows at `epsilon={0.5,1}`; at epsilon `0.5`, LLR-CUSUM reaches AUPRC `0.389` while the best PrivSAF variant reaches `0.212`; at epsilon `1.0`, LLR-CUSUM reaches `0.516` while the best PrivSAF variant reaches `0.332`.
- I-ORS real dropout-QC PM-LDP panel: `results/iors_dropout_qc_pmldp_runs.csv` adds 288 rows with public `SLH_QC=8` missing-value labels; PrivSAF dropout HMM reaches AUROC/AUPRC approximately `1.000/1.000` because the missing symbol is explicitly observed.
- Mechanism extension: `results/mechanism_fault_extension_runs.csv` adds PM and Duchi-binary real-valued LDP channel rows; Duchi-binary segment HMM reaches AUPRC `0.737` at epsilon `1` and `0.939` at epsilon `2`.
- Sensitivity and system panels: `private_stuck_cleaning_sensitivity_summary.csv` covers calibration contamination, bucket count, epsilon mismatch, and multi-device short traces; `private_stuck_cleaning_system_summary.csv` covers stream-state updates, materialized count views, and multi-device grouping.
- Relational pipeline workload: `results/private_telemetry_pipeline_query_benchmark.csv` has 64 rows over in-memory and file-backed SQLite ingestion, mechanism lookup, materialized counts, candidate stuck-bucket likelihood joins, cleaning records, provenance edges, privacy ledgers, windowed budget traces, event windows, repair-aware aggregates, drilldown, and hourly analytics. The 200k-row file-backed WAL run writes a 61.9 MB SQLite database, derives cleaning records at `2.74M` rows/s, computes windowed privacy-budget traces at `1.18M` rows/s, and builds repair-uncertainty aggregates at `1.65M` rows/s. `results/private_telemetry_pipeline_query_plans.csv` records 52 query-plan rows showing indexed mechanism lookup, indexed action/provenance drilldown, and device-time scans for budget windows. The SQL query suite adds 27 timings and 9 p50 summary rows: materialized report counts reduce p50 likelihood-join latency from `3830` ms to `1.19` ms; the materialized budget trace reduces p50 budget-audit latency from `86.1` ms to `9.45` ms; and materialized repair analytics reduce p50 dashboard latency from `127` ms to `1.36` ms. The standalone files `configs/private_telemetry_benchmark_config.json`, `sql/private_telemetry_schema.sql`, `sql/private_telemetry_query_suite.sql`, `sql/private_telemetry_physical_design.sql`, `sql/private_telemetry_duckdb_schema.sql`, and `sql/private_telemetry_postgres_schema.sql` expose the benchmark configuration and database surface without rerunning Python, and `results/private_telemetry_sqlite_verification.csv` records 10 passing relation-count checks plus four passing dashboard/drilldown invariants. The physical-design ablation compares heap, secondary-indexed, batch-materialized, and trigger-maintained incremental designs on the same 200k-report ledger: candidate top-k latency falls from `3822` ms on heap reports to `0.060` ms with batch materialization; privacy-accounting latency falls from `86.3` ms to `0.103` ms with incremental ledgers; dashboard latency falls from `91.4` ms to `1.34/1.67` ms for batch/incremental state; `results/private_telemetry_physical_design_checks.csv` records 17 additional passing invariants; `scripts/run_private_telemetry_benchmark_from_config.py` records a ready six-stage config-driven manifest; and `results/private_telemetry_benchmark_config_check.csv` records 41 passing benchmark-configuration/orchestration checks. `results/private_telemetry_sqlite_operational_stress.csv` adds 27 passing bounded operational metrics for 3 writer clients, 2 reader clients, recovery integrity, storage footprint, and a page-limit pressure case. The DuckDB v1.5.3 integration writes a 9.19 MB DuckDB database over the same 200k-report schema, 15 query timings, 374 plan rows, and 6 passing checks. The PostgreSQL driver and schema are packaged, but `results/private_telemetry_postgres_integration_summary.json` currently records `status=not_run` because the environment lacks `psycopg2`.
- Native flatline event audit: `results/native_flatline_event_audit.csv` records 17 original-stream flatline-like events and 85 matched negatives across Air Quality, Beijing Air, and NAB; lengths range from 12 to 461 samples, with 10 exact-constant and 7 small-range events. `results/native_flatline_event_detail_audit.csv` marks all 17 events as edge-bucket, clipping-risk windows after normalization, so this layer is evidence of native flatline-like structure but not audited deployment fault ground truth. `results/native_flatline_method_audit.csv` records PrivSAF-HMM scan, bucket-count, and identity-channel method rows for the same layer.
- Privacy-utility ledger: `results/privacy_utility_loweps_summary.csv` and `results/device_level_privacy_ledger.csv` report low-epsilon utility and basic-composition device budgets; at 10-minute cadence and epsilon `0.5`, a device spends `72` privacy units per day, so a device budget of `10` covers only 20 reports (`0.139` days).
- Real-label layered analysis: native flatline windows select PrivSAF-HMM scan (AUPRC `0.484`, close to identity-channel `0.505`), I-ORS field-QC labels select PM-window GLR (AUPRC `0.716`), I-ORS low-epsilon rows select LLR-CUSUM (AUPRC `0.389/0.516`), I-ORS dropout labels select dropout HMM (AUPRC approximately `1.000`), and WSN row labels select PM-window GLR (AUPRC `0.463`). The boundary audit records four real or hardware-adjacent layers: three stuck/flatline layers, one weak PrivSAF-compatible native flatline layer, two semantic-mismatch router-boundary stuck-labeled layers, and one availability extension.
- Router map: `main.tex` includes a router map synchronized with `results/private_telemetry_cleaning_router_summary.csv`; the CSV records 10 semantic-layer recommendations and a rollup over PrivSAF posterior, local channel statistic, and availability HMM families.
- WSN labeled-stuck PM-LDP stress panel: over five public prepared Stuck-at files, two privacy budgets, and three seeds, PM-window GLR reaches AUROC/AUPRC `0.634/0.463`, PM-column likelihood scan reaches `0.585/0.354`, and PrivSAF-HMM reaches `0.478/0.296`.
- Channel ablation: segment PrivSAF-HMM drops from `0.930/0.833` with the correct PM channel to `0.671/0.505` with a `2epsilon` channel.
- Channel advantage: at epsilon `0.5`, `53.6%` of settings are weak bucket cases; median bucket AUROC is `0.512`, while known-channel LLR AUROC is `0.779`.
- Native-source weak-label evaluation: PrivSAF-HMM scan reaches AUROC/AUPRC `0.809/0.461`, versus `0.519/0.205` for privatized bucket counting.
- Dropout extension: PrivSAF dropout HMM reaches AUROC/AUPRC `0.992/0.970`, versus `0.983/0.908` for rolling missing fraction.
- Repair: PrivSAF posterior-mask repair reduces MAE from `0.1488` to `0.1240` and closes `32.7%` of the oracle interpolation gap.

## Router Analysis Status

- Real labels are now used to distinguish semantics around scalar stuck-at/flatline cleaning: I-ORS stuck-QC favors PM-window GLR, I-ORS low-epsilon stuck-QC favors LLR-CUSUM, I-ORS dropout-QC follows the missing-symbol path, and WSN prepared stuck labels favor PM-window GLR.
- The mechanism extension covers segment-stuck rows under PM and Duchi-binary channels.
- The reported router summary uses same-protocol privatized-domain, channel-aware, and PM-window baselines generated inside this artifact, while deep/graph/sequence models such as DADA, TSINR, CAROT, FMP-AE, and Ostrich are positioned as non-private related work.

## Audit Conclusion

The Pro/Poe recommendations targeted to this repository are implemented and backed by generated artifacts. The current manuscript presents channel-aware private stuck-at/flatline cleaning for scalar LDP telemetry, with PrivSAF as the posterior operator family and a router/hybrid ledger that selects PrivSAF, GLR, LLR-CUSUM, or dropout HMM according to label semantics. The database evidence is now reproducible as SQL plus SQLite verification and physical-design artifacts; the real-fault evidence remains deliberately narrow because the boundary audit shows that only one stuck/flatline layer is PrivSAF-compatible and that native layer is edge-bucket weak evidence, while I-ORS/WSN labels validate the router boundary.
