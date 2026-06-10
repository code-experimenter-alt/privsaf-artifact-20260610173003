# PrivSAF Artifact

This repository contains the clean artifact for PrivSAF, an auditable database
operator for channel-aware scalar LDP stuck-at and flatline cleaning.

The package is organized for paper compilation and experiment reproduction. It
intentionally omits revision notes, review prompts, build artifacts, generated
PDFs, raw downloaded datasets, local database files, and the former supplement
files. The former supplementary material has been folded into `main.tex`.

## Contents

- `main.tex`, `references.bib`, `IEEEtran.bst`, and paper figures needed to
  rebuild the manuscript.
- `scripts/`: Python experiment, audit, plotting, and verification scripts.
- `sql/`: SQLite, DuckDB, PostgreSQL, physical-design, and workflow SQL.
- `configs/`: benchmark and run configuration files.
- `data/README.md`: notes for external data acquisition.
- `results/`: selected lightweight CSV/JSON ledgers and paper figures used to
  audit the reported results.

The anonymized mirror is:

```text
https://anonymous.4open.science/r/privsaf-artifact-20260610/
```

## Paper

Compile the paper from `main.tex` with BibTeX:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Setup

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Some real-label panels require external public datasets or local utilities
documented by the relevant scripts. Raw downloaded data and local DB files are
not included in this clean package.

## Core Reproduction Commands

Air Quality local experiment:

```bash
python scripts/download_air_quality.py
python scripts/run_air_quality.py
```

Main controlled and stress grids:

```bash
python scripts/run_icde_revision_grid.py
python scripts/run_reviewer_stress_tests.py
python scripts/build_separation_gate_outputs.py
python scripts/build_reviewer_audit_ledgers.py
python scripts/run_reviewer_repair_analytics.py
```

Real-label and boundary panels:

```bash
python scripts/run_iors_stuck_qc_pmldp.py
python scripts/run_iors_dropout_qc_pmldp.py
python scripts/run_wsn_stuck_labeled_pmldp.py
python scripts/audit_hadisd_streak_flags.py
python scripts/run_hadisd_streak_pmldp_panel.py
python scripts/run_hadisd_multistation_streak_pmldp_panel.py
python scripts/screen_coops_verified_flat_flags.py
python scripts/run_coops_verified_flat_pmldp.py
python scripts/run_coops_verified_flat_full_protocol.py
python scripts/summarize_coops_full_protocol_tiers.py
python scripts/summarize_coops_operational_triage.py
python scripts/summarize_real_fault_boundary_audit.py
```

Private telemetry database artifact:

```bash
python scripts/run_private_telemetry_benchmark_from_config.py
python scripts/run_private_telemetry_pipeline_queries.py
python scripts/run_private_telemetry_physical_design_ablation.py
python scripts/run_private_telemetry_sqlite_operational_stress.py
python scripts/run_private_telemetry_end_to_end_workflow.py
python scripts/run_duckdb_private_telemetry_pipeline.py
python scripts/verify_private_telemetry_benchmark_config.py
python scripts/verify_private_telemetry_sqlite_artifact.py
```

PostgreSQL execution is optional and requires a local DSN:

```bash
export PRIVSAF_POSTGRES_DSN="dbname=privsaf_pg_benchmark user=codex host=/var/run/postgresql"
python scripts/run_postgres_private_telemetry_pipeline.py
```

## Key Result Ledgers

The clean package keeps selected lightweight result files that support the
paper's reported tables, figures, router decisions, and database artifact
checks. Regenerated outputs should be written under `results/`.
