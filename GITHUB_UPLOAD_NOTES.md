# GitHub Upload Notes

This package contains the current PrivSAF manuscript source, experiment scripts, SQL/configuration files, documentation, and lightweight result ledgers/figures.

Included:

- `scripts/`: all Python experiment, audit, plotting, and verification scripts
- `sql/`: SQLite, DuckDB, PostgreSQL, physical-design, and workflow SQL
- `configs/`: benchmark and must-run configuration files
- `results/`: CSV/JSON/PNG outputs needed for paper figures and audit ledgers
- `reports/`: artifact documentation and execution/audit notes
- `main.tex`, `references.bib`, and manuscript figures needed by the paper source

Excluded intentionally:

- raw downloaded datasets
- large SQLite/DuckDB database files and WAL/SHM files
- ERDDAP/cache directories
- virtual environments and wheelhouse dependencies
- generated LaTeX intermediate files
- generated PDFs
- separate supplementary TeX/PDF files, because the former supplementary material has been folded into `main.tex`
- previous `release_packages/` contents

To reproduce from source, install `requirements.txt`, download data as documented in `README.md`, then run the relevant scripts under `scripts/`.
