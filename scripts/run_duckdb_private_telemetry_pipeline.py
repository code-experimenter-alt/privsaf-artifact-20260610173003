from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from run_private_telemetry_pipeline_queries import make_mechanism_weights, make_reports


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SQL_DIR = ROOT / "sql"
TMP = Path("/tmp/privsaf_duckdb_pipeline")


def duckdb_cli() -> str | None:
    candidates = [
        os.environ.get("DUCKDB_CLI", ""),
        "/tmp/duckdb_cli/duckdb",
        shutil.which("duckdb") or "",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def q(path: Path) -> str:
    return str(path).replace("'", "''")


def run_duckdb(cli: str, db_path: Path, sql: str, csv_mode: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [cli]
    if csv_mode:
        cmd.extend(["-csv", "-noheader"])
    cmd.extend([str(db_path), "-c", sql])
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=True)


def timed_stage(rows: list[dict[str, object]], cli: str, db_path: Path, stage: str, input_rows: int, sql: str, output_sql: str = "") -> int:
    start = time.perf_counter()
    run_duckdb(cli, db_path, sql)
    elapsed = time.perf_counter() - start
    output_rows = 0
    if output_sql:
        output_rows = int(run_duckdb(cli, db_path, output_sql, csv_mode=True).stdout.strip().splitlines()[0])
    rows.append(
        {
            "stage": stage,
            "input_rows": input_rows,
            "output_rows": output_rows,
            "elapsed_sec": elapsed,
            "throughput_rows_per_sec": input_rows / max(elapsed, 1e-12),
            "sql": " ".join(sql.split()),
        }
    )
    return output_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_input_csvs(total_rows: int, devices: int) -> tuple[Path, Path, Path]:
    TMP.mkdir(parents=True, exist_ok=True)
    reports_path = TMP / "reports.csv"
    weights_path = TMP / "mechanism_weights.csv"
    policy_path = TMP / "router_policy.csv"

    with reports_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["report_id", "device_id", "ts", "report_bucket", "epsilon", "mechanism_id", "calibration_version"])
        writer.writerows(make_reports(total_rows, devices, seed=17001))

    with weights_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["mechanism_id", "epsilon", "report_bucket", "stuck_bucket", "logp_normal", "logp_fault"])
        writer.writerows(make_mechanism_weights())

    with policy_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["operator", "semantic_layer", "output_relation", "privacy_effect"])
        writer.writerows(
            [
                ("privsaf_mixture", "iid stuck-at", "cleaning_records", "post_processing"),
                ("privsaf_hmm", "persistent flatline", "cleaning_records", "post_processing"),
                ("pm_window_glr", "field-QC local regime", "cleaning_records", "post_processing"),
            ]
        )
    return reports_path, weights_path, policy_path


SCHEMA_SQL = """
create table reports(
    report_id bigint primary key,
    device_id integer not null,
    ts integer not null,
    report_bucket integer not null,
    epsilon double not null,
    mechanism_id varchar not null,
    calibration_version varchar not null
);

create table mechanism_weights(
    mechanism_id varchar not null,
    epsilon double not null,
    report_bucket integer not null,
    stuck_bucket integer not null,
    logp_normal double not null,
    logp_fault double not null
);

create table router_policy(
    operator varchar primary key,
    semantic_layer varchar not null,
    output_relation varchar not null,
    privacy_effect varchar not null
);
"""


DERIVE_SQL = """
create table mv_report_counts as
select mechanism_id, epsilon, report_bucket, count(*) as n
from reports
group by mechanism_id, epsilon, report_bucket;

create table candidate_likelihoods as
select
    c.mechanism_id,
    c.epsilon,
    w.stuck_bucket,
    sum(c.n * w.logp_fault) as fault_loglik,
    sum(c.n * w.logp_normal) as normal_loglik,
    sum(c.n * (w.logp_fault - w.logp_normal)) as llr
from mv_report_counts c
join mechanism_weights w
  on w.mechanism_id = c.mechanism_id
 and w.epsilon = c.epsilon
 and w.report_bucket = c.report_bucket
group by c.mechanism_id, c.epsilon, w.stuck_bucket;

create table cleaning_records as
select
    report_id,
    device_id,
    ts,
    case
        when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.80 then 'privsaf_hmm'
        when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.65 then 'pm_window_glr'
        else 'privsaf_mixture'
    end as operator,
    ((report_bucket * 17 + device_id) % 100) / 100.0 as posterior,
    case when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.65 then report_bucket else null end as stuck_bucket,
    0.20 as fault_ratio,
    case when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.80 then 'repair'
         when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.65 then 'flag'
         else 'pass' end as action,
    (2.0 * report_bucket / 31.0) - 1.0 as repair_value,
    'router_v1' as provenance_version
from reports;

create table provenance_edges as
select report_id as output_report_id, 'reports' as source_table, report_id as source_key,
       operator as transform, 'pm32' as mechanism_id, 'cal_v1' as calibration_version
from cleaning_records;

create table privacy_ledger as
select device_id, count(*) as reports, sum(epsilon) as epsilon_sum,
       1000.0 as device_budget, 1000.0 - sum(epsilon) as remaining_budget
from reports
group by device_id;

create table privacy_budget_trace as
select
    report_id,
    device_id,
    ts,
    epsilon,
    cumulative_epsilon,
    1000.0 - cumulative_epsilon as remaining_budget
from (
    select
        report_id,
        device_id,
        ts,
        epsilon,
        sum(epsilon) over (
            partition by device_id
            order by ts
            rows between unbounded preceding and current row
        ) as cumulative_epsilon
    from reports
);

create table cleaned_event_windows as
select
    c.device_id,
    floor(c.ts / 60) as hour_id,
    min(c.ts) as start_ts,
    max(c.ts) as end_ts,
    count(*) as flagged_reports,
    avg(c.posterior) as avg_posterior,
    string_agg(distinct p.semantic_layer, ',') as routed_semantics
from cleaning_records c
join router_policy p on p.operator = c.operator
where c.action in ('flag', 'repair')
group by c.device_id, floor(c.ts / 60)
having count(*) >= 2;

create table repair_uncertainty_analytics as
select
    r.device_id,
    floor(r.ts / 60) as hour_id,
    count(*) as reports,
    avg(case when c.action = 'repair'
             then c.repair_value
             else (2.0 * r.report_bucket / 31.0) - 1.0 end) as cleaned_mean,
    avg((2.0 * r.report_bucket / 31.0) - 1.0) as unrepaired_mean,
    avg(c.posterior * (1.0 - c.posterior)) as uncertainty_proxy,
    sum(case when c.action = 'repair' then 1 else 0 end) as repair_count
from reports r
join cleaning_records c using(report_id)
group by r.device_id, floor(r.ts / 60);
"""


QUERIES = {
    "candidate_likelihood_materialized_counts": """
        select c.mechanism_id, c.epsilon, w.stuck_bucket,
               sum(c.n * w.logp_fault) as fault_loglik,
               sum(c.n * w.logp_normal) as normal_loglik,
               sum(c.n * (w.logp_fault - w.logp_normal)) as llr
        from mv_report_counts c
        join mechanism_weights w
          on w.mechanism_id = c.mechanism_id
         and w.epsilon = c.epsilon
         and w.report_bucket = c.report_bucket
        group by c.mechanism_id, c.epsilon, w.stuck_bucket
    """,
    "candidate_topk": """
        select epsilon, stuck_bucket, llr
        from candidate_likelihoods
        order by llr desc
        limit 10
    """,
    "privacy_budget_trace": """
        select count(*) as reports,
               min(remaining_budget) as min_remaining_budget,
               avg(remaining_budget) as avg_remaining_budget
        from privacy_budget_trace
        where remaining_budget < 1000.0
    """,
    "repair_dashboard": """
        select device_id, count(*) as hourly_groups, avg(cleaned_mean) as avg_cleaned_mean,
               sum(repair_count) as repaired_reports, avg(uncertainty_proxy) as avg_uncertainty_proxy
        from repair_uncertainty_analytics
        group by device_id
    """,
    "end_to_end_dashboard": """
        select case when l.remaining_budget < 980.0 then 'near_cap' else 'ok' end as budget_band,
               count(*) as hourly_groups,
               sum(coalesce(e.flagged_reports, 0)) as flagged_reports,
               avg(h.cleaned_mean) as avg_cleaned_mean,
               avg(h.uncertainty_proxy) as avg_uncertainty_proxy
        from repair_uncertainty_analytics h
        join privacy_ledger l using(device_id)
        left join cleaned_event_windows e
          on e.device_id = h.device_id and e.hour_id = h.hour_id
        group by budget_band
    """,
}


def run_query(cli: str, db_path: Path, name: str, sql: str, input_rows: int, rep: int) -> dict[str, object]:
    count_sql = f"select count(*) from ({sql}) q"
    start = time.perf_counter()
    result = run_duckdb(cli, db_path, count_sql, csv_mode=True)
    elapsed = time.perf_counter() - start
    output_rows = int(result.stdout.strip().splitlines()[0])
    return {
        "query_name": name,
        "rep": rep,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "elapsed_ms": elapsed * 1000.0,
        "throughput_rows_per_sec": input_rows / max(elapsed, 1e-12),
        "query": " ".join(sql.split()),
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    SQL_DIR.mkdir(parents=True, exist_ok=True)
    cli = duckdb_cli()
    env_rows = [
        {"component": "duckdb_cli", "available": int(cli is not None), "path_or_version": ""},
    ]
    if cli is None:
        write_csv(RESULTS / "private_telemetry_duckdb_environment_audit.csv", env_rows, ["component", "available", "path_or_version"])
        (RESULTS / "private_telemetry_duckdb_summary.json").write_text(json.dumps({"status": "not_run", "reason": "duckdb CLI not found"}, indent=2), encoding="utf-8")
        print("DuckDB integration not run: duckdb CLI not found")
        return 0

    version = subprocess.run([cli, "--version"], text=True, capture_output=True, check=True).stdout.strip()
    env_rows[0]["path_or_version"] = f"{cli} {version}"
    write_csv(RESULTS / "private_telemetry_duckdb_environment_audit.csv", env_rows, ["component", "available", "path_or_version"])

    total_rows = int(os.environ.get("PRIVATE_DUCKDB_ROWS", "200000"))
    devices = int(os.environ.get("PRIVATE_DUCKDB_DEVICES", "2000"))
    db_path = RESULTS / "private_telemetry_duckdb_pipeline.duckdb"
    if db_path.exists():
        db_path.unlink()
    reports_path, weights_path, policy_path = write_input_csvs(total_rows, devices)

    schema_file = SQL_DIR / "private_telemetry_duckdb_schema.sql"
    schema_file.write_text(SCHEMA_SQL + "\n" + DERIVE_SQL, encoding="utf-8")

    stage_rows: list[dict[str, object]] = []
    timed_stage(stage_rows, cli, db_path, "create_schema", 1, SCHEMA_SQL)
    timed_stage(stage_rows, cli, db_path, "load_reports", total_rows, f"copy reports from '{q(reports_path)}' (header, delim ',');", "select count(*) from reports")
    timed_stage(stage_rows, cli, db_path, "load_mechanism_weights", 3072, f"copy mechanism_weights from '{q(weights_path)}' (header, delim ',');", "select count(*) from mechanism_weights")
    timed_stage(stage_rows, cli, db_path, "load_router_policy", 3, f"copy router_policy from '{q(policy_path)}' (header, delim ',');", "select count(*) from router_policy")
    timed_stage(stage_rows, cli, db_path, "derive_pipeline_relations", total_rows, DERIVE_SQL, "select count(*) from cleaning_records")

    query_rows: list[dict[str, object]] = []
    plan_rows: list[dict[str, object]] = []
    for name, sql in QUERIES.items():
        for idx, line in enumerate(run_duckdb(cli, db_path, "explain " + sql).stdout.splitlines()):
            if line.strip():
                plan_rows.append({"query_name": name, "plan_step": idx, "detail": line})
        for rep in range(3):
            query_rows.append(run_query(cli, db_path, name, sql, total_rows, rep))

    checks = []
    for check_name, expected, sql in [
        ("reports", total_rows, "select count(*) from reports"),
        ("cleaning_records", total_rows, "select count(*) from cleaning_records"),
        ("provenance_edges", total_rows, "select count(*) from provenance_edges"),
        ("mv_report_counts_sum", total_rows, "select sum(n) from mv_report_counts"),
        ("privacy_ledger_sum", total_rows, "select sum(reports) from privacy_ledger"),
        ("repair_groups_nonempty", 1, "select (count(*) > 0)::integer from repair_uncertainty_analytics"),
    ]:
        observed = int(float(run_duckdb(cli, db_path, sql, csv_mode=True).stdout.strip().splitlines()[0]))
        checks.append({"check": check_name, "expected": expected, "observed": observed, "status": "pass" if expected == observed else "fail"})

    write_csv(
        RESULTS / "private_telemetry_duckdb_build.csv",
        stage_rows,
        ["stage", "input_rows", "output_rows", "elapsed_sec", "throughput_rows_per_sec", "sql"],
    )
    write_csv(
        RESULTS / "private_telemetry_duckdb_queries.csv",
        query_rows,
        ["query_name", "rep", "input_rows", "output_rows", "elapsed_ms", "throughput_rows_per_sec", "query"],
    )
    write_csv(RESULTS / "private_telemetry_duckdb_plans.csv", plan_rows, ["query_name", "plan_step", "detail"])
    write_csv(RESULTS / "private_telemetry_duckdb_verification.csv", checks, ["check", "expected", "observed", "status"])
    summary = {
        "status": "pass" if all(row["status"] == "pass" for row in checks) else "fail",
        "duckdb_version": version,
        "rows": total_rows,
        "devices": devices,
        "build_rows": len(stage_rows),
        "query_rows": len(query_rows),
        "plan_rows": len(plan_rows),
        "checks": len(checks),
        "failed_checks": [row for row in checks if row["status"] != "pass"],
        "database": str(db_path.relative_to(ROOT)),
        "database_bytes": db_path.stat().st_size if db_path.exists() else 0,
        "schema_sql": str(schema_file.relative_to(ROOT)),
    }
    (RESULTS / "private_telemetry_duckdb_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
