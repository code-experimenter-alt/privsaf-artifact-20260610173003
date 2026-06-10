from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


POSTGRES_QUERIES = {
    "candidate_likelihood_base": """
        select
            r.mechanism_id,
            r.epsilon,
            w.stuck_bucket,
            sum(w.logp_fault) as fault_loglik,
            sum(w.logp_normal) as normal_loglik,
            sum(w.logp_fault - w.logp_normal) as llr
        from privsaf_pg.reports r
        join privsaf_pg.mechanism_weights w
          on w.mechanism_id = r.mechanism_id
         and w.epsilon = r.epsilon
         and w.report_bucket = r.report_bucket
        group by r.mechanism_id, r.epsilon, w.stuck_bucket
    """,
    "candidate_likelihood_materialized_counts": """
        select
            c.mechanism_id,
            c.epsilon,
            w.stuck_bucket,
            sum(c.n * w.logp_fault) as fault_loglik,
            sum(c.n * w.logp_normal) as normal_loglik,
            sum(c.n * (w.logp_fault - w.logp_normal)) as llr
        from privsaf_pg.mv_report_counts c
        join privsaf_pg.mechanism_weights w
          on w.mechanism_id = c.mechanism_id
         and w.epsilon = c.epsilon
         and w.report_bucket = c.report_bucket
        group by c.mechanism_id, c.epsilon, w.stuck_bucket
    """,
    "candidate_topk": """
        select epsilon, stuck_bucket, llr
        from privsaf_pg.candidate_likelihoods
        order by llr desc
        limit 10
    """,
    "privacy_budget_trace": """
        select count(*) as reports,
               min(remaining_budget) as min_remaining_budget,
               avg(remaining_budget) as avg_remaining_budget
        from privsaf_pg.privacy_budget_trace
        where remaining_budget < 1000.0
    """,
    "repair_dashboard": """
        select
            device_id,
            count(*) as hourly_groups,
            avg(cleaned_mean) as avg_cleaned_mean,
            sum(repair_count) as repaired_reports,
            avg(uncertainty_proxy) as avg_uncertainty_proxy
        from privsaf_pg.repair_uncertainty_analytics
        group by device_id
    """,
    "provenance_event_drilldown": """
        select
            e.device_id,
            e.hour_id,
            e.flagged_reports,
            e.avg_posterior,
            h.cleaned_mean,
            h.uncertainty_proxy,
            l.remaining_budget
        from privsaf_pg.cleaned_event_windows e
        join privsaf_pg.repair_uncertainty_analytics h
          on h.device_id = e.device_id and h.hour_id = e.hour_id
        join privsaf_pg.privacy_ledger l
          on l.device_id = e.device_id
        where e.avg_posterior >= 0.72
        order by e.flagged_reports desc, e.avg_posterior desc
        limit 100
    """,
    "end_to_end_dashboard": """
        select
            case when l.remaining_budget < 980.0 then 'near_cap' else 'ok' end as budget_band,
            count(*) as hourly_groups,
            sum(coalesce(e.flagged_reports, 0)) as flagged_reports,
            avg(h.cleaned_mean) as avg_cleaned_mean,
            avg(h.uncertainty_proxy) as avg_uncertainty_proxy
        from privsaf_pg.repair_uncertainty_analytics h
        join privsaf_pg.privacy_ledger l
          on l.device_id = h.device_id
        left join privsaf_pg.cleaned_event_windows e
          on e.device_id = h.device_id and e.hour_id = h.hour_id
        group by budget_band
    """,
}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def postgres_binary() -> str:
    path = shutil.which("postgres")
    if path:
        return path
    for candidate in sorted(Path("/usr/lib/postgresql").glob("*/bin/postgres"), reverse=True):
        if candidate.exists():
            return str(candidate)
    return ""


def environment_audit(output_dir: Path, status: str, reason: str, server_version: str = "") -> None:
    rows = [
        {
            "component": "psql",
            "available": int(shutil.which("psql") is not None),
            "path_or_version": shutil.which("psql") or "",
        },
        {
            "component": "postgres",
            "available": int(bool(postgres_binary())),
            "path_or_version": postgres_binary(),
        },
        {
            "component": "python_psycopg2",
            "available": 0,
            "path_or_version": "",
        },
        {
            "component": "server_connection",
            "available": int(status == "complete"),
            "path_or_version": server_version,
        },
    ]
    try:
        import psycopg2  # type: ignore

        for row in rows:
            if row["component"] == "python_psycopg2":
                row["available"] = 1
                row["path_or_version"] = getattr(psycopg2, "__version__", "")
    except Exception:
        pass
    write_csv(output_dir / "private_telemetry_postgres_environment_audit.csv", rows, ["component", "available", "path_or_version"])
    (output_dir / "private_telemetry_postgres_integration_summary.json").write_text(
        json.dumps(
            {
                "status": status,
                "reason": reason,
                "dsn_env": "PRIVSAF_POSTGRES_DSN",
                "outputs_if_available": [
                    "private_telemetry_postgres_query_benchmark.csv",
                    "private_telemetry_postgres_queries.csv",
                    "private_telemetry_postgres_plans.csv",
                    "private_telemetry_postgres_verification.csv",
                    "private_telemetry_postgres_environment_audit.csv",
                ],
                "server_version": server_version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def timed(rows: list[dict[str, object]], stage: str, input_rows: int, fn) -> object:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    rows.append(
        {
            "stage": stage,
            "input_rows": input_rows,
            "output_rows": int(result) if isinstance(result, int) else 0,
            "elapsed_sec": elapsed,
            "throughput_rows_per_sec": input_rows / max(elapsed, 1e-12),
        }
    )
    return result


def timed_sql(rows: list[dict[str, object]], stage: str, input_rows: int, cur, sql: str, output_sql: str = "") -> int:
    start = time.perf_counter()
    cur.execute(sql)
    output_rows = 0
    if output_sql:
        cur.execute(output_sql)
        output_rows = int(cur.fetchone()[0])
    elapsed = time.perf_counter() - start
    rows.append(
        {
            "stage": stage,
            "input_rows": input_rows,
            "output_rows": output_rows,
            "elapsed_sec": elapsed,
            "throughput_rows_per_sec": input_rows / max(elapsed, 1e-12),
        }
    )
    return output_rows


def run_query(cur, name: str, sql: str, input_rows: int, rep: int) -> dict[str, object]:
    start = time.perf_counter()
    cur.execute(f"select count(*) from ({sql}) q")
    output_rows = int(cur.fetchone()[0])
    elapsed = time.perf_counter() - start
    return {
        "query_name": name,
        "rep": rep,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "elapsed_ms": elapsed * 1000.0,
        "throughput_rows_per_sec": input_rows / max(elapsed, 1e-12),
        "query": " ".join(sql.split()),
    }


def explain_query(cur, name: str, sql: str) -> list[dict[str, object]]:
    cur.execute("explain " + sql)
    rows = []
    for idx, item in enumerate(cur.fetchall()):
        detail = str(item[0])
        if detail.strip():
            rows.append({"query_name": name, "plan_step": idx, "detail": detail})
    return rows


def main() -> int:
    output_dir = Path(os.environ.get("PRIVATE_PIPELINE_RESULTS", str(RESULTS)))
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import psycopg2  # type: ignore
        from psycopg2.extras import execute_values  # type: ignore
    except Exception as exc:
        environment_audit(output_dir, "not_run", f"missing psycopg2: {exc}")
        print(f"PostgreSQL integration not run: missing psycopg2 ({exc})")
        return 0

    dsn = os.environ.get("PRIVSAF_POSTGRES_DSN", "")
    if not dsn:
        environment_audit(output_dir, "not_run", "PRIVSAF_POSTGRES_DSN is not set")
        print("PostgreSQL integration not run: PRIVSAF_POSTGRES_DSN is not set")
        return 0

    from run_private_telemetry_pipeline_queries import make_mechanism_weights, make_reports

    total_rows = int(os.environ.get("PRIVATE_POSTGRES_ROWS", "200000"))
    devices = int(os.environ.get("PRIVATE_POSTGRES_DEVICES", "2000"))
    records = make_reports(total_rows, devices, seed=17001)
    mechanism_rows = make_mechanism_weights()
    rows: list[dict[str, object]] = []

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute("select version()")
        server_version = str(cur.fetchone()[0])
        schema_sql = (ROOT / "sql" / "private_telemetry_postgres_schema.sql").read_text(encoding="utf-8")
        timed(rows, "create_schema", 1, lambda: (cur.execute(schema_sql), 1)[1])
        timed(
            rows,
            "load_mechanism_matrix",
            len(mechanism_rows),
            lambda: (
                execute_values(
                    cur,
                    "insert into privsaf_pg.mechanism_weights values %s",
                    mechanism_rows,
                    page_size=1000,
                ),
                len(mechanism_rows),
            )[1],
        )
        policy_rows = [
            ("privsaf_mixture", "iid stuck-at", "cleaning_records", "post_processing"),
            ("privsaf_hmm", "persistent flatline", "cleaning_records", "post_processing"),
            ("pm_window_glr", "field-QC local regime", "cleaning_records", "post_processing"),
        ]
        timed(
            rows,
            "load_router_policy",
            len(policy_rows),
            lambda: (
                execute_values(cur, "insert into privsaf_pg.router_policy values %s", policy_rows),
                len(policy_rows),
            )[1],
        )
        timed(
            rows,
            "ingest_reports",
            total_rows,
            lambda: (
                execute_values(cur, "insert into privsaf_pg.reports values %s", records, page_size=5000),
                total_rows,
            )[1],
        )
        timed(
            rows,
            "refresh_counts",
            total_rows,
            lambda: (
                cur.execute("refresh materialized view privsaf_pg.mv_report_counts"),
                cur.execute("select count(*) from privsaf_pg.mv_report_counts"),
                int(cur.fetchone()[0]),
            )[2],
        )
        timed(
            rows,
            "derive_cleaning_records",
            total_rows,
            lambda: (
                cur.execute(
                    """
                    insert into privsaf_pg.cleaning_records
                    select
                        report_id, device_id, ts,
                        case
                            when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.80 then 'privsaf_hmm'
                            when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.65 then 'pm_window_glr'
                            else 'privsaf_mixture'
                        end,
                        ((report_bucket * 17 + device_id) % 100) / 100.0,
                        case when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.65 then report_bucket else null end,
                        0.20,
                        case when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.80 then 'repair'
                             when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.65 then 'flag'
                             else 'pass' end,
                        (2.0 * report_bucket / 31.0) - 1.0,
                        'router_v1'
                    from privsaf_pg.reports
                    """
                ),
                total_rows,
            )[1],
        )
        timed_sql(
            rows,
            "write_provenance_edges",
            total_rows,
            cur,
            """
            insert into privsaf_pg.provenance_edges
            select
                c.report_id,
                'reports',
                c.report_id,
                c.operator,
                r.mechanism_id,
                r.calibration_version
            from privsaf_pg.cleaning_records c
            join privsaf_pg.reports r using(report_id)
            """,
            "select count(*) from privsaf_pg.provenance_edges",
        )
        timed_sql(
            rows,
            "write_privacy_ledger",
            total_rows,
            cur,
            """
            insert into privsaf_pg.privacy_ledger
            select
                device_id,
                count(*) as reports,
                sum(epsilon) as epsilon_sum,
                1000.0 as device_budget,
                1000.0 - sum(epsilon) as remaining_budget
            from privsaf_pg.reports
            group by device_id
            """,
            "select count(*) from privsaf_pg.privacy_ledger",
        )
        for stage, view in [
            ("refresh_candidate_likelihoods", "candidate_likelihoods"),
            ("refresh_privacy_budget_trace", "privacy_budget_trace"),
            ("refresh_cleaned_event_windows", "cleaned_event_windows"),
            ("refresh_repair_uncertainty_analytics", "repair_uncertainty_analytics"),
        ]:
            timed_sql(
                rows,
                stage,
                total_rows,
                cur,
                f"refresh materialized view privsaf_pg.{view}",
                f"select count(*) from privsaf_pg.{view}",
            )

        query_rows: list[dict[str, object]] = []
        plan_rows: list[dict[str, object]] = []
        for name, sql in POSTGRES_QUERIES.items():
            plan_rows.extend(explain_query(cur, name, sql))
            for rep in range(3):
                query_rows.append(run_query(cur, name, sql, total_rows, rep))

        checks = []
        for check_name, expected, sql in [
            ("count:reports", total_rows, "select count(*) from privsaf_pg.reports"),
            ("count:cleaning_records", total_rows, "select count(*) from privsaf_pg.cleaning_records"),
            ("count:provenance_edges", total_rows, "select count(*) from privsaf_pg.provenance_edges"),
            ("sum:mv_report_counts", total_rows, "select sum(n) from privsaf_pg.mv_report_counts"),
            ("count:candidate_likelihoods", 96, "select count(*) from privsaf_pg.candidate_likelihoods"),
            ("sum:privacy_ledger", total_rows, "select sum(reports) from privsaf_pg.privacy_ledger"),
            ("count:privacy_budget_trace", total_rows, "select count(*) from privsaf_pg.privacy_budget_trace"),
        ]:
            cur.execute(sql)
            observed = int(cur.fetchone()[0])
            checks.append({"check": check_name, "expected": expected, "observed": observed, "status": "pass" if observed == expected else "fail"})
        for check_name, sql in [
            ("nonempty:cleaned_event_windows", "select count(*) from privsaf_pg.cleaned_event_windows"),
            ("nonempty:repair_uncertainty_analytics", "select count(*) from privsaf_pg.repair_uncertainty_analytics"),
        ]:
            cur.execute(sql)
            observed = int(cur.fetchone()[0])
            checks.append({"check": check_name, "expected": ">0", "observed": observed, "status": "pass" if observed > 0 else "fail"})
        checks.extend(
            [
                {
                    "check": "query_rows",
                    "expected": len(POSTGRES_QUERIES) * 3,
                    "observed": len(query_rows),
                    "status": "pass" if len(query_rows) == len(POSTGRES_QUERIES) * 3 else "fail",
                },
                {
                    "check": "plan_rows",
                    "expected": ">0",
                    "observed": len(plan_rows),
                    "status": "pass" if len(plan_rows) > 0 else "fail",
                },
                {
                    "check": "server_version",
                    "expected": "PostgreSQL",
                    "observed": server_version.split(" on ", 1)[0],
                    "status": "pass" if "PostgreSQL" in server_version else "fail",
                },
            ]
        )
        conn.commit()
    conn.close()

    write_csv(
        output_dir / "private_telemetry_postgres_query_benchmark.csv",
        rows,
        ["stage", "input_rows", "output_rows", "elapsed_sec", "throughput_rows_per_sec"],
    )
    write_csv(
        output_dir / "private_telemetry_postgres_verification.csv",
        checks,
        ["check", "expected", "observed", "status"],
    )
    write_csv(
        output_dir / "private_telemetry_postgres_queries.csv",
        query_rows,
        ["query_name", "rep", "input_rows", "output_rows", "elapsed_ms", "throughput_rows_per_sec", "query"],
    )
    write_csv(output_dir / "private_telemetry_postgres_plans.csv", plan_rows, ["query_name", "plan_step", "detail"])
    status = "pass" if all(row["status"] == "pass" for row in checks) else "fail"
    environment_audit(output_dir, "complete" if status == "pass" else "failed", "PostgreSQL pipeline completed", server_version)
    summary = {
        "status": status,
        "postgres_version": server_version,
        "rows": total_rows,
        "devices": devices,
        "build_rows": len(rows),
        "query_rows": len(query_rows),
        "plan_rows": len(plan_rows),
        "checks": len(checks),
        "failed_checks": [row for row in checks if row["status"] != "pass"],
        "database": "privsaf_pg_benchmark",
        "schema": "privsaf_pg",
        "schema_sql": "sql/private_telemetry_postgres_schema.sql",
    }
    (output_dir / "private_telemetry_postgres_integration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
