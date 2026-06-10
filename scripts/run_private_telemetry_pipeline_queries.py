from __future__ import annotations

import csv
import json
import math
import os
import random
import sqlite3
import statistics
import time
from pathlib import Path

from private_telemetry_benchmark_config import load_benchmark_config, result_path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def timed(label: str, rows: int, output_rows: int, query: str, fn) -> dict[str, object]:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    if isinstance(result, int):
        output_rows = result
    return {
        "workload": label,
        "input_rows": rows,
        "output_rows": output_rows,
        "elapsed_sec": elapsed,
        "throughput_rows_per_sec": rows / max(elapsed, 1e-12),
        "query": " ".join(query.split()),
    }


def make_reports(total_rows: int, devices: int, seed: int) -> list[tuple[int, int, int, int, float, str, str]]:
    rng = random.Random(seed)
    records: list[tuple[int, int, int, int, float, str, str]] = []
    eps_values = [0.5, 1.0, 2.0]
    for report_id in range(total_rows):
        device_id = report_id % devices
        ts = report_id // devices
        bucket = rng.randrange(32)
        epsilon = eps_values[(device_id + report_id) % len(eps_values)]
        records.append((report_id, device_id, ts, bucket, epsilon, "pm32", "cal_v1"))
    return records


def make_mechanism_weights() -> list[tuple[str, float, int, int, float, float]]:
    rows: list[tuple[str, float, int, int, float, float]] = []
    for epsilon in [0.5, 1.0, 2.0]:
        normal_prob = 1.0 / 32.0
        for report_bucket in range(32):
            for stuck_bucket in range(32):
                distance = abs(report_bucket - stuck_bucket)
                decay = math.exp(-epsilon * distance / 8.0)
                fault_prob = 0.002 + 0.55 * decay
                rows.append(
                    (
                        "pm32",
                        epsilon,
                        report_bucket,
                        stuck_bucket,
                        math.log(normal_prob),
                        math.log(fault_prob),
                    )
                )
    return rows


def collect_plans(cur: sqlite3.Cursor, scale_rows: int, devices: int, storage_mode: str) -> list[dict[str, object]]:
    plan_queries = {
        "candidate_likelihood_join": """
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
        "provenance_drilldown": """
            select r.report_id, r.device_id, r.ts, c.operator, c.posterior, p.transform
            from cleaning_records c
            join reports r using(report_id)
            join provenance_edges p on p.output_report_id = c.report_id
            where c.action in ('flag', 'repair')
            order by c.posterior desc
            limit 100
        """,
        "privacy_budget_window": """
            select report_id, device_id, ts, epsilon,
                   sum(epsilon) over (
                       partition by device_id
                       order by ts
                       rows between unbounded preceding and current row
                   ) as cumulative_epsilon
            from reports
        """,
        "repair_aggregate_query": """
            select r.device_id, r.ts / 60 as hour_id,
                   count(*) as reports,
                   avg(case when c.action = 'repair'
                            then c.repair_value
                            else (2.0 * r.report_bucket / 31.0) - 1.0 end) as cleaned_mean,
                   avg(c.posterior * (1.0 - c.posterior)) as uncertainty_proxy
            from reports r
            join cleaning_records c using(report_id)
            group by r.device_id, r.ts / 60
        """,
    }
    rows: list[dict[str, object]] = []
    for workload, query in plan_queries.items():
        for idx, (_, _, _, detail) in enumerate(cur.execute("explain query plan " + query).fetchall()):
            rows.append(
                {
                    "scale_rows": scale_rows,
                    "devices": devices,
                    "storage_mode": storage_mode,
                    "workload": workload,
                    "plan_step": idx,
                    "detail": detail,
                }
            )
    return rows


def fetch_timed(
    cur: sqlite3.Cursor,
    query_name: str,
    design: str,
    input_rows: int,
    rep: int,
    query: str,
) -> dict[str, object]:
    start = time.perf_counter()
    rows = cur.execute(query).fetchall()
    elapsed = time.perf_counter() - start
    return {
        "query_name": query_name,
        "design": design,
        "rep": rep,
        "input_rows": input_rows,
        "output_rows": len(rows),
        "elapsed_ms": elapsed * 1000.0,
        "throughput_rows_per_sec": input_rows / max(elapsed, 1e-12),
        "query": " ".join(query.split()),
    }


def run_query_suite(db_path: Path, total_rows: int, devices: int, reps: int = 3) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.executescript(
        """
        create index if not exists idx_candidate_llr on candidate_likelihoods(epsilon, llr desc);
        create index if not exists idx_hourly_device_hour on hourly_cleaned_analytics(device_id, hour_id);
        create index if not exists idx_repair_device_hour on repair_uncertainty_analytics(device_id, hour_id);
        create index if not exists idx_budget_remaining on privacy_budget_trace(remaining_budget, device_id, ts);
        """
    )
    conn.commit()

    queries = [
        (
            "likelihood_join_direct_reports",
            "base_tables",
            total_rows,
            """
            select
                r.mechanism_id,
                r.epsilon,
                w.stuck_bucket,
                sum(w.logp_fault) as fault_loglik,
                sum(w.logp_normal) as normal_loglik,
                sum(w.logp_fault - w.logp_normal) as llr
            from reports as r not indexed
            join mechanism_weights as w
              on w.mechanism_id = r.mechanism_id
             and w.epsilon = r.epsilon
             and w.report_bucket = r.report_bucket
            group by r.mechanism_id, r.epsilon, w.stuck_bucket
            """,
        ),
        (
            "likelihood_join_materialized_counts",
            "materialized_counts",
            total_rows,
            """
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
            group by c.mechanism_id, c.epsilon, w.stuck_bucket
            """,
        ),
        (
            "candidate_topk_lookup",
            "persisted_candidates",
            96,
            """
            select epsilon, stuck_bucket, llr
            from candidate_likelihoods
            order by llr desc
            limit 10
            """,
        ),
        (
            "budget_window_direct_reports",
            "base_window",
            total_rows,
            """
            select count(*) as reports,
                   min(remaining_budget) as min_remaining_budget,
                   avg(remaining_budget) as avg_remaining_budget
            from (
                select 1000.0 - sum(epsilon) over (
                           partition by device_id
                           order by ts
                           rows between unbounded preceding and current row
                       ) as remaining_budget
                from reports as r not indexed
            )
            """,
        ),
        (
            "budget_window_materialized",
            "materialized_window",
            total_rows,
            """
            select count(*) as reports,
                   min(remaining_budget) as min_remaining_budget,
                   avg(remaining_budget) as avg_remaining_budget
            from privacy_budget_trace
            where remaining_budget < 1000.0
            """,
        ),
        (
            "repair_aggregate_direct_join",
            "base_join_aggregate",
            total_rows,
            """
            select
                r.device_id,
                r.ts / 60 as hour_id,
                count(*) as reports,
                avg(case when c.action = 'repair'
                         then c.repair_value
                         else (2.0 * r.report_bucket / 31.0) - 1.0 end) as cleaned_mean,
                avg((2.0 * r.report_bucket / 31.0) - 1.0) as unrepaired_mean,
                avg(c.posterior * (1.0 - c.posterior)) as uncertainty_proxy,
                sum(case when c.action = 'repair' then 1 else 0 end) as repair_count
            from reports r
            join cleaning_records c using(report_id)
            group by r.device_id, r.ts / 60
            """,
        ),
        (
            "repair_dashboard_materialized",
            "materialized_analytics",
            total_rows,
            """
            select
                device_id,
                count(*) as hourly_groups,
                avg(cleaned_mean) as avg_cleaned_mean,
                sum(repair_count) as repaired_reports,
                avg(uncertainty_proxy) as avg_uncertainty_proxy
            from repair_uncertainty_analytics
            group by device_id
            """,
        ),
        (
            "provenance_event_drilldown",
            "indexed_provenance",
            total_rows,
            """
            select
                e.device_id,
                e.hour_id,
                e.flagged_reports,
                e.avg_posterior,
                h.cleaned_mean,
                h.uncertainty_proxy,
                l.remaining_budget
            from cleaned_event_windows e
            join repair_uncertainty_analytics h
              on h.device_id = e.device_id and h.hour_id = e.hour_id
            join privacy_ledger l using(device_id)
            where e.avg_posterior >= 0.72
            order by e.flagged_reports desc, e.avg_posterior desc
            limit 100
            """,
        ),
        (
            "end_to_end_dashboard_query",
            "ingest_clean_analyze",
            total_rows,
            """
            select
                case when l.remaining_budget < 980.0 then 'near_cap' else 'ok' end as budget_band,
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
        ),
    ]

    rows: list[dict[str, object]] = []
    plan_rows: list[dict[str, object]] = []
    for query_name, design, input_rows, query in queries:
        for idx, (_, _, _, detail) in enumerate(cur.execute("explain query plan " + query).fetchall()):
            plan_rows.append(
                {
                    "scale_rows": total_rows,
                    "devices": devices,
                    "query_name": query_name,
                    "design": design,
                    "plan_step": idx,
                    "detail": detail,
                }
            )
        for rep in range(reps):
            row = fetch_timed(cur, query_name, design, input_rows, rep, query)
            row["scale_rows"] = total_rows
            row["devices"] = devices
            rows.append(row)

    conn.close()
    return rows, plan_rows


def run_scale(
    total_rows: int,
    devices: int,
    seed: int,
    storage_mode: str = "memory",
    db_path: Path | None = None,
) -> list[dict[str, object]]:
    if db_path is not None and db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(":memory:" if db_path is None else str(db_path))
    cur = conn.cursor()
    if db_path is None:
        pragmas = """
        pragma journal_mode = off;
        pragma synchronous = off;
        pragma temp_store = memory;
        """
    else:
        pragmas = """
        pragma journal_mode = wal;
        pragma synchronous = normal;
        pragma temp_store = memory;
        """
    cur.executescript(
        pragmas
        + """
        create table reports(
            report_id integer primary key,
            device_id integer not null,
            ts integer not null,
            report_bucket integer not null,
            epsilon real not null,
            mechanism_id text not null,
            calibration_version text not null
        );

        create table cleaning_records(
            report_id integer primary key,
            device_id integer not null,
            ts integer not null,
            operator text not null,
            posterior real not null,
            stuck_bucket integer,
            fault_ratio real not null,
            action text not null,
            repair_value real,
            provenance_version text not null
        );

        create table provenance_edges(
            output_report_id integer not null,
            source_table text not null,
            source_key integer not null,
            transform text not null,
            mechanism_id text not null,
            calibration_version text not null
        );

        create table privacy_ledger(
            device_id integer primary key,
            reports integer not null,
            epsilon_sum real not null,
            device_budget real not null,
            remaining_budget real not null
        );

        create table mechanism_weights(
            mechanism_id text not null,
            epsilon real not null,
            report_bucket integer not null,
            stuck_bucket integer not null,
            logp_normal real not null,
            logp_fault real not null,
            primary key(mechanism_id, epsilon, report_bucket, stuck_bucket)
        );

        create table router_policy(
            operator text primary key,
            semantic_layer text not null,
            output_relation text not null,
            privacy_effect text not null
        );
        """
    )
    conn.commit()

    rows: list[dict[str, object]] = []
    records = make_reports(total_rows, devices, seed)
    mechanism_rows = make_mechanism_weights()

    rows.append(
        timed(
            "load_mechanism_matrix",
            len(mechanism_rows),
            len(mechanism_rows),
            "insert into mechanism_weights values (?, ?, ?, ?, ?, ?)",
            lambda: cur.executemany("insert into mechanism_weights values (?, ?, ?, ?, ?, ?)", mechanism_rows).rowcount,
        )
    )
    conn.commit()

    policy_rows = [
        ("privsaf_mixture", "iid stuck-at", "cleaning_records", "post_processing"),
        ("privsaf_hmm", "persistent flatline", "cleaning_records", "post_processing"),
        ("pm_window_glr", "field-QC local regime", "cleaning_records", "post_processing"),
    ]
    rows.append(
        timed(
            "load_router_policy",
            len(policy_rows),
            len(policy_rows),
            "insert into router_policy values (?, ?, ?, ?)",
            lambda: cur.executemany("insert into router_policy values (?, ?, ?, ?)", policy_rows).rowcount,
        )
    )
    conn.commit()

    rows.append(
        timed(
            "ingest_reports",
            total_rows,
            total_rows,
            "insert into reports values (?, ?, ?, ?, ?, ?, ?)",
            lambda: cur.executemany("insert into reports values (?, ?, ?, ?, ?, ?, ?)", records).rowcount,
        )
    )
    conn.commit()

    rows.append(
        timed(
            "build_indexes",
            total_rows,
            0,
            "create indexes on reports(device_id, ts), reports(epsilon, report_bucket)",
            lambda: cur.executescript(
                """
                create index idx_reports_device_ts on reports(device_id, ts);
                create index idx_reports_channel on reports(mechanism_id, epsilon, report_bucket);
                """
            ),
        )
    )
    conn.commit()

    materialize_counts_sql = """
        create table mv_report_counts as
        select mechanism_id, epsilon, report_bucket, count(*) as n
        from reports
        group by mechanism_id, epsilon, report_bucket
    """
    rows.append(
        timed(
            "materialized_counts",
            total_rows,
            0,
            materialize_counts_sql,
            lambda: (cur.execute(materialize_counts_sql), cur.execute("select count(*) from mv_report_counts").fetchone()[0])[1],
        )
    )
    conn.commit()

    candidate_likelihood_sql = """
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
        group by c.mechanism_id, c.epsilon, w.stuck_bucket
    """
    rows.append(
        timed(
            "candidate_likelihood_join",
            total_rows,
            0,
            candidate_likelihood_sql,
            lambda: (cur.execute(candidate_likelihood_sql), cur.execute("select count(*) from candidate_likelihoods").fetchone()[0])[1],
        )
    )
    conn.commit()

    cleaning_sql = """
        insert into cleaning_records
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
            case when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.65 then report_bucket else null end,
            0.20 as fault_ratio,
            case when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.80 then 'repair'
                 when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.65 then 'flag'
                 else 'pass' end as action,
            (2.0 * report_bucket / 31.0) - 1.0 as repair_value,
            'router_v1' as provenance_version
        from reports
    """
    rows.append(
        timed(
            "derive_cleaning_records",
            total_rows,
            total_rows,
            cleaning_sql,
            lambda: cur.execute(cleaning_sql).rowcount,
        )
    )
    conn.commit()

    provenance_sql = """
        insert into provenance_edges
        select report_id, 'reports', report_id, operator, 'pm32', 'cal_v1'
        from cleaning_records
    """
    rows.append(
        timed(
            "write_provenance",
            total_rows,
            total_rows,
            provenance_sql,
            lambda: cur.execute(provenance_sql).rowcount,
        )
    )
    conn.commit()
    rows.append(
        timed(
            "build_cleaning_indexes",
            total_rows,
            0,
            "create indexes on cleaning_records(action, posterior), cleaning_records(device_id, ts), provenance_edges(output_report_id)",
            lambda: cur.executescript(
                """
                create index idx_cleaning_action on cleaning_records(action, posterior);
                create index idx_cleaning_device_ts on cleaning_records(device_id, ts);
                create index idx_provenance_key on provenance_edges(output_report_id);
                """
            ),
        )
    )
    conn.commit()

    ledger_sql = """
        insert into privacy_ledger
        select
            device_id,
            count(*) as reports,
            sum(epsilon) as epsilon_sum,
            1000.0 as device_budget,
            1000.0 - sum(epsilon) as remaining_budget
        from reports
        group by device_id
    """
    rows.append(
        timed(
            "privacy_ledger",
            total_rows,
            devices,
            ledger_sql,
            lambda: cur.execute(ledger_sql).rowcount,
        )
    )
    conn.commit()

    budget_trace_sql = """
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
        )
    """
    rows.append(
        timed(
            "privacy_budget_window",
            total_rows,
            0,
            budget_trace_sql,
            lambda: (cur.execute(budget_trace_sql), cur.execute("select count(*) from privacy_budget_trace").fetchone()[0])[1],
        )
    )
    conn.commit()
    cur.execute("create index idx_budget_trace_device_ts on privacy_budget_trace(device_id, ts)")
    conn.commit()

    hourly_sql = """
        create table hourly_cleaned_analytics as
        select
            r.device_id,
            r.ts / 60 as hour_id,
            count(*) as reports,
            avg(c.posterior) as avg_fault_posterior,
            sum(case when c.action = 'repair' then 1 else 0 end) as repaired_reports,
            avg(case when c.action = 'repair' then c.repair_value else (2.0 * r.report_bucket / 31.0) - 1.0 end) as cleaned_mean
        from reports r
        join cleaning_records c using(report_id)
        group by r.device_id, r.ts / 60
    """
    rows.append(
        timed(
            "hourly_analytics_view",
            total_rows,
            0,
            hourly_sql,
            lambda: (cur.execute(hourly_sql), cur.execute("select count(*) from hourly_cleaned_analytics").fetchone()[0])[1],
        )
    )
    conn.commit()

    event_window_sql = """
        create table cleaned_event_windows as
        select
            c.device_id,
            c.ts / 60 as hour_id,
            min(c.ts) as start_ts,
            max(c.ts) as end_ts,
            count(*) as flagged_reports,
            avg(c.posterior) as avg_posterior,
            group_concat(distinct p.semantic_layer) as routed_semantics
        from cleaning_records c
        join router_policy p on p.operator = c.operator
        where c.action in ('flag', 'repair')
        group by c.device_id, c.ts / 60
        having count(*) >= 2
    """
    rows.append(
        timed(
            "event_window_materialization",
            total_rows,
            0,
            event_window_sql,
            lambda: (cur.execute(event_window_sql), cur.execute("select count(*) from cleaned_event_windows").fetchone()[0])[1],
        )
    )
    conn.commit()
    cur.execute("create index idx_event_windows_device_hour on cleaned_event_windows(device_id, hour_id)")
    conn.commit()

    repair_aggregate_sql = """
        create table repair_uncertainty_analytics as
        select
            r.device_id,
            r.ts / 60 as hour_id,
            count(*) as reports,
            avg(case when c.action = 'repair' then c.repair_value else (2.0 * r.report_bucket / 31.0) - 1.0 end) as cleaned_mean,
            avg((2.0 * r.report_bucket / 31.0) - 1.0) as unrepaired_mean,
            avg(c.posterior * (1.0 - c.posterior)) as uncertainty_proxy,
            sum(case when c.action = 'repair' then 1 else 0 end) as repair_count
        from reports r
        join cleaning_records c using(report_id)
        group by r.device_id, r.ts / 60
    """
    rows.append(
        timed(
            "repair_uncertainty_aggregate",
            total_rows,
            0,
            repair_aggregate_sql,
            lambda: (cur.execute(repair_aggregate_sql), cur.execute("select count(*) from repair_uncertainty_analytics").fetchone()[0])[1],
        )
    )
    conn.commit()

    drilldown_sql = """
        select r.report_id, r.device_id, r.ts, c.operator, c.posterior, p.transform
        from cleaning_records c
        join reports r using(report_id)
        join provenance_edges p on p.output_report_id = c.report_id
        where c.action in ('flag', 'repair')
        order by c.posterior desc
        limit 100
    """
    rows.append(
        timed(
            "provenance_drilldown",
            total_rows,
            100,
            drilldown_sql,
            lambda: len(cur.execute(drilldown_sql).fetchall()),
        )
    )

    budget_sql = """
        select count(*) from privacy_ledger
        where remaining_budget < 0
    """
    rows.append(
        timed(
            "budget_audit",
            devices,
            1,
            budget_sql,
            lambda: len(cur.execute(budget_sql).fetchall()),
        )
    )

    run_scale.plan_rows = collect_plans(cur, total_rows, devices, storage_mode)  # type: ignore[attr-defined]
    conn.close()
    db_file_bytes = 0
    if db_path is not None and db_path.exists():
        db_file_bytes = db_path.stat().st_size
    for row in rows:
        row["scale_rows"] = total_rows
        row["devices"] = devices
        row["storage_mode"] = storage_mode
        row["db_file_bytes"] = db_file_bytes
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "scale_rows",
        "devices",
        "storage_mode",
        "db_file_bytes",
        "workload",
        "input_rows",
        "output_rows",
        "elapsed_sec",
        "throughput_rows_per_sec",
        "query",
    ]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plan_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["scale_rows", "devices", "storage_mode", "workload", "plan_step", "detail"]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_query_suite_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "scale_rows",
        "devices",
        "query_name",
        "design",
        "rep",
        "input_rows",
        "output_rows",
        "elapsed_ms",
        "throughput_rows_per_sec",
        "query",
    ]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_query_plan_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["scale_rows", "devices", "query_name", "design", "plan_step", "detail"]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_query_suite_summary_csv(path: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["query_name"]), str(row["design"]))
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (query_name, design), group in sorted(grouped.items()):
        elapsed = [float(row["elapsed_ms"]) for row in group]
        input_rows = int(group[0]["input_rows"])
        summary_rows.append(
            {
                "query_name": query_name,
                "design": design,
                "reps": len(group),
                "input_rows": input_rows,
                "output_rows": int(group[0]["output_rows"]),
                "p50_elapsed_ms": statistics.median(elapsed),
                "mean_elapsed_ms": statistics.fmean(elapsed),
                "min_elapsed_ms": min(elapsed),
                "max_elapsed_ms": max(elapsed),
                "p50_throughput_rows_per_sec": input_rows / max(statistics.median(elapsed) / 1000.0, 1e-12),
            }
        )

    fieldnames = [
        "query_name",
        "design",
        "reps",
        "input_rows",
        "output_rows",
        "p50_elapsed_ms",
        "mean_elapsed_ms",
        "min_elapsed_ms",
        "max_elapsed_ms",
        "p50_throughput_rows_per_sec",
    ]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def main() -> None:
    config = load_benchmark_config()
    pipeline_config = config.get("drivers", {}).get("pipeline_query", {})
    scale_config = config.get("scale", {})
    file_storage = next((item for item in config.get("storage_modes", []) if item.get("id") == "file_wal"), {})

    output_dir = Path(os.environ.get("PRIVATE_PIPELINE_RESULTS", str(RESULTS)))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    all_plan_rows: list[dict[str, object]] = []
    memory_scales = [int(value) for value in pipeline_config.get("memory_scales", [50_000, 100_000, 200_000])]
    memory_seed_base = int(pipeline_config.get("memory_seed_base", 9000))
    memory_min_devices = int(pipeline_config.get("memory_min_devices", 250))
    memory_rows_per_device = int(pipeline_config.get("memory_rows_per_device", 100))
    for idx, total_rows in enumerate(memory_scales):
        devices = max(memory_min_devices, total_rows // memory_rows_per_device)
        scale_rows = run_scale(total_rows, devices, memory_seed_base + idx, "memory")
        all_rows.extend(scale_rows)
        all_plan_rows.extend(getattr(run_scale, "plan_rows", []))

    file_rows = int(scale_config.get("report_rows", 200_000))
    file_devices = int(scale_config.get("devices", 2_000))
    file_seed = int(pipeline_config.get("file_seed", 9100))
    configured_db = str(file_storage.get("database_path", "results/private_telemetry_pipeline_200k.sqlite"))
    file_db = result_path(configured_db)
    if "PRIVATE_PIPELINE_RESULTS" in os.environ:
        file_db = output_dir / Path(configured_db).name
    scale_rows = run_scale(file_rows, file_devices, file_seed, "file_wal", file_db)
    all_rows.extend(scale_rows)
    all_plan_rows.extend(getattr(run_scale, "plan_rows", []))
    query_suite_reps = int(pipeline_config.get("query_suite_repetitions", 3))
    query_suite_rows, query_suite_plan_rows = run_query_suite(file_db, file_rows, file_devices, query_suite_reps)

    csv_path = output_dir / "private_telemetry_pipeline_query_benchmark.csv"
    write_csv(csv_path, all_rows)
    plan_csv_path = output_dir / "private_telemetry_pipeline_query_plans.csv"
    write_plan_csv(plan_csv_path, all_plan_rows)
    query_suite_path = output_dir / "private_telemetry_sql_query_suite.csv"
    write_query_suite_csv(query_suite_path, query_suite_rows)
    query_suite_plan_path = output_dir / "private_telemetry_sql_query_suite_plans.csv"
    write_query_plan_csv(query_suite_plan_path, query_suite_plan_rows)
    query_suite_summary_path = output_dir / "private_telemetry_sql_query_suite_summary.csv"
    query_suite_summary_rows = write_query_suite_summary_csv(query_suite_summary_path, query_suite_rows)

    summary = {
        "rows": len(all_rows),
        "plan_rows": len(all_plan_rows),
        "query_suite_rows": len(query_suite_rows),
        "query_suite_plan_rows": len(query_suite_plan_rows),
        "query_suite_summary_rows": len(query_suite_summary_rows),
        "config": str(Path(os.environ.get("PRIVATE_TELEMETRY_BENCHMARK_CONFIG", "configs/private_telemetry_benchmark_config.json"))),
        "scales": memory_scales,
        "storage_modes": sorted({str(row["storage_mode"]) for row in all_rows}),
        "workloads": sorted({str(row["workload"]) for row in all_rows}),
        "query_suite": sorted({str(row["query_name"]) for row in query_suite_rows}),
        "csv": str(csv_path),
        "plans_csv": str(plan_csv_path),
        "query_suite_csv": str(query_suite_path),
        "query_suite_plans_csv": str(query_suite_plan_path),
        "query_suite_summary_csv": str(query_suite_summary_path),
        "file_backed_sqlite": str(file_db),
        "file_backed_sqlite_bytes": file_db.stat().st_size if file_db.exists() else 0,
    }
    (output_dir / "private_telemetry_pipeline_query_benchmark.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
