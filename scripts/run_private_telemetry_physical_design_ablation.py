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
from typing import Callable

from private_telemetry_benchmark_config import load_benchmark_config


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BUCKETS = 32
EPS_VALUES = [0.5, 1.0, 2.0]


def make_reports(total_rows: int, devices: int, seed: int) -> list[tuple[int, int, int, int, float, str, str]]:
    rng = random.Random(seed)
    rows: list[tuple[int, int, int, int, float, str, str]] = []
    for report_id in range(total_rows):
        device_id = report_id % devices
        ts = report_id // devices
        bucket = rng.randrange(BUCKETS)
        epsilon = EPS_VALUES[(device_id + report_id) % len(EPS_VALUES)]
        rows.append((report_id, device_id, ts, bucket, epsilon, "pm32", "cal_v1"))
    return rows


def make_mechanism_weights() -> list[tuple[str, float, int, int, float, float]]:
    rows: list[tuple[str, float, int, int, float, float]] = []
    normal_prob = 1.0 / BUCKETS
    for epsilon in EPS_VALUES:
        for report_bucket in range(BUCKETS):
            for stuck_bucket in range(BUCKETS):
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


BASE_SCHEMA = """
pragma journal_mode = wal;
pragma synchronous = normal;
pragma temp_store = memory;

create table reports(
    report_id integer primary key,
    device_id integer not null,
    ts integer not null,
    report_bucket integer not null,
    epsilon real not null,
    mechanism_id text not null,
    calibration_version text not null
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
"""


INCREMENTAL_SCHEMA = """
create table mv_report_counts_inc(
    mechanism_id text not null,
    epsilon real not null,
    report_bucket integer not null,
    n integer not null,
    primary key(mechanism_id, epsilon, report_bucket)
);

create table privacy_ledger_inc(
    device_id integer primary key,
    reports integer not null,
    epsilon_sum real not null,
    device_budget real not null,
    remaining_budget real not null
);

create table repair_uncertainty_inc(
    device_id integer not null,
    hour_id integer not null,
    reports integer not null,
    posterior_sum real not null,
    repair_count integer not null,
    cleaned_sum real not null,
    raw_sum real not null,
    uncertainty_sum real not null,
    primary key(device_id, hour_id)
);

create table event_windows_inc(
    device_id integer not null,
    hour_id integer not null,
    start_ts integer not null,
    end_ts integer not null,
    flagged_reports integer not null,
    posterior_sum real not null,
    primary key(device_id, hour_id)
);

create trigger reports_to_count_inc after insert on reports
begin
    insert into mv_report_counts_inc(mechanism_id, epsilon, report_bucket, n)
    values(new.mechanism_id, new.epsilon, new.report_bucket, 1)
    on conflict(mechanism_id, epsilon, report_bucket)
    do update set n = mv_report_counts_inc.n + 1;

    insert into privacy_ledger_inc(device_id, reports, epsilon_sum, device_budget, remaining_budget)
    values(new.device_id, 1, new.epsilon, 1000.0, 1000.0 - new.epsilon)
    on conflict(device_id)
    do update set
        reports = privacy_ledger_inc.reports + 1,
        epsilon_sum = privacy_ledger_inc.epsilon_sum + excluded.epsilon_sum,
        remaining_budget = privacy_ledger_inc.device_budget
            - (privacy_ledger_inc.epsilon_sum + excluded.epsilon_sum);
end;

create trigger cleaning_to_repair_inc after insert on cleaning_records
begin
    insert into repair_uncertainty_inc(
        device_id, hour_id, reports, posterior_sum, repair_count,
        cleaned_sum, raw_sum, uncertainty_sum
    )
    values(
        new.device_id,
        new.ts / 60,
        1,
        new.posterior,
        case when new.action = 'repair' then 1 else 0 end,
        new.repair_value,
        new.repair_value,
        new.posterior * (1.0 - new.posterior)
    )
    on conflict(device_id, hour_id)
    do update set
        reports = repair_uncertainty_inc.reports + 1,
        posterior_sum = repair_uncertainty_inc.posterior_sum + excluded.posterior_sum,
        repair_count = repair_uncertainty_inc.repair_count + excluded.repair_count,
        cleaned_sum = repair_uncertainty_inc.cleaned_sum + excluded.cleaned_sum,
        raw_sum = repair_uncertainty_inc.raw_sum + excluded.raw_sum,
        uncertainty_sum = repair_uncertainty_inc.uncertainty_sum + excluded.uncertainty_sum;
end;

create trigger cleaning_to_event_inc after insert on cleaning_records
when new.action in ('flag', 'repair')
begin
    insert into event_windows_inc(device_id, hour_id, start_ts, end_ts, flagged_reports, posterior_sum)
    values(new.device_id, new.ts / 60, new.ts, new.ts, 1, new.posterior)
    on conflict(device_id, hour_id)
    do update set
        start_ts = min(event_windows_inc.start_ts, excluded.start_ts),
        end_ts = max(event_windows_inc.end_ts, excluded.end_ts),
        flagged_reports = event_windows_inc.flagged_reports + 1,
        posterior_sum = event_windows_inc.posterior_sum + excluded.posterior_sum;
end;
"""


DERIVE_CLEANING_SQL = """
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


BATCH_MATERIALIZATION_SQL = """
create table mv_report_counts as
select mechanism_id, epsilon, report_bucket, count(*) as n
from reports
group by mechanism_id, epsilon, report_bucket;

create index idx_mv_report_counts_channel
on mv_report_counts(mechanism_id, epsilon, report_bucket);

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

create index idx_candidate_likelihoods_llr
on candidate_likelihoods(epsilon, llr desc);

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

create index idx_budget_trace_device_ts
on privacy_budget_trace(device_id, ts);

create index idx_budget_trace_remaining
on privacy_budget_trace(remaining_budget, device_id);

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
having count(*) >= 2;

create index idx_cleaned_event_windows_device_hour
on cleaned_event_windows(device_id, hour_id);

create table repair_uncertainty_analytics as
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
group by r.device_id, r.ts / 60;

create index idx_repair_uncertainty_device_hour
on repair_uncertainty_analytics(device_id, hour_id);
"""


def clean_db_files(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def time_stage(
    stage_rows: list[dict[str, object]],
    design: str,
    stage: str,
    input_rows: int,
    query: str,
    fn: Callable[[], int],
) -> None:
    start = time.perf_counter()
    output_rows = fn()
    elapsed = time.perf_counter() - start
    stage_rows.append(
        {
            "design": design,
            "stage": stage,
            "input_rows": input_rows,
            "output_rows": output_rows,
            "elapsed_sec": elapsed,
            "throughput_rows_per_sec": input_rows / max(elapsed, 1e-12),
            "query": " ".join(query.split()),
        }
    )


def build_indexes(cur: sqlite3.Cursor) -> None:
    cur.executescript(
        """
        create index idx_reports_device_ts on reports(device_id, ts);
        create index idx_reports_channel on reports(mechanism_id, epsilon, report_bucket);
        create index idx_cleaning_action_posterior on cleaning_records(action, posterior desc);
        create index idx_cleaning_device_ts on cleaning_records(device_id, ts);
        create index idx_provenance_output on provenance_edges(output_report_id);
        """
    )


def build_database(
    db_path: Path,
    design: str,
    reports: list[tuple[int, int, int, int, float, str, str]],
    mechanism_rows: list[tuple[str, float, int, int, float, float]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    clean_db_files(db_path)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    stage_rows: list[dict[str, object]] = []
    total_rows = len(reports)
    devices = len({row[1] for row in reports})

    time_stage(stage_rows, design, "create_base_schema", 1, BASE_SCHEMA, lambda: (cur.executescript(BASE_SCHEMA), 1)[1])
    conn.commit()

    if design == "incremental_mv":
        time_stage(
            stage_rows,
            design,
            "create_incremental_triggers",
            1,
            INCREMENTAL_SCHEMA,
            lambda: (cur.executescript(INCREMENTAL_SCHEMA), 1)[1],
        )
        conn.commit()

    policy_rows = [
        ("privsaf_mixture", "iid stuck-at", "cleaning_records", "post_processing"),
        ("privsaf_hmm", "persistent flatline", "cleaning_records", "post_processing"),
        ("pm_window_glr", "field-QC local regime", "cleaning_records", "post_processing"),
    ]
    time_stage(
        stage_rows,
        design,
        "load_mechanism_matrix",
        len(mechanism_rows),
        "insert into mechanism_weights values (?, ?, ?, ?, ?, ?)",
        lambda: cur.executemany("insert into mechanism_weights values (?, ?, ?, ?, ?, ?)", mechanism_rows).rowcount,
    )
    conn.commit()
    time_stage(
        stage_rows,
        design,
        "load_router_policy",
        len(policy_rows),
        "insert into router_policy values (?, ?, ?, ?)",
        lambda: cur.executemany("insert into router_policy values (?, ?, ?, ?)", policy_rows).rowcount,
    )
    conn.commit()
    time_stage(
        stage_rows,
        design,
        "ingest_reports",
        total_rows,
        "insert into reports values (?, ?, ?, ?, ?, ?, ?)",
        lambda: cur.executemany("insert into reports values (?, ?, ?, ?, ?, ?, ?)", reports).rowcount,
    )
    conn.commit()

    if design in {"indexed_base", "materialized_batch"}:
        time_stage(
            stage_rows,
            design,
            "build_secondary_indexes",
            total_rows,
            "create indexes for report, cleaning, provenance access paths",
            lambda: (build_indexes(cur), 0)[1],
        )
        conn.commit()

    if design != "incremental_mv":
        ledger_sql = """
            insert into privacy_ledger
            select device_id, count(*), sum(epsilon), 1000.0, 1000.0 - sum(epsilon)
            from reports
            group by device_id
        """
        time_stage(
            stage_rows,
            design,
            "materialize_device_privacy_ledger",
            total_rows,
            ledger_sql,
            lambda: cur.execute(ledger_sql).rowcount,
        )
        conn.commit()

    time_stage(
        stage_rows,
        design,
        "derive_cleaning_records",
        total_rows,
        DERIVE_CLEANING_SQL,
        lambda: cur.execute(DERIVE_CLEANING_SQL).rowcount,
    )
    conn.commit()

    provenance_sql = """
        insert into provenance_edges
        select report_id, 'reports', report_id, operator, 'pm32', 'cal_v1'
        from cleaning_records
    """
    time_stage(
        stage_rows,
        design,
        "write_provenance_edges",
        total_rows,
        provenance_sql,
        lambda: cur.execute(provenance_sql).rowcount,
    )
    conn.commit()

    if design == "materialized_batch":
        time_stage(
            stage_rows,
            design,
            "batch_materialize_query_state",
            total_rows,
            BATCH_MATERIALIZATION_SQL,
            lambda: (
                cur.executescript(BATCH_MATERIALIZATION_SQL),
                cur.execute("select count(*) from repair_uncertainty_analytics").fetchone()[0],
            )[1],
        )
        conn.commit()

    if design == "incremental_mv":
        time_stage(
            stage_rows,
            design,
            "index_incremental_state",
            total_rows,
            "create indexes on incrementally maintained summaries",
            lambda: (
                cur.executescript(
                    """
                    create index idx_repair_inc_device_hour on repair_uncertainty_inc(device_id, hour_id);
                    create index idx_event_inc_device_hour on event_windows_inc(device_id, hour_id);
                    create index idx_privacy_inc_remaining on privacy_ledger_inc(remaining_budget, device_id);
                    """
                ),
                0,
            )[1],
        )
        conn.commit()

    counts = {
        "reports": cur.execute("select count(*) from reports").fetchone()[0],
        "cleaning_records": cur.execute("select count(*) from cleaning_records").fetchone()[0],
        "provenance_edges": cur.execute("select count(*) from provenance_edges").fetchone()[0],
        "devices": devices,
    }
    if design == "materialized_batch":
        counts["mv_report_counts_sum"] = cur.execute("select sum(n) from mv_report_counts").fetchone()[0]
        counts["repair_groups"] = cur.execute("select count(*) from repair_uncertainty_analytics").fetchone()[0]
    if design == "incremental_mv":
        counts["mv_report_counts_sum"] = cur.execute("select sum(n) from mv_report_counts_inc").fetchone()[0]
        counts["repair_groups"] = cur.execute("select count(*) from repair_uncertainty_inc").fetchone()[0]

    conn.close()
    return stage_rows, counts


def direct_dashboard_query() -> str:
    return """
    with hourly as (
        select
            r.device_id,
            r.ts / 60 as hour_id,
            count(*) as reports,
            avg(case when c.action = 'repair'
                     then c.repair_value
                     else (2.0 * r.report_bucket / 31.0) - 1.0 end) as cleaned_mean,
            avg(c.posterior * (1.0 - c.posterior)) as uncertainty_proxy,
            sum(case when c.action = 'repair' then 1 else 0 end) as repair_count
        from reports r
        join cleaning_records c using(report_id)
        group by r.device_id, r.ts / 60
    ),
    events as (
        select
            device_id,
            ts / 60 as hour_id,
            count(*) as flagged_reports,
            avg(posterior) as avg_posterior
        from cleaning_records
        where action in ('flag', 'repair')
        group by device_id, ts / 60
        having count(*) >= 2
    )
    select
        case when l.remaining_budget < 980.0 then 'near_cap' else 'ok' end as budget_band,
        count(*) as hourly_groups,
        sum(coalesce(e.flagged_reports, 0)) as flagged_reports,
        avg(h.cleaned_mean) as avg_cleaned_mean,
        avg(h.uncertainty_proxy) as avg_uncertainty_proxy,
        sum(h.repair_count) as repair_count
    from hourly h
    join privacy_ledger l using(device_id)
    left join events e on e.device_id = h.device_id and e.hour_id = h.hour_id
    group by budget_band
    """


def query_specs(design: str, total_rows: int) -> list[tuple[str, str, int, str]]:
    if design == "materialized_batch":
        candidate_query = """
            select mechanism_id, epsilon, stuck_bucket, fault_loglik, normal_loglik, llr
            from candidate_likelihoods
            order by llr desc
            limit 96
        """
        budget_query = """
            select count(*) as reports,
                   min(remaining_budget) as min_remaining_budget,
                   avg(remaining_budget) as avg_remaining_budget
            from privacy_budget_trace
            where remaining_budget < 1000.0
        """
        dashboard_query = """
            select
                case when l.remaining_budget < 980.0 then 'near_cap' else 'ok' end as budget_band,
                count(*) as hourly_groups,
                sum(coalesce(e.flagged_reports, 0)) as flagged_reports,
                avg(h.cleaned_mean) as avg_cleaned_mean,
                avg(h.uncertainty_proxy) as avg_uncertainty_proxy,
                sum(h.repair_count) as repair_count
            from repair_uncertainty_analytics h
            join privacy_ledger l using(device_id)
            left join cleaned_event_windows e
              on e.device_id = h.device_id and e.hour_id = h.hour_id
            group by budget_band
        """
    elif design == "incremental_mv":
        candidate_query = """
            select
                c.mechanism_id,
                c.epsilon,
                w.stuck_bucket,
                sum(c.n * w.logp_fault) as fault_loglik,
                sum(c.n * w.logp_normal) as normal_loglik,
                sum(c.n * (w.logp_fault - w.logp_normal)) as llr
            from mv_report_counts_inc c
            join mechanism_weights w
              on w.mechanism_id = c.mechanism_id
             and w.epsilon = c.epsilon
             and w.report_bucket = c.report_bucket
            group by c.mechanism_id, c.epsilon, w.stuck_bucket
            order by llr desc
            limit 96
        """
        budget_query = """
            select count(*) as devices,
                   min(remaining_budget) as min_remaining_budget,
                   avg(remaining_budget) as avg_remaining_budget
            from privacy_ledger_inc
            where remaining_budget < 1000.0
        """
        dashboard_query = """
            with h as (
                select
                    device_id,
                    hour_id,
                    reports,
                    cleaned_sum / reports as cleaned_mean,
                    uncertainty_sum / reports as uncertainty_proxy,
                    repair_count
                from repair_uncertainty_inc
            ),
            e as (
                select
                    device_id,
                    hour_id,
                    flagged_reports,
                    posterior_sum / flagged_reports as avg_posterior
                from event_windows_inc
                where flagged_reports >= 2
            )
            select
                case when l.remaining_budget < 980.0 then 'near_cap' else 'ok' end as budget_band,
                count(*) as hourly_groups,
                sum(coalesce(e.flagged_reports, 0)) as flagged_reports,
                avg(h.cleaned_mean) as avg_cleaned_mean,
                avg(h.uncertainty_proxy) as avg_uncertainty_proxy,
                sum(h.repair_count) as repair_count
            from h
            join privacy_ledger_inc l using(device_id)
            left join e on e.device_id = h.device_id and e.hour_id = h.hour_id
            group by budget_band
        """
    else:
        candidate_query = """
            select
                r.mechanism_id,
                r.epsilon,
                w.stuck_bucket,
                sum(w.logp_fault) as fault_loglik,
                sum(w.logp_normal) as normal_loglik,
                sum(w.logp_fault - w.logp_normal) as llr
            from reports r
            join mechanism_weights w
              on w.mechanism_id = r.mechanism_id
             and w.epsilon = r.epsilon
             and w.report_bucket = r.report_bucket
            group by r.mechanism_id, r.epsilon, w.stuck_bucket
            order by llr desc
            limit 96
        """
        budget_query = """
            select count(*) as reports,
                   min(remaining_budget) as min_remaining_budget,
                   avg(remaining_budget) as avg_remaining_budget
            from (
                select 1000.0 - sum(epsilon) over (
                           partition by device_id
                           order by ts
                           rows between unbounded preceding and current row
                       ) as remaining_budget
                from reports
            )
        """
        dashboard_query = direct_dashboard_query()

    provenance_query = """
        select
            c.report_id,
            r.device_id,
            r.ts,
            c.operator,
            c.posterior,
            p.transform,
            w.logp_fault - w.logp_normal as llr_component
        from cleaning_records c
        join reports r using(report_id)
        join provenance_edges p on p.output_report_id = c.report_id
        join mechanism_weights w
          on w.mechanism_id = r.mechanism_id
         and w.epsilon = r.epsilon
         and w.report_bucket = r.report_bucket
         and w.stuck_bucket = coalesce(c.stuck_bucket, r.report_bucket)
        where c.action in ('flag', 'repair')
        order by c.posterior desc, r.ts desc
        limit 100
    """
    return [
        ("candidate_likelihood_topk", "query_processing", total_rows, candidate_query),
        ("privacy_budget_accounting", "privacy_accounting", total_rows, budget_query),
        ("uncertainty_privacy_dashboard", "end_to_end_analytics", total_rows, dashboard_query),
        ("provenance_event_drilldown", "provenance", total_rows, provenance_query),
    ]


def run_queries(db_path: Path, design: str, total_rows: int, reps: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    rows: list[dict[str, object]] = []
    plans: list[dict[str, object]] = []
    for query_name, query_family, input_rows, query in query_specs(design, total_rows):
        compact_query = " ".join(query.split())
        for idx, (_, _, _, detail) in enumerate(cur.execute("explain query plan " + query).fetchall()):
            plans.append(
                {
                    "design": design,
                    "query_name": query_name,
                    "query_family": query_family,
                    "plan_step": idx,
                    "detail": detail,
                }
            )
        for rep in range(reps):
            start = time.perf_counter()
            result = cur.execute(query).fetchall()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            rows.append(
                {
                    "design": design,
                    "query_name": query_name,
                    "query_family": query_family,
                    "rep": rep,
                    "input_rows": input_rows,
                    "output_rows": len(result),
                    "elapsed_ms": elapsed_ms,
                    "throughput_rows_per_sec": input_rows / max(elapsed_ms / 1000.0, 1e-12),
                    "query": compact_query,
                }
            )
    conn.close()
    return rows, plans


def collect_checks(db_path: Path, design: str, expected_rows: int) -> list[dict[str, object]]:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    checks = [
        ("reports", expected_rows, cur.execute("select count(*) from reports").fetchone()[0]),
        ("cleaning_records", expected_rows, cur.execute("select count(*) from cleaning_records").fetchone()[0]),
        ("provenance_edges", expected_rows, cur.execute("select count(*) from provenance_edges").fetchone()[0]),
    ]
    if design == "materialized_batch":
        checks.append(("mv_report_counts_sum", expected_rows, cur.execute("select sum(n) from mv_report_counts").fetchone()[0]))
        checks.append(("repair_groups_nonempty", 1, int(cur.execute("select count(*) > 0 from repair_uncertainty_analytics").fetchone()[0])))
    if design == "incremental_mv":
        checks.append(("mv_report_counts_sum", expected_rows, cur.execute("select sum(n) from mv_report_counts_inc").fetchone()[0]))
        checks.append(("privacy_ledger_inc_sum", expected_rows, cur.execute("select sum(reports) from privacy_ledger_inc").fetchone()[0]))
        checks.append(("repair_groups_nonempty", 1, int(cur.execute("select count(*) > 0 from repair_uncertainty_inc").fetchone()[0])))
    conn.close()
    return [
        {
            "design": design,
            "check": check,
            "expected": expected,
            "observed": observed,
            "status": "pass" if expected == observed else "fail",
        }
        for check, expected, observed in checks
    ]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_queries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["design"]), str(row["query_name"]), str(row["query_family"]))
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, object]] = []
    for (design, query_name, query_family), group in sorted(groups.items()):
        elapsed = [float(row["elapsed_ms"]) for row in group]
        input_rows = int(group[0]["input_rows"])
        summary.append(
            {
                "design": design,
                "query_name": query_name,
                "query_family": query_family,
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
    return summary


def summarize_stages(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "design": row["design"],
            "stage": row["stage"],
            "input_rows": row["input_rows"],
            "output_rows": row["output_rows"],
            "elapsed_sec": row["elapsed_sec"],
            "throughput_rows_per_sec": row["throughput_rows_per_sec"],
        }
        for row in rows
    ]


def main() -> None:
    config = load_benchmark_config()
    scale_config = config.get("scale", {})
    physical_config = config.get("drivers", {}).get("physical_design", {})
    ablation_config = config.get("workload_mix", {}).get("physical_design_ablation", {})

    output_dir = Path(os.environ.get("PRIVATE_PIPELINE_RESULTS", str(RESULTS)))
    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = int(os.environ.get("PRIVATE_PHYSICAL_ROWS", str(scale_config.get("report_rows", 200000))))
    devices = int(os.environ.get("PRIVATE_PHYSICAL_DEVICES", str(scale_config.get("devices", 2000))))
    reps = int(os.environ.get("PRIVATE_PHYSICAL_REPS", str(ablation_config.get("repetitions", 3))))
    seed = int(physical_config.get("seed", 12027))

    reports = make_reports(total_rows, devices, seed=seed)
    mechanism_rows = make_mechanism_weights()
    designs = [str(item) for item in ablation_config.get("designs", ["heap_base", "indexed_base", "materialized_batch", "incremental_mv"])]

    all_stage_rows: list[dict[str, object]] = []
    all_query_rows: list[dict[str, object]] = []
    all_plan_rows: list[dict[str, object]] = []
    all_checks: list[dict[str, object]] = []

    for design in designs:
        db_path = output_dir / f"private_telemetry_physical_{design}_{total_rows}.sqlite"
        stage_rows, _ = build_database(db_path, design, reports, mechanism_rows)
        for row in stage_rows:
            row["scale_rows"] = total_rows
            row["devices"] = devices
            row["db_file_bytes"] = db_path.stat().st_size if db_path.exists() else 0
        query_rows, plan_rows = run_queries(db_path, design, total_rows, reps)
        all_stage_rows.extend(stage_rows)
        all_query_rows.extend(query_rows)
        all_plan_rows.extend(plan_rows)
        all_checks.extend(collect_checks(db_path, design, total_rows))

    query_summary_rows = summarize_queries(all_query_rows)
    stage_summary_rows = summarize_stages(all_stage_rows)

    write_csv(
        output_dir / "private_telemetry_physical_design_build.csv",
        all_stage_rows,
        [
            "scale_rows",
            "devices",
            "db_file_bytes",
            "design",
            "stage",
            "input_rows",
            "output_rows",
            "elapsed_sec",
            "throughput_rows_per_sec",
            "query",
        ],
    )
    write_csv(
        output_dir / "private_telemetry_physical_design_build_summary.csv",
        stage_summary_rows,
        ["design", "stage", "input_rows", "output_rows", "elapsed_sec", "throughput_rows_per_sec"],
    )
    write_csv(
        output_dir / "private_telemetry_physical_design_queries.csv",
        all_query_rows,
        [
            "design",
            "query_name",
            "query_family",
            "rep",
            "input_rows",
            "output_rows",
            "elapsed_ms",
            "throughput_rows_per_sec",
            "query",
        ],
    )
    write_csv(
        output_dir / "private_telemetry_physical_design_summary.csv",
        query_summary_rows,
        [
            "design",
            "query_name",
            "query_family",
            "reps",
            "input_rows",
            "output_rows",
            "p50_elapsed_ms",
            "mean_elapsed_ms",
            "min_elapsed_ms",
            "max_elapsed_ms",
            "p50_throughput_rows_per_sec",
        ],
    )
    write_csv(
        output_dir / "private_telemetry_physical_design_plans.csv",
        all_plan_rows,
        ["design", "query_name", "query_family", "plan_step", "detail"],
    )
    write_csv(
        output_dir / "private_telemetry_physical_design_checks.csv",
        all_checks,
        ["design", "check", "expected", "observed", "status"],
    )

    summary = {
        "rows": total_rows,
        "devices": devices,
        "reps": reps,
        "seed": seed,
        "config": str(Path(os.environ.get("PRIVATE_TELEMETRY_BENCHMARK_CONFIG", "configs/private_telemetry_benchmark_config.json"))),
        "designs": designs,
        "query_rows": len(all_query_rows),
        "query_summary_rows": len(query_summary_rows),
        "plan_rows": len(all_plan_rows),
        "checks": len(all_checks),
        "failed_checks": [row for row in all_checks if row["status"] != "pass"],
    }
    (output_dir / "private_telemetry_physical_design_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
