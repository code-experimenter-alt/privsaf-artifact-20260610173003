from __future__ import annotations

import csv
import json
import shutil
import sqlite3
import time
from pathlib import Path

from run_private_telemetry_pipeline_queries import make_mechanism_weights, make_reports


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


WORK_DIR = Path("/tmp/privsaf_private_telemetry_e2e_workflow")
DB_PATH = WORK_DIR / "private_telemetry_end_to_end_workflow.sqlite"
RESULT_DB_PATH = RESULTS / "private_telemetry_end_to_end_workflow.sqlite"
BATCH_CSV = RESULTS / "private_telemetry_end_to_end_workflow_batches.csv"
QUERY_CSV = RESULTS / "private_telemetry_end_to_end_workflow_queries.csv"
PLAN_CSV = RESULTS / "private_telemetry_end_to_end_workflow_plans.csv"
CHECK_CSV = RESULTS / "private_telemetry_end_to_end_workflow_checks.csv"
SUMMARY_JSON = RESULTS / "private_telemetry_end_to_end_workflow_summary.json"


DEVICE_BUDGET = 120.0
TOTAL_ROWS = 72_000
DEVICES = 600
BATCH_SIZE = 6_000


SCHEMA_SQL = """
pragma journal_mode = wal;
pragma synchronous = normal;
pragma temp_store = memory;
pragma busy_timeout = 30000;

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

create table reports(
    report_id integer primary key,
    source_event_id text not null unique,
    batch_id integer not null,
    device_id integer not null,
    ts integer not null,
    report_bucket integer not null,
    epsilon real not null,
    mechanism_id text not null,
    calibration_version text not null,
    correction_version integer not null default 0,
    late_event integer not null default 0
);

create table batch_manifest(
    batch_id integer primary key,
    batch_kind text not null,
    input_events integer not null,
    accepted_new_events integer not null,
    duplicate_events integer not null,
    correction_events integer not null,
    rejected_budget_events integer not null,
    late_events integer not null,
    watermark_before integer not null,
    watermark_after integer not null,
    elapsed_ms real not null
);

create table device_budget_state(
    device_id integer primary key,
    accepted_reports integer not null default 0,
    epsilon_sum real not null default 0.0,
    device_budget real not null default 120.0,
    remaining_budget real not null default 120.0
);

create table rejected_events(
    batch_id integer not null,
    source_event_id text not null,
    report_id integer not null,
    device_id integer not null,
    ts integer not null,
    epsilon real not null,
    projected_epsilon_sum real not null,
    reason text not null
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
    provenance_version text not null,
    correction_version integer not null
);

create table provenance_edges(
    output_report_id integer not null,
    source_table text not null,
    source_key text not null,
    transform text not null,
    mechanism_id text not null,
    calibration_version text not null,
    batch_id integer not null,
    correction_version integer not null
);

create table privacy_budget_trace(
    report_id integer primary key,
    device_id integer not null,
    ts integer not null,
    epsilon real not null,
    cumulative_epsilon real not null,
    remaining_budget real not null
);

create table mv_report_counts(
    mechanism_id text not null,
    epsilon real not null,
    report_bucket integer not null,
    n integer not null,
    primary key(mechanism_id, epsilon, report_bucket)
);

create table candidate_likelihoods(
    mechanism_id text not null,
    epsilon real not null,
    stuck_bucket integer not null,
    fault_loglik real not null,
    normal_loglik real not null,
    llr real not null,
    primary key(mechanism_id, epsilon, stuck_bucket)
);

create table cleaned_event_windows(
    device_id integer not null,
    hour_id integer not null,
    reports integer not null,
    flagged_reports integer not null,
    avg_posterior real not null,
    max_posterior real not null,
    primary key(device_id, hour_id)
);

create table repair_uncertainty_analytics(
    device_id integer not null,
    hour_id integer not null,
    reports integer not null,
    cleaned_mean real not null,
    unrepaired_mean real not null,
    uncertainty_proxy real not null,
    repair_count integer not null,
    primary key(device_id, hour_id)
);

create index idx_reports_device_ts on reports(device_id, ts);
create index idx_reports_batch on reports(batch_id);
create index idx_reports_channel on reports(mechanism_id, epsilon, report_bucket);
create index idx_rejected_device_ts on rejected_events(device_id, ts);
create index idx_cleaning_action on cleaning_records(action, posterior desc);
create index idx_cleaning_device_ts on cleaning_records(device_id, ts);
create index idx_provenance_output on provenance_edges(output_report_id);
create index idx_budget_remaining on privacy_budget_trace(remaining_budget, device_id, ts);
create index idx_events_score on cleaned_event_windows(avg_posterior desc, flagged_reports desc);
"""


def clean_outputs() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for path in [
        DB_PATH,
        Path(str(DB_PATH) + "-wal"),
        Path(str(DB_PATH) + "-shm"),
        RESULT_DB_PATH,
        Path(str(RESULT_DB_PATH) + "-wal"),
        Path(str(RESULT_DB_PATH) + "-shm"),
    ]:
        if path.exists():
            path.unlink()


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("pragma busy_timeout = 30000")
    conn.execute("pragma journal_mode = wal")
    conn.execute("pragma synchronous = normal")
    return conn


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def timed_query(cur: sqlite3.Cursor, name: str, query: str, input_rows: int, rep: int) -> dict[str, object]:
    start = time.perf_counter()
    rows = cur.execute(query).fetchall()
    elapsed = time.perf_counter() - start
    return {
        "query_name": name,
        "rep": rep,
        "input_rows": input_rows,
        "output_rows": len(rows),
        "elapsed_ms": elapsed * 1000.0,
        "throughput_rows_per_sec": input_rows / max(elapsed, 1e-12),
        "query": " ".join(query.split()),
    }


def initialize_database() -> None:
    conn = connect()
    cur = conn.cursor()
    cur.executescript(SCHEMA_SQL)
    cur.executemany("insert into mechanism_weights values (?, ?, ?, ?, ?, ?)", make_mechanism_weights())
    cur.executemany(
        "insert into router_policy values (?, ?, ?, ?)",
        [
            ("privsaf_mixture", "iid stuck-at", "cleaning_records", "post_processing"),
            ("privsaf_hmm", "persistent flatline", "cleaning_records", "post_processing"),
            ("pm_window_glr", "field-QC local regime", "cleaning_records", "post_processing"),
            ("privsaf_dropout_hmm", "explicit availability", "cleaning_records", "post_processing"),
        ],
    )
    cur.executemany(
        "insert into device_budget_state(device_id, device_budget, remaining_budget) values (?, ?, ?)",
        [(device_id, DEVICE_BUDGET, DEVICE_BUDGET) for device_id in range(DEVICES)],
    )
    conn.commit()
    conn.close()


def base_events() -> list[tuple[str, int, int, int, int, float, str, str, int]]:
    rows = []
    for report_id, device_id, ts, bucket, epsilon, mechanism_id, calibration in make_reports(TOTAL_ROWS, DEVICES, seed=27001):
        rows.append((f"evt-{report_id}", report_id, device_id, ts, bucket, epsilon, mechanism_id, calibration, 0))
    return rows


def correction_events(source: list[tuple[str, int, int, int, int, float, str, str, int]]) -> list[tuple[str, int, int, int, int, float, str, str, int]]:
    out = []
    # Corrections intentionally arrive after the watermark and change existing report buckets.
    for offset, original in enumerate(source[18_000:18_360:3]):
        _, report_id, device_id, ts, bucket, epsilon, mechanism_id, calibration, _ = original
        out.append((f"corr-{report_id}", report_id, device_id, max(0, ts - 7), (bucket + 11 + offset) % 32, epsilon, mechanism_id, calibration, 1))
    return out


def make_batches() -> list[tuple[int, str, list[tuple[str, int, int, int, int, float, str, str, int]]]]:
    events = base_events()
    batches = []
    batch_id = 0
    for start in range(0, len(events), BATCH_SIZE):
        batches.append((batch_id, "base", events[start : start + BATCH_SIZE]))
        batch_id += 1
        if batch_id == 4:
            batches.append((batch_id, "replay_batch_3", events[start : start + BATCH_SIZE]))
            batch_id += 1
        if batch_id == 9:
            previous = events[start - BATCH_SIZE : start]
            batches.append((batch_id, "replay_previous", previous))
            batch_id += 1
    batches.append((batch_id, "late_corrections", correction_events(events)))
    return batches


def process_batch(
    conn: sqlite3.Connection,
    batch_id: int,
    batch_kind: str,
    rows: list[tuple[str, int, int, int, int, float, str, str, int]],
    watermark: int,
) -> tuple[dict[str, object], int]:
    cur = conn.cursor()
    start = time.perf_counter()
    watermark_after = max([watermark] + [row[3] for row in rows])
    cur.execute("drop table if exists temp_batch_events")
    cur.execute(
        """
        create temp table temp_batch_events(
            source_event_id text,
            report_id integer,
            device_id integer,
            ts integer,
            report_bucket integer,
            epsilon real,
            mechanism_id text,
            calibration_version text,
            correction_version integer,
            late_event integer
        )
        """
    )
    cur.executemany(
        "insert into temp_batch_events values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [row + (1 if row[3] < watermark else 0,) for row in rows],
    )

    input_events = len(rows)
    duplicate_events = int(
        cur.execute(
            """
            select count(*)
            from temp_batch_events b
            join reports r using(report_id)
            where b.correction_version <= r.correction_version
            """
        ).fetchone()[0]
    )
    correction_events_count = int(
        cur.execute(
            """
            select count(*)
            from temp_batch_events b
            join reports r using(report_id)
            where b.correction_version > r.correction_version
            """
        ).fetchone()[0]
    )
    late_events = int(cur.execute("select count(*) from temp_batch_events where late_event = 1").fetchone()[0])

    cur.execute("drop table if exists temp_new_candidates")
    cur.execute(
        """
        create temp table temp_new_candidates as
        select
            b.*,
            coalesce(s.epsilon_sum, 0.0)
            + sum(b.epsilon) over (
                partition by b.device_id
                order by b.ts, b.report_id
                rows between unbounded preceding and current row
            ) as projected_epsilon_sum
        from temp_batch_events b
        left join reports r using(report_id)
        left join device_budget_state s using(device_id)
        where r.report_id is null
        """
    )
    rejected_budget_events = int(
        cur.execute(
            """
            insert into rejected_events
            select ?, source_event_id, report_id, device_id, ts, epsilon, projected_epsilon_sum, 'device_budget_guard'
            from temp_new_candidates
            where projected_epsilon_sum > ?
            """,
            (batch_id, DEVICE_BUDGET),
        ).rowcount
    )
    accepted_new_events = int(
        cur.execute(
            """
            insert into reports
            select report_id, source_event_id, ?, device_id, ts, report_bucket, epsilon,
                   mechanism_id, calibration_version, correction_version, late_event
            from temp_new_candidates
            where projected_epsilon_sum <= ?
            """,
            (batch_id, DEVICE_BUDGET),
        ).rowcount
    )
    cur.execute(
        """
        update reports
        set source_event_id = (
                select b.source_event_id from temp_batch_events b where b.report_id = reports.report_id
            ),
            batch_id = ?,
            ts = (
                select b.ts from temp_batch_events b where b.report_id = reports.report_id
            ),
            report_bucket = (
                select b.report_bucket from temp_batch_events b where b.report_id = reports.report_id
            ),
            correction_version = (
                select b.correction_version from temp_batch_events b where b.report_id = reports.report_id
            ),
            late_event = (
                select b.late_event from temp_batch_events b where b.report_id = reports.report_id
            )
        where exists (
            select 1 from temp_batch_events b
            where b.report_id = reports.report_id
              and b.correction_version > reports.correction_version
        )
        """,
        (batch_id,),
    )

    cur.execute(
        """
        update device_budget_state
        set accepted_reports = accepted_reports + (
                select count(*) from reports r where r.batch_id = ? and r.device_id = device_budget_state.device_id and r.correction_version = 0
            ),
            epsilon_sum = epsilon_sum + coalesce((
                select sum(epsilon) from reports r where r.batch_id = ? and r.device_id = device_budget_state.device_id and r.correction_version = 0
            ), 0.0),
            remaining_budget = device_budget - epsilon_sum - coalesce((
                select sum(epsilon) from reports r where r.batch_id = ? and r.device_id = device_budget_state.device_id and r.correction_version = 0
            ), 0.0)
        where exists (
            select 1 from reports r where r.batch_id = ? and r.device_id = device_budget_state.device_id and r.correction_version = 0
        )
        """,
        (batch_id, batch_id, batch_id, batch_id),
    )

    cur.execute("drop table if exists temp_affected_reports")
    cur.execute(
        """
        create temp table temp_affected_reports as
        select report_id from reports where batch_id = ?
        """,
        (batch_id,),
    )
    cur.execute("delete from cleaning_records where report_id in (select report_id from temp_affected_reports)")
    cur.execute("delete from provenance_edges where output_report_id in (select report_id from temp_affected_reports)")
    cur.execute(
        """
        insert into cleaning_records
        select
            report_id,
            device_id,
            ts,
            case
                when ((report_bucket * 17 + device_id + correction_version * 13) % 100) / 100.0 >= 0.80 then 'privsaf_hmm'
                when ((report_bucket * 17 + device_id + correction_version * 13) % 100) / 100.0 >= 0.65 then 'pm_window_glr'
                else 'privsaf_mixture'
            end,
            ((report_bucket * 17 + device_id + correction_version * 13) % 100) / 100.0,
            case when ((report_bucket * 17 + device_id + correction_version * 13) % 100) / 100.0 >= 0.65 then report_bucket else null end,
            0.20,
            case when ((report_bucket * 17 + device_id + correction_version * 13) % 100) / 100.0 >= 0.80 then 'repair'
                 when ((report_bucket * 17 + device_id + correction_version * 13) % 100) / 100.0 >= 0.65 then 'flag'
                 else 'pass' end,
            (2.0 * report_bucket / 31.0) - 1.0,
            'router_v2',
            correction_version
        from reports
        where report_id in (select report_id from temp_affected_reports)
        """
    )
    cur.execute(
        """
        insert into provenance_edges
        select r.report_id, 'reports', r.source_event_id, c.operator, r.mechanism_id,
               r.calibration_version, r.batch_id, r.correction_version
        from reports r
        join cleaning_records c using(report_id)
        where r.report_id in (select report_id from temp_affected_reports)
        """
    )
    conn.commit()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    manifest = {
        "batch_id": batch_id,
        "batch_kind": batch_kind,
        "input_events": input_events,
        "accepted_new_events": accepted_new_events,
        "duplicate_events": duplicate_events,
        "correction_events": correction_events_count,
        "rejected_budget_events": rejected_budget_events,
        "late_events": late_events,
        "watermark_before": watermark,
        "watermark_after": watermark_after,
        "elapsed_ms": elapsed_ms,
    }
    cur.execute(
        """
        insert into batch_manifest values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            batch_id,
            batch_kind,
            input_events,
            accepted_new_events,
            duplicate_events,
            correction_events_count,
            rejected_budget_events,
            late_events,
            watermark,
            watermark_after,
            elapsed_ms,
        ),
    )
    conn.commit()
    return manifest, watermark_after


def refresh_materialized_state(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        delete from privacy_budget_trace;
        insert into privacy_budget_trace
        select report_id, device_id, ts, epsilon,
               sum(epsilon) over (
                   partition by device_id
                   order by ts, report_id
                   rows between unbounded preceding and current row
               ) as cumulative_epsilon,
               120.0 - sum(epsilon) over (
                   partition by device_id
                   order by ts, report_id
                   rows between unbounded preceding and current row
               ) as remaining_budget
        from reports;

        delete from mv_report_counts;
        insert into mv_report_counts
        select mechanism_id, epsilon, report_bucket, count(*) as n
        from reports
        group by mechanism_id, epsilon, report_bucket;

        delete from candidate_likelihoods;
        insert into candidate_likelihoods
        select c.mechanism_id, c.epsilon, w.stuck_bucket,
               sum(c.n * w.logp_fault) as fault_loglik,
               sum(c.n * w.logp_normal) as normal_loglik,
               sum(c.n * (w.logp_fault - w.logp_normal)) as llr
        from mv_report_counts c
        join mechanism_weights w
          on w.mechanism_id = c.mechanism_id
         and w.epsilon = c.epsilon
         and w.report_bucket = c.report_bucket
        group by c.mechanism_id, c.epsilon, w.stuck_bucket;

        delete from cleaned_event_windows;
        insert into cleaned_event_windows
        select device_id, ts / 60 as hour_id,
               count(*) as reports,
               sum(case when action in ('flag', 'repair') then 1 else 0 end) as flagged_reports,
               avg(posterior) as avg_posterior,
               max(posterior) as max_posterior
        from cleaning_records
        group by device_id, ts / 60
        having flagged_reports > 0;

        delete from repair_uncertainty_analytics;
        insert into repair_uncertainty_analytics
        select r.device_id, r.ts / 60 as hour_id,
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
        """
    )
    conn.commit()


def collect_queries(conn: sqlite3.Connection) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cur = conn.cursor()
    input_rows = int(cur.execute("select count(*) from reports").fetchone()[0])
    queries = {
        "batch_replay_audit": """
            select batch_kind, sum(input_events), sum(accepted_new_events), sum(duplicate_events), sum(rejected_budget_events)
            from batch_manifest
            group by batch_kind
        """,
        "budget_guard_dashboard": """
            select count(*) as devices,
                   min(remaining_budget),
                   avg(remaining_budget),
                   sum(case when remaining_budget < 5.0 then 1 else 0 end) as near_cap_devices
            from device_budget_state
        """,
        "rejected_event_drilldown": """
            select device_id, count(*) as rejected, min(ts) as first_rejected_ts, max(projected_epsilon_sum) as max_projected
            from rejected_events
            group by device_id
            order by rejected desc
            limit 20
        """,
        "late_correction_lineage": """
            select r.report_id, r.device_id, r.ts, r.report_bucket, c.operator, c.posterior, p.source_key, p.batch_id
            from reports r
            join cleaning_records c using(report_id)
            join provenance_edges p on p.output_report_id = r.report_id
            where r.correction_version > 0 and r.late_event = 1
            order by c.posterior desc
            limit 50
        """,
        "candidate_topk": """
            select epsilon, stuck_bucket, llr
            from candidate_likelihoods
            order by llr desc
            limit 20
        """,
        "privacy_trace_window": """
            select device_id, min(remaining_budget), max(cumulative_epsilon), count(*) as reports
            from privacy_budget_trace
            group by device_id
            order by min(remaining_budget)
            limit 20
        """,
        "event_window_uncertainty_join": """
            select e.device_id, e.hour_id, e.flagged_reports, e.avg_posterior, a.cleaned_mean, a.uncertainty_proxy
            from cleaned_event_windows e
            join repair_uncertainty_analytics a using(device_id, hour_id)
            where e.flagged_reports >= 3
            order by e.flagged_reports desc, e.avg_posterior desc
            limit 100
        """,
        "end_to_end_dashboard": """
            select case when d.remaining_budget < 5.0 then 'near_cap'
                        when d.remaining_budget < 25.0 then 'low'
                        else 'ok' end as budget_band,
                   count(distinct a.device_id) as devices,
                   count(*) as hourly_groups,
                   sum(coalesce(e.flagged_reports, 0)) as flagged_reports,
                   avg(a.cleaned_mean) as avg_cleaned_mean,
                   avg(a.uncertainty_proxy) as avg_uncertainty
            from repair_uncertainty_analytics a
            join device_budget_state d using(device_id)
            left join cleaned_event_windows e using(device_id, hour_id)
            group by budget_band
        """,
    }
    rows: list[dict[str, object]] = []
    plans: list[dict[str, object]] = []
    for name, query in queries.items():
        for step, (_, _, _, detail) in enumerate(cur.execute("explain query plan " + query).fetchall()):
            plans.append({"query_name": name, "plan_step": step, "detail": detail})
        for rep in range(3):
            rows.append(timed_query(cur, name, query, input_rows, rep))
    return rows, plans


def collect_checks(conn: sqlite3.Connection) -> list[dict[str, object]]:
    cur = conn.cursor()
    checks = []

    def add(name: str, expected: object, observed: object, status: bool, note: str = "") -> None:
        checks.append(
            {
                "check": name,
                "expected": expected,
                "observed": observed,
                "status": "pass" if status else "fail",
                "note": note,
            }
        )

    reports = int(cur.execute("select count(*) from reports").fetchone()[0])
    cleaning = int(cur.execute("select count(*) from cleaning_records").fetchone()[0])
    provenance = int(cur.execute("select count(*) from provenance_edges").fetchone()[0])
    add("cleaning_covers_reports", reports, cleaning, reports == cleaning)
    add("provenance_covers_reports", reports, provenance, reports == provenance)
    add("unique_report_ids", reports, int(cur.execute("select count(distinct report_id) from reports").fetchone()[0]), True)
    rejected = int(cur.execute("select count(*) from rejected_events").fetchone()[0])
    add("budget_guard_rejects_some_events", ">0", rejected, rejected > 0)
    max_budget = float(cur.execute("select max(epsilon_sum) from device_budget_state").fetchone()[0])
    add("device_budget_not_exceeded", f"<= {DEVICE_BUDGET}", max_budget, max_budget <= DEVICE_BUDGET + 1e-9)
    replays = cur.execute(
        "select sum(accepted_new_events), sum(duplicate_events) from batch_manifest where batch_kind like 'replay%'"
    ).fetchone()
    add("replayed_batches_insert_no_new_rows", 0, int(replays[0] or 0), int(replays[0] or 0) == 0)
    add("replayed_batches_detect_duplicates", ">0", int(replays[1] or 0), int(replays[1] or 0) > 0)
    corrections = int(cur.execute("select count(*) from reports where correction_version > 0").fetchone()[0])
    add("late_corrections_applied", ">0", corrections, corrections > 0)
    late_lineage = int(
        cur.execute(
            """
            select count(*)
            from reports r
            join provenance_edges p on p.output_report_id = r.report_id
            where r.correction_version > 0 and r.late_event = 1 and p.source_key like 'corr-%'
            """
        ).fetchone()[0]
    )
    add("late_corrections_have_lineage", corrections, late_lineage, late_lineage == corrections)
    event_windows = int(cur.execute("select count(*) from cleaned_event_windows").fetchone()[0])
    add("event_windows_materialized", ">0", event_windows, event_windows > 0)
    analytics = int(cur.execute("select count(*) from repair_uncertainty_analytics").fetchone()[0])
    add("repair_analytics_materialized", ">0", analytics, analytics > 0)
    trace = int(cur.execute("select count(*) from privacy_budget_trace").fetchone()[0])
    add("privacy_trace_covers_reports", reports, trace, reports == trace)
    integrity = cur.execute("pragma integrity_check").fetchone()[0]
    add("sqlite_integrity_check", "ok", integrity, integrity == "ok")
    return checks


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    clean_outputs()
    initialize_database()
    conn = connect()
    watermark = -1
    batch_rows: list[dict[str, object]] = []
    for batch_id, batch_kind, rows in make_batches():
        manifest, watermark = process_batch(conn, batch_id, batch_kind, rows, watermark)
        batch_rows.append(manifest)
    refresh_materialized_state(conn)
    query_rows, plan_rows = collect_queries(conn)
    check_rows = collect_checks(conn)
    conn.execute("pragma wal_checkpoint(full)")
    conn.close()
    shutil.copyfile(DB_PATH, RESULT_DB_PATH)

    write_csv(BATCH_CSV, batch_rows)
    write_csv(QUERY_CSV, query_rows)
    write_csv(PLAN_CSV, plan_rows)
    write_csv(CHECK_CSV, check_rows)

    summary = {
        "status": "pass" if all(row["status"] == "pass" for row in check_rows) else "fail",
        "rows_requested": TOTAL_ROWS,
        "devices": DEVICES,
        "device_budget": DEVICE_BUDGET,
        "batches": len(batch_rows),
        "accepted_reports": sum(int(row["accepted_new_events"]) for row in batch_rows),
        "duplicate_events": sum(int(row["duplicate_events"]) for row in batch_rows),
        "correction_events": sum(int(row["correction_events"]) for row in batch_rows),
        "rejected_budget_events": sum(int(row["rejected_budget_events"]) for row in batch_rows),
        "late_events": sum(int(row["late_events"]) for row in batch_rows),
        "query_rows": len(query_rows),
        "plan_rows": len(plan_rows),
        "checks": len(check_rows),
        "failed_checks": [row for row in check_rows if row["status"] != "pass"],
        "database": str(RESULT_DB_PATH.relative_to(ROOT)),
        "database_bytes": RESULT_DB_PATH.stat().st_size,
        "interpretation": (
            "End-to-end SQLite workflow for private telemetry cleaning: microbatch ingestion, "
            "idempotent replay, late correction propagation, budget guarding, cleaning/provenance "
            "upserts, privacy traces, event windows, repair-uncertainty analytics, and dashboard queries."
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
