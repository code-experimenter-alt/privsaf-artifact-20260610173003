from __future__ import annotations

import csv
import json
import sqlite3
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from private_telemetry_benchmark_config import load_benchmark_config
from run_private_telemetry_pipeline_queries import make_mechanism_weights, make_reports


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


BASE_SCHEMA = """
pragma journal_mode = wal;
pragma synchronous = normal;
pragma temp_store = memory;
pragma busy_timeout = 30000;
pragma wal_autocheckpoint = 1000000;

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

create index idx_reports_device_ts on reports(device_id, ts);
create index idx_reports_channel on reports(mechanism_id, epsilon, report_bucket);
create index idx_cleaning_action on cleaning_records(action, posterior desc);
create index idx_cleaning_device_ts on cleaning_records(device_id, ts);
create index idx_provenance_key on provenance_edges(output_report_id);
"""


DERIVE_SQL = """
insert into cleaning_records
select
    report_id,
    device_id,
    ts,
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
from reports;

insert into provenance_edges
select report_id, 'reports', report_id, operator, 'pm32', 'cal_v1'
from cleaning_records;

insert into privacy_ledger
select device_id, count(*), sum(epsilon), 1000.0, 1000.0 - sum(epsilon)
from reports
group by device_id;

create table mv_report_counts as
select mechanism_id, epsilon, report_bucket, count(*) as n
from reports
group by mechanism_id, epsilon, report_bucket;

create table repair_uncertainty_analytics as
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
group by r.device_id, r.ts / 60;
"""


def clean_db(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
    conn.execute("pragma busy_timeout = 10000")
    conn.execute("pragma journal_mode = wal")
    conn.execute("pragma synchronous = normal")
    return conn


def add_metric(rows: list[dict[str, object]], scenario: str, metric: str, value: object, unit: str, status: str, note: str = "") -> None:
    rows.append(
        {
            "scenario": scenario,
            "metric": metric,
            "value": value,
            "unit": unit,
            "status": status,
            "note": note,
        }
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def chunked(rows: list[tuple[int, int, int, int, float, str, str]], size: int) -> list[list[tuple[int, int, int, int, float, str, str]]]:
    return [rows[idx : idx + size] for idx in range(0, len(rows), size)]


def setup_database(db_path: Path) -> None:
    conn = connect(db_path)
    conn.executescript(BASE_SCHEMA)
    conn.executemany("insert into mechanism_weights values (?, ?, ?, ?, ?, ?)", make_mechanism_weights())
    conn.executemany(
        "insert into router_policy values (?, ?, ?, ?)",
        [
            ("privsaf_mixture", "iid stuck-at", "cleaning_records", "post_processing"),
            ("privsaf_hmm", "persistent flatline", "cleaning_records", "post_processing"),
            ("pm_window_glr", "field-QC local regime", "cleaning_records", "post_processing"),
        ],
    )
    conn.commit()
    conn.close()


def writer_task(db_path: Path, writer_id: int, batches: list[list[tuple[int, int, int, int, float, str, str]]], start: threading.Event) -> dict[str, object]:
    start.wait()
    conn = connect(db_path)
    inserted = 0
    busy_errors = 0
    t0 = time.perf_counter()
    for batch in batches:
        try:
            conn.execute("begin immediate")
            conn.executemany("insert into reports values (?, ?, ?, ?, ?, ?, ?)", batch)
            conn.commit()
            inserted += len(batch)
        except sqlite3.OperationalError as exc:
            conn.rollback()
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                busy_errors += 1
            else:
                conn.close()
                raise
    elapsed = time.perf_counter() - t0
    conn.close()
    return {
        "writer_id": writer_id,
        "inserted_rows": inserted,
        "elapsed_sec": elapsed,
        "throughput_rows_per_sec": inserted / max(elapsed, 1e-12),
        "busy_errors": busy_errors,
    }


def reader_task(
    db_path: Path,
    reader_id: int,
    start: threading.Event,
    stop: threading.Event,
    latencies: list[float],
    lock: threading.Lock,
    sleep_sec: float,
) -> dict[str, object]:
    start.wait()
    conn = connect(db_path)
    reads = 0
    errors = 0
    while not stop.is_set():
        t0 = time.perf_counter()
        try:
            conn.execute(
                """
                select count(*), coalesce(avg(epsilon), 0.0)
                from reports
                where device_id between ? and ?
                """,
                (reader_id * 10, reader_id * 10 + 199),
            ).fetchone()
            latency = (time.perf_counter() - t0) * 1000.0
            with lock:
                latencies.append(latency)
            reads += 1
        except sqlite3.OperationalError:
            errors += 1
        time.sleep(sleep_sec)
    conn.close()
    return {"reader_id": reader_id, "reads": reads, "errors": errors}


def run_concurrency(
    db_path: Path,
    metrics: list[dict[str, object]],
    detail_rows: list[dict[str, object]],
    stress_config: dict[str, object],
) -> int:
    total_rows = int(stress_config.get("report_rows", 12_000))
    devices = int(stress_config.get("devices", 300))
    writers = int(stress_config.get("writer_clients", 3))
    readers = int(stress_config.get("reader_clients", 2))
    chunk_size = int(stress_config.get("writer_chunk_rows", 2_000))
    reader_sleep_sec = float(stress_config.get("reader_sleep_sec", 0.01))
    clean_db(db_path)
    setup_database(db_path)
    reports = make_reports(total_rows, devices, seed=31127)
    writer_batches = [chunked(reports[writer_id::writers], chunk_size) for writer_id in range(writers)]

    start = threading.Event()
    stop = threading.Event()
    read_latencies: list[float] = []
    latency_lock = threading.Lock()
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=writers + readers) as pool:
        writer_futures = [pool.submit(writer_task, db_path, idx, writer_batches[idx], start) for idx in range(writers)]
        reader_futures = [pool.submit(reader_task, db_path, idx, start, stop, read_latencies, latency_lock, reader_sleep_sec) for idx in range(readers)]
        start.set()
        writer_results = [future.result() for future in writer_futures]
        stop.set()
        reader_results = [future.result() for future in reader_futures]
    wall_elapsed = time.perf_counter() - wall_start

    conn = connect(db_path)
    observed_rows = conn.execute("select count(*) from reports").fetchone()[0]
    conn.executescript(DERIVE_SQL)
    conn.commit()
    derived_rows = conn.execute("select count(*) from cleaning_records").fetchone()[0]
    analytics_groups = conn.execute("select count(*) from repair_uncertainty_analytics").fetchone()[0]
    conn.close()

    for row in writer_results:
        detail_rows.append({"scenario": "concurrent_ingest", "actor": f"writer_{row['writer_id']}", **row})
    for row in reader_results:
        detail_rows.append({"scenario": "concurrent_ingest", "actor": f"reader_{row['reader_id']}", **row})

    add_metric(metrics, "concurrent_ingest", "writer_clients", writers, "clients", "pass")
    add_metric(metrics, "concurrent_ingest", "reader_clients", readers, "clients", "pass")
    add_metric(metrics, "concurrent_ingest", "inserted_rows", observed_rows, "rows", "pass" if observed_rows == total_rows else "fail")
    add_metric(metrics, "concurrent_ingest", "wall_throughput", total_rows / max(wall_elapsed, 1e-12), "rows_per_sec", "pass")
    add_metric(metrics, "concurrent_ingest", "writer_busy_errors", sum(int(row["busy_errors"]) for row in writer_results), "errors", "pass")
    add_metric(metrics, "concurrent_ingest", "reader_errors", sum(int(row["errors"]) for row in reader_results), "errors", "pass")
    add_metric(metrics, "concurrent_ingest", "reader_queries", sum(int(row["reads"]) for row in reader_results), "queries", "pass")
    if read_latencies:
        add_metric(metrics, "concurrent_ingest", "reader_p50_latency", statistics.median(read_latencies), "ms", "pass")
        add_metric(metrics, "concurrent_ingest", "reader_max_latency", max(read_latencies), "ms", "pass")
    add_metric(metrics, "ingestion_cleaning_analytics", "cleaning_records", derived_rows, "rows", "pass" if derived_rows == total_rows else "fail")
    add_metric(metrics, "ingestion_cleaning_analytics", "repair_analytics_groups", analytics_groups, "groups", "pass" if analytics_groups > 0 else "fail")
    return total_rows


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_recovery(db_path: Path, metrics: list[dict[str, object]], expected_rows: int) -> None:
    wal_before = file_size(Path(str(db_path) + "-wal"))
    shm_before = file_size(Path(str(db_path) + "-shm"))
    start = time.perf_counter()
    conn = connect(db_path)
    reopen_sec = time.perf_counter() - start
    integrity_start = time.perf_counter()
    integrity = conn.execute("pragma integrity_check").fetchone()[0]
    integrity_sec = time.perf_counter() - integrity_start
    count_start = time.perf_counter()
    report_count = conn.execute("select count(*) from reports").fetchone()[0]
    dashboard_rows = conn.execute(
        """
        select l.device_id, l.remaining_budget, coalesce(sum(r.repair_count), 0)
        from privacy_ledger l
        left join repair_uncertainty_analytics r using(device_id)
        group by l.device_id, l.remaining_budget
        limit 50
        """
    ).fetchall()
    count_sec = time.perf_counter() - count_start
    checkpoint_start = time.perf_counter()
    checkpoint = conn.execute("pragma wal_checkpoint(truncate)").fetchone()
    checkpoint_sec = time.perf_counter() - checkpoint_start
    conn.close()

    add_metric(metrics, "recovery_restart", "reopen_time", reopen_sec, "sec", "pass")
    add_metric(metrics, "recovery_restart", "integrity_check", integrity, "status", "pass" if integrity == "ok" else "fail")
    add_metric(metrics, "recovery_restart", "integrity_check_time", integrity_sec, "sec", "pass")
    add_metric(metrics, "recovery_restart", "report_count_after_reopen", report_count, "rows", "pass" if report_count == expected_rows else "fail")
    add_metric(metrics, "recovery_restart", "dashboard_rows_after_reopen", len(dashboard_rows), "rows", "pass" if dashboard_rows else "fail")
    add_metric(metrics, "recovery_restart", "dashboard_query_time", count_sec, "sec", "pass")
    add_metric(metrics, "recovery_restart", "wal_checkpoint_time", checkpoint_sec, "sec", "pass", str(checkpoint))
    add_metric(metrics, "storage_footprint", "main_db_bytes", file_size(db_path), "bytes", "pass")
    add_metric(metrics, "storage_footprint", "wal_bytes_before_checkpoint", wal_before, "bytes", "pass")
    add_metric(metrics, "storage_footprint", "shm_bytes_before_checkpoint", shm_before, "bytes", "pass")
    add_metric(metrics, "storage_footprint", "wal_bytes_after_checkpoint", file_size(Path(str(db_path) + "-wal")), "bytes", "pass")


def run_page_limit_pressure(db_path: Path, metrics: list[dict[str, object]], stress_config: dict[str, object]) -> None:
    clean_db(db_path)
    conn = sqlite3.connect(str(db_path))
    conn.execute("pragma page_size = 4096")
    conn.execute("pragma journal_mode = delete")
    conn.execute(f"pragma max_page_count = {int(stress_config.get('page_limit_pages', 384))}")
    conn.executescript(
        """
        create table reports(
            report_id integer primary key,
            device_id integer not null,
            ts integer not null,
            report_bucket integer not null,
            epsilon real not null,
            mechanism_id text not null,
            calibration_version text not null
        );
        create index idx_reports_device_ts on reports(device_id, ts);
        create index idx_reports_channel on reports(mechanism_id, epsilon, report_bucket);
        """
    )
    inserted = 0
    error_text = ""
    pressure_rows = int(stress_config.get("page_limit_pressure_rows", 40_000))
    pressure_devices = max(100, int(stress_config.get("devices", 300)))
    for batch in chunked(make_reports(pressure_rows, pressure_devices, seed=31128), 500):
        try:
            conn.executemany("insert into reports values (?, ?, ?, ?, ?, ?, ?)", batch)
            conn.commit()
            inserted += len(batch)
        except sqlite3.OperationalError as exc:
            conn.rollback()
            error_text = str(exc)
            break
    page_count = conn.execute("pragma page_count").fetchone()[0]
    max_page_count = conn.execute("pragma max_page_count").fetchone()[0]
    integrity = conn.execute("pragma integrity_check").fetchone()[0]
    conn.close()
    full_observed = "full" in error_text.lower()
    add_metric(metrics, "sqlite_page_limit_pressure", "inserted_before_limit", inserted, "rows", "pass" if full_observed else "fail", error_text)
    add_metric(metrics, "sqlite_page_limit_pressure", "page_count", page_count, "pages", "pass")
    add_metric(metrics, "sqlite_page_limit_pressure", "max_page_count", max_page_count, "pages", "pass")
    add_metric(metrics, "sqlite_page_limit_pressure", "integrity_after_limit", integrity, "status", "pass" if integrity == "ok" else "fail")
    add_metric(metrics, "sqlite_page_limit_pressure", "db_bytes_at_limit", file_size(db_path), "bytes", "pass")


def main() -> int:
    config = load_benchmark_config()
    stress_config = config.get("drivers", {}).get("sqlite_operational_stress", {})
    output_dir = RESULTS
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_db_dir = Path(str(stress_config.get("temp_database_dir", "/tmp/privsaf_sqlite_operational_stress")))
    temp_db_dir.mkdir(parents=True, exist_ok=True)
    db_path = temp_db_dir / str(stress_config.get("database_path", "private_telemetry_operational_stress.sqlite"))
    pressure_db = temp_db_dir / str(stress_config.get("pressure_database_path", "private_telemetry_page_limit_pressure.sqlite"))

    metrics: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    expected_rows = run_concurrency(db_path, metrics, detail_rows, stress_config)
    run_recovery(db_path, metrics, expected_rows)
    run_page_limit_pressure(pressure_db, metrics, stress_config)

    metric_fields = ["scenario", "metric", "value", "unit", "status", "note"]
    detail_fields = ["scenario", "actor", "writer_id", "reader_id", "inserted_rows", "elapsed_sec", "throughput_rows_per_sec", "busy_errors", "reads", "errors"]
    write_csv(output_dir / "private_telemetry_sqlite_operational_stress.csv", metrics, metric_fields)
    write_csv(output_dir / "private_telemetry_sqlite_operational_stress_detail.csv", detail_rows, detail_fields)
    failed = [row for row in metrics if row["status"] != "pass"]
    summary = {
        "status": "pass" if not failed else "fail",
        "metrics": len(metrics),
        "failed": failed,
        "database_path": display_path(db_path),
        "pressure_database_path": display_path(pressure_db),
    }
    (output_dir / "private_telemetry_sqlite_operational_stress.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
