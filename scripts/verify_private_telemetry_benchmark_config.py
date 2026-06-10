from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "private_telemetry_benchmark_config.json"
RESULTS = ROOT / "results"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["check", "expected", "observed", "status"])
        writer.writeheader()
        writer.writerows(rows)


def check(rows: list[dict[str, object]], name: str, expected: object, observed: object, ok: bool) -> None:
    rows.append(
        {
            "check": name,
            "expected": expected,
            "observed": observed,
            "status": "pass" if ok else "fail",
        }
    )


def main() -> int:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []

    query_ids = [item["id"] for item in config["query_mix"]]
    required_query_count = int(config["acceptance_checks"]["required_query_count"])
    expected_query_ids = [f"Q{i}" for i in range(1, required_query_count + 1)]
    check(rows, "query_count", required_query_count, len(query_ids), len(query_ids) == required_query_count)
    check(rows, "query_ids_unique", len(query_ids), len(set(query_ids)), len(query_ids) == len(set(query_ids)))
    check(rows, "query_ids_q1_to_q9", f"Q1..Q{required_query_count}", ",".join(query_ids), query_ids == expected_query_ids)

    sql_text = (ROOT / "sql" / "private_telemetry_query_suite.sql").read_text(encoding="utf-8")
    sql_query_ids = re.findall(r"--\s*(Q[0-9]+)\.", sql_text)
    check(rows, "sql_suite_query_ids", ",".join(query_ids), ",".join(sql_query_ids), sql_query_ids == query_ids)

    observed_categories = sorted({item["category"] for item in config["query_mix"]})
    required_categories = sorted(config["acceptance_checks"]["required_categories"])
    check(rows, "query_categories", ",".join(required_categories), ",".join(observed_categories), set(required_categories).issubset(observed_categories))

    design_ids = [item["id"] for item in config["physical_designs"]]
    required_designs = config["acceptance_checks"]["required_physical_designs"]
    check(rows, "physical_designs", ",".join(required_designs), ",".join(design_ids), set(required_designs).issubset(design_ids))

    drivers = config["drivers"]
    for driver_id in [
        "pipeline_query",
        "physical_design",
        "sqlite_verifier",
        "sqlite_operational_stress",
        "end_to_end_workflow",
        "duckdb_integration",
        "config_verifier",
        "postgres_optional",
    ]:
        script = drivers[driver_id]["script"]
        check(rows, f"driver_script:{driver_id}", "exists", script, (ROOT / script).exists())

    physical_reps = int(config["workload_mix"]["physical_design_ablation"]["repetitions"])
    check(rows, "physical_repetitions", 3, physical_reps, physical_reps == 3)

    scale = config["scale"]
    check(rows, "scale_report_rows", 200000, scale["report_rows"], scale["report_rows"] == 200000)
    check(rows, "scale_devices", 2000, scale["devices"], scale["devices"] == 2000)
    check(rows, "scale_buckets", 32, scale["buckets"], scale["buckets"] == 32)

    e2e_driver = drivers["end_to_end_workflow"]
    e2e_workload = config["workload_mix"]["end_to_end_workflow"]
    e2e_acceptance = config["acceptance_checks"]
    e2e_sql_path = ROOT / e2e_driver["sql"]
    check(rows, "end_to_end_sql_file", "exists", e2e_driver["sql"], e2e_sql_path.exists())
    e2e_sql_query_ids: list[str] = []
    if e2e_sql_path.exists():
        e2e_sql_query_ids = re.findall(r"--\s*(Q[0-9]+)\.", e2e_sql_path.read_text(encoding="utf-8"))
    required_e2e_query_count = int(e2e_acceptance["required_end_to_end_query_count"])
    expected_e2e_query_ids = [f"Q{i}" for i in range(1, required_e2e_query_count + 1)]
    check(
        rows,
        "end_to_end_sql_query_ids",
        ",".join(expected_e2e_query_ids),
        ",".join(e2e_sql_query_ids),
        e2e_sql_query_ids == expected_e2e_query_ids,
    )
    required_e2e_features = set(e2e_acceptance["required_end_to_end_features"])
    observed_e2e_features = set(e2e_workload["features"])
    check(
        rows,
        "end_to_end_workflow_features",
        ",".join(sorted(required_e2e_features)),
        ",".join(sorted(observed_e2e_features)),
        required_e2e_features.issubset(observed_e2e_features),
    )

    e2e_summary_path = ROOT / "results" / "private_telemetry_end_to_end_workflow_summary.json"
    e2e_summary: dict[str, object] = {}
    if e2e_summary_path.exists():
        e2e_summary = json.loads(e2e_summary_path.read_text(encoding="utf-8"))
    check(rows, "end_to_end_summary_status", "pass", e2e_summary.get("status"), e2e_summary.get("status") == "pass")
    check(
        rows,
        "end_to_end_requested_events",
        e2e_driver["requested_events"],
        e2e_summary.get("rows_requested"),
        e2e_summary.get("rows_requested") == e2e_driver["requested_events"],
    )
    check(
        rows,
        "end_to_end_accepted_reports",
        e2e_driver["accepted_reports"],
        e2e_summary.get("accepted_reports"),
        e2e_summary.get("accepted_reports") == e2e_driver["accepted_reports"],
    )
    check(rows, "end_to_end_devices", e2e_driver["devices"], e2e_summary.get("devices"), e2e_summary.get("devices") == e2e_driver["devices"])
    check(
        rows,
        "end_to_end_device_budget",
        e2e_driver["device_budget"],
        e2e_summary.get("device_budget"),
        e2e_summary.get("device_budget") == e2e_driver["device_budget"],
    )
    check(
        rows,
        "end_to_end_failed_checks",
        0,
        len(e2e_summary.get("failed_checks", [])) if isinstance(e2e_summary.get("failed_checks", []), list) else "invalid",
        e2e_summary.get("failed_checks") == [],
    )
    check(
        rows,
        "end_to_end_query_rows",
        required_e2e_query_count * int(e2e_workload["query_repetitions"]),
        e2e_summary.get("query_rows"),
        e2e_summary.get("query_rows") == required_e2e_query_count * int(e2e_workload["query_repetitions"]),
    )
    check(rows, "end_to_end_plan_rows", ">0", e2e_summary.get("plan_rows"), int(e2e_summary.get("plan_rows", 0) or 0) > 0)

    e2e_checks_path = ROOT / "results" / "private_telemetry_end_to_end_workflow_checks.csv"
    e2e_check_rows: list[dict[str, str]] = []
    if e2e_checks_path.exists():
        with e2e_checks_path.open("r", encoding="utf-8", newline="") as fin:
            e2e_check_rows = list(csv.DictReader(fin))
    required_e2e_checks = int(e2e_acceptance["required_end_to_end_checks"])
    check(rows, "end_to_end_check_count", required_e2e_checks, len(e2e_check_rows), len(e2e_check_rows) == required_e2e_checks)
    check(
        rows,
        "end_to_end_checks_pass",
        required_e2e_checks,
        sum(1 for row in e2e_check_rows if row.get("status") == "pass"),
        len(e2e_check_rows) == required_e2e_checks and all(row.get("status") == "pass" for row in e2e_check_rows),
    )

    for output in config["required_outputs"]:
        path = ROOT / output
        observed = path.stat().st_size if path.exists() else 0
        check(rows, f"required_output:{output}", "exists_nonempty", observed, observed > 0)

    for output in config.get("orchestration_outputs", []):
        path = ROOT / output
        observed = path.stat().st_size if path.exists() else 0
        check(rows, f"orchestration_output:{output}", "exists_nonempty", observed, observed > 0)

    optional_present = []
    for output in config["optional_server_dbms_outputs"]:
        path = ROOT / output
        if path.exists() and path.stat().st_size > 0:
            optional_present.append(output)
    check(rows, "optional_server_dbms_outputs", "optional", len(optional_present), True)

    postgres_driver = drivers["postgres_optional"]
    postgres_sql_path = ROOT / postgres_driver["sql"]
    check(rows, "postgres_schema_sql_file", "exists", postgres_driver["sql"], postgres_sql_path.exists())
    postgres_summary_path = ROOT / "results" / "private_telemetry_postgres_integration_summary.json"
    postgres_summary: dict[str, object] = {}
    if postgres_summary_path.exists():
        postgres_summary = json.loads(postgres_summary_path.read_text(encoding="utf-8"))
    if postgres_summary.get("status") == "pass":
        for output in config["optional_server_dbms_outputs"]:
            path = ROOT / output
            observed = path.stat().st_size if path.exists() else 0
            check(rows, f"postgres_output:{output}", "exists_nonempty", observed, observed > 0)
        check(rows, "postgres_summary_status", "pass", postgres_summary.get("status"), postgres_summary.get("status") == "pass")
        check(
            rows,
            "postgres_rows",
            postgres_driver["report_rows"],
            postgres_summary.get("rows"),
            postgres_summary.get("rows") == postgres_driver["report_rows"],
        )
        check(
            rows,
            "postgres_devices",
            postgres_driver["devices"],
            postgres_summary.get("devices"),
            postgres_summary.get("devices") == postgres_driver["devices"],
        )
        required_postgres_query_count = int(config["acceptance_checks"]["required_postgres_query_count_if_available"])
        check(
            rows,
            "postgres_query_rows",
            required_postgres_query_count * int(postgres_driver["query_repetitions"]),
            postgres_summary.get("query_rows"),
            postgres_summary.get("query_rows") == required_postgres_query_count * int(postgres_driver["query_repetitions"]),
        )
        check(rows, "postgres_plan_rows", ">0", postgres_summary.get("plan_rows"), int(postgres_summary.get("plan_rows", 0) or 0) > 0)
        check(
            rows,
            "postgres_check_count",
            config["acceptance_checks"]["required_postgres_checks_if_available"],
            postgres_summary.get("checks"),
            postgres_summary.get("checks") == config["acceptance_checks"]["required_postgres_checks_if_available"],
        )
        check(rows, "postgres_failed_checks", 0, postgres_summary.get("failed_checks"), postgres_summary.get("failed_checks") == [])
        postgres_check_path = ROOT / "results" / "private_telemetry_postgres_verification.csv"
        postgres_check_rows: list[dict[str, str]] = []
        if postgres_check_path.exists():
            with postgres_check_path.open("r", encoding="utf-8", newline="") as fin:
                postgres_check_rows = list(csv.DictReader(fin))
        check(
            rows,
            "postgres_verification_pass",
            len(postgres_check_rows),
            sum(1 for row in postgres_check_rows if row.get("status") == "pass"),
            bool(postgres_check_rows) and all(row.get("status") == "pass" for row in postgres_check_rows),
        )
    else:
        check(rows, "postgres_summary_status", "optional_not_run_or_unavailable", postgres_summary.get("status", "missing"), True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "private_telemetry_benchmark_config_check.csv", rows)
    status = "pass" if all(row["status"] == "pass" for row in rows) else "fail"
    (RESULTS / "private_telemetry_benchmark_config_check.json").write_text(
        json.dumps(
            {
                "status": status,
                "checks": len(rows),
                "passes": sum(1 for row in rows if row["status"] == "pass"),
                "failures": [row for row in rows if row["status"] != "pass"],
                "config": str(CONFIG.relative_to(ROOT)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"benchmark config check {status}: {sum(1 for row in rows if row['status'] == 'pass')}/{len(rows)} checks passed")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
