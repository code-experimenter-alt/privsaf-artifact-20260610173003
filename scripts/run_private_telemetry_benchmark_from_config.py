from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from private_telemetry_benchmark_config import CONFIG_ENV, ROOT, config_path_from_env, load_benchmark_config, result_path


RESULTS = ROOT / "results"


def as_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def output_status(outputs: list[str]) -> tuple[int, int, list[str]]:
    present = 0
    total_bytes = 0
    missing: list[str] = []
    for output in outputs:
        path = result_path(output)
        if path.exists() and path.stat().st_size > 0:
            present += 1
            total_bytes += path.stat().st_size
        else:
            missing.append(output)
    return present, total_bytes, missing


def stage_rows(config: dict[str, Any], config_path: Path, include_postgres: bool) -> list[dict[str, Any]]:
    drivers = config["drivers"]
    required = config["required_outputs"]
    stages: list[dict[str, Any]] = [
        {
            "stage": "pipeline_query_workload",
            "script": drivers["pipeline_query"]["script"],
            "command": [sys.executable, "-B", drivers["pipeline_query"]["script"]],
            "outputs": [
                "results/private_telemetry_pipeline_query_benchmark.csv",
                "results/private_telemetry_pipeline_query_benchmark.json",
                "results/private_telemetry_pipeline_query_plans.csv",
                "results/private_telemetry_pipeline_200k.sqlite",
                "results/private_telemetry_sql_query_suite.csv",
                "results/private_telemetry_sql_query_suite_summary.csv",
                "results/private_telemetry_sql_query_suite_plans.csv",
            ],
            "config_fields": "scale,storage_modes,drivers.pipeline_query",
            "required": True,
        },
        {
            "stage": "physical_design_ablation",
            "script": drivers["physical_design"]["script"],
            "command": [sys.executable, "-B", drivers["physical_design"]["script"]],
            "outputs": [
                "results/private_telemetry_physical_design_build.csv",
                "results/private_telemetry_physical_design_build_summary.csv",
                "results/private_telemetry_physical_design_queries.csv",
                "results/private_telemetry_physical_design_summary.csv",
                "results/private_telemetry_physical_design_plans.csv",
                "results/private_telemetry_physical_design_checks.csv",
                "results/private_telemetry_physical_design_summary.json",
            ],
            "config_fields": "scale,workload_mix.physical_design_ablation,drivers.physical_design",
            "required": True,
        },
        {
            "stage": "sqlite_artifact_verification",
            "script": drivers["sqlite_verifier"]["script"],
            "command": [
                sys.executable,
                "-B",
                drivers["sqlite_verifier"]["script"],
                "--db",
                drivers["sqlite_verifier"]["db"],
                "--out",
                drivers["sqlite_verifier"]["out"],
            ],
            "outputs": [drivers["sqlite_verifier"]["out"]],
            "config_fields": "drivers.sqlite_verifier",
            "required": True,
        },
        {
            "stage": "sqlite_operational_stress",
            "script": drivers["sqlite_operational_stress"]["script"],
            "command": [sys.executable, "-B", drivers["sqlite_operational_stress"]["script"]],
            "outputs": [
                "results/private_telemetry_sqlite_operational_stress.csv",
                "results/private_telemetry_sqlite_operational_stress_detail.csv",
                "results/private_telemetry_sqlite_operational_stress.json",
            ],
            "config_fields": "drivers.sqlite_operational_stress",
            "required": True,
        },
        {
            "stage": "end_to_end_workflow",
            "script": drivers["end_to_end_workflow"]["script"],
            "command": [sys.executable, "-B", drivers["end_to_end_workflow"]["script"]],
            "outputs": [
                "results/private_telemetry_end_to_end_workflow.sqlite",
                "results/private_telemetry_end_to_end_workflow_batches.csv",
                "results/private_telemetry_end_to_end_workflow_queries.csv",
                "results/private_telemetry_end_to_end_workflow_plans.csv",
                "results/private_telemetry_end_to_end_workflow_checks.csv",
                "results/private_telemetry_end_to_end_workflow_summary.json",
            ],
            "config_fields": "workload_mix.end_to_end_workflow,drivers.end_to_end_workflow",
            "required": True,
        },
        {
            "stage": "duckdb_integration",
            "script": drivers["duckdb_integration"]["script"],
            "command": [sys.executable, "-B", drivers["duckdb_integration"]["script"]],
            "outputs": [
                "results/private_telemetry_duckdb_environment_audit.csv",
                "results/private_telemetry_duckdb_build.csv",
                "results/private_telemetry_duckdb_queries.csv",
                "results/private_telemetry_duckdb_plans.csv",
                "results/private_telemetry_duckdb_verification.csv",
                "results/private_telemetry_duckdb_summary.json",
                "results/private_telemetry_duckdb_pipeline.duckdb",
            ],
            "config_fields": "drivers.duckdb_integration",
            "required": True,
        },
        {
            "stage": "benchmark_config_verification",
            "script": drivers["config_verifier"]["script"],
            "command": [sys.executable, "-B", drivers["config_verifier"]["script"]],
            "outputs": [
                "results/private_telemetry_benchmark_config_check.csv",
                "results/private_telemetry_benchmark_config_check.json",
            ],
            "config_fields": "query_mix,physical_designs,required_outputs,acceptance_checks",
            "required": True,
        },
    ]
    if include_postgres:
        stages.append(
            {
                "stage": "postgres_optional_integration",
                "script": drivers["postgres_optional"]["script"],
                "command": [sys.executable, "-B", drivers["postgres_optional"]["script"]],
                "outputs": [
                    "results/private_telemetry_postgres_environment_audit.csv",
                    "results/private_telemetry_postgres_integration_summary.json",
                    "results/private_telemetry_postgres_query_benchmark.csv",
                    "results/private_telemetry_postgres_queries.csv",
                    "results/private_telemetry_postgres_plans.csv",
                    "results/private_telemetry_postgres_verification.csv",
                ],
                "config_fields": "drivers.postgres_optional,optional_server_dbms_outputs,acceptance_checks.required_postgres_*",
                "required": False,
            }
        )

    for stage in stages:
        stage["config"] = as_rel(config_path)
        stage["required_outputs_intersection"] = sorted(set(stage["outputs"]).intersection(required))
    return stages


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "stage",
        "required",
        "config",
        "config_fields",
        "script",
        "script_exists",
        "executed",
        "returncode",
        "status",
        "present_outputs",
        "total_outputs",
        "output_bytes",
        "missing_outputs",
        "command",
    ]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or manifest the private telemetry benchmark from JSON config.")
    parser.add_argument("--config", default=None, help="Benchmark config path. Defaults to PRIVATE_TELEMETRY_BENCHMARK_CONFIG or the project config.")
    parser.add_argument("--execute", action="store_true", help="Execute required benchmark stages instead of writing a manifest only.")
    parser.add_argument("--include-postgres", action="store_true", help="Include the optional PostgreSQL stage in the manifest or execution.")
    args = parser.parse_args()

    config_path = config_path_from_env(args.config)
    config = load_benchmark_config(config_path)
    RESULTS.mkdir(parents=True, exist_ok=True)
    stages = stage_rows(config, config_path, args.include_postgres)
    env = os.environ.copy()
    env[CONFIG_ENV] = str(config_path)

    manifest_rows: list[dict[str, Any]] = []
    for stage in stages:
        script_path = result_path(stage["script"])
        command = [str(item) for item in stage["command"]]
        executed = bool(args.execute and (stage["required"] or stage["stage"] == "postgres_optional_integration"))
        returncode: int | str = "not_executed"
        if executed:
            result = subprocess.run(command, cwd=str(ROOT), env=env, check=False)
            returncode = result.returncode
        present, output_bytes, missing = output_status(stage["outputs"])
        script_exists = script_path.exists()
        if executed and returncode != 0:
            status = "failed"
        elif not script_exists:
            status = "missing_script"
        elif missing and stage["required"]:
            status = "missing_outputs"
        elif executed:
            status = "executed"
        else:
            status = "ready"
        manifest_rows.append(
            {
                "stage": stage["stage"],
                "required": int(bool(stage["required"])),
                "config": stage["config"],
                "config_fields": stage["config_fields"],
                "script": stage["script"],
                "script_exists": int(script_exists),
                "executed": int(executed),
                "returncode": returncode,
                "status": status,
                "present_outputs": present,
                "total_outputs": len(stage["outputs"]),
                "output_bytes": output_bytes,
                "missing_outputs": ";".join(missing),
                "command": " ".join(command),
            }
        )

    manifest_csv = RESULTS / "private_telemetry_benchmark_config_manifest.csv"
    manifest_json = RESULTS / "private_telemetry_benchmark_config_manifest.json"
    write_manifest(manifest_csv, manifest_rows)
    overall_status = "pass" if all(row["status"] in {"ready", "executed"} for row in manifest_rows if int(row["required"])) else "fail"
    manifest_json.write_text(
        json.dumps(
            {
                "status": overall_status,
                "execute": bool(args.execute),
                "include_postgres": bool(args.include_postgres),
                "config": as_rel(config_path),
                "stages": manifest_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"benchmark config manifest {overall_status}: {len(manifest_rows)} stages, execute={bool(args.execute)}")
    return 0 if overall_status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
