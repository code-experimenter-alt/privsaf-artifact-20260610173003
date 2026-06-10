from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path


EXPECTED_COUNTS = {
    "reports": 200_000,
    "cleaning_records": 200_000,
    "provenance_edges": 200_000,
    "privacy_budget_trace": 200_000,
    "privacy_ledger": 2_000,
    "mv_report_counts": 96,
    "candidate_likelihoods": 96,
    "hourly_cleaned_analytics": 4_000,
    "cleaned_event_windows": 4_000,
    "repair_uncertainty_analytics": 4_000,
}


def verify_counts(conn: sqlite3.Connection) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for table, expected in EXPECTED_COUNTS.items():
        observed = conn.execute(f"select count(*) from {table}").fetchone()[0]
        rows.append(
            {
                "check": f"count:{table}",
                "expected": expected,
                "observed": observed,
                "status": "pass" if observed == expected else "fail",
            }
        )
    return rows


def verify_queries(conn: sqlite3.Connection) -> list[dict[str, object]]:
    checks = [
        (
            "candidate_topk_nonempty",
            "select count(*) from (select epsilon, stuck_bucket, llr from candidate_likelihoods order by llr desc limit 10)",
            10,
        ),
        (
            "budget_window_nonempty",
            "select count(*) from privacy_budget_trace where remaining_budget < 1000.0",
            200_000,
        ),
        (
            "provenance_drilldown_nonempty",
            """
            select count(*) from (
                select e.device_id, e.hour_id
                from cleaned_event_windows e
                join repair_uncertainty_analytics h
                  on h.device_id = e.device_id and h.hour_id = e.hour_id
                join privacy_ledger l using(device_id)
                where e.avg_posterior >= 0.72
                limit 100
            )
            """,
            100,
        ),
        (
            "dashboard_nonempty",
            """
            select count(*) from (
                select case when l.remaining_budget < 980.0 then 'near_cap' else 'ok' end as budget_band
                from repair_uncertainty_analytics h
                join privacy_ledger l using(device_id)
                left join cleaned_event_windows e
                  on e.device_id = h.device_id and e.hour_id = h.hour_id
                group by budget_band
            )
            """,
            1,
        ),
    ]
    rows: list[dict[str, object]] = []
    for name, query, expected in checks:
        observed = conn.execute(query).fetchone()[0]
        rows.append(
            {
                "check": name,
                "expected": expected,
                "observed": observed,
                "status": "pass" if observed == expected else "fail",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=["check", "expected", "observed", "status"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    conn = sqlite3.connect(str(args.db))
    rows = verify_counts(conn) + verify_queries(conn)
    conn.close()

    failures = [row for row in rows if row["status"] != "pass"]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.out, rows)
    for row in rows:
        print(f"{row['status']}: {row['check']} observed={row['observed']} expected={row['expected']}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
