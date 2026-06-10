-- PostgreSQL schema for the private telemetry cleaning pipeline.
-- This is the server-DBMS counterpart to sql/private_telemetry_schema.sql.

CREATE SCHEMA IF NOT EXISTS privsaf_pg;
SET search_path TO privsaf_pg;

DROP MATERIALIZED VIEW IF EXISTS mv_report_counts CASCADE;
DROP MATERIALIZED VIEW IF EXISTS repair_uncertainty_analytics CASCADE;
DROP MATERIALIZED VIEW IF EXISTS cleaned_event_windows CASCADE;
DROP MATERIALIZED VIEW IF EXISTS privacy_budget_trace CASCADE;
DROP MATERIALIZED VIEW IF EXISTS candidate_likelihoods CASCADE;
DROP TABLE IF EXISTS privacy_ledger CASCADE;
DROP TABLE IF EXISTS provenance_edges CASCADE;
DROP TABLE IF EXISTS cleaning_records CASCADE;
DROP TABLE IF EXISTS router_policy CASCADE;
DROP TABLE IF EXISTS mechanism_weights CASCADE;
DROP TABLE IF EXISTS reports CASCADE;

CREATE TABLE reports (
    report_id BIGINT PRIMARY KEY,
    device_id INTEGER NOT NULL,
    ts INTEGER NOT NULL,
    report_bucket INTEGER NOT NULL,
    epsilon DOUBLE PRECISION NOT NULL,
    mechanism_id TEXT NOT NULL,
    calibration_version TEXT NOT NULL
);

CREATE TABLE mechanism_weights (
    mechanism_id TEXT NOT NULL,
    epsilon DOUBLE PRECISION NOT NULL,
    report_bucket INTEGER NOT NULL,
    stuck_bucket INTEGER NOT NULL,
    logp_normal DOUBLE PRECISION NOT NULL,
    logp_fault DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (mechanism_id, epsilon, report_bucket, stuck_bucket)
);

CREATE TABLE router_policy (
    operator TEXT PRIMARY KEY,
    semantic_layer TEXT NOT NULL,
    output_relation TEXT NOT NULL,
    privacy_effect TEXT NOT NULL
);

CREATE TABLE cleaning_records (
    report_id BIGINT PRIMARY KEY,
    device_id INTEGER NOT NULL,
    ts INTEGER NOT NULL,
    operator TEXT NOT NULL,
    posterior DOUBLE PRECISION NOT NULL,
    stuck_bucket INTEGER,
    fault_ratio DOUBLE PRECISION NOT NULL,
    action TEXT NOT NULL,
    repair_value DOUBLE PRECISION,
    provenance_version TEXT NOT NULL
);

CREATE TABLE provenance_edges (
    output_report_id BIGINT NOT NULL,
    source_table TEXT NOT NULL,
    source_key BIGINT NOT NULL,
    transform TEXT NOT NULL,
    mechanism_id TEXT NOT NULL,
    calibration_version TEXT NOT NULL
);

CREATE TABLE privacy_ledger (
    device_id INTEGER PRIMARY KEY,
    reports BIGINT NOT NULL,
    epsilon_sum DOUBLE PRECISION NOT NULL,
    device_budget DOUBLE PRECISION NOT NULL,
    remaining_budget DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_pg_reports_device_ts
ON reports(device_id, ts);

CREATE INDEX idx_pg_reports_channel
ON reports(mechanism_id, epsilon, report_bucket);

CREATE INDEX idx_pg_cleaning_action
ON cleaning_records(action, posterior DESC);

CREATE INDEX idx_pg_cleaning_device_ts
ON cleaning_records(device_id, ts);

CREATE INDEX idx_pg_provenance_output
ON provenance_edges(output_report_id);

CREATE MATERIALIZED VIEW mv_report_counts AS
SELECT mechanism_id, epsilon, report_bucket, COUNT(*) AS n
FROM reports
GROUP BY mechanism_id, epsilon, report_bucket;

CREATE INDEX idx_pg_mv_report_counts_channel
ON mv_report_counts(mechanism_id, epsilon, report_bucket);

CREATE MATERIALIZED VIEW candidate_likelihoods AS
SELECT
    c.mechanism_id,
    c.epsilon,
    w.stuck_bucket,
    SUM(c.n * w.logp_fault) AS fault_loglik,
    SUM(c.n * w.logp_normal) AS normal_loglik,
    SUM(c.n * (w.logp_fault - w.logp_normal)) AS llr
FROM mv_report_counts c
JOIN mechanism_weights w
  ON w.mechanism_id = c.mechanism_id
 AND w.epsilon = c.epsilon
 AND w.report_bucket = c.report_bucket
GROUP BY c.mechanism_id, c.epsilon, w.stuck_bucket;

CREATE INDEX idx_pg_candidate_llr
ON candidate_likelihoods(epsilon, llr DESC);

CREATE MATERIALIZED VIEW privacy_budget_trace AS
SELECT
    report_id,
    device_id,
    ts,
    epsilon,
    cumulative_epsilon,
    1000.0 - cumulative_epsilon AS remaining_budget
FROM (
    SELECT
        report_id,
        device_id,
        ts,
        epsilon,
        SUM(epsilon) OVER (
            PARTITION BY device_id
            ORDER BY ts
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_epsilon
    FROM reports
) budget_window;

CREATE INDEX idx_pg_budget_trace_device_ts
ON privacy_budget_trace(device_id, ts);

CREATE INDEX idx_pg_budget_remaining
ON privacy_budget_trace(remaining_budget, device_id, ts);

CREATE MATERIALIZED VIEW cleaned_event_windows AS
SELECT
    c.device_id,
    c.ts / 60 AS hour_id,
    MIN(c.ts) AS start_ts,
    MAX(c.ts) AS end_ts,
    COUNT(*) AS flagged_reports,
    AVG(c.posterior) AS avg_posterior,
    STRING_AGG(DISTINCT p.semantic_layer, ',' ORDER BY p.semantic_layer) AS routed_semantics
FROM cleaning_records c
JOIN router_policy p ON p.operator = c.operator
WHERE c.action IN ('flag', 'repair')
GROUP BY c.device_id, c.ts / 60
HAVING COUNT(*) >= 2;

CREATE INDEX idx_pg_event_windows_device_hour
ON cleaned_event_windows(device_id, hour_id);

CREATE MATERIALIZED VIEW repair_uncertainty_analytics AS
SELECT
    r.device_id,
    r.ts / 60 AS hour_id,
    COUNT(*) AS reports,
    AVG(CASE WHEN c.action = 'repair'
             THEN c.repair_value
             ELSE (2.0 * r.report_bucket / 31.0) - 1.0 END) AS cleaned_mean,
    AVG((2.0 * r.report_bucket / 31.0) - 1.0) AS unrepaired_mean,
    AVG(c.posterior * (1.0 - c.posterior)) AS uncertainty_proxy,
    SUM(CASE WHEN c.action = 'repair' THEN 1 ELSE 0 END) AS repair_count
FROM reports r
JOIN cleaning_records c USING(report_id)
GROUP BY r.device_id, r.ts / 60;

CREATE INDEX idx_pg_repair_device_hour
ON repair_uncertainty_analytics(device_id, hour_id);
