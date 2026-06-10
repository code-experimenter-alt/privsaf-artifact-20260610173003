-- Physical-design ablation SQL for the private telemetry cleaning pipeline.
-- The executable benchmark is scripts/run_private_telemetry_physical_design_ablation.py.

-- Incrementally maintained report-count and privacy-accounting state.
CREATE TABLE mv_report_counts_inc(
    mechanism_id TEXT NOT NULL,
    epsilon REAL NOT NULL,
    report_bucket INTEGER NOT NULL,
    n INTEGER NOT NULL,
    PRIMARY KEY(mechanism_id, epsilon, report_bucket)
);

CREATE TABLE privacy_ledger_inc(
    device_id INTEGER PRIMARY KEY,
    reports INTEGER NOT NULL,
    epsilon_sum REAL NOT NULL,
    device_budget REAL NOT NULL,
    remaining_budget REAL NOT NULL
);

CREATE TABLE repair_uncertainty_inc(
    device_id INTEGER NOT NULL,
    hour_id INTEGER NOT NULL,
    reports INTEGER NOT NULL,
    posterior_sum REAL NOT NULL,
    repair_count INTEGER NOT NULL,
    cleaned_sum REAL NOT NULL,
    raw_sum REAL NOT NULL,
    uncertainty_sum REAL NOT NULL,
    PRIMARY KEY(device_id, hour_id)
);

CREATE TABLE event_windows_inc(
    device_id INTEGER NOT NULL,
    hour_id INTEGER NOT NULL,
    start_ts INTEGER NOT NULL,
    end_ts INTEGER NOT NULL,
    flagged_reports INTEGER NOT NULL,
    posterior_sum REAL NOT NULL,
    PRIMARY KEY(device_id, hour_id)
);

CREATE TRIGGER reports_to_count_inc AFTER INSERT ON reports
BEGIN
    INSERT INTO mv_report_counts_inc(mechanism_id, epsilon, report_bucket, n)
    VALUES(NEW.mechanism_id, NEW.epsilon, NEW.report_bucket, 1)
    ON CONFLICT(mechanism_id, epsilon, report_bucket)
    DO UPDATE SET n = mv_report_counts_inc.n + 1;

    INSERT INTO privacy_ledger_inc(device_id, reports, epsilon_sum, device_budget, remaining_budget)
    VALUES(NEW.device_id, 1, NEW.epsilon, 1000.0, 1000.0 - NEW.epsilon)
    ON CONFLICT(device_id)
    DO UPDATE SET
        reports = privacy_ledger_inc.reports + 1,
        epsilon_sum = privacy_ledger_inc.epsilon_sum + excluded.epsilon_sum,
        remaining_budget = privacy_ledger_inc.device_budget
            - (privacy_ledger_inc.epsilon_sum + excluded.epsilon_sum);
END;

CREATE TRIGGER cleaning_to_repair_inc AFTER INSERT ON cleaning_records
BEGIN
    INSERT INTO repair_uncertainty_inc(
        device_id, hour_id, reports, posterior_sum, repair_count,
        cleaned_sum, raw_sum, uncertainty_sum
    )
    VALUES(
        NEW.device_id,
        NEW.ts / 60,
        1,
        NEW.posterior,
        CASE WHEN NEW.action = 'repair' THEN 1 ELSE 0 END,
        NEW.repair_value,
        NEW.repair_value,
        NEW.posterior * (1.0 - NEW.posterior)
    )
    ON CONFLICT(device_id, hour_id)
    DO UPDATE SET
        reports = repair_uncertainty_inc.reports + 1,
        posterior_sum = repair_uncertainty_inc.posterior_sum + excluded.posterior_sum,
        repair_count = repair_uncertainty_inc.repair_count + excluded.repair_count,
        cleaned_sum = repair_uncertainty_inc.cleaned_sum + excluded.cleaned_sum,
        raw_sum = repair_uncertainty_inc.raw_sum + excluded.raw_sum,
        uncertainty_sum = repair_uncertainty_inc.uncertainty_sum + excluded.uncertainty_sum;
END;

CREATE TRIGGER cleaning_to_event_inc AFTER INSERT ON cleaning_records
WHEN NEW.action IN ('flag', 'repair')
BEGIN
    INSERT INTO event_windows_inc(device_id, hour_id, start_ts, end_ts, flagged_reports, posterior_sum)
    VALUES(NEW.device_id, NEW.ts / 60, NEW.ts, NEW.ts, 1, NEW.posterior)
    ON CONFLICT(device_id, hour_id)
    DO UPDATE SET
        start_ts = min(event_windows_inc.start_ts, excluded.start_ts),
        end_ts = max(event_windows_inc.end_ts, excluded.end_ts),
        flagged_reports = event_windows_inc.flagged_reports + 1,
        posterior_sum = event_windows_inc.posterior_sum + excluded.posterior_sum;
END;

-- Q1. Direct candidate likelihood from base reports.
SELECT
    r.mechanism_id,
    r.epsilon,
    w.stuck_bucket,
    SUM(w.logp_fault) AS fault_loglik,
    SUM(w.logp_normal) AS normal_loglik,
    SUM(w.logp_fault - w.logp_normal) AS llr
FROM reports r
JOIN mechanism_weights w
  ON w.mechanism_id = r.mechanism_id
 AND w.epsilon = r.epsilon
 AND w.report_bucket = r.report_bucket
GROUP BY r.mechanism_id, r.epsilon, w.stuck_bucket
ORDER BY llr DESC
LIMIT 96;

-- Q2. Candidate likelihood from incrementally maintained counts.
SELECT
    c.mechanism_id,
    c.epsilon,
    w.stuck_bucket,
    SUM(c.n * w.logp_fault) AS fault_loglik,
    SUM(c.n * w.logp_normal) AS normal_loglik,
    SUM(c.n * (w.logp_fault - w.logp_normal)) AS llr
FROM mv_report_counts_inc c
JOIN mechanism_weights w
  ON w.mechanism_id = c.mechanism_id
 AND w.epsilon = c.epsilon
 AND w.report_bucket = c.report_bucket
GROUP BY c.mechanism_id, c.epsilon, w.stuck_bucket
ORDER BY llr DESC
LIMIT 96;

-- Q3. End-to-end uncertainty, privacy, and event dashboard from maintained state.
WITH h AS (
    SELECT
        device_id,
        hour_id,
        reports,
        cleaned_sum / reports AS cleaned_mean,
        uncertainty_sum / reports AS uncertainty_proxy,
        repair_count
    FROM repair_uncertainty_inc
),
e AS (
    SELECT
        device_id,
        hour_id,
        flagged_reports,
        posterior_sum / flagged_reports AS avg_posterior
    FROM event_windows_inc
    WHERE flagged_reports >= 2
)
SELECT
    CASE WHEN l.remaining_budget < 980.0 THEN 'near_cap' ELSE 'ok' END AS budget_band,
    COUNT(*) AS hourly_groups,
    SUM(COALESCE(e.flagged_reports, 0)) AS flagged_reports,
    AVG(h.cleaned_mean) AS avg_cleaned_mean,
    AVG(h.uncertainty_proxy) AS avg_uncertainty_proxy,
    SUM(h.repair_count) AS repair_count
FROM h
JOIN privacy_ledger_inc l USING(device_id)
LEFT JOIN e ON e.device_id = h.device_id AND e.hour_id = h.hour_id
GROUP BY budget_band;

-- Q4. Provenance drilldown with mechanism-likelihood context.
SELECT
    c.report_id,
    r.device_id,
    r.ts,
    c.operator,
    c.posterior,
    p.transform,
    w.logp_fault - w.logp_normal AS llr_component
FROM cleaning_records c
JOIN reports r USING(report_id)
JOIN provenance_edges p ON p.output_report_id = c.report_id
JOIN mechanism_weights w
  ON w.mechanism_id = r.mechanism_id
 AND w.epsilon = r.epsilon
 AND w.report_bucket = r.report_bucket
 AND w.stuck_bucket = COALESCE(c.stuck_bucket, r.report_bucket)
WHERE c.action IN ('flag', 'repair')
ORDER BY c.posterior DESC, r.ts DESC
LIMIT 100;
