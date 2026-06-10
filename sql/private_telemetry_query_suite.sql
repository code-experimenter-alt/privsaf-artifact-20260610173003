-- SQL query suite used for the physical-design evaluation.
-- Each query is also timed by scripts/run_private_telemetry_pipeline_queries.py.

-- Q1. Candidate likelihood from base reports.
SELECT
    r.mechanism_id,
    r.epsilon,
    w.stuck_bucket,
    SUM(w.logp_fault) AS fault_loglik,
    SUM(w.logp_normal) AS normal_loglik,
    SUM(w.logp_fault - w.logp_normal) AS llr
FROM reports AS r NOT INDEXED
JOIN mechanism_weights AS w
  ON w.mechanism_id = r.mechanism_id
 AND w.epsilon = r.epsilon
 AND w.report_bucket = r.report_bucket
GROUP BY r.mechanism_id, r.epsilon, w.stuck_bucket;

-- Q2. Candidate likelihood from materialized report counts.
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

-- Q3. Top candidate stuck buckets from persisted candidate likelihoods.
SELECT epsilon, stuck_bucket, llr
FROM candidate_likelihoods
ORDER BY llr DESC
LIMIT 10;

-- Q4. Windowed privacy-budget audit from base reports.
SELECT COUNT(*) AS reports,
       MIN(remaining_budget) AS min_remaining_budget,
       AVG(remaining_budget) AS avg_remaining_budget
FROM (
    SELECT 1000.0 - SUM(epsilon) OVER (
               PARTITION BY device_id
               ORDER BY ts
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
           ) AS remaining_budget
    FROM reports AS r NOT INDEXED
);

-- Q5. Privacy-budget audit from materialized budget trace.
SELECT COUNT(*) AS reports,
       MIN(remaining_budget) AS min_remaining_budget,
       AVG(remaining_budget) AS avg_remaining_budget
FROM privacy_budget_trace
WHERE remaining_budget < 1000.0;

-- Q6. Repair-aware aggregate from base report and cleaning tables.
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

-- Q7. Repair dashboard from materialized analytics.
SELECT
    device_id,
    COUNT(*) AS hourly_groups,
    AVG(cleaned_mean) AS avg_cleaned_mean,
    SUM(repair_count) AS repaired_reports,
    AVG(uncertainty_proxy) AS avg_uncertainty_proxy
FROM repair_uncertainty_analytics
GROUP BY device_id;

-- Q8. Provenance/event drilldown.
SELECT
    e.device_id,
    e.hour_id,
    e.flagged_reports,
    e.avg_posterior,
    h.cleaned_mean,
    h.uncertainty_proxy,
    l.remaining_budget
FROM cleaned_event_windows e
JOIN repair_uncertainty_analytics h
  ON h.device_id = e.device_id AND h.hour_id = e.hour_id
JOIN privacy_ledger l USING(device_id)
WHERE e.avg_posterior >= 0.72
ORDER BY e.flagged_reports DESC, e.avg_posterior DESC
LIMIT 100;

-- Q9. End-to-end ingestion-cleaning-analytics dashboard.
SELECT
    CASE WHEN l.remaining_budget < 980.0 THEN 'near_cap' ELSE 'ok' END AS budget_band,
    COUNT(*) AS hourly_groups,
    SUM(COALESCE(e.flagged_reports, 0)) AS flagged_reports,
    AVG(h.cleaned_mean) AS avg_cleaned_mean,
    AVG(h.uncertainty_proxy) AS avg_uncertainty_proxy
FROM repair_uncertainty_analytics h
JOIN privacy_ledger l USING(device_id)
LEFT JOIN cleaned_event_windows e
  ON e.device_id = h.device_id AND e.hour_id = h.hour_id
GROUP BY budget_band;
