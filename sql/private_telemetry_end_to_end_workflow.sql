-- End-to-end private telemetry cleaning workflow checks and dashboard queries.
-- Timed by scripts/run_private_telemetry_end_to_end_workflow.py.

-- Q1. Replay/idempotence audit by batch kind.
SELECT batch_kind,
       SUM(input_events) AS input_events,
       SUM(accepted_new_events) AS accepted_new_events,
       SUM(duplicate_events) AS duplicate_events,
       SUM(rejected_budget_events) AS rejected_budget_events
FROM batch_manifest
GROUP BY batch_kind;

-- Q2. Device privacy-budget dashboard.
SELECT COUNT(*) AS devices,
       MIN(remaining_budget) AS min_remaining_budget,
       AVG(remaining_budget) AS avg_remaining_budget,
       SUM(CASE WHEN remaining_budget < 5.0 THEN 1 ELSE 0 END) AS near_cap_devices
FROM device_budget_state;

-- Q3. Budget-rejected event drilldown.
SELECT device_id,
       COUNT(*) AS rejected,
       MIN(ts) AS first_rejected_ts,
       MAX(projected_epsilon_sum) AS max_projected
FROM rejected_events
GROUP BY device_id
ORDER BY rejected DESC
LIMIT 20;

-- Q4. Late-correction lineage.
SELECT r.report_id,
       r.device_id,
       r.ts,
       r.report_bucket,
       c.operator,
       c.posterior,
       p.source_key,
       p.batch_id
FROM reports r
JOIN cleaning_records c USING(report_id)
JOIN provenance_edges p ON p.output_report_id = r.report_id
WHERE r.correction_version > 0
  AND r.late_event = 1
ORDER BY c.posterior DESC
LIMIT 50;

-- Q5. Candidate stuck buckets from materialized counts and mechanism weights.
SELECT epsilon, stuck_bucket, llr
FROM candidate_likelihoods
ORDER BY llr DESC
LIMIT 20;

-- Q6. Per-device privacy trace window.
SELECT device_id,
       MIN(remaining_budget) AS min_remaining_budget,
       MAX(cumulative_epsilon) AS max_cumulative_epsilon,
       COUNT(*) AS reports
FROM privacy_budget_trace
GROUP BY device_id
ORDER BY min_remaining_budget
LIMIT 20;

-- Q7. Event-window/uncertainty join.
SELECT e.device_id,
       e.hour_id,
       e.flagged_reports,
       e.avg_posterior,
       a.cleaned_mean,
       a.uncertainty_proxy
FROM cleaned_event_windows e
JOIN repair_uncertainty_analytics a USING(device_id, hour_id)
WHERE e.flagged_reports >= 3
ORDER BY e.flagged_reports DESC, e.avg_posterior DESC
LIMIT 100;

-- Q8. End-to-end budget/cleaning/analytics dashboard.
SELECT CASE WHEN d.remaining_budget < 5.0 THEN 'near_cap'
            WHEN d.remaining_budget < 25.0 THEN 'low'
            ELSE 'ok' END AS budget_band,
       COUNT(DISTINCT a.device_id) AS devices,
       COUNT(*) AS hourly_groups,
       SUM(COALESCE(e.flagged_reports, 0)) AS flagged_reports,
       AVG(a.cleaned_mean) AS avg_cleaned_mean,
       AVG(a.uncertainty_proxy) AS avg_uncertainty
FROM repair_uncertainty_analytics a
JOIN device_budget_state d USING(device_id)
LEFT JOIN cleaned_event_windows e USING(device_id, hour_id)
GROUP BY budget_band;
