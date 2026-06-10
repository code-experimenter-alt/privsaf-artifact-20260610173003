
create table reports(
    report_id bigint primary key,
    device_id integer not null,
    ts integer not null,
    report_bucket integer not null,
    epsilon double not null,
    mechanism_id varchar not null,
    calibration_version varchar not null
);

create table mechanism_weights(
    mechanism_id varchar not null,
    epsilon double not null,
    report_bucket integer not null,
    stuck_bucket integer not null,
    logp_normal double not null,
    logp_fault double not null
);

create table router_policy(
    operator varchar primary key,
    semantic_layer varchar not null,
    output_relation varchar not null,
    privacy_effect varchar not null
);


create table mv_report_counts as
select mechanism_id, epsilon, report_bucket, count(*) as n
from reports
group by mechanism_id, epsilon, report_bucket;

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

create table cleaning_records as
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
    case when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.65 then report_bucket else null end as stuck_bucket,
    0.20 as fault_ratio,
    case when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.80 then 'repair'
         when ((report_bucket * 17 + device_id) % 100) / 100.0 >= 0.65 then 'flag'
         else 'pass' end as action,
    (2.0 * report_bucket / 31.0) - 1.0 as repair_value,
    'router_v1' as provenance_version
from reports;

create table provenance_edges as
select report_id as output_report_id, 'reports' as source_table, report_id as source_key,
       operator as transform, 'pm32' as mechanism_id, 'cal_v1' as calibration_version
from cleaning_records;

create table privacy_ledger as
select device_id, count(*) as reports, sum(epsilon) as epsilon_sum,
       1000.0 as device_budget, 1000.0 - sum(epsilon) as remaining_budget
from reports
group by device_id;

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

create table cleaned_event_windows as
select
    c.device_id,
    floor(c.ts / 60) as hour_id,
    min(c.ts) as start_ts,
    max(c.ts) as end_ts,
    count(*) as flagged_reports,
    avg(c.posterior) as avg_posterior,
    string_agg(distinct p.semantic_layer, ',') as routed_semantics
from cleaning_records c
join router_policy p on p.operator = c.operator
where c.action in ('flag', 'repair')
group by c.device_id, floor(c.ts / 60)
having count(*) >= 2;

create table repair_uncertainty_analytics as
select
    r.device_id,
    floor(r.ts / 60) as hour_id,
    count(*) as reports,
    avg(case when c.action = 'repair'
             then c.repair_value
             else (2.0 * r.report_bucket / 31.0) - 1.0 end) as cleaned_mean,
    avg((2.0 * r.report_bucket / 31.0) - 1.0) as unrepaired_mean,
    avg(c.posterior * (1.0 - c.posterior)) as uncertainty_proxy,
    sum(case when c.action = 'repair' then 1 else 0 end) as repair_count
from reports r
join cleaning_records c using(report_id)
group by r.device_id, floor(r.ts / 60);
