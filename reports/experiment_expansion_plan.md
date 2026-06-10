# PrivSAF Experiment Expansion Plan

This report records the experiment-focused review, Poe GPT-5.5 Pro query history, concrete public datasets, five-dataset evaluation outputs, and generated visualizations.

## Current Experimental Position

The current paper now has three reproducible empirical layers: a detailed UCI Air Quality anchor, a five-dataset evaluation covering Air Quality, Household Power, Bike Sharing, Beijing Air, and NAB machine temperature, and targeted ICDE acceptability extensions for channel advantage, direct channel-aware baselines, native weak labels, dropout, a downloaded I-ORS real stuck-QC PM-LDP panel, and an external WSN labeled-stuck PM-LDP stress panel. The manuscript also records a source audit for I-ORS real scalar stuck QC labels, MiFaD real induced stuck labels, AHU semi-real fault labels, restricted automotive HIL sensor stuck-at labels, and public prepared WSN stuck-at labels.

The experiment section is organized around clear empirical claims:

- report verified results only from generated CSV logs;
- state that epsilon is per scalar report/frame;
- present the paper as channel-aware private stuck-at/flatline cleaning for scalar LDP telemetry;
- present PrivSAF mixture/HMM as posterior operators for iid and persistent scalar stuck-at faults, and present the dropout HMM as the availability operator;
- use I-ORS and WSN real or hardware-adjacent labels as router evidence for label semantics around scalar stuck-at/flatline cleaning;
- keep the manuscript's router map synchronized with generated CSV summaries;
- use the channel-advantage diagnostic to show why privatized-domain flatline/bucket tests can fail under PM-LDP;
- label native-source flatline evaluation as weak-label evidence, not audited deployment ground truth;
- report repair as a posterior-mask post-processing operator with oracle-mask repair controls;
- keep audio-derived CHiME/DESED/SINS as optional future inputs unless full feature extraction and logs are added.

## Poe GPT-5.5 Pro Round-5 Query

The requested `$poe-balance-gpt55-pro` call checked the three configured balances and selected the highest-balance key. The single GPT-5.5-Pro request timed out while reading the response, so no second paid model call was made under the skill's one-call rule. The local revision below therefore implements the user's explicit ICDE requirements directly from the repository state.

## Poe GPT-5.5 Pro Round-2 Query

The skill selected the highest-balance Poe key and made exactly one model call.

Output file:

- `reviews/poe_gpt55_round2_experiment_plan_response.txt`

Core Poe recommendations:

- add multiple public scalar stream datasets;
- add real-label and mechanism checks around scalar stuck-at/flatline cleaning;
- add local baselines that consume the same PM-LDP reports;
- add blind budget-estimation diagnostics;
- add an unlabeled HMM-vs-mixture model-selection rule;
- generate every figure from CSV inputs.

## Downloaded Datasets

Concrete source and local status are recorded in:

- `results/dataset_manifest_plan.csv`
- `results/downloaded_dataset_inventory.csv`

Downloaded or already present datasets:

1. UCI Air Quality
   - Source: https://archive.ics.uci.edu/dataset/360/air+quality
   - Status: already run
   - Primary target: `C6H6(GT)`

2. UCI Individual Household Electric Power Consumption
   - Source: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
   - Local file: `data/household_power/raw/household_power_consumption.txt`
   - Rows: 2,075,259
   - Targets: `Global_active_power`, `Voltage`

3. UCI Bike Sharing Dataset
   - Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
   - Local file: `data/bike_sharing/raw/hour.csv`
   - Rows: 17,379
   - Targets: `cnt`, `temp`, `hum`

4. UCI Beijing Multi-Site Air Quality
   - Source: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
   - Local files: 12 station CSVs under `data/beijing_air/raw/PRSA2017_Data_20130301-20170228/`
   - Rows: 35,064 per station, 420,768 total station-hours
   - Targets: `PM2.5`, `TEMP`, `WSPM`

5. Numenta Anomaly Benchmark subset
   - Source: https://github.com/numenta/NAB
   - Local files:
     - `machine_temperature_system_failure.csv`, 22,695 rows
     - `ambient_temperature_system_failure.csv`, 7,267 rows
     - `occupancy_6005.csv`, 2,380 rows

6. UCI Gas Sensor Array Drift
   - Source: https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset
   - Local files: `data/gas_drift/raw/Dataset/batch1.dat` through `batch10.dat`
   - Role: optional dataset inventory item

## Execution Steps

Current completed steps:

```powershell
cd C:\Users\Public\ICDE_PrivSAF_project
py scripts\download_expansion_datasets.py
py scripts\build_experiment_visualizations.py
```

Existing core experiment:

```powershell
py scripts\download_air_quality.py
py scripts\run_air_quality.py
```

Five-dataset evaluation:

```powershell
py scripts\run_icde_revision_grid.py
```

ICDE acceptability extension:

```powershell
py scripts\run_icde_acceptability_extensions.py
```

The current ICDE revision evaluation writes 3,420 detection rows, 1,620 repair rows, 360 diagnostic-output rows, 19 access-regime rows, 72 sensitivity rows, 420 channel-advantage rows, 1,836 native weak-label rows, 540 injected-dropout rows, and 20 theory-diagnostic rows. Post-audit extensions add 336 low-epsilon I-ORS stuck-QC rows, 288 real I-ORS dropout-QC rows, 360 PM/Duchi-binary mechanism-fault rows, 32 low-epsilon privacy-utility summary rows, and a 24-row device-level privacy ledger.

## Experiment Design

Must-run datasets:

- Air Quality: verified anchor and continuity with current results.
- Household Power: dense energy telemetry; strong for dropout and repair.
- Bike Sharing: seasonal non-Gaussian count stream.
- Beijing Multi-Site Air Quality: multi-station environmental telemetry.
- NAB subset: benchmark scalar time series outside chemistry/air-quality domain.

Optional stress dataset:

- Gas Sensor Array Drift: optional sensor dataset inventory item.

Fault modes:

- iid stuck-at;
- segment stuck-at;
- dropout bursts as the measured availability side case;
- real-label and mechanism-extension rows remain optional additions around the stuck-at/flatline target.

Baseline suite:

- Raw-domain upper bounds: `raw_nonprivate_hmm`, `raw_hampel`, `raw_cusum`, `raw_bocpd`, `raw_rolling_median`.
- Privatized-domain baselines: `privatized_zscore`, `privatized_rolling_median`.
- Privacy-aware baselines: `ldp_distribution_surprise`, `privsaf_mixture`, `privsaf_hmm`.
- Repair methods: `no_repair`, `linear_interpolation`, `locf`, `rolling_median`, `privsaf_repair`.

Budget-estimation experiment:

- compute PM support constraints from privatized reports;
- fit candidate epsilon values by heldout validation likelihood;
- bootstrap over windows/users;
- report `epsilon_true`, `epsilon_hat`, CI, stability flag, and downstream metric change.

Model-selection experiment:

- run mixture and HMM on privatized train reports;
- evaluate heldout privatized validation negative log likelihood;
- select HMM only if validation/BIC improvement clears a one-standard-error threshold;
- show the mixture/HMM switch point as segment length grows from 1 to 64.

## Generated Visualizations

All generated figures are in `results/`.

- `fig_icde_revision_baselines.png`: measured baseline-family comparison from `icde_revision_detection_summary.csv`.
- `fig_icde_revision_repair.png`: repair comparison from `icde_revision_repair_summary.csv`, filtered to stuck-at faults.
- `fig_icde_channel_advantage.png`: bucket-count score versus known-channel LLR diagnostic.
- `fig_icde_native_weaklabel.png`: native-source weak-label event-level comparison.
- `fig_icde_dropout.png`: dropout extension comparison.
- `fig_dataset_gap_coverage.png`: heatmap mapping datasets to review gaps.
- `fig_dataset_scale.png`: log-scale size comparison of downloaded/planned datasets.
- `fig_downloaded_stream_previews.png`: actual normalized previews from downloaded scalar streams.
- `fig_fault_taxonomy_examples.png`: clean vs injected fault examples.
- `fig_expected_metric_trends.png`: planning diagram for qualitative metric trends.
- `fig_model_selection_expectation.png`: expected HMM-vs-mixture switch point.
- `fig_budget_estimation_plan.png`: planned blind budget-estimation diagnostics.
- `fig_experiment_workplan.png`: work-package roadmap.

## High-Level Results

- The evaluation has complete coverage for Air Quality, Household Power, Bike Sharing, Beijing Air, and NAB machine temperature: each dataset has iid stuck-at and segment stuck-at rows.
- On iid stuck-at faults, PrivSAF mixture is the strongest privacy-aware method with AUROC/AUPRC `0.761/0.493`.
- On segment stuck-at faults, PrivSAF-HMM is the strongest privacy-aware method with AUROC/AUPRC `0.930/0.833`, close to the raw-domain HMM upper bound `0.948/0.923`.
- On stuck-at repair, PrivSAF posterior-mask repair improves no repair from MAE/RMSE `0.1488/0.3867` to `0.1240/0.3233`, and reduces downstream mean error from `0.1400` to `0.0916`.
- Oracle-mask repair controls quantify remaining headroom: linear interpolation reaches MAE/RMSE `0.0432/0.1320` when fault locations are known.
- Local channel statistics are retained as router candidates for deployed QC labels.
- Theory diagnostics connect the empirical results to the separation argument: average normal-fault PM-emission L1 separation increases from `0.867` at epsilon `2` to `1.423` at epsilon `4`.
- Channel-advantage diagnostics show that bucket-count tests are weak in `53.6%` of epsilon `0.5` settings, while the known-channel LLR median AUROC is `0.779`.
- Native-source weak-label evaluation uses 17 original-location weak flatline episodes and 85 matched negatives; PrivSAF-HMM scan reaches AUROC/AUPRC `0.809/0.461`.
- Dropout extension rows show PrivSAF dropout HMM reaches AUROC/AUPRC `0.992/0.970`, improving event recall from `0.899` to `0.973` versus rolling missing fraction.
- Direct channel-aware baseline rows show generic PM-HMM reaches `0.840/0.704` on segment faults and `0.834/0.583` on template faults, below PrivSAF-HMM; PM-column likelihood scan reaches `0.625/0.315` on iid faults.
- The WSN labeled-stuck PM-LDP stress panel adds 180 rows from five public prepared Stuck-at files; PM-window GLR reaches AUROC/AUPRC `0.634/0.463`, PM-column likelihood scan reaches `0.585/0.354`, and PrivSAF-HMM reaches `0.478/0.296` on this row-level multivariate injection benchmark.
- The I-ORS real stuck-QC PM-LDP panel adds 336 rows from eight annual slices of a 2003-2022 deployed sea-level rangefinder record; PM-window GLR reaches AUROC/AUPRC `0.940/0.716`, generic PM-HMM reaches `0.745/0.370`, and global PrivSAF-HMM reaches `0.713/0.372`.
- Low-epsilon I-ORS stuck-QC reruns add 336 rows at epsilon `0.5` and `1.0`; the best AUPRC is `0.389` and `0.516` from LLR-CUSUM, while the best PrivSAF variant reaches `0.212` and `0.332`.
- I-ORS real dropout-QC rows add 288 PM-LDP rows from public `SLH_QC=8` missing-value labels; PrivSAF dropout HMM reaches approximately `1.000/1.000` because the missing symbol is explicitly observed.
- Mechanism extension rows add PM/Duchi-binary channel rows over segment-stuck faults; Duchi-binary segment HMM reaches AUPRC `0.737` at epsilon `1` and `0.939` at epsilon `2`.
- Device-level privacy ledger shows basic-composition pressure: at 10-minute cadence and epsilon `0.5`, device-level cost is `72` per day, so a budget of `10` covers only `0.139` days.
- Real-label router: I-ORS stuck-QC labels favor local PM-window GLR or LLR-CUSUM, native flatline windows favor PrivSAF-HMM scan, WSN prepared stuck-at labels favor PM-window GLR, and I-ORS dropout labels favor the dropout HMM.
- The source audit identifies I-ORS public real scalar stuck QC labels, MiFaD public real induced stuck-label MEMS microphone data, AHU labeled semi-real fault data, restricted automotive HIL sensor stuck-at labels, and WSN public prepared stuck-at labels. MiFaD is not included in the scalar PM-LDP result ledger because the accessible data archive is a large raw-audio corpus.

## Expected Qualitative Results for Optional Extensions

Planning expectations for future extensions:

- HMM should beat mixture for persistent segment faults, especially as segment length increases.
- Mixture should match or beat HMM for iid faults.
- Higher epsilon and larger sample size should improve all methods.
- Stuck values at extreme quantiles should be easier than stuck values near the median.
- Dropout should reward methods that track report counts.
- Optional external datasets should not change the scalar stuck-at/flatline claim.

## Writing Principles

- Lead with the empirical result before caveats.
- Use "evaluation", "comparison", and "stress test" instead of revision-oriented wording.
- Keep result sections focused on operator selection, channel-aware evidence, and generated artifacts.
- Organize results by router role: mixture for iid stuck-at, HMM for segment stuck-at, dropout HMM for explicit availability labels, and GLR/CUSUM for field-QC labels with local regime evidence.
