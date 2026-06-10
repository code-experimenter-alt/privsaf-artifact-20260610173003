# Data Access Notes

## UCI Air Quality

The reproducible local experiment downloads the official UCI Air Quality archive:

- Page: https://archive.ics.uci.edu/dataset/360/air+quality
- DOI: https://doi.org/10.24432/C59K5F
- Archive: https://archive.ics.uci.edu/static/public/360/air+quality.zip
- Files: `AirQualityUCI.csv`, `AirQualityUCI.xlsx`

The dataset has 9,358 instances and 15 features. Missing values are tagged as `-200`.

Local files:

- `data/air_quality/air-quality.zip`
- `data/air_quality/raw/AirQualityUCI.csv`
- `data/air_quality/raw/AirQualityUCI.xlsx`

## CHiME, DESED, and SINS

The manuscript now treats these as optional audio-derived univariate stream workloads. Full audio downloads may require large storage, registration, or license-specific access. Do not claim reproduced acoustic results unless a local manifest, checksums, cached scalar features, PM-LDP streams, labels, and generated CSV logs are present.

Reference access pages checked during this revision:

- CHiME: https://www.chimechallenge.org/
- DESED: https://project.inria.fr/desed/ and https://github.com/turpaultn/DESED
- SINS: https://dcase-repo.github.io/dcase_datalist/datasets/sounds/sins.html

These audio datasets were not downloaded in this local run. CHiME data requires challenge-specific/licensed source data, and the SINS datalist reports a 563 GB download. The paper therefore lists them as planned large-data extensions rather than reproduced evidence.

## Round-2 Expansion Datasets

The following public scalar-stream datasets were downloaded for the experiment expansion plan:

- UCI Individual Household Electric Power Consumption: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
- UCI Bike Sharing Dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- UCI Beijing Multi-Site Air Quality: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- UCI Gas Sensor Array Drift: https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset
- Numenta Anomaly Benchmark subset: https://github.com/numenta/NAB

The exact local files and row counts are recorded in `results/downloaded_dataset_inventory.csv`. These datasets are downloaded and inspected, but only UCI Air Quality has been fully run through PrivSAF at this stage. The other datasets are staged for the expanded experiment grid and are marked as plans/expected trends until their run logs exist.

## Real-Stuck Source Audit

The closest real scalar stuck-label source found in the latest audit is the I-ORS sea level height QC dataset:

- Article: https://doi.org/10.5194/os-21-2085-2025
- Dataset DOI: https://doi.org/10.22808/DATA-2024-8
- Local article cache: `data/iors_slh_article.pdf`
- Local supplement cache: `data/iors_slh_supplement.pdf`
- Local NetCDF file: `data/iors_slh/I-ORS_2003_2022_D_SLH.nc`
- Local result CSVs: `results/iors_stuck_qc_pmldp_runs.csv`, `results/iors_stuck_qc_pmldp_summary.csv`, `results/iors_stuck_qc_inventory.csv`, and `results/iors_stuck_qc_source_metadata.csv`

The article reports MIROS SM-140 rangefinder observations from 2003 to 2022 at 10 min intervals, QC labels including `stuck`, and a maintenance context with relocations and instrument replacement. The NetCDF file was retrieved through the repository bitstream discovered from OAI-PMH metadata; its MD5 is `a8601d3cf013742412ee523d172b2ab8`, matching the KIOST metadata. The file contains `SLH` and `SLH_QC`, with flag `5` meaning `stuck`; the local PM-LDP panel uses good rows (`SLH_QC=1`) for previous-year calibration and persistent stuck episodes (`SLH_QC=5`, length at least 8 reports) as positives.

The manuscript now cites MiFaD as the closest public real-stuck sensor-fault source found during revision:

- Zenodo record: https://zenodo.org/records/17641389
- DOI: https://doi.org/10.5281/zenodo.17641389
- Code repository: https://gitlab.com/etrovub/embedded-systems/publications/lightweight-AI-for-sensor-fault-monitoring

MiFaD metadata lists real induced `stuck`, `clipping`, `spike`, and `normal` MEMS microphone fault classes. The code repository README also documents pre-merged CSV names, but the GitLab archive checked on 2026-05-26 did not include those CSVs; the Zenodo data file is a 17.16 GB `datasets.7z` archive. It was therefore audited as an external real-stuck source and cited in the paper, not incorporated into the scalar PM-LDP result tables.

The broader source audit is saved in `results/real_fault_source_audit.csv`. It also records the Scientific Data AHU fault datasets:

- Article: https://doi.org/10.1038/s41597-025-06179-y
- figshare dataset: https://doi.org/10.6084/m9.figshare.29297999

The AHU source provides labeled real-building, hardware-in-the-loop, and simulated air-handling-unit fault scenarios in CSV form, with stuck/leaking dampers and valves plus sensor-bias faults. It is a stronger semi-real system-level fault-label source than mined flatlines, but it is not a public scalar PM-LDP stuck-at sensor-maintenance log, so it remains an audited source rather than a main result table in this artifact.

The audit also records a Scientific Reports automotive HIL/continuous-integration dataset:

- Article: https://doi.org/10.1038/s41598-025-21416-5

That dataset explicitly includes APP/RPM sensor-related stuck-at labels together with gain, noise, drift, and delay faults. Its data availability statement restricts access to request, so it is useful evidence that labeled sensor stuck-at corpora exist, but it cannot be a fully reproducible local PM-LDP benchmark in this repository without obtaining the restricted files.

The repository now also includes a downloaded public prepared WSN stuck-at benchmark:

- Dataset repository: https://github.com/tmoulahi/Dataset-for-WSN-fault-detection
- Local archive: `data/wsn_fault_detection/raw/Stuck-at.rar`
- Extracted files: `data/wsn_fault_detection/raw/Stuck-at/*.xlsx`
- Local result CSVs: `results/wsn_stuck_labeled_pmldp_runs.csv`, `results/wsn_stuck_labeled_pmldp_summary.csv`, and `results/wsn_stuck_labeled_dataset_inventory.csv`

This WSN benchmark is based on real TelosB temperature/humidity measurements, with prepared Offset/Gain/Stuck-at/Out-of-bounds sensor faults. The local PM-LDP stress panel uses the five public Stuck-at Excel files and their binary labels. It is stronger than mined weak flatlines because labels are supplied by the dataset, but the labels are row-level prepared injections over 12-dimensional vectors rather than public maintenance-log scalar stuck-at labels.
