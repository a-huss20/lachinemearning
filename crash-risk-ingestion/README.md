<!--
Student Name: Aftab Hussaini
Student FAN:  [YourFAN]
File:         README.md
Date:         01-10-2026
Description:  Run instructions, design notes and known limitations for the crash-risk data ingestion module.
-->

# Crash Risk Data Ingestion Module (COMP3742 Artefact 1)

This is the automated data ingestion subsystem of a hybrid (machine learning + symbolic reasoning) system that predicts crash risk and severity at South Australian road sites. It fetches five open datasets from data.sa.gov.au. It then validates and cleans them, fuses them spatially, and publishes three model-ready tables plus a data contract for the team's ML and reasoning modules.

Repository: https://github.com/a-huss20/lachinemearning

## Requirements

- Python **3.11 or newer** (pandas 3 requires it)
- `pip install -r requirements.txt` (use a virtual environment)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Running

| Command | What it does |
|---|---|
| `python data_ingestion.py` | **Online (default).** Resolves the current file URLs through the CKAN API, downloads only what changed (ETag / Last-Modified), then processes everything. |
| `python data_ingestion.py --offline --local-dir "C:\path\to\Data"` | Uses archives you downloaded manually (any folder layout; matched by file name). No network access. |
| `python data_ingestion.py --offline` | Re-processes the last cached download. |
| `python eda.py` | Creates the report figures and `eda_summary.json` from the processed tables. |
| `python -m pytest tests` | Runs 12 unit tests (security, validation, transform, mocked download). |

If a download fails in online mode, the pipeline falls back to `--local-dir` when one is given, and otherwise to the last cached copy. It logs which one it used. The process exits with code 1 on failure, so a scheduler can detect it.

**Scheduling:** the source data is republished at most yearly, so a weekly Windows Task Scheduler job (or a cron job) running `python data_ingestion.py` is enough. A conditional GET makes unchanged weeks almost free.

## Outputs (`data/`)

```
data/
├── raw/            cached .zip archives, manifest.json (URL, SHA-256, ETag, time), extracted GeoJSON
├── processed/      crash_locations / intersection_sites / segment_sites  (.parquet + .csv)
│                   data_contract.json, run_report.json
├── rejected/       rows that failed validation, with the failed check name
├── eda/            figures + eda_summary.json
└── logs/           one log file per run
```

| Table | Grain | Typical target | Exposure |
|---|---|---|---|
| `crash_locations` | one geocoded crash location (43,522) | `max_severity`, `any_fsi` | - |
| `intersection_sites` | one signalised intersection (677), **including zero-crash sites** | `total_crashes`, `fsi_crashes` | `million_entering_vehicles` |
| `segment_sites` | one state-road volume segment (2,685), **including zero-crash segments** | `total_crashes`, `fsi_crashes` | `vkt_100m` |

Column meanings, units and null counts are listed in `data/processed/data_contract.json`.

## Pipeline stages

1. **Fetch** (`ingestion/fetch.py`): HTTPS only, host allow-list (checked again after redirects), timeouts, a 150 MB size cap, a check that the payload really is a zip, an atomic write via a temporary file, and a SHA-256 manifest. The cached archive's hash is re-checked on every run to detect tampering. Extraction rejects path traversal (zip-slip), empty files and zip-bomb ratios.
2. **Validate** (`ingestion/validate.py`): pandera schemas. A missing required column is a hard failure (upstream schema drift). Rows that break a rule are quarantined to `data/rejected/`. The rules check that severity counts partition the total, that serious injuries are a subset of injuries, that crash types sum to the total, that coordinates fall inside South Australia, and that keys are unique.
3. **Transform** (`ingestion/transform.py`): declares each file's CRS (the GeoJSON files carry none) and reprojects to GDA2020 Lambert (EPSG:7845), a metre-based projection valid across the whole state. It merges multi-point infrastructure sites into one point each and builds the site universe from infrastructure, so zero-crash sites exist. It then assigns crashes to sites by nearest-neighbour joins: a signalised intersection within 20 m first, otherwise a state-road segment within 25 m. Finally it derives exposure, rates and context features.
4. **Load** (`ingestion/contract.py`): writes GeoParquet (published in GDA2020, EPSG:7844), CSV, and a data contract generated from the actual outputs.

## Known bugs, limitations and incomplete features

- **The online download was tested only with mocked HTTP responses.** The development environment could not reach data.sa.gov.au, so the end-to-end runs used `--offline --local-dir` on archives downloaded from the portal. The first online run should be checked against its log.
- **The crash data is pre-aggregated.** DIT publishes one row per location with counts for 2020–2024, with no crash date, speed limit, weather or road surface. As a result:
  - severity can only be modelled per location (worst outcome or counts), not per crash;
  - a temporal train/test split inside this file is impossible. The older five-year releases (e.g. 2015–2019) on the same CKAN page are the intended source for a before/after split, but this module does not ingest them yet.
- Only **state-maintained roads** have traffic volume estimates. Crash locations on local roads (about 40% of locations) are kept as `other_road`, with no exposure.
- **Divided roads appear as a single line** in the volume layer (carriageway code `U` or `L`; there is no right-carriageway line). Crashes on the far carriageway are still captured by the 25 m radius, but the AADT is assumed to be the two-way total.
- Some volume estimates are projections from old counts (`volume_age_years` reaches 24 years; 10.8% of segments are more than 5 years old). These are flagged with `volume_stale`.
- **Segment crash rates are unstable on low-exposure segments.** The standard deviation of the rate is about 12 times higher in the lowest exposure decile than in the highest. Downstream models should use counts with an exposure offset, or Empirical Bayes, rather than raw rates.
- The 20 m and 25 m matching radii are global constants. Large intersections, such as 3-arm junctions with slip lanes, may lose some crashes to the segment class.
- Signal sites whose mapped points spread over more than 50 m are flagged (`spread_flag`) but kept. For six sites the id may have been reused.
- Site types other than `TS` in the signals layer (`RS`, `ES`, `PP`) are used only as context features, because their meaning is not documented in the metadata.

## Data sources and licence

All data comes from the Government of South Australia, Department for Infrastructure and Transport, via data.sa.gov.au, under CC BY 4.0: Road Crash Locations in SA, Traffic Volumes, Traffic Signals, Pedestrian Crossings and School Crossings. This code is released under the MIT Licence.

## For teammates

- The output tables are **not** committed (they are regenerated in about 10 seconds). Run the pipeline, then read `data/processed/*.parquet` (or `.csv`).
- `docs/data_contract.json` is a snapshot of the output schema: the grain, key, units and null counts of every column. Validate your module's inputs against it.
- `docs/figures/` holds the exploratory analysis figures produced by `eda.py`.
