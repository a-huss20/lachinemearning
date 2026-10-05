# Student Name: Aftab Hussaini
# Student FAN:  [YourFAN]
# File:         data_ingestion.py
# Date:         01-10-2026
# Description:  Entry point: automated fetch -> validate -> transform -> load pipeline for SA crash-risk data.
# Usage:        python data_ingestion.py            (online)
#               python data_ingestion.py --offline --local-dir "C:/Users/me/Desktop/Data"
# Licence:      MIT Licence
"""Data ingestion pipeline for the hybrid crash-risk prediction system.

Stages (each in its own module so it can be tested and replaced alone):

    fetch      ingestion/fetch.py      download, cache, hash, safe unzip
    validate   ingestion/validate.py   schema + domain rules, quarantine
    transform  ingestion/transform.py  clean, reproject, spatial features
    load       ingestion/contract.py   GeoParquet/CSV + data contract

A run report (row counts, rejections, timings, source hashes) is written
to data/processed/run_report.json for every run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ingestion import config
from ingestion.contract import write_outputs
from ingestion.fetch import acquire_all
from ingestion.transform import build_tables, load_layer, normalise_points
from ingestion.validate import validate

POINT_LAYERS = ("signals", "ped_crossings", "school_crossings")


def setup_logging(verbose: bool) -> None:
    """Logs to the console and to a timestamped file in data/logs."""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_DIR / f"run_{stamp}.log", encoding="utf-8"),
        ],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--offline", action="store_true",
                   help="Do not use the network; use the cache or --local-dir.")
    p.add_argument("--local-dir", type=Path, default=None,
                   help="Folder of manually downloaded .zip archives (offline seed / fallback).")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    return p.parse_args(argv)


def run(offline: bool = False, local_dir: Path | None = None) -> dict:
    """Runs the full pipeline and returns the run report."""
    log = logging.getLogger("pipeline")
    report: dict = {"started_at": datetime.now(timezone.utc).isoformat(), "stages": {}}
    t0 = time.perf_counter()

    # 1. Fetch -----------------------------------------------------------------
    t = time.perf_counter()
    paths = acquire_all(offline=offline, local_dir=local_dir)
    report["stages"]["fetch_s"] = round(time.perf_counter() - t, 2)

    # 2. Validate ----------------------------------------------------------------
    t = time.perf_counter()
    layers, report["rows"] = {}, {}
    for key, path in paths.items():
        gdf = load_layer(path, config.SOURCES[key].crs)
        if key in POINT_LAYERS:
            gdf = normalise_points(gdf)
        schema = "points" if key in POINT_LAYERS else key
        clean, rejected = validate(gdf, schema, key)
        layers[key] = clean
        report["rows"][key] = {"raw": len(gdf), "clean": len(clean), "rejected": len(rejected)}
    report["stages"]["validate_s"] = round(time.perf_counter() - t, 2)

    # 3. Transform ---------------------------------------------------------------
    t = time.perf_counter()
    tables = build_tables(layers)
    report["stages"]["transform_s"] = round(time.perf_counter() - t, 2)

    # 4. Load --------------------------------------------------------------------
    t = time.perf_counter()
    write_outputs(tables)
    report["stages"]["load_s"] = round(time.perf_counter() - t, 2)

    report["outputs"] = {name: len(df) for name, df in tables.items()}
    report["crash_site_assignment"] = (
        tables["crash_locations"]["site_type"].value_counts().to_dict())
    report["total_s"] = round(time.perf_counter() - t0, 2)
    manifest = json.loads((config.RAW_DIR / "manifest.json").read_text(encoding="utf-8"))
    report["sources"] = {k: {"sha256": v.get("sha256"), "origin": v.get("origin"), "url": v.get("url")}
                         for k, v in manifest.items()}
    (config.PROCESSED_DIR / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("pipeline finished in %.1f s -> %s", report["total_s"], config.PROCESSED_DIR)
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    try:
        run(offline=args.offline, local_dir=args.local_dir)
    except Exception:  # top-level guard: log full traceback, non-zero exit for schedulers
        logging.getLogger("pipeline").exception("pipeline failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
