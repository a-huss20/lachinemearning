# Student Name: Aftab Hussaini
# Student FAN:  [YourFAN]
# File:         contract.py
# Date:         01-10-2026
# Description:  Writes outputs and the machine-readable data contract consumed by the team's ML/reasoning modules.
# Licence:      MIT Licence
"""Load stage and data contract.

The data contract is the agreed interface between this ingestion module
and the downstream ML and symbolic-reasoning modules. It is generated from
the actual output (so it can never drift from the data) and lists, for every
table: grain, primary key, row count, CRS, and each column's type, null
count, unit and meaning. Downstream code should validate against it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import geopandas as gpd
import pandas as pd

from . import config

log = logging.getLogger(__name__)

CONTRACT_VERSION = "1.0.0"

TABLE_META = {
    "crash_locations": {
        "grain": "One row per geocoded crash location; counts aggregate all crashes at that point over the period.",
        "primary_key": "unique_loc",
        "intended_use": "Severity classification (target: max_severity or any_fsi).",
    },
    "intersection_sites": {
        "grain": "One row per signalised intersection (site_type TS), including sites with zero crashes.",
        "primary_key": "site_key",
        "intended_use": "Crash frequency / rate modelling with exposure offset million_entering_vehicles.",
    },
    "segment_sites": {
        "grain": "One row per state-road traffic-volume segment, including segments with zero crashes.",
        "primary_key": "segment_id",
        "intended_use": "Crash frequency / rate modelling with exposure offset vkt_100m.",
    },
}

DESCRIPTIONS = {
    "unique_loc": ("id", "Crash location identifier from DIT."),
    "total_crashes": ("count", "All reported crashes at the location/site, 2020-2024."),
    "cse_pdo": ("count", "Property-damage-only crashes."),
    "cse_inj": ("count", "Injury crashes (includes serious injury)."),
    "cse_minor_inj": ("count", "Minor injury crashes (cse_inj - cse_si)."),
    "cse_si": ("count", "Serious injury crashes."),
    "cse_fat": ("count", "Fatal crashes."),
    "fsi_crashes": ("count", "Fatal + serious injury crashes (primary safety outcome)."),
    "any_fsi": ("0/1", "1 if at least one fatal or serious injury crash."),
    "max_severity": ("ordinal 0-3", "Worst outcome: 0 PDO, 1 minor, 2 serious, 3 fatal."),
    "night_share": ("ratio", "Share of crashes in night lighting conditions."),
    "vru_involved": ("0/1", "Any pedestrian or cyclist involved."),
    "site_type": ("category", "Assigned site: intersection, state_road_segment or other_road."),
    "intersection_site_key": ("id", "Matched signalised intersection (within 20 m)."),
    "segment_id": ("id", "DIT traffic estimate section id."),
    "road_aadt": ("vehicles/day", "AADT of nearest state-road segment within 25 m (NaN = local road)."),
    "on_state_road": ("0/1", "Location lies within 25 m of a state road with a volume estimate."),
    "aadt": ("vehicles/day", "Annual average daily traffic estimate."),
    "entering_aadt": ("vehicles/day", "Sum over roads of max segment AADT within 50 m (entering-traffic proxy)."),
    "million_entering_vehicles": ("MEV", "entering_aadt x 365 x 5 / 1e6 - exposure for intersections."),
    "crash_rate_per_mev": ("crashes/MEV", "total_crashes / million_entering_vehicles."),
    "vkt_100m": ("1e8 veh-km", "aadt x 365 x 5 x length_km / 1e8 - exposure for segments."),
    "crash_rate_per_100m_vkt": ("crashes/1e8 VKT", "total_crashes / vkt_100m."),
    "cv_percent": ("%", "Commercial (heavy) vehicle share of traffic."),
    "volume_age_years": ("years", "2024 minus the volume estimate's base count year."),
    "volume_stale": ("0/1", "Volume estimate based on a count more than 5 years old."),
    "dist_cbd_km": ("km", "Straight-line distance to the Adelaide GPO (remoteness proxy)."),
    "ped_crossings_200m": ("count", "Pedestrian crossing sites within 200 m."),
    "school_crossings_300m": ("count", "School crossing sites within 300 m."),
    "dist_school_crossing_m": ("m", "Distance to the nearest school crossing."),
    "length_m": ("m", "Projected segment length (GDA2020 Lambert)."),
}


def _to_output(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.to_crs(config.OUTPUT_CRS)


def write_outputs(tables: dict[str, gpd.GeoDataFrame]) -> dict:
    """Writes GeoParquet + CSV for each table and the data contract.

    GeoParquet keeps geometry and dtypes for the team's Python modules;
    the CSV (with lon/lat of a representative point) is for quick
    inspection and non-spatial tools.

    Returns:
        The contract dictionary.
    """
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    contract = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "crash_period": config.CRASH_PERIOD,
        "crs": config.OUTPUT_CRS,
        "tables": {},
    }
    for name, gdf in tables.items():
        out = _to_output(gdf)
        out.to_parquet(config.PROCESSED_DIR / f"{name}.parquet", index=False)

        pts = out.geometry.representative_point()
        flat = pd.DataFrame(out.drop(columns="geometry")).assign(lon=pts.x.round(6), lat=pts.y.round(6))
        flat.to_csv(config.PROCESSED_DIR / f"{name}.csv", index=False)

        cols = []
        for col in flat.columns:
            unit, desc = DESCRIPTIONS.get(col, ("", ""))
            cols.append({"name": col, "dtype": str(flat[col].dtype),
                         "nulls": int(flat[col].isna().sum()), "unit": unit, "description": desc})
        contract["tables"][name] = {**TABLE_META[name], "rows": len(out), "columns": cols}
        log.info("wrote %s: %d rows, %d columns", name, len(out), len(flat.columns))

    path = config.PROCESSED_DIR / "data_contract.json"
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract
