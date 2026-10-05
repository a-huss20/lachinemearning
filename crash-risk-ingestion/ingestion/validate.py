# Student Name: Aftab Hussaini
# Student FAN:  [YourFAN]
# File:         validate.py
# Date:         01-10-2026
# Description:  Schema and business-rule validation; invalid rows are quarantined, not silently dropped.
# Licence:      MIT Licence
"""Validation stage.

Two levels of checking:

* Structural (hard fail): a required column is missing. This signals
  upstream schema drift, and continuing would silently produce wrong
  features, so the run stops.
* Row level (quarantine): a row breaks a domain rule (e.g. severity counts
  that do not add up, coordinates outside South Australia). The row is
  written to ``data/rejected/<source>.csv`` with the failed check name and
  removed from the clean set, so nothing disappears without a trace.

Schemas are declared with pandera so the rules are readable and double as
documentation of each dataset's contract.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

from . import config

log = logging.getLogger(__name__)

_BBOX = config.SA_BBOX
_LON = pa.Column(float, pa.Check.in_range(_BBOX["min_lon"], _BBOX["max_lon"]), nullable=False)
_LAT = pa.Column(float, pa.Check.in_range(_BBOX["min_lat"], _BBOX["max_lat"]), nullable=False)
_COUNT = pa.Column(int, pa.Check.ge(0), coerce=True)

CRASH_SEVERITY = ["cse_pdo", "cse_inj", "cse_si", "cse_fat"]
CRASH_TYPES = [
    "cty_rear_end", "cty_hit_fixed_object", "cty_side_swipe", "cty_right_angle",
    "cty_head_on", "cty_hit_pedestrian", "cty_roll_over", "cty_right_turn",
    "cty_hit_parked_vehile", "cty_hit_animal", "cty_hit_object_on_road",
    "cty_left_road_oc", "cty_other", "cty_unknown",
]
CRASH_OTHER = [
    "total_crashes", "total_casualties", "total_fatalities",
    "total_serious_injuries", "uty_bicycle", "uty_pedestrian", "lco_night",
]

SCHEMAS: dict[str, pa.DataFrameSchema] = {
    "crashes": pa.DataFrameSchema(
        {
            "unique_loc": pa.Column(str, pa.Check.str_matches(r"^\d{14}$"), unique=True),
            **{c: _COUNT for c in CRASH_SEVERITY + CRASH_TYPES + CRASH_OTHER},
            "_lon": _LON,
            "_lat": _LAT,
            "_geom_type": pa.Column(str, pa.Check.isin(["Point"])),
        },
        checks=[
            # Severity categories are mutually exclusive: PDO + injury + fatal
            # = total (serious injury is a subset of injury in this file).
            pa.Check(
                lambda d: d.cse_pdo + d.cse_inj + d.cse_fat == d.total_crashes,
                name="severity_partition",
            ),
            pa.Check(lambda d: d.cse_si <= d.cse_inj, name="serious_subset_of_injury"),
            pa.Check(lambda d: d[CRASH_TYPES].sum(axis=1) == d.total_crashes, name="crash_type_partition"),
            pa.Check(lambda d: d.total_fatalities >= d.cse_fat, name="fatalities_ge_fatal_crashes"),
            pa.Check(lambda d: d.total_serious_injuries >= d.cse_si, name="si_persons_ge_si_crashes"),
            pa.Check(lambda d: d.total_crashes >= 1, name="at_least_one_crash"),
        ],
        strict=False,
    ),
    "volumes": pa.DataFrameSchema(
        {
            "TESECN_ID": pa.Column(int, unique=True, coerce=True),
            "ROAD_NO": pa.Column(str),
            "RLCWY_CODE": pa.Column(str, nullable=True),
            "TESECN_VOLUME": pa.Column(float, pa.Check.gt(0), coerce=True),
            "TESECN_BASE_YEAR": pa.Column(
                int, pa.Check.in_range(1990, config.VOLUME_REFERENCE_YEAR), coerce=True
            ),
            "CV_PERCENT": pa.Column(float, pa.Check.in_range(0, 100), nullable=True, coerce=True),
            "_lon": _LON,
            "_lat": _LAT,
            "_geom_type": pa.Column(str, pa.Check.isin(["LineString", "MultiLineString"])),
        },
        strict=False,
    ),
    # The three point layers share one structure; duplicates of the site id
    # are legitimate (one site can have several mapped points) and are
    # resolved later in transform, so uniqueness is not enforced here.
    "points": pa.DataFrameSchema(
        {
            "site_type": pa.Column(str, nullable=False),
            "site_id": pa.Column(str, nullable=False),
            "_lon": _LON,
            "_lat": _LAT,
            "_geom_type": pa.Column(str, pa.Check.isin(["Point"])),
        },
        strict=False,
    ),
}

REQUIRED = {
    "crashes": ["unique_loc", *CRASH_SEVERITY, *CRASH_TYPES, *CRASH_OTHER],
    "volumes": ["TESECN_ID", "ROAD_NO", "TESECN_VOLUME", "TESECN_BASE_YEAR", "CV_PERCENT"],
    "points": ["site_type", "site_id"],
}


class SchemaDriftError(RuntimeError):
    """Raised when an upstream file no longer has a required column."""


def _with_geometry_columns(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Adds lon/lat/type helper columns so geometry can be validated in pandera."""
    df = pd.DataFrame(gdf.drop(columns="geometry"))
    geom = gdf.geometry
    valid = geom.notna() & ~geom.is_empty
    df["_lon"] = float("nan")
    df["_lat"] = float("nan")
    if valid.any():
        # Representative point always lies on the geometry (unlike a line centroid).
        pts = geom[valid].representative_point().to_crs("EPSG:4326")
        df.loc[valid, "_lon"] = pts.x
        df.loc[valid, "_lat"] = pts.y
    df["_geom_type"] = geom.geom_type.fillna("None")
    return df


def validate(gdf: gpd.GeoDataFrame, schema_key: str, name: str) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Validates a dataset and splits it into clean and rejected rows.

    Args:
        gdf: Raw dataset with a CRS set.
        schema_key: Which schema in SCHEMAS to apply.
        name: Dataset name for logging and the rejected-rows file.

    Returns:
        (clean GeoDataFrame, DataFrame of rejected rows with reasons)

    Raises:
        SchemaDriftError: If a required column is missing.
    """
    missing = [c for c in REQUIRED[schema_key] if c not in gdf.columns]
    if missing:
        raise SchemaDriftError(f"{name}: required columns missing upstream: {missing}")

    df = _with_geometry_columns(gdf)
    try:
        SCHEMAS[schema_key].validate(df, lazy=True)  # lazy: collect every failure
        failures = pd.DataFrame(columns=["index", "check"])
    except SchemaErrors as err:
        fc = err.failure_cases
        # Row-level failures carry an index; DataFrame-wide checks evaluated
        # element-wise also report the failing index.
        failures = fc.loc[fc["index"].notna(), ["index", "check"]].copy()
        failures["check"] = failures["check"].astype(str)
        unlocated = fc[fc["index"].isna()]
        if not unlocated.empty:
            log.warning("%s: %d non-row failures: %s", name, len(unlocated),
                        unlocated["check"].astype(str).unique().tolist())

    if failures.empty:
        log.info("%s: %d rows, all valid", name, len(gdf))
        return gdf, pd.DataFrame()

    reasons = failures.groupby("index")["check"].agg(lambda s: "; ".join(sorted(set(s))))
    bad_idx = reasons.index.intersection(gdf.index)
    rejected = df.loc[bad_idx].assign(reject_reason=reasons.loc[bad_idx])
    clean = gdf.drop(index=bad_idx)

    config.REJECTED_DIR.mkdir(parents=True, exist_ok=True)
    rejected.to_csv(config.REJECTED_DIR / f"{name}.csv", index_label="row")
    log.warning("%s: %d of %d rows quarantined (%s)", name, len(rejected), len(gdf),
                rejected["reject_reason"].value_counts().to_dict())
    return clean, rejected
