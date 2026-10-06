# Student Name: Aftab Hussaini
# Student FAN:  huss0138
# File:         transform.py
# Date:         01-10-2026
# Description:  Cleaning, reprojection, deduplication and spatial feature engineering for crash-risk sites.
# Licence:      MIT Licence
"""Transformation stage.

Turns five independent layers into three model-ready tables:

* ``crash_locations`` - one row per crash location (the classification
  unit), labelled with its worst severity and enriched with road context.
* ``intersection_sites`` - one row per signalised intersection, *including
  sites with zero crashes*, with exposure (entering traffic) and rates.
* ``segment_sites`` - one row per state road traffic-volume segment, again
  including zero-crash segments, with vehicle-kilometres travelled (VKT)
  and crash rates.

Why sites and not just crash points: the published crash file only
contains locations where at least one crash happened. A model trained on
it alone can never learn what a *safe* location looks like. Defining the
site universe from the infrastructure layers restores the zeros.

All distance work is done in a metre-based projected CRS (config.WORK_CRS).
"""

from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from . import config
from .validate import CRASH_OTHER, CRASH_SEVERITY, CRASH_TYPES

log = logging.getLogger(__name__)

SEVERITY_LABELS = {0: "property_damage_only", 1: "minor_injury", 2: "serious_injury", 3: "fatal"}
CRASH_COUNT_COLS = CRASH_SEVERITY + CRASH_TYPES + CRASH_OTHER + ["cse_minor_inj", "fsi_crashes"]


# ---------------------------------------------------------------------------
# Loading and per-layer cleaning
# ---------------------------------------------------------------------------
def load_layer(path, crs: str) -> gpd.GeoDataFrame:
    """Reads a GeoJSON file and declares its CRS (the files carry none)."""
    gdf = gpd.read_file(path)
    return gdf.set_crs(crs, allow_override=True)


def normalise_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Gives the three point layers a common id column name."""
    gdf = gdf.rename(columns={"ts_no": "site_id", "ts_desc": "site_desc", "site_no": "site_id"})
    for col in ("site_type", "site_id"):
        gdf[col] = gdf[col].astype("string").str.strip()
    return gdf


def clean_crashes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Derives severity labels and rates from the per-location counts.

    Severity hierarchy (Austroads convention, worst first): fatal > serious
    injury > minor injury > property damage only. Because each row aggregates
    up to 119 crashes, the location label is its worst observed outcome; the
    counts are kept so modellers can use the full distribution instead.
    """
    gdf = gdf.copy()
    count_cols = CRASH_SEVERITY + CRASH_TYPES + CRASH_OTHER
    gdf[count_cols] = gdf[count_cols].astype("int32")
    gdf["cse_minor_inj"] = gdf["cse_inj"] - gdf["cse_si"]
    gdf["fsi_crashes"] = gdf["cse_fat"] + gdf["cse_si"]  # fatal + serious injury
    gdf["max_severity"] = np.select(
        [gdf.cse_fat > 0, gdf.cse_si > 0, gdf.cse_minor_inj > 0], [3, 2, 1], default=0
    ).astype("int8")
    gdf["max_severity_label"] = gdf["max_severity"].map(SEVERITY_LABELS)
    gdf["any_fsi"] = (gdf["fsi_crashes"] > 0).astype("int8")
    gdf["night_share"] = (gdf["lco_night"] / gdf["total_crashes"]).round(4)
    gdf["vru_involved"] = ((gdf.uty_pedestrian + gdf.uty_bicycle) > 0).astype("int8")
    # cty_unknown is all zero in 2020-2024; kept for schema stability, flagged for modellers.
    return gdf


def clean_volumes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Standardises the traffic-volume segments.

    * Volume estimates have different base years (some are projections from
      counts over a decade old), so the age is exposed as a quality feature.
    * Missing heavy-vehicle percentage only occurs on very low-volume roads;
      it is left missing with an indicator rather than imputed, so the ML
      module chooses the imputation strategy explicitly.
    * ``SHAPE_Length`` is in degrees (meaningless as a length) and is
      replaced by a projected length in metres.
    """
    gdf = gdf.copy()
    gdf["aadt"] = gdf["TESECN_VOLUME"].astype(float)
    gdf["volume_base_year"] = gdf["TESECN_BASE_YEAR"].astype(int)
    gdf["volume_age_years"] = config.VOLUME_REFERENCE_YEAR - gdf["volume_base_year"]
    gdf["volume_stale"] = (gdf["volume_age_years"] > 5).astype("int8")
    gdf["cv_percent"] = gdf["CV_PERCENT"].astype(float)
    gdf["cv_missing"] = gdf["cv_percent"].isna().astype("int8")
    gdf["heavy_vehicle_aadt"] = gdf["aadt"] * gdf["cv_percent"] / 100
    gdf = gdf.rename(columns={"TESECN_ID": "segment_id", "ROAD_NO": "road_no",
                              "RLCWY_CODE": "carriageway"})
    keep = ["segment_id", "road_no", "carriageway", "TESECN_START_RRD", "TESECN_END_RRD", "aadt",
            "volume_base_year", "volume_age_years", "volume_stale", "cv_percent",
            "cv_missing", "heavy_vehicle_aadt", "geometry"]
    gdf = gdf[keep].rename(columns={"TESECN_START_RRD": "start_rrd_km", "TESECN_END_RRD": "end_rrd_km"})
    gdf["length_m"] = gdf.to_crs(config.WORK_CRS).length.round(1)
    return gdf


def dedupe_points(gdf: gpd.GeoDataFrame, layer: str) -> gpd.GeoDataFrame:
    """Collapses multiple mapped points of one site into a single centroid.

    The signal layer has 1,552 points for 883 sites: one intersection is
    often mapped once per pole or approach. Counting raw points would
    double-count infrastructure, so points sharing (site_type, site_id) are
    dissolved to their centroid. Sites whose points spread over > 50 m are
    flagged because the id may have been reused.
    """
    work = gdf.to_crs(config.WORK_CRS)
    work["site_key"] = layer + ":" + work["site_type"] + ":" + work["site_id"]
    centroids = work.dissolve("site_key", aggfunc="first").centroid
    spread = work.geometry.distance(work["site_key"].map(centroids)).groupby(work["site_key"]).max()
    n_points = work.groupby("site_key").size()
    attrs = work.drop(columns="geometry").groupby("site_key").first()
    out = gpd.GeoDataFrame(attrs, geometry=centroids.reindex(attrs.index), crs=config.WORK_CRS)
    out["n_points"] = n_points
    out["point_spread_m"] = spread.round(1)
    out["spread_flag"] = (out["point_spread_m"] > 50).astype("int8")
    flagged = out.index[out.spread_flag == 1].tolist()
    if flagged:
        log.warning("%s: %d site(s) with points spread > 50 m: %s", layer, len(flagged), flagged[:10])
    log.info("%s: %d points -> %d sites", layer, len(gdf), len(out))
    return out.reset_index()


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------
def nearest(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame, cols: list[str],
            max_distance: float | None, prefix: str) -> pd.DataFrame:
    """Nearest-neighbour join (STRtree index) returning one match per left row.

    Ties at equal distance are broken deterministically by keeping the
    first match, so reruns produce identical outputs.
    """
    # Positional index on the left and prefixed names on the right avoid any
    # name collision between the two frames' index/columns.
    lhs = gpd.GeoDataFrame(geometry=left.geometry.values, crs=left.crs)
    rhs = (right[cols + ["geometry"]].rename(columns={c: f"{prefix}_{c}" for c in cols})
           .reset_index(drop=True))
    joined = gpd.sjoin_nearest(lhs, rhs, how="left", max_distance=max_distance,
                               distance_col=f"{prefix}_dist_m")
    joined = joined[~joined.index.duplicated(keep="first")].sort_index()
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))
    out[f"{prefix}_dist_m"] = out[f"{prefix}_dist_m"].round(1)
    out.index = left.index
    return out


def count_within(targets: gpd.GeoDataFrame, features: gpd.GeoDataFrame, radius: float) -> pd.Series:
    """Counts features within ``radius`` metres of each target geometry."""
    buf = gpd.GeoDataFrame(geometry=targets.geometry.buffer(radius), index=targets.index,
                           crs=targets.crs)
    hits = gpd.sjoin(buf, features[["geometry"]], how="inner", predicate="intersects")
    return hits.groupby(level=0).size().reindex(targets.index, fill_value=0).astype("int32")


def distance_to_cbd_km(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Straight-line distance to the Adelaide GPO, a simple remoteness proxy."""
    gpo = gpd.GeoSeries(gpd.points_from_xy([config.ADELAIDE_GPO[0]], [config.ADELAIDE_GPO[1]]),
                        crs="EPSG:7844").to_crs(config.WORK_CRS).iloc[0]
    return (gdf.geometry.distance(gpo) / 1000).round(2)


# ---------------------------------------------------------------------------
# Site construction
# ---------------------------------------------------------------------------
def build_tables(raw: dict[str, gpd.GeoDataFrame]) -> dict[str, gpd.GeoDataFrame]:
    """Builds the three output tables from validated layers.

    Args:
        raw: Validated layers keyed by source key.

    Returns:
        Mapping of output table name to GeoDataFrame (in WORK_CRS).
    """
    crashes = clean_crashes(raw["crashes"]).to_crs(config.WORK_CRS)
    volumes = clean_volumes(raw["volumes"]).to_crs(config.WORK_CRS)
    signals = dedupe_points(normalise_points(raw["signals"]), "signal")
    peds = dedupe_points(normalise_points(raw["ped_crossings"]), "ped")
    schools = dedupe_points(normalise_points(raw["school_crossings"]), "school")

    # Signalised intersections are the 'TS' sites; other signal types
    # (e.g. 'RS', 'ES', 'PP') are retained as context features only.
    intersections = signals[signals.site_type == "TS"].reset_index(drop=True)
    log.info("intersection sites: %d", len(intersections))

    # ---- 1. Assign each crash location to a site ---------------------------
    to_int = nearest(crashes, intersections, ["site_key"], config.INTERSECTION_RADIUS_M, "int")
    crashes = crashes.join(to_int)
    unassigned = crashes["int_site_key"].isna()
    to_seg = nearest(crashes[unassigned], volumes, ["segment_id"], config.SEGMENT_RADIUS_M, "seg")
    crashes = crashes.join(to_seg)
    crashes["site_type"] = np.select(
        [crashes.int_site_key.notna(), crashes.seg_segment_id.notna()],
        ["intersection", "state_road_segment"], default="other_road",
    )
    log.info("crash location assignment: %s", crashes.site_type.value_counts().to_dict())

    # ---- 2. Road-context features for every crash location -----------------
    ctx = nearest(crashes, volumes, ["aadt", "cv_percent", "volume_stale", "road_no"],
                  config.SEGMENT_RADIUS_M, "road")
    crashes = crashes.join(ctx)
    crashes["on_state_road"] = crashes["road_aadt"].notna().astype("int8")
    crashes["dist_signal_m"] = nearest(crashes, signals, ["site_key"], None, "sig")["sig_dist_m"]
    crashes["dist_ped_crossing_m"] = nearest(crashes, peds, ["site_key"], None, "ped")["ped_dist_m"]
    crashes["dist_school_crossing_m"] = nearest(crashes, schools, ["site_key"], None, "sch")["sch_dist_m"]
    crashes["ped_crossings_200m"] = count_within(crashes, peds, config.PED_CROSSING_BUFFER_M)
    crashes["school_crossings_300m"] = count_within(crashes, schools, config.SCHOOL_CROSSING_BUFFER_M)
    crashes["signals_200m"] = count_within(crashes, signals, config.SIGNAL_BUFFER_M)
    crashes["dist_cbd_km"] = distance_to_cbd_km(crashes)

    # ---- 3. Intersection site table (zeros included) ------------------------
    agg = crashes.groupby("int_site_key")[CRASH_COUNT_COLS].sum()
    ints = intersections.set_index("site_key").join(agg)
    ints[CRASH_COUNT_COLS] = ints[CRASH_COUNT_COLS].fillna(0).astype("int32")
    ints["n_crash_locations"] = crashes.groupby("int_site_key").size().reindex(ints.index, fill_value=0)

    # Entering-traffic proxy: for each road meeting the intersection take its
    # highest segment volume, then sum across roads. A vehicle passes through
    # on one road, so the road totals approximate total entering vehicles.
    near = gpd.sjoin(
        gpd.GeoDataFrame(geometry=ints.geometry.buffer(config.SIGNAL_VOLUME_RADIUS_M), crs=ints.crs),
        volumes[["road_no", "aadt", "cv_percent", "volume_stale", "geometry"]],
        how="inner", predicate="intersects",
    )
    per_road = near.groupby([near.index, "road_no"]).agg(
        aadt=("aadt", "max"), cv=("cv_percent", "mean"), stale=("volume_stale", "max"))
    by_site = per_road.groupby(level=0).agg(
        entering_aadt=("aadt", "sum"), major_road_aadt=("aadt", "max"),
        n_state_roads=("aadt", "size"), cv_percent=("cv", "mean"), volume_stale=("stale", "max"))
    ints = ints.join(by_site)
    ints["n_state_roads"] = ints["n_state_roads"].fillna(0).astype("int8")
    ints["has_volume"] = ints["entering_aadt"].notna().astype("int8")
    mev = ints["entering_aadt"] * 365 * config.CRASH_YEARS / 1e6  # million entering vehicles
    ints["million_entering_vehicles"] = mev.round(3)
    ints["crash_rate_per_mev"] = (ints["total_crashes"] / mev).round(4)
    ints["fsi_rate_per_mev"] = (ints["fsi_crashes"] / mev).round(4)
    ints["ped_crossings_200m"] = count_within(ints, peds, config.PED_CROSSING_BUFFER_M)
    ints["school_crossings_300m"] = count_within(ints, schools, config.SCHOOL_CROSSING_BUFFER_M)
    ints["dist_school_crossing_m"] = nearest(ints, schools, ["site_key"], None, "sch")["sch_dist_m"]
    ints["dist_cbd_km"] = distance_to_cbd_km(ints)
    ints["any_fsi"] = (ints["fsi_crashes"] > 0).astype("int8")
    ints = ints.reset_index()

    # ---- 4. Segment site table (zeros included) -----------------------------
    agg = crashes.groupby("seg_segment_id")[CRASH_COUNT_COLS].sum()
    segs = volumes.set_index("segment_id").join(agg)
    segs[CRASH_COUNT_COLS] = segs[CRASH_COUNT_COLS].fillna(0).astype("int32")
    segs["n_crash_locations"] = crashes.groupby("seg_segment_id").size().reindex(segs.index, fill_value=0)
    vkt = segs["aadt"] * 365 * config.CRASH_YEARS * (segs["length_m"] / 1000)
    segs["vkt_100m"] = (vkt / 1e8).round(5)  # hundred-million vehicle-km
    segs["crash_rate_per_100m_vkt"] = (segs["total_crashes"] / segs["vkt_100m"]).round(3)
    segs["fsi_rate_per_100m_vkt"] = (segs["fsi_crashes"] / segs["vkt_100m"]).round(3)
    length_km = (segs["length_m"] / 1000).replace(0, np.nan)
    segs["signals_per_km"] = (count_within(segs, signals, 30) / length_km).round(3)
    segs["ped_crossings_per_km"] = (count_within(segs, peds, 30) / length_km).round(3)
    segs["school_crossings_on_segment"] = count_within(segs, schools, 30)
    segs["dist_cbd_km"] = distance_to_cbd_km(segs.set_geometry(segs.representative_point()))
    segs["any_fsi"] = (segs["fsi_crashes"] > 0).astype("int8")
    segs = segs.reset_index()

    crashes = crashes.rename(columns={"int_site_key": "intersection_site_key",
                                      "seg_segment_id": "segment_id",
                                      "int_dist_m": "intersection_dist_m",
                                      "seg_dist_m": "segment_dist_m"})
    return {"crash_locations": crashes, "intersection_sites": ints, "segment_sites": segs}
