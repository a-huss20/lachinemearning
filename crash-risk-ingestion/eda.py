# Student Name: Aftab Hussaini
# Student FAN:  [YourFAN]
# File:         eda.py
# Date:         01-10-2026
# Description:  Exploratory data analysis of the processed tables; writes report figures and summary statistics.
# Usage:        python eda.py   (run after data_ingestion.py)
# Licence:      MIT Licence
"""Exploratory data analysis for the technical report.

Reads the processed outputs (not the raw files) so the figures describe
exactly what the ML module will receive. Figures are saved to
data/eda/ as 200-dpi PNGs; key numbers go to data/eda/eda_summary.json.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless: no display needed when scheduled
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter
import pandas as pd

from ingestion import config

EDA_DIR = config.DATA_DIR / "eda"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEVERITY_RAMP = ["#c6dbf5", "#7fb0ea", "#2a78d6", "#123f78"]  # light -> dark = less -> more severe

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 200,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(EDA_DIR / name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    crashes = pd.read_csv(config.PROCESSED_DIR / "crash_locations.csv")
    ints = pd.read_csv(config.PROCESSED_DIR / "intersection_sites.csv")
    segs = pd.read_csv(config.PROCESSED_DIR / "segment_sites.csv")
    summary: dict = {}

    # ---- Fig 1: severity distribution (class imbalance) ----------------------
    sev = {"Property damage only": crashes.cse_pdo.sum(), "Minor injury": crashes.cse_minor_inj.sum(),
           "Serious injury": crashes.cse_si.sum(), "Fatal": crashes.cse_fat.sum()}
    total = sum(sev.values())
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(list(sev), list(sev.values()), color=SEVERITY_RAMP, height=0.6)
    ax.invert_yaxis()
    for b, v in zip(bars, sev.values()):
        ax.text(b.get_width() + total * 0.01, b.get_y() + b.get_height() / 2,
                f"{v:,} ({v / total:.1%})", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Crashes, South Australia 2020-2024")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlim(0, max(sev.values()) * 1.3)
    ax.grid(axis="y", visible=False)
    save(fig, "fig1_severity_distribution.png")
    summary["severity_counts"] = {k: int(v) for k, v in sev.items()}
    summary["fsi_share"] = round((sev["Serious injury"] + sev["Fatal"]) / total, 4)

    # ---- Fig 2: distance to nearest signal (justifies matching radius) -------
    near = crashes[crashes.dist_signal_m < 100]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(near.dist_signal_m, bins=np.arange(0, 101, 5), weights=near.total_crashes,
            color=BLUE, edgecolor="white", linewidth=1)
    ax.axvline(config.INTERSECTION_RADIUS_M, color=ORANGE, linewidth=2)
    ax.text(config.INTERSECTION_RADIUS_M + 2, ax.get_ylim()[1] * 0.85,
            f"matching radius = {config.INTERSECTION_RADIUS_M:.0f} m", color=INK, fontsize=9)
    ax.set_xlabel("Distance from crash location to nearest traffic signal (m)")
    ax.set_ylabel("Crashes (per 5 m bin)")
    save(fig, "fig2_signal_distance.png")
    background = near.loc[near.dist_signal_m.between(50, 100), "total_crashes"].sum() / 10
    summary["background_crashes_per_5m_bin_50_100m"] = round(float(background), 1)

    # ---- Fig 3: crash types --------------------------------------------------
    types = {c.replace("cty_", "").replace("_", " ").replace("vehile", "vehicle").replace(" oc", " (off carriageway)"): crashes[c].sum()
             for c in crashes.columns if c.startswith("cty_")}
    types = pd.Series(types).sort_values(ascending=True)
    types = types[types > 0].tail(10)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(types.index, types.values, color=BLUE, height=0.6)
    ax.set_xlabel("Crashes, 2020-2024")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", visible=False)
    save(fig, "fig3_crash_types.png")

    # ---- Fig 4: rate instability vs exposure (motivates Empirical Bayes) -----
    s = segs[segs.vkt_100m > 0]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.scatter(s.vkt_100m, s.crash_rate_per_100m_vkt, s=10, color=BLUE, alpha=0.45,
               edgecolors="white", linewidths=0.3)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylim(bottom=-0.5)
    ax.set_xlabel("Exposure: hundred-million vehicle-km travelled, 2020-2024 (log)")
    ax.set_ylabel("Crashes per 100M VKT")
    save(fig, "fig4_rate_vs_exposure.png")
    q = s.vkt_100m.quantile([0.1, 0.9])
    summary["rate_sd_low_exposure_decile"] = round(float(s[s.vkt_100m <= q[0.1]].crash_rate_per_100m_vkt.std()), 1)
    summary["rate_sd_high_exposure_decile"] = round(float(s[s.vkt_100m >= q[0.9]].crash_rate_per_100m_vkt.std()), 1)

    # ---- Fig 5: age of traffic volume estimates (data quality) ---------------
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.hist(segs.volume_age_years, bins=np.arange(-0.5, segs.volume_age_years.max() + 1.5, 1),
            color=BLUE, edgecolor="white", linewidth=1)
    ax.set_xlabel("Age of traffic count behind the 2024 estimate (years)")
    ax.set_ylabel("Segments")
    save(fig, "fig5_volume_age.png")

    # ---- Summary numbers for the report --------------------------------------
    summary.update({
        "crash_locations": len(crashes),
        "total_crashes": int(crashes.total_crashes.sum()),
        "single_crash_location_share": round(float((crashes.total_crashes == 1).mean()), 3),
        "crash_assignment_locations": crashes.site_type.value_counts().to_dict(),
        "crash_assignment_crashes": crashes.groupby("site_type").total_crashes.sum().astype(int).to_dict(),
        "fsi_by_site_type": crashes.groupby("site_type").fsi_crashes.sum().astype(int).to_dict(),
        "intersections": len(ints),
        "intersections_zero_crash_share": round(float((ints.total_crashes == 0).mean()), 3),
        "intersections_without_volume": int((ints.has_volume == 0).sum()),
        "segments": len(segs),
        "segments_zero_crash_share": round(float((segs.total_crashes == 0).mean()), 3),
        "segments_stale_volume_share": round(float(segs.volume_stale.mean()), 3),
        "segments_cv_missing": int(segs.cv_missing.sum()),
        "max_severity_label_counts": crashes.max_severity_label.value_counts().to_dict(),
        "top_intersections_by_crashes": ints.nlargest(5, "total_crashes")[
            ["site_desc", "total_crashes", "fsi_crashes", "crash_rate_per_mev"]].to_dict("records"),
    })
    (EDA_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
