# Student Name: Aftab Hussaini
# Student FAN:  [YourFAN]
# File:         config.py
# Date:         01-10-2026
# Description:  Central configuration: data sources, CRS choices, spatial tolerances and limits.
# Licence:      MIT Licence
"""Pipeline configuration.

Everything that a maintainer may need to tune lives here so the processing
code itself never contains magic numbers. Values are grouped by concern:
sources, security limits, coordinate reference systems and spatial
matching tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SOURCE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = SOURCE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # downloaded archives (immutable cache)
PROCESSED_DIR = DATA_DIR / "processed"  # cleaned, model-ready outputs
REJECTED_DIR = DATA_DIR / "rejected"    # quarantined rows that failed validation
LOG_DIR = DATA_DIR / "logs"

# ---------------------------------------------------------------------------
# Data sources (data.sa.gov.au CKAN catalogue, CC-BY 4.0)
# ---------------------------------------------------------------------------
CKAN_API = "https://data.sa.gov.au/data/api/3/action/package_show"


@dataclass(frozen=True)
class Source:
    """Describes one upstream dataset.

    Attributes:
        key: Short internal name used for file names and logs.
        ckan_id: Package slug on data.sa.gov.au, used to resolve the
            current download URL at run time (so new releases are picked up
            without code changes).
        resource_pattern: Case-insensitive regex matched against the CKAN
            resource name to pick the right file from the package.
        fallback_url: Direct URL used if the catalogue API is unreachable.
        member: File inside the zip archive to read.
        crs: CRS of the coordinates in ``member`` (GeoJSON files from this
            portal carry no ``crs`` member, so it must be declared).
    """

    key: str
    ckan_id: str
    resource_pattern: str
    fallback_url: str
    member: str
    crs: str


SOURCES: dict[str, Source] = {
    s.key: s
    for s in (
        Source(
            key="crashes",
            ckan_id="road-crashes-in-sa",
            resource_pattern=r"road crashes 2020-2024 \(geojson\)",
            fallback_url=(
                "https://data.sa.gov.au/data/dataset/cf8d8549-b976-4523-88ad-46dd32053696/"
                "resource/5fcbce14-ad30-4737-aa7d-5b1cfb939eb9/download/"
                "roadcrashes_2020_2024_geojson.zip"
            ),
            member="RoadCrashes_GDA2020.geojson",
            crs="EPSG:7844",  # GDA2020 geographic
        ),
        Source(
            key="volumes",
            ckan_id="traffic-volumes",
            resource_pattern=r"2024.*geojson|geojson.*2024",
            fallback_url=(
                "https://data.sa.gov.au/data/dataset/e5d6588a-f163-4f6a-bc57-25e95c87b5bd/"
                "resource/daf8098d-4ffb-4b07-b347-6b3c204add43/download/"
                "trafficvolumeestimates2024_geojson.zip"
            ),
            member="TrafficVolumeEstimates_WGS1984.geojson",
            crs="EPSG:4326",  # WGS84
        ),
        Source(
            key="signals",
            ckan_id="traffic-signals",
            resource_pattern=r"geojson",
            fallback_url="https://www.dptiapps.com.au/dataportal/TrafficSignals_geojson.zip",
            # NB: the GDA94 file in this archive is 0 bytes upstream; use GDA2020.
            member="TrafficSignals_GDA2020.geojson",
            crs="EPSG:7844",
        ),
        Source(
            key="ped_crossings",
            ckan_id="pedestrian-crossings",
            resource_pattern=r"geojson",
            fallback_url="https://www.dptiapps.com.au/dataportal/PedestrianCrossings_geojson.zip",
            member="PedestrianCrossings_GDA2020.geojson",
            crs="EPSG:7844",
        ),
        Source(
            key="school_crossings",
            ckan_id="school-crossings",
            resource_pattern=r"geojson",
            fallback_url="https://www.dptiapps.com.au/dataportal/SchoolCrossings_geojson.zip",
            member="SchoolCrossings_GDA2020.geojson",
            crs="EPSG:7844",
        ),
    )
}

# ---------------------------------------------------------------------------
# Security limits (defence against oversized / malicious payloads)
# ---------------------------------------------------------------------------
ALLOWED_HOSTS = {"data.sa.gov.au", "www.dptiapps.com.au", "location.sa.gov.au"}
HTTP_TIMEOUT_S = (10, 120)            # (connect, read)
MAX_DOWNLOAD_BYTES = 150 * 1024**2    # 150 MB per archive
MAX_UNCOMPRESSED_BYTES = 500 * 1024**2  # zip-bomb guard per member
MAX_COMPRESSION_RATIO = 200           # zip-bomb guard
USER_AGENT = "COMP3742-A1-crash-risk-ingestion/1.0 (educational use)"

# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------
# South Australia spans MGA zones 52-54, so a single UTM zone would distort
# distances at the state's edges. GDA2020 / Geoscience Australia Lambert is a
# conformal, metre-based projection valid state-wide, used for all buffers and
# nearest-neighbour joins.
WORK_CRS = "EPSG:7845"
OUTPUT_CRS = "EPSG:7844"  # publish in GDA2020 geographic (Australian standard)

# Sanity bounding box for South Australia (lon/lat, generous margin).
SA_BBOX = {"min_lon": 128.9, "max_lon": 141.1, "min_lat": -38.2, "max_lat": -25.9}
ADELAIDE_GPO = (138.6007, -34.9285)  # lon, lat - remoteness proxy origin

# ---------------------------------------------------------------------------
# Spatial matching tolerances (metres). Justified in the report.
# ---------------------------------------------------------------------------
INTERSECTION_RADIUS_M = 20.0    # crash location -> signalised intersection (knee of distance histogram, see eda.py)
SEGMENT_RADIUS_M = 25.0         # remaining crash location -> traffic-volume segment
SIGNAL_VOLUME_RADIUS_M = 50.0   # segments whose volume feeds an intersection
PED_CROSSING_BUFFER_M = 200.0
SCHOOL_CROSSING_BUFFER_M = 300.0
SIGNAL_BUFFER_M = 200.0

# ---------------------------------------------------------------------------
# Crash data window (the published file aggregates five calendar years)
# ---------------------------------------------------------------------------
CRASH_YEARS = 5
CRASH_PERIOD = "2020-2024"
VOLUME_REFERENCE_YEAR = 2024
