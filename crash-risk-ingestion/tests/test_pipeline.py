# Student Name: Aftab Hussaini
# Student FAN:  [YourFAN]
# File:         test_pipeline.py
# Date:         01-10-2026
# Description:  Unit tests for the security and validation behaviour of the ingestion pipeline.
# Usage:        python -m pytest tests
"""Unit tests (synthetic data only, no network or real datasets needed)."""

import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion import config, fetch, transform, validate  # noqa: E402


# ---- security --------------------------------------------------------------
def test_rejects_plain_http():
    with pytest.raises(fetch.FetchError):
        fetch.check_url("http://data.sa.gov.au/file.zip")


def test_rejects_unknown_host():
    with pytest.raises(fetch.FetchError):
        fetch.check_url("https://evil.example.com/file.zip")


def test_accepts_allowed_host():
    fetch.check_url("https://data.sa.gov.au/data/x.zip")


def test_zip_slip_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escape.geojson", "{}")
    with pytest.raises(fetch.FetchError):
        fetch.extract_member(archive, "../escape.geojson", tmp_path / "out")


def test_empty_member_rejected(tmp_path):
    archive = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("TrafficSignals_GDA94.geojson", "")  # mirrors the real 0-byte upstream file
    with pytest.raises(fetch.FetchError):
        fetch.extract_member(archive, "TrafficSignals_GDA94.geojson", tmp_path / "out")


# ---- validation ------------------------------------------------------------
def _crash_row(loc, pdo, inj, fat, lon=138.6, lat=-34.9):
    row = {c: 0 for c in validate.CRASH_SEVERITY + validate.CRASH_TYPES + validate.CRASH_OTHER}
    row.update(unique_loc=loc, cse_pdo=pdo, cse_inj=inj, cse_fat=fat,
               total_crashes=pdo + inj + fat, cty_rear_end=pdo + inj + fat,
               total_fatalities=fat, geometry=Point(lon, lat))
    return row


def test_bad_rows_quarantined_not_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REJECTED_DIR", tmp_path)
    good = _crash_row("00000000000001", 1, 0, 0)
    bad_sum = _crash_row("00000000000002", 1, 0, 0)
    bad_sum["total_crashes"] = 5                      # breaks severity partition
    outside = _crash_row("00000000000003", 1, 0, 0, lon=151.2, lat=-33.9)  # Sydney
    gdf = gpd.GeoDataFrame([good, bad_sum, outside], crs="EPSG:7844")
    clean, rejected = validate.validate(gdf, "crashes", "crashes")
    assert clean.unique_loc.tolist() == ["00000000000001"]
    assert len(rejected) == 2
    assert (tmp_path / "crashes.csv").exists()


def test_schema_drift_is_fatal():
    gdf = gpd.GeoDataFrame({"unique_loc": ["1"]}, geometry=[Point(138.6, -34.9)], crs="EPSG:7844")
    with pytest.raises(validate.SchemaDriftError):
        validate.validate(gdf, "crashes", "crashes")


# ---- transform -------------------------------------------------------------
def test_severity_label_uses_worst_outcome():
    gdf = gpd.GeoDataFrame([_crash_row("00000000000001", 3, 1, 1)], crs="EPSG:7844")
    gdf.loc[0, "cse_si"] = 1
    out = transform.clean_crashes(gdf)
    assert out.loc[0, "max_severity"] == 3
    assert out.loc[0, "fsi_crashes"] == 2
    assert out.loc[0, "cse_minor_inj"] == 0


def test_nearest_respects_max_distance():
    left = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(100, 0)], crs=config.WORK_CRS)
    right = gpd.GeoDataFrame({"site_key": ["a"]}, geometry=[Point(5, 0)], crs=config.WORK_CRS)
    out = transform.nearest(left, right, ["site_key"], 20, "int")
    assert out.loc[0, "int_site_key"] == "a"
    assert out["int_site_key"].isna().iloc[1]


# ---- download path (mocked HTTP, no network) -------------------------------
class _FakeResp:
    def __init__(self, status, body=b"", headers=None, url="https://data.sa.gov.au/x.zip"):
        self.status_code, self._body, self.headers, self.url = status, body, headers or {}, url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise fetch.requests.HTTPError(self.status_code)

    def iter_content(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]

    def json(self):
        return {"result": {"resources": [{"name": "Road Crashes 2020-2024 (geojson)",
                                          "url": "https://data.sa.gov.au/x.zip"}]}}


class _FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kw):
        self.calls.append((url, kw.get("headers", {})))
        return self.responses.pop(0)


def _zip_bytes(tmp_path):
    p = tmp_path / "z.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("a.geojson", "{}")
    return p.read_bytes()


def test_download_then_conditional_304(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    src = config.SOURCES["crashes"]
    body = _zip_bytes(tmp_path)
    manifest = {}
    s1 = _FakeSession([_FakeResp(200), _FakeResp(200, body, {"ETag": '"v1"'})])
    path = fetch.download(src, s1, manifest)
    assert path.exists() and manifest["crashes"]["etag"] == '"v1"'
    s2 = _FakeSession([_FakeResp(200), _FakeResp(304)])
    fetch.download(src, s2, manifest)
    assert s2.calls[1][1].get("If-None-Match") == '"v1"'   # conditional GET sent


def test_download_rejects_non_zip_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    s = _FakeSession([_FakeResp(200), _FakeResp(200, b"<html>error page</html>")])
    with pytest.raises(fetch.FetchError):
        fetch.download(config.SOURCES["crashes"], s, {})


def test_download_rejects_redirect_to_foreign_host(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    s = _FakeSession([_FakeResp(200), _FakeResp(200, b"x", url="https://evil.example.com/x.zip")])
    with pytest.raises(fetch.FetchError):
        fetch.download(config.SOURCES["crashes"], s, {})
