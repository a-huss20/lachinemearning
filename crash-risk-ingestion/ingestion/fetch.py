# Student Name: Aftab Hussaini
# Student FAN:  [YourFAN]
# File:         fetch.py
# Date:         01-10-2026
# Description:  Secure, cached, conditional download of source archives from data.sa.gov.au.
# Licence:      MIT Licence
"""Acquisition stage.

Responsibilities:
  1. Resolve the current download URL through the CKAN catalogue API, so a
     re-published file is picked up automatically (fallback: pinned URL).
  2. Download over HTTPS only, from an allow-list of hosts, with timeouts,
     a size cap and a conditional GET (ETag / Last-Modified) so unchanged
     files are not re-downloaded.
  3. Record provenance (URL, hash, timestamps) in a manifest for auditability.
  4. Safely extract a single named member from the archive, rejecting path
     traversal and zip bombs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import config
from .config import Source

log = logging.getLogger(__name__)

MANIFEST = config.RAW_DIR / "manifest.json"


class FetchError(RuntimeError):
    """Raised when a source cannot be obtained from network or cache."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """Returns the SHA-256 hex digest of a file, read in 1 MB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(MANIFEST)  # atomic on the same filesystem


def check_url(url: str) -> None:
    """Enforces HTTPS and the host allow-list.

    Raises:
        FetchError: If the URL is not HTTPS or its host is not allowed.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise FetchError(f"Refusing non-HTTPS URL: {url}")
    if parsed.hostname not in config.ALLOWED_HOSTS:
        raise FetchError(f"Host not in allow-list: {parsed.hostname}")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = config.USER_AGENT
    return s  # TLS certificate verification is on by default and never disabled


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------
def resolve_url(source: Source, session: requests.Session) -> str:
    """Finds the current resource URL via the CKAN package_show API.

    Falls back to the pinned URL in config if the API is unreachable or
    no resource matches, so a catalogue outage does not stop the pipeline.
    """
    try:
        resp = session.get(
            config.CKAN_API, params={"id": source.ckan_id}, timeout=config.HTTP_TIMEOUT_S
        )
        resp.raise_for_status()
        resources = resp.json()["result"]["resources"]
        pattern = re.compile(source.resource_pattern, re.IGNORECASE)
        for res in resources:
            name = f"{res.get('name', '')} {res.get('url', '')}"
            if pattern.search(name) and res.get("url", "").lower().endswith(".zip"):
                log.info("[%s] catalogue resolved -> %s", source.key, res["url"])
                return res["url"]
        log.warning("[%s] no catalogue resource matched; using fallback URL", source.key)
    except (requests.RequestException, KeyError, ValueError) as exc:
        log.warning("[%s] catalogue lookup failed (%s); using fallback URL", source.key, exc)
    return source.fallback_url


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download(source: Source, session: requests.Session, manifest: dict) -> Path:
    """Downloads a source archive if it changed upstream.

    Uses ETag / Last-Modified from the previous run for a conditional GET;
    an HTTP 304 means the cached copy is current. The body is streamed to a
    temporary file and only moved into place once complete and within the
    size limit, so a failed transfer never corrupts the cache.

    Returns:
        Path to the cached archive.
    """
    url = resolve_url(source, session)
    check_url(url)
    dest = config.RAW_DIR / f"{source.key}.zip"
    prev = manifest.get(source.key, {})

    headers = {}
    if dest.exists() and prev.get("url") == url:
        if prev.get("etag"):
            headers["If-None-Match"] = prev["etag"]
        if prev.get("last_modified"):
            headers["If-Modified-Since"] = prev["last_modified"]

    with session.get(url, headers=headers, stream=True, timeout=config.HTTP_TIMEOUT_S) as resp:
        # Redirects may change host: re-check the final URL.
        check_url(resp.url)
        if resp.status_code == 304:
            log.info("[%s] not modified upstream; using cache", source.key)
            return dest
        resp.raise_for_status()

        declared = int(resp.headers.get("Content-Length", 0) or 0)
        if declared > config.MAX_DOWNLOAD_BYTES:
            raise FetchError(f"[{source.key}] declared size {declared} exceeds limit")

        tmp = dest.with_suffix(".part")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                written += len(chunk)
                if written > config.MAX_DOWNLOAD_BYTES:
                    fh.close()
                    tmp.unlink(missing_ok=True)
                    raise FetchError(f"[{source.key}] download exceeded size limit")
                fh.write(chunk)

        if not zipfile.is_zipfile(tmp):  # content check, not just extension
            tmp.unlink(missing_ok=True)
            raise FetchError(f"[{source.key}] payload is not a zip archive")
        tmp.replace(dest)

        manifest[source.key] = {
            "url": url,
            "etag": resp.headers.get("ETag"),
            "last_modified": resp.headers.get("Last-Modified"),
            "bytes": written,
            "sha256": sha256_file(dest),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "origin": "network",
        }
        log.info("[%s] downloaded %.1f MB", source.key, written / 1e6)
    return dest


def seed_from_local(source: Source, local_dir: Path, manifest: dict) -> Path:
    """Copies a manually downloaded archive into the cache (offline mode).

    Searches ``local_dir`` recursively for a zip whose name matches the
    fallback URL's file name, so the folder layout produced by downloading
    from data.sa.gov.au in a browser works unchanged.
    """
    wanted = Path(urlparse(source.fallback_url).path).name.lower()
    matches = [p for p in local_dir.rglob("*.zip") if p.name.lower() == wanted]
    if not matches:
        raise FetchError(f"[{source.key}] {wanted} not found under {local_dir}")
    dest = config.RAW_DIR / f"{source.key}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], dest)
    manifest[source.key] = {
        "url": source.fallback_url,
        "bytes": dest.stat().st_size,
        "sha256": sha256_file(dest),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "origin": f"local:{matches[0]}",
    }
    log.info("[%s] seeded cache from %s", source.key, matches[0])
    return dest


# ---------------------------------------------------------------------------
# Safe extraction
# ---------------------------------------------------------------------------
def extract_member(archive: Path, member: str, out_dir: Path) -> Path:
    """Extracts one named file from a zip archive with safety checks.

    Rejects absolute paths and ``..`` components (zip-slip), and members
    whose uncompressed size or compression ratio indicates a zip bomb.

    Returns:
        Path to the extracted file.
    """
    with zipfile.ZipFile(archive) as zf:
        try:
            info = zf.getinfo(member)
        except KeyError as exc:
            raise FetchError(
                f"{member} missing from {archive.name}; contents: {zf.namelist()}"
            ) from exc
        name = Path(info.filename)
        if name.is_absolute() or ".." in name.parts:
            raise FetchError(f"Unsafe path in archive: {info.filename}")
        if info.file_size > config.MAX_UNCOMPRESSED_BYTES:
            raise FetchError(f"{member} too large when uncompressed ({info.file_size} B)")
        if info.compress_size and info.file_size / info.compress_size > config.MAX_COMPRESSION_RATIO:
            raise FetchError(f"{member} has suspicious compression ratio")
        if info.file_size == 0:
            raise FetchError(f"{member} is empty in {archive.name}")
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / name.name
        with zf.open(info) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return target


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def acquire_all(offline: bool = False, local_dir: Path | None = None) -> dict[str, Path]:
    """Obtains every configured source and returns paths to the extracted files.

    Args:
        offline: If True, never touch the network; use the cache, seeding it
            from ``local_dir`` when given.
        local_dir: Folder of manually downloaded archives (offline seeding,
            or automatic fallback if the network fetch fails).

    Returns:
        Mapping of source key to extracted GeoJSON path.
    """
    manifest = _load_manifest()
    session = None if offline else _session()
    extracted: dict[str, Path] = {}

    for key, source in config.SOURCES.items():
        cached = config.RAW_DIR / f"{key}.zip"
        try:
            if offline:
                archive = seed_from_local(source, local_dir, manifest) if local_dir else cached
            else:
                archive = download(source, session, manifest)
        except (requests.RequestException, FetchError) as exc:
            # Degrade gracefully: local copy, then last good cache.
            log.error("[%s] fetch failed: %s", key, exc)
            if local_dir:
                archive = seed_from_local(source, local_dir, manifest)
            elif cached.exists():
                log.warning("[%s] using last cached copy", key)
                archive = cached
            else:
                raise

        if not archive.exists():
            raise FetchError(f"[{key}] no cached archive; run online or pass --local-dir")

        # Integrity: cached file must match the hash recorded when it was fetched.
        recorded = manifest.get(key, {}).get("sha256")
        actual = sha256_file(archive)
        if recorded and recorded != actual:
            raise FetchError(f"[{key}] cache hash mismatch - possible tampering")
        manifest.setdefault(key, {})["sha256"] = actual

        extracted[key] = extract_member(archive, source.member, config.RAW_DIR / "extracted")

    _save_manifest(manifest)
    return extracted
