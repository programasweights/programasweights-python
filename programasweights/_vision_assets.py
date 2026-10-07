"""Resolve verified GGUF model/projector files; never instantiate a model.

V2 assets use base_models/sha256/<digest>.gguf, independent of source filename
or runtime ID. Existing filename-based cache entries may be adopted via a
verified hard link (or a copy if links are unavailable). Text caching is unchanged.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import NamedTuple

import httpx

from . import cache, config
from ._output import ProgressCallback, report_progress
from ._vision_contract import VISION_MANIFEST_VERSION


class VisionAssetPaths(NamedTuple):
    base_model: Path
    projector: Path


def _manifest(value: dict) -> dict:
    normalized = cache._normalize_runtime_manifest(value, require_local_model=True)
    if (
        normalized is None
        or normalized["manifest_version"] != VISION_MANIFEST_VERSION
        or not normalized["local_sdk"]["supported"]
    ):
        raise ValueError("A valid, locally supported vision runtime manifest is required.")
    return normalized


def _asset_path(asset: dict) -> Path:
    return config.get_base_models_dir() / "sha256" / (asset["sha256"].lower() + ".gguf")


def _valid(path: Path, asset: dict) -> bool:
    return cache._valid_gguf_file(
        path, expected_size=asset["size_bytes"], expected_sha256=asset["sha256"],
    )


def get_cached_vision_asset_paths(runtime_manifest: dict) -> VisionAssetPaths | None:
    """Return both verified content-addressed files, or None; never download."""
    local = _manifest(runtime_manifest)["local_sdk"]
    base, projector = local["base_model"], local["vision"]["projector"]
    paths = VisionAssetPaths(_asset_path(base), _asset_path(projector))
    if _valid(paths.base_model, base) and _valid(paths.projector, projector):
        return paths
    return None


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError(f"Unsafe vision asset cache directory: {path}")


def _adopt_existing(asset: dict, destination: Path) -> bool:
    source = config.get_base_models_dir() / asset["file"]
    if not _valid(source, asset):
        return False
    # The content-addressed path remains stable if a legacy downloader later
    # atomically replaces its filename-based entry. Do not move/remove that entry.
    fd, name = tempfile.mkstemp(prefix=".adopt-", suffix=".tmp", dir=str(destination.parent))
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.unlink()
        try:
            os.link(str(source), str(temporary))
        except OSError:
            shutil.copyfile(str(source), str(temporary))
        if not _valid(temporary, asset):
            raise RuntimeError("Cached vision asset changed while being adopted.")
        os.replace(str(temporary), str(destination))
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _download_verified(
    asset: dict, destination: Path, stage: str, runtime_id: str,
    progress: ProgressCallback | None,
) -> None:
    url = asset.get("url") or cache._build_hf_url(asset["repo"], asset["file"])
    expected_size = asset["size_bytes"]
    fd, name = tempfile.mkstemp(prefix=".download-", suffix=".tmp", dir=str(destination.parent))
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as output:
            digest = hashlib.sha256()
            size = 0
            magic = b""
            with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    size += len(chunk)
                    if size > expected_size:
                        raise RuntimeError(f"{stage} download exceeds its declared size.")
                    magic += chunk[:max(0, len(cache.GGUF_MAGIC) - len(magic))]
                    digest.update(chunk)
                    output.write(chunk)
                    if progress is not None:
                        progress({
                            "stage": stage, "status": "downloading",
                            "runtime_id": runtime_id, "path": str(destination),
                            "downloaded_bytes": size, "total_bytes": expected_size,
                        })
            if (
                size != expected_size or magic != cache.GGUF_MAGIC
                or digest.hexdigest() != asset["sha256"].lower()
            ):
                raise RuntimeError(
                    f"{stage} download failed GGUF magic, file size, or SHA-256 validation."
                )
            output.flush()
            os.fsync(output.fileno())
        # Only a fully verified file becomes visible at its shared cache path.
        os.replace(str(temporary), str(destination))
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_asset(
    asset: dict, stage: str, runtime_id: str, *,
    offline: bool, progress: ProgressCallback | None,
) -> Path:
    destination = _asset_path(asset)

    def report(status: str, message: str | None = None) -> None:
        report_progress(progress, {
            "stage": stage, "status": status,
            "runtime_id": runtime_id, "path": str(destination),
        }, message)

    if _valid(destination, asset):
        report("cached")
        return destination

    _ensure_directory(destination.parent)
    lock_dir = cache._locks_dir() / "vision_assets"
    _ensure_directory(lock_dir)
    lock = lock_dir / (asset["sha256"].lower() + ".lock")
    if os.path.lexists(lock) and not stat.S_ISREG(lock.lstat().st_mode):
        raise ValueError(f"Unsafe vision asset cache lock: {lock}")
    with cache._cross_process_lock(lock):
        if _valid(destination, asset) or _adopt_existing(asset, destination):
            report("cached")
            return destination
        if offline:
            raise RuntimeError(
                f"{stage} for runtime {runtime_id!r} is missing or invalid; "
                "offline mode prohibits network downloads."
            )
        report("downloading", f"Downloading {stage} for {runtime_id} (one-time download)...")
        _download_verified(asset, destination, stage, runtime_id, progress)
        report("ready", f"Saved to {destination}")
        return destination


def get_vision_asset_paths(
    runtime_manifest: dict, *, offline: bool = False,
    progress: ProgressCallback | None = None,
) -> VisionAssetPaths:
    """Resolve both exact files, reusing verified cache entries before downloading.

    Explicit offline mode and PAW_OFFLINE prohibit network access even if only
    one asset is missing. A completed asset is retained when the other fails,
    so retries fetch only what is still missing. This does not enable inference.
    """
    manifest = _manifest(runtime_manifest)
    local = manifest["local_sdk"]
    offline = offline or os.environ.get("PAW_OFFLINE", "").strip().lower() in (
        "1", "true", "yes",
    )
    base = _resolve_asset(
        local["base_model"], "base_model", manifest["runtime_id"],
        offline=offline, progress=progress,
    )
    projector = _resolve_asset(
        local["vision"]["projector"], "vision_projector", manifest["runtime_id"],
        offline=offline, progress=progress,
    )
    return VisionAssetPaths(base, projector)

