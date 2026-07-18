"""
HTTP client for the PAW API.

Handles compilation, program download, and authentication.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Optional, TypedDict, cast

import httpx

from . import config
from ._output import ProgressCallback, report_progress

MAX_PAW_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_PAW_ARCHIVE_MEMBERS = 256
MAX_PAW_EXPANDED_BYTES = 512 * 1024 * 1024


@dataclass
class Program:
    """Result of a compilation."""
    id: str
    status: str
    slug: Optional[str] = None
    compiler_snapshot: Optional[str] = None
    compiler_kind: Optional[str] = None
    pseudo_program_strategy: Optional[str] = None
    runtime_id: Optional[str] = None
    runtime_manifest_version: Optional[int] = None
    timings: Optional[dict] = None
    error: Optional[str] = None
    version: Optional[int] = None
    version_action: Optional[str] = None


class CompilePrecheck(TypedDict):
    """Result returned by :meth:`PAWClient.precheck_compile`."""

    cached: bool
    program_id: str | None
    compiler_snapshot: str | None
    compiler_kind: str | None
    queue_length: int | None
    estimated_wait_s: float | None


class CompileJob(TypedDict):
    """Queued compile returned by :meth:`PAWClient.compile_async`."""

    job_id: str
    status: str
    program_id: str | None
    compiler_snapshot: str | None
    compiler_kind: str | None
    pseudo_program: str | None
    pseudo_program_strategy: str | None
    runtime_id: str | None
    runtime_manifest_version: int | None
    queue_position: int | None
    estimated_wait_s: float | None
    timings: dict | None
    error: str | None
    cached: bool
    slug: str | None
    version: int | None
    version_action: str | None


class CompileStatus(TypedDict):
    """Current state returned by :meth:`PAWClient.get_compile_status`."""

    job_id: str
    status: str
    program_id: str | None
    compiler_snapshot: str | None
    compiler_kind: str | None
    error: str | None
    created_at: str | None
    completed_at: str | None
    percent: float | None
    pseudo_program: str | None
    pseudo_program_strategy: str | None
    runtime_id: str | None
    runtime_manifest_version: int | None
    cached: bool
    queue_length: int | None
    estimated_wait_s: float | None
    slug: str | None
    version: int | None
    version_action: str | None


class CompileCancellation(TypedDict):
    """Cancellation acknowledgement returned by the compile API."""

    job_id: str
    status: str


class PAWClient:
    """HTTP client for the PAW API."""

    def __init__(self, api_url: str | None = None, api_key: str | None = None):
        self._api_url = (api_url or config.get_api_url()).rstrip("/")
        self._api_key = api_key or config.get_api_key()

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["X-API-Key"] = self._api_key
        return h

    @staticmethod
    def _compile_body(
        spec: str,
        compiler: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        public: bool = True,
        slug: str | None = None,
        ephemeral: bool = False,
    ) -> dict:
        """Build the shared request body for sync and async compilation."""
        body: dict = {"spec": spec, "public": public}
        if compiler:
            body["compiler"] = compiler
        if ephemeral:
            body["ephemeral"] = True
        if name:
            body["name"] = name
        if tags:
            body["tags"] = tags
        if slug:
            body["slug"] = slug
        return body

    def compile(
        self,
        spec: str,
        compiler: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        public: bool = True,
        slug: str | None = None,
        ephemeral: bool = False,
    ) -> Program:
        """Compile a spec into a neural program on the server.

        Args:
            spec: Natural language specification. Include examples in the text.
            compiler: Compiler name (alias or snapshot).
            name: Human-readable program name (display title).
            tags: Tags for hub discovery.
            public: Whether to list on the public hub.
            slug: URL-safe handle (e.g. 'message-classifier'). Creates a
                ``username/slug`` alias for easy reference. Requires auth.

        Returns:
            Program with id, slug, status, and timings.

        Raises:
            httpx.HTTPStatusError: On API errors (422 for validation, 429 for rate limit).
        """
        body = self._compile_body(
            spec,
            compiler=compiler,
            name=name,
            tags=tags,
            public=public,
            slug=slug,
            ephemeral=ephemeral,
        )

        resp = httpx.post(
            f"{self._api_url}/api/v1/compile",
            json=body,
            headers=self._headers(),
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()

        return Program(
            id=data.get("program_id", ""),
            status=data.get("status", "unknown"),
            slug=data.get("slug"),
            compiler_snapshot=data.get("compiler_snapshot"),
            compiler_kind=data.get("compiler_kind"),
            pseudo_program_strategy=data.get("pseudo_program_strategy"),
            runtime_id=data.get("runtime_id"),
            runtime_manifest_version=data.get("runtime_manifest_version"),
            timings=data.get("timings"),
            error=data.get("error"),
            version=data.get("version"),
            version_action=data.get("version_action"),
        )

    def precheck_compile(
        self,
        spec: str,
        compiler: str | None = None,
    ) -> CompilePrecheck:
        """Check cache and queue state without submitting a compile."""
        body: dict = {"spec": spec}
        if compiler:
            body["compiler"] = compiler
        resp = httpx.post(
            f"{self._api_url}/api/v1/compile/precheck",
            json=body,
            headers=self._headers(),
            timeout=10.0,
        )
        resp.raise_for_status()
        return cast(CompilePrecheck, resp.json())

    def compile_async(
        self,
        spec: str,
        compiler: str,
        name: str | None = None,
        tags: list[str] | None = None,
        public: bool = True,
        slug: str | None = None,
        ephemeral: bool = False,
    ) -> CompileJob:
        """Queue a long-running compile and return its job metadata.

        This is a synchronous HTTP call that submits work to the asynchronous
        compile endpoint. Poll :meth:`get_compile_status` for progress.
        ``compiler`` is required because this endpoint only accepts explicit
        finetune compilers.
        """
        if not isinstance(compiler, str) or not compiler.strip():
            raise ValueError(
                "compile_async requires an explicit finetune compiler."
            )
        body = self._compile_body(
            spec,
            compiler=compiler,
            name=name,
            tags=tags,
            public=public,
            slug=slug,
            ephemeral=ephemeral,
        )
        resp = httpx.post(
            f"{self._api_url}/api/v1/compile/async",
            json=body,
            headers=self._headers(),
            timeout=30.0,
        )
        resp.raise_for_status()
        return cast(CompileJob, resp.json())

    def get_compile_status(self, job_id: str) -> CompileStatus:
        """Get live status for an asynchronous compile job."""
        resp = httpx.get(
            f"{self._api_url}/api/v1/compile/{job_id}",
            headers=self._headers(),
            timeout=10.0,
        )
        resp.raise_for_status()
        return cast(CompileStatus, resp.json())

    def cancel_compile(self, job_id: str) -> CompileCancellation:
        """Cancel a queued compile job.

        The server returns HTTP 409 if the job has already started.
        """
        resp = httpx.delete(
            f"{self._api_url}/api/v1/compile/{job_id}",
            headers=self._headers(),
            timeout=10.0,
        )
        resp.raise_for_status()
        return cast(CompileCancellation, resp.json())

    def resolve_slug(self, slug: str) -> str:
        """Resolve a human-readable slug to a program ID."""
        resp = httpx.get(
            f"{self._api_url}/api/v1/programs/resolve/{slug}",
            headers=self._headers(),
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()["program_id"]

    def download_paw(
        self,
        program_id: str,
        progress: ProgressCallback | None = None,
    ) -> Path:
        """Download a .paw bundle to the local cache.

        Returns the path to the extracted program directory.
        Handles 202 Accepted (assets still generating) and 302 redirects (HF CDN).
        """
        from . import cache

        if not isinstance(program_id, str) or not cache.is_program_id(program_id):
            raise ValueError(f"Invalid program ID: {program_id!r}")
        program_dir = config.get_programs_dir() / program_id
        if cache.has_valid_program_assets(program_id):
            report_progress(
                progress,
                {
                    "stage": "program",
                    "status": "cached",
                    "program_id": program_id,
                    "path": str(program_dir),
                },
            )
            return program_dir

        with cache.program_cache_lock(program_id):
            if cache.has_valid_program_assets(program_id):
                report_progress(
                    progress,
                    {
                        "stage": "program",
                        "status": "cached",
                        "program_id": program_id,
                        "path": str(program_dir),
                    },
                )
                return program_dir

            report_progress(
                progress,
                {
                    "stage": "program",
                    "status": "downloading",
                    "program_id": program_id,
                    "path": str(program_dir),
                },
                f"Downloading program {program_id[:12]}...",
            )
            max_wait = 60
            elapsed = 0
            waiting_logged = False
            programs_dir = config.get_programs_dir()
            staging_parent = programs_dir / ".staging"
            staging_parent.mkdir(parents=True, exist_ok=True)
            staging_root = None
            try:
                while elapsed < max_wait:
                    retry_after = None
                    downloaded = False
                    with httpx.stream(
                        "GET",
                        f"{self._api_url}/api/v1/programs/{program_id}/download",
                        headers=self._headers(),
                        timeout=60.0,
                        follow_redirects=True,
                    ) as resp:
                        if resp.status_code == 202:
                            if not waiting_logged:
                                report_progress(
                                    progress,
                                    {
                                        "stage": "program",
                                        "status": "waiting",
                                        "program_id": program_id,
                                        "path": str(program_dir),
                                    },
                                    "Waiting for program to be ready...",
                                )
                                waiting_logged = True
                            retry_after = max(
                                1,
                                int(resp.headers.get("Retry-After", "3")),
                            )
                        elif resp.status_code == 404:
                            try:
                                resp.read()
                                detail = resp.json().get("detail", "")
                            except Exception:
                                detail = getattr(resp, "text", "")
                            if "not found" in str(detail).lower():
                                raise RuntimeError(
                                    f"Program {program_id} not found on server."
                                )
                            if elapsed < max_wait - 3:
                                if not waiting_logged:
                                    report_progress(
                                        progress,
                                        {
                                            "stage": "program",
                                            "status": "waiting",
                                            "program_id": program_id,
                                            "path": str(program_dir),
                                        },
                                        "Waiting for program to be ready...",
                                    )
                                    waiting_logged = True
                                retry_after = 3
                            else:
                                resp.raise_for_status()
                        else:
                            resp.raise_for_status()
                            staging_root = Path(
                                tempfile.mkdtemp(
                                    prefix=f"{program_id}.",
                                    dir=str(staging_parent),
                                )
                            )
                            paw_path = staging_root / f"{program_id}.paw"
                            self._stream_response_to_file(resp, paw_path)
                            downloaded = True

                    if downloaded:
                        break
                    if retry_after is not None:
                        time.sleep(retry_after)
                        elapsed += retry_after
                        continue
                    raise RuntimeError(
                        f"Program {program_id} download did not return assets."
                    )
                else:
                    raise RuntimeError(
                        f"Program {program_id} assets not ready after {max_wait}s. "
                        "The program may still be generating. Try again shortly."
                    )

                if staging_root is None:
                    raise RuntimeError(
                        f"Program {program_id} download produced no archive."
                    )
                paw_path = staging_root / f"{program_id}.paw"
                extracted_dir = staging_root / "extracted"
                extracted_dir.mkdir()
                self._safe_extract_paw(paw_path, extracted_dir)

                if not cache.validate_program_assets_dir(
                    extracted_dir,
                    program_id,
                ):
                    raise RuntimeError(
                        f"Downloaded program {program_id} has missing, "
                        "malformed, or mismatched staged assets."
                    )

                self._hydrate_runtime_manifest(extracted_dir)
                if not cache.validate_program_assets_dir(
                    extracted_dir,
                    program_id,
                ):
                    raise RuntimeError(
                        f"Downloaded program {program_id} became invalid while "
                        "hydrating its runtime metadata."
                    )

                self._install_staged_program(
                    extracted_dir,
                    program_dir,
                    program_id,
                )
            finally:
                if staging_root is not None:
                    shutil.rmtree(staging_root, ignore_errors=True)

            if not cache.has_valid_program_assets(program_id):
                raise RuntimeError(
                    f"Installed program {program_id} failed strict asset "
                    "validation."
                )

        report_progress(
            progress,
            {
                "stage": "program",
                "status": "ready",
                "program_id": program_id,
                "path": str(program_dir),
            },
        )
        return program_dir

    @staticmethod
    def _validate_stream_content_length(resp) -> int | None:
        declared = resp.headers.get("content-length")
        if declared is None:
            declared = resp.headers.get("Content-Length")
        if declared is None:
            return None
        try:
            declared_size = int(declared)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid program archive Content-Length: {declared!r}"
            ) from exc
        if declared_size < 0 or declared_size > MAX_PAW_ARCHIVE_BYTES:
            raise ValueError(
                f"Program archive is too large: {declared_size} bytes "
                f"(limit {MAX_PAW_ARCHIVE_BYTES})."
            )
        return declared_size

    @staticmethod
    def _stream_response_to_file(resp, destination: Path) -> int:
        PAWClient._validate_stream_content_length(resp)
        downloaded = 0
        with open(destination, "xb") as output:
            for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                if not chunk:
                    continue
                next_size = downloaded + len(chunk)
                if next_size > MAX_PAW_ARCHIVE_BYTES:
                    raise ValueError(
                        f"Program archive is too large: more than "
                        f"{MAX_PAW_ARCHIVE_BYTES} bytes."
                    )
                output.write(chunk)
                downloaded = next_size
            output.flush()
            os.fsync(output.fileno())
        return downloaded

    @staticmethod
    def _safe_extract_paw(paw_path: Path, destination: Path) -> None:
        """Extract regular archive members without traversal or symlinks."""
        if paw_path.stat().st_size > MAX_PAW_ARCHIVE_BYTES:
            raise ValueError(
                f"Program archive exceeds {MAX_PAW_ARCHIVE_BYTES} bytes."
            )
        seen: set[str] = set()
        try:
            archive = zipfile.ZipFile(paw_path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise RuntimeError(f"Invalid .paw archive: {exc}") from exc

        with archive:
            members = archive.infolist()
            if len(members) > MAX_PAW_ARCHIVE_MEMBERS:
                raise ValueError(
                    f"Program archive has {len(members)} members "
                    f"(limit {MAX_PAW_ARCHIVE_MEMBERS})."
                )
            expanded_size = sum(info.file_size for info in members)
            if expanded_size > MAX_PAW_EXPANDED_BYTES:
                raise ValueError(
                    f"Program archive expands to {expanded_size} bytes "
                    f"(limit {MAX_PAW_EXPANDED_BYTES})."
                )

            for info in members:
                member = info.filename
                pure_path = PurePosixPath(member)
                parts = pure_path.parts
                if (
                    not member
                    or "\\" in member
                    or pure_path.is_absolute()
                    or not parts
                    or ":" in parts[0]
                    or any(part in ("", ".", "..") for part in parts)
                ):
                    raise ValueError(
                        f"Unsafe path in .paw archive: {member!r}"
                    )
                normalized = "/".join(parts)
                if normalized in seen:
                    raise ValueError(
                        f"Duplicate path in .paw archive: {member!r}"
                    )
                seen.add(normalized)

                unix_mode = (info.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(unix_mode)
                if file_type and not (
                    stat.S_ISREG(unix_mode) or stat.S_ISDIR(unix_mode)
                ):
                    raise ValueError(
                        f"Non-regular path in .paw archive: {member!r}"
                    )

                target = destination.joinpath(*parts)
                try:
                    target.relative_to(destination)
                except ValueError as exc:
                    raise ValueError(
                        f"Unsafe path in .paw archive: {member!r}"
                    ) from exc

                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=False)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, open(
                    target,
                    "xb",
                ) as output:
                    shutil.copyfileobj(source, output)

    @staticmethod
    def _install_staged_program(
        staged_dir: Path,
        program_dir: Path,
        program_id: str,
    ) -> None:
        """Atomically publish a complete staged directory under its ID."""
        parent = program_dir.parent
        backup = parent / (
            f".{program_id}.replaced.{os.getpid()}.{time.time_ns()}"
        )
        moved_old = False
        try:
            if program_dir.exists() or program_dir.is_symlink():
                os.replace(str(program_dir), str(backup))
                moved_old = True
            os.replace(str(staged_dir), str(program_dir))
        except BaseException:
            if moved_old and not program_dir.exists() and backup.exists():
                os.replace(str(backup), str(program_dir))
            raise
        else:
            if moved_old:
                if backup.is_dir() and not backup.is_symlink():
                    shutil.rmtree(backup, ignore_errors=True)
                else:
                    backup.unlink(missing_ok=True)

    def _hydrate_runtime_manifest(self, program_dir: Path) -> None:
        from . import cache

        meta_path = program_dir / "meta.json"
        if not meta_path.is_file() or meta_path.is_symlink():
            return

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(meta, dict):
            return

        runtime_manifest = cache.resolve_runtime_manifest(
            meta,
            api_url=self._api_url,
            api_key=self._api_key,
            offline=False,
        )
        if runtime_manifest is None:
            return

        meta["runtime"] = runtime_manifest
        meta.setdefault(
            "runtime_id",
            runtime_manifest.get("runtime_id"),
        )
        meta.setdefault(
            "runtime_manifest_version",
            runtime_manifest.get("manifest_version"),
        )
        cache._atomic_write_json(meta_path, meta)

    def get_program_meta(self, program_id: str) -> dict:
        """Get program metadata from the server."""
        resp = httpx.get(
            f"{self._api_url}/api/v1/programs/{program_id}",
            headers=self._headers(),
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()

    def get_runtime_manifest(self, runtime_id: str) -> dict:
        """Fetch a runtime manifest from the server and cache it locally."""
        from . import cache

        cached = cache.get_cached_runtime_manifest(runtime_id)
        if cached:
            return cached

        return cache.fetch_runtime_manifest(
            runtime_id,
            api_url=self._api_url,
            api_key=self._api_key,
        )

    def list_compilers(self) -> list[dict]:
        """List available compilers from the server."""
        resp = httpx.get(
            f"{self._api_url}/api/v1/models/compilers",
            headers=self._headers(),
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()["compilers"]

    def list_slug_versions(self, slug: str) -> dict:
        """List all versions of a slug. Slug format: 'username/slug-name' or bare 'slug-name'."""
        resp = httpx.get(
            f"{self._api_url}/api/v1/programs/{slug}/versions",
            headers=self._headers(),
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()

    def list_programs(self, sort: str = "recent", per_page: int = 20, page: int = 1) -> dict:
        """List programs for the authenticated user."""
        resp = httpx.get(
            f"{self._api_url}/api/v1/programs",
            params={"mine": "true", "sort": sort, "per_page": per_page, "page": page},
            headers=self._headers(),
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()
