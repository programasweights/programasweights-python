"""
HTTP client for the PAW API.

Handles compilation, program download, and authentication.
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from . import config


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

    def resolve_slug(self, slug: str) -> str:
        """Resolve a human-readable slug to a program ID."""
        resp = httpx.get(
            f"{self._api_url}/api/v1/programs/resolve/{slug}",
            headers=self._headers(),
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()["program_id"]

    def download_paw(self, program_id: str) -> Path:
        """Download a .paw bundle to the local cache.

        Returns the path to the extracted program directory.
        Handles 202 Accepted (assets still generating) and 302 redirects (HF CDN).
        """
        program_dir = config.get_programs_dir() / program_id
        if (program_dir / "prompt_template.txt").exists():
            return program_dir

        from ._output import status

        status(f"Downloading program {program_id[:12]}...")
        max_wait = 60
        elapsed = 0
        resp = None
        waiting_logged = False
        while elapsed < max_wait:
            resp = httpx.get(
                f"{self._api_url}/api/v1/programs/{program_id}/download",
                headers=self._headers(),
                timeout=60.0,
                follow_redirects=True,
            )
            if resp.status_code == 202:
                if not waiting_logged:
                    status("Waiting for program to be ready...")
                    waiting_logged = True
                retry_after = int(resp.headers.get("Retry-After", "3"))
                time.sleep(retry_after)
                elapsed += retry_after
                continue
            if resp.status_code == 404:
                try:
                    detail = resp.json().get("detail", "")
                except Exception:
                    detail = resp.text
                if "not found" in detail.lower():
                    raise RuntimeError(f"Program {program_id} not found on server.")
                if elapsed < max_wait - 3:
                    if not waiting_logged:
                        status("Waiting for program to be ready...")
                        waiting_logged = True
                    time.sleep(3)
                    elapsed += 3
                    continue
            resp.raise_for_status()
            break
        else:
            raise RuntimeError(
                f"Program {program_id} assets not ready after {max_wait}s. "
                "The program may still be generating. Try again shortly."
            )

        program_dir.mkdir(parents=True, exist_ok=True)
        paw_path = program_dir / f"{program_id}.paw"
        tmp_path = paw_path.with_suffix(".paw.tmp")
        tmp_path.write_bytes(resp.content)
        os.replace(str(tmp_path), str(paw_path))

        with zipfile.ZipFile(paw_path) as zf:
            for member in zf.namelist():
                if os.path.isabs(member) or ".." in member.split("/"):
                    raise ValueError(f"Unsafe path in .paw archive: {member}")
            zf.extractall(program_dir)

        self._hydrate_runtime_manifest(program_dir)

        return program_dir

    def _hydrate_runtime_manifest(self, program_dir: Path) -> None:
        meta_path = program_dir / "meta.json"
        if not meta_path.exists():
            return

        try:
            meta = json.loads(meta_path.read_text())
        except (json.JSONDecodeError, OSError):
            return

        runtime_id = meta.get("runtime_id")
        runtime = meta.get("runtime")
        if isinstance(runtime, dict) and runtime.get("runtime_id") and runtime.get("local_sdk", {}).get("base_model"):
            return
        if not runtime_id:
            return

        try:
            runtime_manifest = self.get_runtime_manifest(runtime_id)
        except Exception:
            return

        meta["runtime"] = runtime_manifest
        meta.setdefault("runtime_manifest_version", runtime_manifest.get("manifest_version"))
        meta_path.write_text(json.dumps(meta, indent=2))

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
