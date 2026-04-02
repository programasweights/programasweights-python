"""
HTTP client for the PAW API.

Handles compilation, program download, and authentication.
"""

from __future__ import annotations

import json
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
    timings: Optional[dict] = None
    error: Optional[str] = None


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
        compiler: str = "paw-4b-qwen3-0.6b",
        name: str | None = None,
        tags: list[str] | None = None,
        public: bool = True,
        slug: str | None = None,
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
        body: dict = {"spec": spec, "compiler": compiler, "public": public}
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
            timings=data.get("timings"),
            error=data.get("error"),
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

        max_wait = 30
        elapsed = 0
        resp = None
        while elapsed < max_wait:
            resp = httpx.get(
                f"{self._api_url}/api/v1/programs/{program_id}/download",
                headers=self._headers(),
                timeout=60.0,
                follow_redirects=True,
            )
            if resp.status_code == 202:
                retry_after = int(resp.headers.get("Retry-After", "3"))
                time.sleep(retry_after)
                elapsed += retry_after
                continue
            if resp.status_code == 404 and elapsed < max_wait - 3:
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
        paw_path.write_bytes(resp.content)

        with zipfile.ZipFile(paw_path) as zf:
            zf.extractall(program_dir)

        return program_dir

    def get_program_meta(self, program_id: str) -> dict:
        """Get program metadata from the server."""
        resp = httpx.get(
            f"{self._api_url}/api/v1/programs/{program_id}",
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


_default_client: PAWClient | None = None


def get_client() -> PAWClient:
    global _default_client
    if _default_client is None:
        _default_client = PAWClient()
    return _default_client
