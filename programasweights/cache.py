"""
Local cache management for base models and compiled programs.

Cache structure:
    ~/.cache/programasweights/
        base_models/
            qwen3-0.6b-q6_k.gguf      # ~594 MB, downloaded once
            gpt2-q8_0.gguf            # ~134 MB, downloaded once
        programs/
            <program_id>/
                adapter.gguf            # ~23 MB, Q4_0 LoRA
                prompt_template.txt
                meta.json
        slug_cache.json                 # slug -> program_id mapping
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

from . import config

BASE_MODEL_URLS = {
    "qwen3-0.6b-q6_k": "https://huggingface.co/programasweights/Qwen3-0.6B-GGUF-Q6_K/resolve/main/qwen3-0.6b-q6_k.gguf",
    "gpt2-q8_0": "https://huggingface.co/programasweights/GPT2-GGUF-Q8_0/resolve/main/gpt2-q8_0.gguf",
}

INTERPRETER_TO_GGUF = {
    "Qwen/Qwen3-0.6B": "qwen3-0.6b-q6_k",
    "gpt2": "gpt2-q8_0",
}


LEGACY_RUNTIME_MANIFESTS = {
    "qwen3-0.6b-q6_k": {
        "runtime_id": "qwen3-0.6b-q6_k",
        "manifest_version": 1,
        "display_name": "Qwen3 0.6B (Q6_K)",
        "interpreter": "Qwen/Qwen3-0.6B",
        "adapter_format": "gguf_lora",
        "local_sdk": {
            "supported": True,
            "base_model": {
                "provider": "huggingface",
                "repo": "programasweights/Qwen3-0.6B-GGUF-Q6_K",
                "file": "qwen3-0.6b-q6_k.gguf",
                "url": BASE_MODEL_URLS["qwen3-0.6b-q6_k"],
                "sha256": None,
            },
            "n_ctx": 2048,
        },
        "js_sdk": {
            "supported": False,
            "base_model": None,
            "prefix_cache_supported": False,
        },
    },
    "gpt2-q8_0": {
        "runtime_id": "gpt2-q8_0",
        "manifest_version": 1,
        "display_name": "GPT-2 124M (Q8_0)",
        "interpreter": "gpt2",
        "adapter_format": "gguf_lora",
        "local_sdk": {
            "supported": True,
            "base_model": {
                "provider": "huggingface",
                "repo": "programasweights/GPT2-GGUF-Q8_0",
                "file": "gpt2-q8_0.gguf",
                "url": BASE_MODEL_URLS["gpt2-q8_0"],
                "sha256": None,
            },
            "n_ctx": 2048,
        },
        "js_sdk": {
            "supported": True,
            "base_model": {
                "provider": "huggingface",
                "repo": "programasweights/GPT2-GGUF-Q8_0",
                "file": "gpt2-q8_0.gguf",
                "url": BASE_MODEL_URLS["gpt2-q8_0"],
                "sha256": None,
            },
            "prefix_cache_supported": True,
        },
    },
}


def _runtime_cache_dir() -> Path:
    d = config.get_cache_dir() / "runtimes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _runtime_manifest_path(runtime_id: str) -> Path:
    return _runtime_cache_dir() / f"{runtime_id}.json"


def get_cached_runtime_manifest(runtime_id: str) -> dict | None:
    path = _runtime_manifest_path(runtime_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def save_runtime_manifest(runtime_manifest: dict) -> None:
    runtime_id = runtime_manifest.get("runtime_id")
    if not runtime_id:
        return
    path = _runtime_manifest_path(runtime_id)
    path.write_text(json.dumps(runtime_manifest, indent=2))


def _legacy_runtime_manifest(interpreter: str | None) -> dict | None:
    if not interpreter:
        return None
    runtime_id = INTERPRETER_TO_GGUF.get(interpreter)
    if not runtime_id:
        return None
    manifest = LEGACY_RUNTIME_MANIFESTS.get(runtime_id)
    return json.loads(json.dumps(manifest)) if manifest else None


def _is_runtime_manifest_complete(runtime_manifest: dict | None) -> bool:
    if not runtime_manifest:
        return False
    base_model = runtime_manifest.get("local_sdk", {}).get("base_model")
    return bool(runtime_manifest.get("runtime_id") and base_model and base_model.get("file"))


def fetch_runtime_manifest(
    runtime_id: str,
    api_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    base_url = (api_url or config.get_api_url()).rstrip("/")
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
    resp = httpx.get(
        f"{base_url}/api/v1/models/runtimes/{runtime_id}",
        headers=headers,
        timeout=10.0,
    )
    resp.raise_for_status()
    runtime_manifest = resp.json()
    save_runtime_manifest(runtime_manifest)
    return runtime_manifest


def resolve_runtime_manifest(
    program_meta: dict,
    api_url: str | None = None,
    api_key: str | None = None,
    offline: bool = False,
) -> dict | None:
    embedded = program_meta.get("runtime")
    if isinstance(embedded, dict) and _is_runtime_manifest_complete(embedded):
        save_runtime_manifest(embedded)
        return embedded

    runtime_id = program_meta.get("runtime_id")
    if runtime_id:
        cached = get_cached_runtime_manifest(runtime_id)
        if _is_runtime_manifest_complete(cached):
            return cached
        if not offline:
            try:
                return fetch_runtime_manifest(runtime_id, api_url=api_url, api_key=api_key)
            except Exception:
                pass

    return _legacy_runtime_manifest(program_meta.get("interpreter"))


def _base_model_info_from_runtime(runtime_manifest: dict) -> dict | None:
    return runtime_manifest.get("local_sdk", {}).get("base_model")


def _build_hf_url(repo: str, file_name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{file_name}"


def get_base_model_path(
    interpreter: str = "Qwen/Qwen3-0.6B",
    runtime_manifest: dict | None = None,
) -> Path:
    """Get the path to the base model GGUF, downloading if needed."""
    if runtime_manifest:
        local_sdk = runtime_manifest.get("local_sdk", {})
        if not local_sdk.get("supported", False):
            raise ValueError(
                f"Runtime '{runtime_manifest.get('runtime_id', interpreter)}' is not supported by the local SDK."
            )

        base_model = _base_model_info_from_runtime(runtime_manifest)
        if base_model:
            file_name = base_model.get("file")
            if not file_name:
                raise ValueError(
                    f"Runtime '{runtime_manifest.get('runtime_id', interpreter)}' is missing a base model file."
                )

            gguf_path = config.get_base_models_dir() / file_name
            if gguf_path.exists():
                return gguf_path

            url = base_model.get("url")
            if not url and base_model.get("provider") == "huggingface":
                url = _build_hf_url(base_model["repo"], file_name)
            if not url:
                raise ValueError(
                    f"Runtime '{runtime_manifest.get('runtime_id', interpreter)}' is missing a downloadable base model URL."
                )

            from ._output import status
            label = runtime_manifest.get("display_name") or runtime_manifest.get("runtime_id") or interpreter
            status(f"Downloading interpreter {label} (one-time download)...")
            _download_file(url, gguf_path)
            status(f"Saved to {gguf_path}")
            return gguf_path

    gguf_name = INTERPRETER_TO_GGUF.get(interpreter)
    if not gguf_name:
        raise ValueError(
            f"Unknown interpreter: '{interpreter}'. "
            f"Supported: {list(INTERPRETER_TO_GGUF.keys())}"
        )
    gguf_path = config.get_base_models_dir() / f"{gguf_name}.gguf"

    if gguf_path.exists():
        return gguf_path

    url = BASE_MODEL_URLS.get(gguf_name)
    if not url:
        raise ValueError(
            f"Unknown interpreter model: {interpreter}. "
            f"Available: {list(INTERPRETER_TO_GGUF.keys())}"
        )

    from ._output import status
    status(f"Downloading interpreter {gguf_name} (one-time download)...")
    _download_file(url, gguf_path)
    status(f"Saved to {gguf_path}")
    return gguf_path


def get_program_dir(program_id: str) -> Path:
    """Get the local cache directory for a program."""
    return config.get_programs_dir() / program_id


def is_program_cached(program_id: str) -> bool:
    """Check if a program's artifacts are already cached locally."""
    d = get_program_dir(program_id)
    return (d / "adapter.gguf").exists() and (d / "prompt_template.txt").exists()


def _download_file(url: str, dest: Path):
    """Download a file atomically with progress indication."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(tmp, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        mb = downloaded / 1024 / 1024
                        from ._output import status_inline, status_end
                        status_inline(f"  {mb:.1f} MB ({pct:.0f}%)")
            from ._output import status_end
            status_end()
        os.replace(str(tmp), str(dest))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _slug_cache_path() -> Path:
    return config.get_cache_dir() / "slug_cache.json"


def get_cached_slug(slug: str) -> str | None:
    """Look up a slug in the local cache. Returns program_id or None."""
    path = _slug_cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        program_id = data.get(slug)
        if program_id and is_program_cached(program_id):
            return program_id
    except (json.JSONDecodeError, OSError):
        pass
    return None


def save_slug_mapping(slug: str, program_id: str) -> None:
    """Save a slug -> program_id mapping to the local cache."""
    path = _slug_cache_path()
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    data[slug] = program_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
