"""
Local cache management for base models and compiled programs.

Cache structure:
    ~/.cache/programasweights/
        base_models/
            qwen3-0.6b-q6_k.gguf      # ~594 MB, downloaded once
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
    "gpt2-q6_k": "https://huggingface.co/programasweights/GPT2-GGUF-Q6_K/resolve/main/gpt2-q6_k.gguf",
}

INTERPRETER_TO_GGUF = {
    "Qwen/Qwen3-0.6B": "qwen3-0.6b-q6_k",
    "gpt2": "gpt2-q6_k",
}


def get_base_model_path(interpreter: str = "Qwen/Qwen3-0.6B") -> Path:
    """Get the path to the base model GGUF, downloading if needed."""
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

    print(f"Downloading base model {gguf_name} (one-time download)...")
    _download_file(url, gguf_path)
    print(f"Saved to {gguf_path}")
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
                        print(f"\r  {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)
            print()
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
