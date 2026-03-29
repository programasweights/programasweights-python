"""
Local cache management for base models and compiled programs.

Cache structure:
    ~/.cache/programasweights/
        base_models/
            qwen3-0.6b-q6k.gguf       # ~623 MB, downloaded once
        programs/
            <program_id>/
                adapter.gguf            # ~23 MB, Q4_0 LoRA
                prompt_template.txt
                meta.json
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx

from . import config

BASE_MODEL_URLS = {
    "qwen3-0.6b-q6_k": "https://huggingface.co/yuntian-deng/Qwen3-0.6B-GGUF-Q6_K/resolve/main/qwen3-0.6b-q6_k.gguf",
    "gpt2-q6_k": "https://huggingface.co/yuntian-deng/GPT2-GGUF-Q6_K/resolve/main/gpt2-q6_k.gguf",
}

INTERPRETER_TO_GGUF = {
    "Qwen/Qwen3-0.6B": "qwen3-0.6b-q6_k",
    "gpt2": "gpt2-q6_k",
}


def get_base_model_path(interpreter: str = "Qwen/Qwen3-0.6B") -> Path:
    """Get the path to the base model GGUF, downloading if needed."""
    gguf_name = INTERPRETER_TO_GGUF.get(interpreter, "qwen3-0.6b-q6k")
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
    """Download a file with progress indication."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=300.0) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    mb = downloaded / 1024 / 1024
                    print(f"\r  {mb:.1f} MB ({pct:.0f}%)", end="", flush=True)
        print()
