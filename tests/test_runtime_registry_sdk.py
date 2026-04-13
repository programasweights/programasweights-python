from __future__ import annotations

import json
from pathlib import Path

import pytest

from programasweights import cache


RUNTIME = {
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
            "url": "https://example.com/gpt2-q8_0.gguf",
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
            "url": "https://example.com/gpt2-q8_0.gguf",
            "sha256": None,
        },
        "prefix_cache_supported": True,
    },
}


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("PAW_CACHE_DIR", str(tmp_path / "cache"))


def test_resolve_runtime_manifest_prefers_embedded_manifest():
    meta = {
        "interpreter": "gpt2",
        "runtime_id": "gpt2-q8_0",
        "runtime": RUNTIME,
    }
    resolved = cache.resolve_runtime_manifest(meta, offline=True)
    assert resolved["runtime_id"] == "gpt2-q8_0"
    cached = cache.get_cached_runtime_manifest("gpt2-q8_0")
    assert cached["runtime_id"] == "gpt2-q8_0"


def test_resolve_runtime_manifest_uses_legacy_fallback_when_offline():
    meta = {
        "interpreter": "gpt2",
        "runtime_id": "gpt2-q8_0",
    }
    resolved = cache.resolve_runtime_manifest(meta, offline=True)
    assert resolved is not None
    assert resolved["runtime_id"] == "gpt2-q8_0"
    assert resolved["local_sdk"]["base_model"]["file"] == "gpt2-q8_0.gguf"


def test_get_base_model_path_uses_runtime_manifest(tmp_path, monkeypatch):
    target = tmp_path / "cache" / "base_models" / "gpt2-q8_0.gguf"

    def fake_download(url: str, dest: Path):
        assert url == "https://example.com/gpt2-q8_0.gguf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"gguf")

    monkeypatch.setattr(cache, "_download_file", fake_download)
    path = cache.get_base_model_path("gpt2", runtime_manifest=RUNTIME)
    assert path == target
    assert path.read_bytes() == b"gguf"
