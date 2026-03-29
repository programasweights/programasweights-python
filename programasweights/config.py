"""
Configuration management for the PAW SDK.

Stores API key and settings in ~/.config/programasweights/config.json.
All settings can be overridden via environment variables (PAW_API_URL, PAW_API_KEY).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_DEFAULT_API_URL = "https://programasweights.com"
_CONFIG_DIR = Path.home() / ".config" / "programasweights"
_CONFIG_FILE = _CONFIG_DIR / "config.json"
_CACHE_DIR = Path.home() / ".cache" / "programasweights"


def _load_config() -> dict:
    if _CONFIG_FILE.exists():
        with open(_CONFIG_FILE) as f:
            return json.load(f)
    return {}


def _save_config(config: dict):
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_api_url() -> str:
    return os.environ.get("PAW_API_URL") or _load_config().get("api_url") or _DEFAULT_API_URL


def get_api_key() -> str | None:
    return os.environ.get("PAW_API_KEY") or _load_config().get("api_key")


def set_api_key(key: str):
    config = _load_config()
    config["api_key"] = key
    _save_config(config)


def get_cache_dir() -> Path:
    d = Path(os.environ.get("PAW_CACHE_DIR", str(_CACHE_DIR)))
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_base_models_dir() -> Path:
    d = get_cache_dir() / "base_models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_programs_dir() -> Path:
    d = get_cache_dir() / "programs"
    d.mkdir(parents=True, exist_ok=True)
    return d
