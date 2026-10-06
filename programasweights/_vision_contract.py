"""Inert v2 runtime metadata for local text/image functions.

The existing prompt_template.txt member holds literal system instructions.
Each call appends one user message containing its ordered text/image parts;
there are no string placeholders or named image slots. The qwen3.5 chat
profile adds the assistant generation prompt with thinking disabled.

Preprocessing v1 uses native input dimensions, RGB with transparency over
white, stored pixel orientation, and no color-profile transform. Resizing and
patch construction belong to the backend, with explicit image-token limits.
This contract describes assets and rendering; it does not load or execute them.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit


VISION_MANIFEST_VERSION = 2
_SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")
_INPUT = {"format": "content_parts", "types": ["text", "image"]}
_PROMPT = {
    "format": "chat_messages",
    "system_prompt_file": "prompt_template.txt",
    "chat_format": "qwen3.5",
    "enable_thinking": False,
}
_PREPROCESSING = {
    "version": 1,
    "color_mode": "RGB",
    "alpha": "composite_white",
    "orientation": "stored",
    "color_profile": "ignore",
    "resize": "backend",
}

# A base call is one user message containing the ordered input parts. The
# pinned GGUF chat template adds the assistant prefix with thinking disabled;
# no system message or compiled prompt is inserted in adapter-free mode.
VISION_BASE_INFERENCE = {"contract_version": 1, "format": "chat_messages"}

BUILTIN_VISION_RUNTIMES = {
    "qwen3.5-0.8b-q8_0": {
        "runtime_id": "qwen3.5-0.8b-q8_0",
        "manifest_version": VISION_MANIFEST_VERSION,
        "interpreter": "Qwen/Qwen3.5-0.8B",
        "adapter_format": "gguf_lora",
        "input": dict(_INPUT),
        "prompt_template": dict(_PROMPT),
        "program_assets": {"adapter_filename": "adapter.gguf"},
        "base_inference": dict(VISION_BASE_INFERENCE),
        "local_sdk": {
            "supported": True,
            "n_ctx": 4096,
            "base_model": {
                "provider": "huggingface",
                "repo": "ggml-org/Qwen3.5-0.8B-GGUF",
                "file": "Qwen3.5-0.8B-Q8_0.gguf",
                "url": "https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/resolve/8fea620810c4afa23dd6443f999a48574c1611a3/Qwen3.5-0.8B-Q8_0.gguf",
                "size_bytes": 833592096,
                "sha256": "37ae482d336108d23516fa35e8e0c4126688d81018b87178a18d752a1357814f",
            },
            "vision": {
                "projector": {
                    "provider": "huggingface",
                    "repo": "unsloth/Qwen3.5-0.8B-GGUF",
                    "file": "mmproj-BF16.gguf",
                    "url": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0/mmproj-BF16.gguf",
                    "size_bytes": 207346528,
                    "sha256": "d312c4d02fd46eea7a16e4f3bbb58840e6222209322ca1e33ca03247ad8935d6",
                },
                "preprocessing": {
                    **_PREPROCESSING,
                    "image_min_tokens": 1,
                    "image_max_tokens": 512,
                },
            },
        },
        "js_sdk": {"supported": False},
    },
}


def _positive_int(value) -> bool:
    return type(value) is int and value > 0


def _asset_valid(asset) -> bool:
    if not isinstance(asset, dict):
        return False
    file = asset.get("file")
    if (
        not isinstance(file, str) or not file or file in (".", "..")
        or any(char in file for char in ("/", "\\", ":", "\x00"))
    ):
        return False
    digest = asset.get("sha256")
    if (
        not isinstance(digest, str) or not _SHA256.fullmatch(digest)
        or not _positive_int(asset.get("size_bytes"))
    ):
        return False
    provider, repo, url = (asset.get(key) for key in ("provider", "repo", "url"))
    if any(value is not None and not isinstance(value, str)
           for value in (provider, repo, url)):
        return False
    if url:
        try:
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https" or not parsed.hostname
                or parsed.username or parsed.password
            ):
                return False
        except ValueError:
            return False
    elif provider != "huggingface" or not repo:
        return False
    return True


def valid_vision_contract(manifest: dict) -> bool:
    """Validate v2's required fields after the common runtime identity checks."""
    if manifest.get("input") != _INPUT:
        return False
    prompt = manifest.get("prompt_template")
    if prompt != _PROMPT or prompt.get("enable_thinking") is not False:
        return False
    assets = manifest.get("program_assets")
    if (
        not isinstance(assets, dict) or assets.get("adapter_filename") != "adapter.gguf"
        or assets.get("prefix_cache_required", False) is not False
        or assets.get("prefix_cache_filename") is not None
        or assets.get("prefix_tokens_filename") is not None
    ):
        return False
    local = manifest.get("local_sdk")
    if (
        not isinstance(local, dict) or not isinstance(local.get("supported"), bool)
        or not _positive_int(local.get("n_ctx"))
    ):
        return False
    vision = local.get("vision")
    if not isinstance(vision, dict) or set(vision) != {"projector", "preprocessing"}:
        return False
    if not _asset_valid(local.get("base_model")) or not _asset_valid(vision.get("projector")):
        return False
    if local["base_model"]["file"].casefold() == vision["projector"]["file"].casefold():
        return False
    preprocessing = vision.get("preprocessing")
    if not isinstance(preprocessing, dict):
        return False
    if set(preprocessing) != set(_PREPROCESSING) | {"image_min_tokens", "image_max_tokens"}:
        return False
    if any(preprocessing.get(key) != value for key, value in _PREPROCESSING.items()):
        return False
    if type(preprocessing["version"]) is not int:
        return False
    minimum, maximum = (preprocessing[key] for key in ("image_min_tokens", "image_max_tokens"))
    if not _positive_int(minimum) or not _positive_int(maximum) or minimum > maximum:
        return False
    base = manifest.get("base_inference")
    if base is not None and (
        base != VISION_BASE_INFERENCE or type(base.get("contract_version")) is not int
    ):
        return False
    js = manifest.get("js_sdk")
    return js is None or (isinstance(js, dict) and js.get("supported") is False)


def declares_vision(program_meta: dict) -> bool:
    """Recognize image intent, including malformed or downgraded manifests."""
    if program_meta.get("runtime_manifest_version") == VISION_MANIFEST_VERSION:
        return True
    runtime = program_meta.get("runtime")
    if not isinstance(runtime, dict):
        return False
    local = runtime.get("local_sdk")
    prompt = runtime.get("prompt_template")
    return (
        runtime.get("manifest_version") == VISION_MANIFEST_VERSION
        or "input" in runtime
        or (isinstance(local, dict) and "vision" in local)
        or (isinstance(prompt, dict) and prompt.get("format") == "chat_messages")
    )


def require_text_execution(program_meta: dict) -> None:
    """Prevent validated image metadata from reaching the legacy text runtime."""
    if declares_vision(program_meta):
        raise ValueError(
            "Image programs require the vision runtime; load them with paw.function()."
        )
