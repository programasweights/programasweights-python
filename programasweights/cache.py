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

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TypedDict

import httpx

from . import config
from ._output import ProgressCallback, report_progress

BASE_MODEL_URLS = {
    "qwen3-0.6b-q6_k": "https://huggingface.co/programasweights/Qwen3-0.6B-GGUF-Q6_K/resolve/main/qwen3-0.6b-q6_k.gguf",
    "gpt2-q8_0": "https://huggingface.co/programasweights/GPT2-GGUF-Q8_0/resolve/main/gpt2-q8_0.gguf",
}

INTERPRETER_TO_GGUF = {
    "Qwen/Qwen3-0.6B": "qwen3-0.6b-q6_k",
    "gpt2": "gpt2-q8_0",
}

INPUT_PLACEHOLDER = "{INPUT_PLACEHOLDER}"
BASE_INFERENCE_CONTRACT_VERSION = 1
# Frozen v1 rendering of one raw user message with Qwen3's
# add_generation_prompt=True and enable_thinking=False chat-template options.
QWEN3_BASE_PROMPT_TEMPLATE = (
    "<|im_start|>user\n"
    "{INPUT_PLACEHOLDER}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)
GPT2_BASE_PROMPT_TEMPLATE = "{INPUT_PLACEHOLDER}"

_PROGRAM_ID_RE = re.compile(r"^[a-f0-9]{16,64}$")
_RUNTIME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[a-fA-F0-9]{64}$")
SUPPORTED_RUNTIME_MANIFEST_VERSIONS = frozenset({1})
GGUF_MAGIC = b"GGUF"
MIN_ADAPTER_GGUF_SIZE = 1024
# A waiter may be behind a slow first download of the 622 MB Qwen GGUF.
WINDOWS_LOCK_TIMEOUT_S = 2 * 60 * 60
WINDOWS_LOCK_RETRY_S = 0.05
_LOCKS_GUARD = threading.Lock()
_IN_PROCESS_LOCKS: dict[str, threading.Lock] = {}


class CachedProgram(TypedDict):
    """JSON-serializable metadata for one valid local program cache."""

    program_id: str
    slugs: list[str]
    spec: str | None
    compiler_snapshot: str | None
    runtime_id: str | None
    runtime_manifest_version: int | None
    created_at: str | None
    program_dir: str
    adapter_path: str
    prompt_template_path: str
    base_model_path: str | None
    offline_ready: bool


LEGACY_RUNTIME_MANIFESTS = {
    "qwen3-0.6b-q6_k": {
        "runtime_id": "qwen3-0.6b-q6_k",
        "manifest_version": 1,
        "display_name": "Qwen3 0.6B (Q6_K)",
        "interpreter": "Qwen/Qwen3-0.6B",
        "adapter_format": "gguf_lora",
        "prompt_template": {
            "format": "rendered_text",
            "placeholder": INPUT_PLACEHOLDER,
        },
        "program_assets": {
            "adapter_filename": "adapter.gguf",
            "prefix_cache_required": False,
            "prefix_cache_filename": None,
            "prefix_tokens_filename": None,
        },
        "base_inference": {
            "contract_version": BASE_INFERENCE_CONTRACT_VERSION,
            "format": "rendered_text",
            "placeholder": INPUT_PLACEHOLDER,
            "template": QWEN3_BASE_PROMPT_TEMPLATE,
        },
        "local_sdk": {
            "supported": True,
            "base_model": {
                "provider": "huggingface",
                "repo": "programasweights/Qwen3-0.6B-GGUF-Q6_K",
                "file": "qwen3-0.6b-q6_k.gguf",
                "url": BASE_MODEL_URLS["qwen3-0.6b-q6_k"],
                "size_bytes": 622733120,
                "sha256": "9a16ed5cacba959e63b62e2b6840c3eca2b51c3c3e51d31367ef8e4aafeae33c",
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
        "prompt_template": {
            "format": "rendered_text",
            "placeholder": INPUT_PLACEHOLDER,
        },
        "program_assets": {
            "adapter_filename": "adapter.gguf",
            "prefix_cache_required": True,
            "prefix_cache_filename": "prefix_cache.bin",
            "prefix_tokens_filename": "prefix_tokens.json",
        },
        "base_inference": {
            "contract_version": BASE_INFERENCE_CONTRACT_VERSION,
            "format": "rendered_text",
            "placeholder": INPUT_PLACEHOLDER,
            "template": GPT2_BASE_PROMPT_TEMPLATE,
        },
        "local_sdk": {
            "supported": True,
            "base_model": {
                "provider": "huggingface",
                "repo": "programasweights/GPT2-GGUF-Q8_0",
                "file": "gpt2-q8_0.gguf",
                "url": BASE_MODEL_URLS["gpt2-q8_0"],
                "size_bytes": 139804832,
                "sha256": "0aa260efb2cce9def922e0546b88ad731cf1a68554db73fa2d4a0949cfa958c5",
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
                "size_bytes": 139804832,
                "sha256": "0aa260efb2cce9def922e0546b88ad731cf1a68554db73fa2d4a0949cfa958c5",
            },
            "prefix_cache_supported": True,
        },
    },
}


def _lock_for_path(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        lock = _IN_PROCESS_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _IN_PROCESS_LOCKS[key] = lock
        return lock


def _acquire_windows_file_lock(
    lock_file,
    msvcrt_module,
    *,
    timeout_s: float = WINDOWS_LOCK_TIMEOUT_S,
    retry_s: float = WINDOWS_LOCK_RETRY_S,
    monotonic=time.monotonic,
    sleep=time.sleep,
) -> None:
    """Acquire one byte with non-blocking retries and a bounded deadline."""
    if timeout_s <= 0 or retry_s <= 0:
        raise ValueError("Windows lock timeout and retry interval must be positive.")
    deadline = monotonic() + timeout_s
    while True:
        lock_file.seek(0)
        try:
            msvcrt_module.locking(
                lock_file.fileno(),
                msvcrt_module.LK_NBLCK,
                1,
            )
            return
        except OSError as exc:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Timed out after {timeout_s:.0f}s waiting for cache lock."
                ) from exc
            sleep(min(retry_s, remaining))


@contextmanager
def _cross_process_lock(path: Path) -> Iterator[None]:
    """Serialize cache mutations across threads and Python processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    thread_lock = _lock_for_path(path)
    with thread_lock:
        with open(path, "a+b") as lock_file:
            if os.name == "nt":
                import msvcrt

                lock_file.seek(0, os.SEEK_END)
                if lock_file.tell() == 0:
                    lock_file.write(b"\0")
                    lock_file.flush()
                _acquire_windows_file_lock(lock_file, msvcrt)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    try:
                        msvcrt.locking(
                            lock_file.fileno(),
                            msvcrt.LK_UNLCK,
                            1,
                        )
                    except OSError:
                        pass
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _locks_dir() -> Path:
    path = config.get_cache_dir() / ".locks"
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def program_cache_lock(program_id: str) -> Iterator[None]:
    """Lock installation of one immutable program cache entry."""
    if not is_program_id(program_id):
        raise ValueError(f"Invalid program ID: {program_id!r}")
    with _cross_process_lock(
        _locks_dir() / "programs" / f"{program_id}.lock"
    ):
        yield


@contextmanager
def _base_model_cache_lock(file_name: str) -> Iterator[None]:
    if not file_name or Path(file_name).name != file_name:
        raise ValueError(f"Invalid base-model filename: {file_name!r}")
    with _cross_process_lock(
        _locks_dir() / "base_models" / f"{file_name}.lock"
    ):
        yield


@contextmanager
def prefix_cache_lock(program_dir: Path) -> Iterator[None]:
    """Lock one program's derived prefix state across processes."""
    identity = hashlib.sha256(
        str(program_dir.resolve(strict=False)).encode("utf-8")
    ).hexdigest()
    with _cross_process_lock(
        _locks_dir() / "prefix_cache" / f"{identity}.lock"
    ):
        yield


def _atomic_write_json(path: Path, value: object) -> None:
    """Write JSON by replacing a complete same-filesystem temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(tmp_path), str(path))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _valid_slug(slug: object) -> bool:
    return bool(
        isinstance(slug, str)
        and slug
        and slug == slug.strip()
        and "\x00" not in slug
    )


def _runtime_manifest_has_valid_shape(
    runtime_manifest: object,
    *,
    require_local_model: bool,
) -> bool:
    if not isinstance(runtime_manifest, dict):
        return False

    runtime_id = runtime_manifest.get("runtime_id")
    manifest_version = runtime_manifest.get("manifest_version")
    interpreter = runtime_manifest.get("interpreter")
    if (
        not isinstance(runtime_id, str)
        or not _RUNTIME_ID_RE.fullmatch(runtime_id)
        or not isinstance(manifest_version, int)
        or isinstance(manifest_version, bool)
        or manifest_version not in SUPPORTED_RUNTIME_MANIFEST_VERSIONS
        or not isinstance(interpreter, str)
        or not interpreter
        or runtime_manifest.get("adapter_format") != "gguf_lora"
    ):
        return False

    prompt_template = runtime_manifest.get("prompt_template")
    if prompt_template is not None and (
        not isinstance(prompt_template, dict)
        or prompt_template.get("format") != "rendered_text"
        or prompt_template.get("placeholder") != INPUT_PLACEHOLDER
    ):
        return False

    program_assets = runtime_manifest.get("program_assets")
    if program_assets is not None and (
        not isinstance(program_assets, dict)
        or program_assets.get("adapter_filename") != "adapter.gguf"
    ):
        return False

    local_sdk = runtime_manifest.get("local_sdk")
    if not isinstance(local_sdk, dict):
        return False
    supported = local_sdk.get("supported")
    if not isinstance(supported, bool):
        return False

    n_ctx = local_sdk.get("n_ctx")
    if (
        n_ctx is not None
        and (
            not isinstance(n_ctx, int)
            or isinstance(n_ctx, bool)
            or n_ctx <= 0
        )
    ):
        return False

    base_model = local_sdk.get("base_model")
    if supported or require_local_model:
        if not isinstance(base_model, dict):
            return False
        file_name = base_model.get("file")
        if (
            not isinstance(file_name, str)
            or not file_name
            or Path(file_name).name != file_name
        ):
            return False
        provider = base_model.get("provider")
        repo = base_model.get("repo")
        url = base_model.get("url")
        if provider is not None and not isinstance(provider, str):
            return False
        if repo is not None and not isinstance(repo, str):
            return False
        if url is not None and not isinstance(url, str):
            return False
        if not url and not (provider == "huggingface" and repo):
            return False
        sha256 = base_model.get("sha256")
        if sha256 is not None and (
            not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256)
        ):
            return False
        size_bytes = base_model.get("size_bytes")
        if size_bytes is not None and (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes <= 0
        ):
            return False
    elif base_model is not None and not isinstance(base_model, dict):
        return False

    base_inference = runtime_manifest.get("base_inference")
    if base_inference is not None:
        if not isinstance(base_inference, dict):
            return False
        if (
            base_inference.get("contract_version")
            != BASE_INFERENCE_CONTRACT_VERSION
            or base_inference.get("format") != "rendered_text"
            or base_inference.get("placeholder") != INPUT_PLACEHOLDER
        ):
            return False
        template = base_inference.get("template")
        if (
            not isinstance(template, str)
            or template.count(INPUT_PLACEHOLDER) != 1
        ):
            return False

    return True


def _normalize_runtime_manifest(
    runtime_manifest: object,
    *,
    expected_runtime_id: str | None = None,
    require_local_model: bool = False,
) -> dict | None:
    """Validate a manifest and pin known runtimes to canonical integrity."""
    if not _runtime_manifest_has_valid_shape(
        runtime_manifest,
        require_local_model=require_local_model,
    ):
        return None
    assert isinstance(runtime_manifest, dict)
    if (
        expected_runtime_id is not None
        and runtime_manifest.get("runtime_id") != expected_runtime_id
    ):
        return None

    normalized = json.loads(json.dumps(runtime_manifest))
    runtime_id = normalized["runtime_id"]
    canonical = LEGACY_RUNTIME_MANIFESTS.get(runtime_id)
    if canonical is None:
        return normalized

    for field in (
        "interpreter",
        "manifest_version",
        "adapter_format",
    ):
        if normalized.get(field) != canonical.get(field):
            return None

    normalized_local = normalized.get("local_sdk")
    canonical_local = canonical.get("local_sdk")
    if not isinstance(normalized_local, dict) or not isinstance(
        canonical_local,
        dict,
    ):
        return None
    if (
        normalized_local.get("supported")
        != canonical_local.get("supported")
        or normalized_local.get("n_ctx") != canonical_local.get("n_ctx")
    ):
        return None

    normalized_base = normalized_local.get("base_model")
    canonical_base = canonical_local.get("base_model")
    if not isinstance(normalized_base, dict) or not isinstance(
        canonical_base,
        dict,
    ):
        return None
    if normalized_base.get("file") != canonical_base.get("file"):
        return None
    for field in ("provider", "repo", "url"):
        incoming = normalized_base.get(field)
        canonical_value = canonical_base.get(field)
        if incoming is not None and incoming != canonical_value:
            return None

    for field in ("size_bytes", "sha256"):
        incoming = normalized_base.get(field)
        canonical_value = canonical_base.get(field)
        if canonical_value is None or (
            incoming is not None and incoming != canonical_value
        ):
            return None

    def merge_canonical_contract(section_name: str) -> bool:
        canonical_section = canonical.get(section_name)
        incoming_section = normalized.get(section_name)
        if not isinstance(canonical_section, dict):
            return incoming_section is None
        if incoming_section is not None:
            if not isinstance(incoming_section, dict):
                return False
            for key, canonical_value in canonical_section.items():
                if (
                    key in incoming_section
                    and incoming_section[key] != canonical_value
                ):
                    return False
        merged = dict(incoming_section or {})
        merged.update(json.loads(json.dumps(canonical_section)))
        normalized[section_name] = merged
        return True

    for section_name in (
        "prompt_template",
        "program_assets",
        "base_inference",
    ):
        if not merge_canonical_contract(section_name):
            return None

    normalized_base.update(
        {
            field: canonical_base[field]
            for field in (
                "provider",
                "repo",
                "file",
                "url",
                "size_bytes",
                "sha256",
            )
        }
    )
    if not _runtime_manifest_has_valid_shape(
        normalized,
        require_local_model=require_local_model,
    ):
        return None
    return normalized


def _runtime_cache_dir() -> Path:
    d = config.get_cache_dir() / "runtimes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _runtime_manifest_path(
    runtime_id: str,
    manifest_version: int | None = None,
) -> Path:
    if not _RUNTIME_ID_RE.fullmatch(runtime_id):
        raise ValueError(f"Invalid runtime ID: {runtime_id!r}")
    if manifest_version is None:
        file_name = f"{runtime_id}.json"
    else:
        if (
            not isinstance(manifest_version, int)
            or isinstance(manifest_version, bool)
            or manifest_version < 1
        ):
            raise ValueError(
                f"Invalid runtime manifest version: {manifest_version!r}"
            )
        file_name = f"{runtime_id}.v{manifest_version}.json"
    return _runtime_cache_dir() / file_name


def _read_cached_runtime_manifest(
    path: Path,
    runtime_id: str,
    manifest_version: int | None,
) -> dict | None:
    if not path.exists() or path.is_symlink():
        return None
    try:
        runtime_manifest = json.loads(path.read_text(encoding="utf-8"))
        normalized = _normalize_runtime_manifest(
            runtime_manifest,
            expected_runtime_id=runtime_id,
            require_local_model=False,
        )
        if normalized is None or (
            manifest_version is not None
            and normalized.get("manifest_version") != manifest_version
        ):
            return None
        return normalized
    except (json.JSONDecodeError, OSError):
        return None


def get_cached_runtime_manifest(
    runtime_id: str,
    manifest_version: int | None = None,
) -> dict | None:
    """Read an exact manifest version, with the legacy current file fallback."""
    try:
        legacy_path = _runtime_manifest_path(runtime_id)
        if manifest_version is None:
            candidates = [legacy_path]
            candidates.extend(
                _runtime_manifest_path(runtime_id, version)
                for version in sorted(
                    SUPPORTED_RUNTIME_MANIFEST_VERSIONS,
                    reverse=True,
                )
            )
        else:
            candidates = [
                _runtime_manifest_path(runtime_id, manifest_version),
                legacy_path,
            ]
    except (TypeError, ValueError):
        return None

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        cached = _read_cached_runtime_manifest(
            path,
            runtime_id,
            manifest_version,
        )
        if cached is not None:
            return cached
    return None


def save_runtime_manifest(runtime_manifest: dict) -> None:
    normalized = _normalize_runtime_manifest(
        runtime_manifest,
        require_local_model=False,
    )
    if normalized is None:
        raise ValueError("Invalid runtime manifest shape.")
    runtime_id = normalized["runtime_id"]
    manifest_version = normalized["manifest_version"]
    versioned_path = _runtime_manifest_path(runtime_id, manifest_version)
    legacy_path = _runtime_manifest_path(runtime_id)
    with _cross_process_lock(
        _locks_dir() / "runtimes" / f"{runtime_id}.lock"
    ):
        _atomic_write_json(versioned_path, normalized)
        _atomic_write_json(legacy_path, normalized)


def _legacy_runtime_manifest(interpreter: str | None) -> dict | None:
    if not interpreter:
        return None
    runtime_id = INTERPRETER_TO_GGUF.get(interpreter)
    if not runtime_id:
        return None
    manifest = LEGACY_RUNTIME_MANIFESTS.get(runtime_id)
    return json.loads(json.dumps(manifest)) if manifest else None


def _is_runtime_manifest_complete(runtime_manifest: dict | None) -> bool:
    normalized = _normalize_runtime_manifest(
        runtime_manifest,
        require_local_model=True,
    )
    return bool(normalized and normalized["local_sdk"]["supported"])


def get_base_runtime_manifest(interpreter: str) -> dict:
    """Return the built-in runtime and base prompt contract for an interpreter."""
    runtime_id = INTERPRETER_TO_GGUF.get(interpreter)
    if runtime_id is None:
        raise ValueError(
            f"Unknown interpreter: {interpreter!r}. "
            f"Supported: {list(INTERPRETER_TO_GGUF.keys())}"
        )
    manifest = LEGACY_RUNTIME_MANIFESTS.get(runtime_id)
    if (
        not _is_runtime_manifest_complete(manifest)
        or manifest.get("interpreter") != interpreter
    ):
        raise RuntimeError(
            f"Built-in runtime manifest for {interpreter!r} is invalid."
        )
    return json.loads(json.dumps(manifest))


def get_base_prompt_template(runtime_manifest: dict) -> str:
    """Read and validate the versioned base-inference prompt contract."""
    normalized = _normalize_runtime_manifest(
        runtime_manifest,
        require_local_model=True,
    )
    if normalized is None:
        raise ValueError("Invalid runtime manifest shape.")
    contract = normalized.get("base_inference")
    if not isinstance(contract, dict):
        raise ValueError(
            f"Runtime {normalized.get('runtime_id')!r} has no "
            "base-inference prompt contract."
        )
    if contract.get("contract_version") != BASE_INFERENCE_CONTRACT_VERSION:
        raise ValueError(
            "Unsupported base-inference prompt contract version: "
            f"{contract.get('contract_version')!r}."
        )
    template = contract.get("template")
    placeholder = contract.get("placeholder")
    if (
        placeholder != INPUT_PLACEHOLDER
        or not isinstance(template, str)
        or template.count(INPUT_PLACEHOLDER) != 1
    ):
        raise ValueError(
            "Base-inference prompt template must contain exactly one "
            f"{INPUT_PLACEHOLDER} placeholder."
        )
    return template


def _normalize_runtime_manifest_for_program(
    runtime_manifest: dict | None,
    program_meta: dict,
) -> dict | None:
    normalized = _normalize_runtime_manifest(
        runtime_manifest,
        require_local_model=True,
    )
    if normalized is None:
        return None
    runtime_id = program_meta.get("runtime_id")
    if runtime_id and normalized.get("runtime_id") != runtime_id:
        return None
    interpreter = program_meta.get("interpreter")
    if (
        interpreter
        and normalized.get("interpreter") != interpreter
    ):
        return None
    manifest_version = program_meta.get("runtime_manifest_version")
    if (
        manifest_version is not None
        and normalized.get("manifest_version") != manifest_version
    ):
        return None
    return normalized


def _runtime_manifest_matches_program(
    runtime_manifest: dict | None,
    program_meta: dict,
) -> bool:
    return (
        _normalize_runtime_manifest_for_program(runtime_manifest, program_meta)
        is not None
    )


def get_offline_runtime_manifest(program_meta: dict) -> dict | None:
    """Resolve the exact runtime manifest without performing network I/O."""
    embedded = program_meta.get("runtime")
    if isinstance(embedded, dict):
        normalized_embedded = _normalize_runtime_manifest_for_program(
            embedded,
            program_meta,
        )
        if normalized_embedded is not None:
            return normalized_embedded

    runtime_id = program_meta.get("runtime_id")
    if isinstance(runtime_id, str) and runtime_id:
        manifest_version = program_meta.get("runtime_manifest_version")
        cached = get_cached_runtime_manifest(
            runtime_id,
            (
                manifest_version
                if isinstance(manifest_version, int)
                and not isinstance(manifest_version, bool)
                else None
            ),
        )
        normalized_cached = _normalize_runtime_manifest_for_program(
            cached,
            program_meta,
        )
        if normalized_cached is not None:
            return normalized_cached

    legacy = _legacy_runtime_manifest(program_meta.get("interpreter"))
    return _normalize_runtime_manifest_for_program(legacy, program_meta)


def fetch_runtime_manifest(
    runtime_id: str,
    api_url: str | None = None,
    api_key: str | None = None,
) -> dict:
    if not isinstance(runtime_id, str) or not _RUNTIME_ID_RE.fullmatch(runtime_id):
        raise ValueError(f"Invalid runtime ID: {runtime_id!r}")
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
    if (
        isinstance(runtime_manifest, dict)
        and runtime_manifest.get("runtime_id") != runtime_id
    ):
        raise ValueError(
            f"Server returned runtime ID "
            f"{runtime_manifest.get('runtime_id')!r} for requested "
            f"{runtime_id!r}."
        )
    normalized = _normalize_runtime_manifest(
        runtime_manifest,
        expected_runtime_id=runtime_id,
        require_local_model=False,
    )
    if normalized is None:
        raise ValueError(
            f"Server returned an invalid runtime manifest for {runtime_id!r}."
        )
    save_runtime_manifest(normalized)
    return normalized


def resolve_runtime_manifest(
    program_meta: dict,
    api_url: str | None = None,
    api_key: str | None = None,
    offline: bool = False,
) -> dict | None:
    embedded = program_meta.get("runtime")
    if isinstance(embedded, dict):
        normalized_embedded = _normalize_runtime_manifest_for_program(
            embedded,
            program_meta,
        )
        if normalized_embedded is not None:
            try:
                save_runtime_manifest(normalized_embedded)
            except OSError:
                pass
            return normalized_embedded

    runtime_id = program_meta.get("runtime_id")
    if isinstance(runtime_id, str) and _RUNTIME_ID_RE.fullmatch(runtime_id):
        manifest_version = program_meta.get("runtime_manifest_version")
        cached = get_cached_runtime_manifest(
            runtime_id,
            (
                manifest_version
                if isinstance(manifest_version, int)
                and not isinstance(manifest_version, bool)
                else None
            ),
        )
        normalized_cached = _normalize_runtime_manifest_for_program(
            cached,
            program_meta,
        )
        if normalized_cached is not None:
            return normalized_cached
        if not offline:
            try:
                fetched = fetch_runtime_manifest(
                    runtime_id,
                    api_url=api_url,
                    api_key=api_key,
                )
                normalized_fetched = _normalize_runtime_manifest_for_program(
                    fetched,
                    program_meta,
                )
                if normalized_fetched is not None:
                    return normalized_fetched
            except Exception:
                pass

    legacy = _legacy_runtime_manifest(program_meta.get("interpreter"))
    return _normalize_runtime_manifest_for_program(legacy, program_meta)


def _base_model_info_from_runtime(runtime_manifest: dict) -> dict | None:
    local_sdk = runtime_manifest.get("local_sdk")
    if not isinstance(local_sdk, dict):
        return None
    base_model = local_sdk.get("base_model")
    return base_model if isinstance(base_model, dict) else None


def _build_hf_url(repo: str, file_name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{file_name}"


def get_cached_base_model_path(runtime_manifest: dict) -> Path | None:
    """Return the exact cached base-model file, without downloading it."""
    normalized = _normalize_runtime_manifest(
        runtime_manifest,
        require_local_model=True,
    )
    if normalized is None:
        return None
    local_sdk = normalized.get("local_sdk")
    if not isinstance(local_sdk, dict) or not local_sdk.get("supported", False):
        return None
    base_model = _base_model_info_from_runtime(normalized)
    file_name = base_model.get("file") if base_model else None
    if (
        not isinstance(file_name, str)
        or not file_name
        or Path(file_name).name != file_name
    ):
        return None
    path = config.get_base_models_dir() / file_name
    expected_sha256 = base_model.get("sha256") if base_model else None
    expected_size = base_model.get("size_bytes") if base_model else None
    if not _valid_gguf_file(
        path,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    ):
        return None
    return path


def get_base_model_path(
    interpreter: str = "Qwen/Qwen3-0.6B",
    runtime_manifest: dict | None = None,
    progress: ProgressCallback | None = None,
    offline: bool = False,
) -> Path:
    """Get the path to the base model GGUF, downloading if needed."""
    candidate_manifest = (
        runtime_manifest
        if runtime_manifest is not None
        else get_base_runtime_manifest(interpreter)
    )
    manifest = _normalize_runtime_manifest(
        candidate_manifest,
        require_local_model=True,
    )
    if manifest is None:
        runtime_label = (
            candidate_manifest.get("runtime_id", interpreter)
            if isinstance(candidate_manifest, dict)
            else interpreter
        )
        raise ValueError(
            f"Runtime {runtime_label!r} is not "
            "supported by the local SDK or has an invalid base model."
        )
    manifest_interpreter = manifest.get("interpreter")
    if (
        isinstance(manifest_interpreter, str)
        and manifest_interpreter != interpreter
    ):
        raise ValueError(
            f"Runtime {manifest.get('runtime_id')!r} is for interpreter "
            f"{manifest_interpreter!r}, not {interpreter!r}."
        )

    local_sdk = manifest["local_sdk"]
    base_model = local_sdk["base_model"]
    file_name = base_model["file"]
    runtime_id = str(manifest["runtime_id"])
    gguf_path = config.get_base_models_dir() / file_name
    cached_path = get_cached_base_model_path(manifest)
    if cached_path is not None:
        report_progress(
            progress,
            {
                "stage": "base_model",
                "status": "cached",
                "runtime_id": runtime_id,
                "path": str(cached_path),
            },
        )
        return cached_path

    if offline:
        raise RuntimeError(
            f"Base model for runtime {runtime_id!r} is not cached; "
            "offline mode prohibits network downloads."
        )

    url = base_model.get("url")
    if not url and base_model.get("provider") == "huggingface":
        url = _build_hf_url(base_model["repo"], file_name)
    if not url:
        raise ValueError(
            f"Runtime {runtime_id!r} is missing a downloadable base-model URL."
        )

    with _base_model_cache_lock(file_name):
        cached_path = get_cached_base_model_path(manifest)
        if cached_path is not None:
            report_progress(
                progress,
                {
                    "stage": "base_model",
                    "status": "cached",
                    "runtime_id": runtime_id,
                    "path": str(cached_path),
                },
            )
            return cached_path

        label = (
            manifest.get("display_name")
            or manifest.get("runtime_id")
            or interpreter
        )
        report_progress(
            progress,
            {
                "stage": "base_model",
                "status": "downloading",
                "runtime_id": runtime_id,
                "path": str(gguf_path),
            },
            f"Downloading interpreter {label} (one-time download)...",
        )
        if progress is None:
            _download_file(url, gguf_path)
        else:
            _download_file(url, gguf_path, progress=progress)

        cached_path = get_cached_base_model_path(manifest)
        if cached_path is None:
            gguf_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded base model for runtime {runtime_id!r} failed "
                "GGUF magic, file size, or SHA-256 validation."
            )

    report_progress(
        progress,
        {
            "stage": "base_model",
            "status": "ready",
            "runtime_id": runtime_id,
            "path": str(cached_path),
        },
        f"Saved to {cached_path}",
    )
    return cached_path


def get_program_dir(program_id: str) -> Path:
    """Get the local cache directory for a program."""
    return config.get_programs_dir() / program_id


def is_program_id(value: str) -> bool:
    """Return whether *value* is an immutable PAW program ID."""
    return bool(_PROGRAM_ID_RE.fullmatch(value))


def _valid_regular_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
        stat_result = path.stat()
        if stat_result.st_size <= 0:
            return False
        if expected_size is not None and stat_result.st_size != expected_size:
            return False
        if expected_sha256 is not None:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest().lower() != expected_sha256.lower():
                return False
        return True
    except OSError:
        return False


def _valid_gguf_file(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    minimum_size: int = len(GGUF_MAGIC),
) -> bool:
    if not _valid_regular_file(
        path,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    ):
        return False
    try:
        if path.stat().st_size < minimum_size:
            return False
        with open(path, "rb") as handle:
            return handle.read(len(GGUF_MAGIC)) == GGUF_MAGIC
    except OSError:
        return False


def is_valid_adapter_file(path: Path) -> bool:
    """Return whether an adapter is a nontrivial GGUF regular file."""
    return _valid_gguf_file(
        path,
        minimum_size=MIN_ADAPTER_GGUF_SIZE,
    )


def validate_program_assets_dir(
    program_dir: Path,
    expected_program_id: str,
) -> bool:
    """Validate only one directory's required immutable program assets."""
    if not is_program_id(expected_program_id):
        return False
    try:
        if not program_dir.is_dir() or program_dir.is_symlink():
            return False
    except OSError:
        return False

    adapter_path = program_dir / "adapter.gguf"
    template_path = program_dir / "prompt_template.txt"
    meta_path = program_dir / "meta.json"
    if (
        not is_valid_adapter_file(adapter_path)
        or not _valid_regular_file(template_path)
        or not _valid_regular_file(meta_path)
    ):
        return False

    try:
        template = template_path.read_text(encoding="utf-8")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return False

    if template.count(INPUT_PLACEHOLDER) != 1:
        return False
    if not isinstance(meta, dict) or meta.get("program_id") != expected_program_id:
        return False
    interpreter = meta.get("interpreter")
    if not isinstance(interpreter, str) or not interpreter:
        return False
    runtime_id = meta.get("runtime_id")
    if runtime_id is not None and (
        not isinstance(runtime_id, str)
        or not _RUNTIME_ID_RE.fullmatch(runtime_id)
    ):
        return False
    manifest_version = meta.get("runtime_manifest_version")
    if manifest_version is not None and (
        not isinstance(manifest_version, int)
        or isinstance(manifest_version, bool)
        or manifest_version not in SUPPORTED_RUNTIME_MANIFEST_VERSIONS
    ):
        return False
    embedded_runtime = meta.get("runtime")
    if embedded_runtime not in (None, {}) and _normalize_runtime_manifest(
        embedded_runtime,
        expected_runtime_id=(
            runtime_id if isinstance(runtime_id, str) else None
        ),
        require_local_model=False,
    ) is None:
        return False
    return True


def _load_valid_program_meta(program_id: str) -> dict | None:
    if not is_program_id(program_id):
        return None
    program_dir = get_program_dir(program_id)
    if not validate_program_assets_dir(program_dir, program_id):
        return None
    try:
        meta = json.loads(
            (program_dir / "meta.json").read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError):
        return None
    return meta


def has_valid_program_assets(program_id: str) -> bool:
    """Check adapter, template, and exact metadata without checking the model."""
    return _load_valid_program_meta(program_id) is not None


def load_cached_program_meta(program_id: str) -> dict | None:
    """Load validated metadata for an exact immutable program ID."""
    meta = _load_valid_program_meta(program_id)
    return dict(meta) if meta is not None else None


def is_program_cached(program_id: str) -> bool:
    """Check if a program's artifacts are already cached locally."""
    return has_valid_program_assets(program_id)


def get_cached_program_metadata(
    program_id: str,
    *,
    slugs: list[str] | None = None,
) -> CachedProgram | None:
    """Inspect one valid program cache without performing network I/O."""
    meta = _load_valid_program_meta(program_id)
    if meta is None:
        return None

    runtime_manifest = get_offline_runtime_manifest(meta)
    base_model_path = (
        get_cached_base_model_path(runtime_manifest)
        if runtime_manifest is not None
        else None
    )
    runtime_id = (
        runtime_manifest.get("runtime_id")
        if runtime_manifest is not None
        else meta.get("runtime_id")
    )
    runtime_manifest_version = (
        runtime_manifest.get("manifest_version")
        if runtime_manifest is not None
        else meta.get("runtime_manifest_version")
    )
    if slugs is None:
        slugs = [
            slug
            for slug, mapped_program_id in _load_slug_mappings().items()
            if mapped_program_id == program_id
        ]
    normalized_slugs = sorted(
        {
            slug
            for slug in slugs
            if _valid_slug(slug)
        }
    )
    program_dir = get_program_dir(program_id)
    return {
        "program_id": program_id,
        "slugs": normalized_slugs,
        "spec": meta.get("spec") if isinstance(meta.get("spec"), str) else None,
        "compiler_snapshot": (
            meta.get("compiler_snapshot")
            if isinstance(meta.get("compiler_snapshot"), str)
            else None
        ),
        "runtime_id": runtime_id if isinstance(runtime_id, str) else None,
        "runtime_manifest_version": (
            runtime_manifest_version
            if isinstance(runtime_manifest_version, int)
            else None
        ),
        "created_at": (
            meta.get("created_at")
            if isinstance(meta.get("created_at"), str)
            else None
        ),
        "program_dir": str(program_dir),
        "adapter_path": str(program_dir / "adapter.gguf"),
        "prompt_template_path": str(program_dir / "prompt_template.txt"),
        "base_model_path": (
            str(base_model_path) if base_model_path is not None else None
        ),
        "offline_ready": runtime_manifest is not None
        and base_model_path is not None,
    }


def _download_file(
    url: str,
    dest: Path,
    progress: ProgressCallback | None = None,
):
    """Download a file atomically with progress indication."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{dest.name}.",
        suffix=".tmp",
        dir=str(dest.parent),
    )
    os.close(fd)
    tmp = Path(tmp_name)
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
                        if progress is not None:
                            progress(
                                {
                                    "stage": "base_model",
                                    "status": "downloading",
                                    "path": str(dest),
                                    "downloaded_bytes": downloaded,
                                    "total_bytes": total,
                                }
                            )
                        else:
                            pct = downloaded / total * 100
                            mb = downloaded / 1024 / 1024
                            from ._output import status_inline
                            status_inline(f"  {mb:.1f} MB ({pct:.0f}%)")
            if progress is None:
                from ._output import status_end
                status_end()
        os.replace(str(tmp), str(dest))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _slug_cache_path() -> Path:
    return config.get_cache_dir() / "slug_cache.json"


def _load_slug_mappings() -> dict[str, str]:
    path = _slug_cache_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        slug: program_id
        for slug, program_id in data.items()
        if _valid_slug(slug)
        and isinstance(program_id, str)
        and is_program_id(program_id)
    }


def get_cached_slug(slug: str) -> str | None:
    """Look up a slug in the local cache. Returns program_id or None."""
    if not _valid_slug(slug):
        return None
    program_id = _load_slug_mappings().get(slug)
    if program_id and has_valid_program_assets(program_id):
        return program_id
    return None


def resolve_cached_program_id(program_id_or_slug: str) -> str | None:
    """Resolve an immutable ID or cached slug without network access."""
    if is_program_id(program_id_or_slug):
        return program_id_or_slug
    return get_cached_slug(program_id_or_slug)


def list_cached_programs() -> list[CachedProgram]:
    """List valid cached program directories and their offline readiness."""
    slug_mappings = _load_slug_mappings()
    slugs_by_program: dict[str, list[str]] = {}
    for slug, program_id in slug_mappings.items():
        slugs_by_program.setdefault(program_id, []).append(slug)

    programs: list[CachedProgram] = []
    for program_dir in sorted(config.get_programs_dir().iterdir()):
        if (
            not program_dir.is_dir()
            or program_dir.is_symlink()
            or not is_program_id(program_dir.name)
        ):
            continue
        metadata = get_cached_program_metadata(
            program_dir.name,
            slugs=slugs_by_program.get(program_dir.name),
        )
        if metadata is not None:
            programs.append(metadata)
    return programs


def save_slug_mapping(slug: str, program_id: str) -> None:
    """Save a slug -> program_id mapping to the local cache."""
    if not _valid_slug(slug):
        raise ValueError(f"Invalid slug: {slug!r}")
    if not is_program_id(program_id):
        raise ValueError(f"Invalid program ID: {program_id!r}")
    path = _slug_cache_path()
    lock_path = _locks_dir() / "slug_cache.lock"
    with _cross_process_lock(lock_path):
        data = _load_slug_mappings()
        data[slug] = program_id
        _atomic_write_json(path, data)
