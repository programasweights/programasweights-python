"""Image artifacts carry an exact, inert contract and cannot run as text."""

import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import httpx
import pytest

import programasweights as paw
from programasweights import cache, config
from programasweights.local_program import import_local_program


PID = "c" * 20
MODEL = b"GGUF" + b"M" * 4092
PROJECTOR = b"GGUF" + b"P" * 4092
ADAPTER = b"GGUF" + b"A" * (cache.MIN_ADAPTER_GGUF_SIZE - 4)


def asset(file, data):
    return {
        "file": file, "url": "https://example.invalid/" + file,
        "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
    }


@pytest.fixture
def manifest():
    return {
        "runtime_id": "qwen3.5-0.8b-test",
        "manifest_version": 2,
        "interpreter": "Qwen/Qwen3.5-0.8B",
        "adapter_format": "gguf_lora",
        "input": {"format": "content_parts", "types": ["text", "image"]},
        "prompt_template": {
            "format": "chat_messages",
            "system_prompt_file": "prompt_template.txt",
            "chat_format": "qwen3.5",
            "enable_thinking": False,
        },
        "program_assets": {"adapter_filename": "adapter.gguf"},
        "local_sdk": {
            "supported": True,
            "n_ctx": 4096,
            "base_model": asset("base.gguf", MODEL),
            "vision": {
                "projector": asset("mmproj.gguf", PROJECTOR),
                "preprocessing": {
                    "version": 1, "color_mode": "RGB",
                    "alpha": "composite_white", "orientation": "stored",
                    "color_profile": "ignore", "resize": "backend",
                    "image_min_tokens": 64, "image_max_tokens": 512,
                },
            },
        },
        "js_sdk": {"supported": False},
    }


def metadata(manifest):
    return {
        "version": 4, "program_id": PID,
        "interpreter": manifest["interpreter"],
        "runtime_id": manifest["runtime_id"],
        "runtime_manifest_version": manifest["manifest_version"],
        "runtime": manifest,
    }


def write_program(directory, meta, prompt="Locate the target in the supplied images."):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (directory / "prompt_template.txt").write_text(prompt, encoding="utf-8")
    (directory / "adapter.gguf").write_bytes(ADAPTER)
    return directory


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PAW_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("PAW_OFFLINE", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Metadata validation must not download assets or make API calls")

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setattr(cache, "_download_file", forbidden)


def test_valid_manifest_roundtrip_preserves_contract_and_copies_it(manifest):
    expected = copy.deepcopy(manifest)
    resolved = cache.resolve_runtime_manifest(metadata(manifest))
    assert resolved == expected
    resolved["local_sdk"]["vision"]["preprocessing"]["image_max_tokens"] = 256
    assert manifest == expected
    cache.save_runtime_manifest(manifest)
    assert cache.get_cached_runtime_manifest(manifest["runtime_id"], 2) == expected
    assert cache.get_offline_runtime_manifest(metadata(manifest)) == expected


@pytest.mark.parametrize("path", [
    ("input",), ("prompt_template",), ("program_assets",),
    ("local_sdk", "vision"), ("local_sdk", "base_model"),
    ("local_sdk", "n_ctx"),
    ("local_sdk", "vision", "projector"),
    ("local_sdk", "vision", "preprocessing"),
])
def test_required_contract_sections_cannot_be_omitted(manifest, path):
    current = manifest
    for key in path[:-1]:
        current = current[key]
    del current[path[-1]]
    assert cache._normalize_runtime_manifest(manifest) is None


@pytest.mark.parametrize("path,value", [
    (("manifest_version",), True), (("manifest_version",), 2.0),
    (("manifest_version",), 3), (("manifest_version",), 1),
    (("input", "types"), ["text"]), (("input", "types"), ["image", "audio"]),
    (("input", "format"), "named_slots"), (("input", "extra"), True),
    (("prompt_template", "format"), "rendered_text"),
    (("prompt_template", "chat_format"), "unknown"),
    (("prompt_template", "system_prompt_file"), "../prompt.txt"),
    (("prompt_template", "enable_thinking"), True),
    (("prompt_template", "enable_thinking"), 0),
    (("prompt_template", "placeholder"), cache.INPUT_PLACEHOLDER),
    (("program_assets", "adapter_filename"), "other.gguf"),
    (("program_assets", "prefix_cache_required"), True),
    (("program_assets", "prefix_tokens_filename"), "prefix_tokens.json"),
    (("local_sdk", "supported"), "yes"),
    (("local_sdk", "n_ctx"), True), (("local_sdk", "n_ctx"), 0),
    (("local_sdk", "vision"), []),
    (("local_sdk", "vision", "extra"), True),
    (("local_sdk", "vision", "preprocessing", "version"), True),
    (("local_sdk", "vision", "preprocessing", "version"), 2),
    (("local_sdk", "vision", "preprocessing", "alpha"), "discard"),
    (("local_sdk", "vision", "preprocessing", "color_mode"), "RGBA"),
    (("local_sdk", "vision", "preprocessing", "orientation"), "exif"),
    (("local_sdk", "vision", "preprocessing", "color_profile"), "apply"),
    (("local_sdk", "vision", "preprocessing", "resize"), "512x512"),
    (("local_sdk", "vision", "preprocessing", "image_min_tokens"), 513),
    (("local_sdk", "vision", "preprocessing", "image_min_tokens"), 0),
    (("local_sdk", "vision", "preprocessing", "image_max_tokens"), True),
    (("local_sdk", "vision", "preprocessing", "image_max_tokens"), 512.0),
    (("local_sdk", "vision", "preprocessing", "crop"), "center"),
    (("js_sdk", "supported"), True),
    (("base_inference",), {"format": "rendered_text"}),
])
def test_incompatible_or_ambiguous_settings_rejected(manifest, path, value):
    current = manifest
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
    assert cache._normalize_runtime_manifest(manifest) is None
    with pytest.raises(ValueError):
        cache.save_runtime_manifest(manifest)


@pytest.mark.parametrize("which", ["base", "projector"])
@pytest.mark.parametrize("key,value", [
    ("sha256", None), ("sha256", "a" * 63), ("sha256", "z" * 64),
    ("size_bytes", None), ("size_bytes", True), ("size_bytes", 0), ("size_bytes", 4.0),
    ("file", "../file.gguf"), ("file", "..\\file.gguf"),
    ("file", "C:file.gguf"), ("file", ".."), ("file", "file\x00.gguf"),
    ("url", "file:///tmp/model.gguf"), ("url", "https://"),
    ("url", "https://user:secret@example.invalid/a"),
])
def test_asset_identity_is_required_for_both_files(manifest, which, key, value):
    local = manifest["local_sdk"]
    model = local["base_model"] if which == "base" else local["vision"]["projector"]
    if value is None:
        del model[key]
    else:
        model[key] = value
    assert cache._normalize_runtime_manifest(manifest) is None


def test_huggingface_source_can_use_existing_repo_file_convention(manifest):
    for model in [manifest["local_sdk"]["base_model"], manifest["local_sdk"]["vision"]["projector"]]:
        del model["url"]
        model.update(provider="huggingface", repo="example/model")
    assert cache._normalize_runtime_manifest(manifest) == manifest


@pytest.mark.parametrize("file", ["base.gguf", "BASE.GGUF"])
def test_base_and_projector_cannot_collide_by_filename(manifest, file):
    manifest["local_sdk"]["vision"]["projector"]["file"] = file
    assert cache._normalize_runtime_manifest(manifest) is None


@pytest.mark.parametrize("prompt", [
    "Locate the target.", "Treat {INPUT_PLACEHOLDER} literally.",
    "{INPUT_PLACEHOLDER} and {INPUT_PLACEHOLDER} are examples.",
])
def test_image_archive_import_preserves_literal_instructions(tmp_path, manifest, prompt):
    source = write_program(tmp_path / "source", metadata(manifest), prompt)
    archive = tmp_path / "function.paw"
    with zipfile.ZipFile(archive, "w") as output:
        for path in source.iterdir():
            output.write(path, path.name)
    before = archive.read_bytes()
    imported = import_local_program(archive)
    assert cache.validate_program_assets_dir(imported, PID)
    assert (imported / "prompt_template.txt").read_text() == prompt
    assert json.loads((imported / "meta.json").read_text()) == metadata(manifest)
    assert archive.read_bytes() == before


@pytest.mark.parametrize("field,value", [
    ("runtime_id", None), ("runtime_id", "different-runtime"),
    ("interpreter", None), ("interpreter", "gpt2"),
    ("runtime_manifest_version", None), ("runtime_manifest_version", 1),
    ("runtime_manifest_version", 2.0),
])
def test_program_and_embedded_runtime_must_agree(tmp_path, manifest, field, value):
    meta = metadata(manifest)
    if value is None:
        del meta[field]
    else:
        meta[field] = value
    path = write_program(tmp_path / "program", meta)
    assert not cache.validate_program_assets_dir(path, PID)
    assert cache.get_offline_runtime_manifest(meta) is None
    assert cache.resolve_runtime_manifest(meta) is None


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("damage", ["missing", "bad_alpha", "wrong_version"])
def test_bad_embedded_image_contract_never_falls_back(manifest, offline, damage, monkeypatch):
    cache.save_runtime_manifest(manifest)
    meta = metadata(copy.deepcopy(manifest))
    if damage == "missing":
        del meta["runtime"]
    elif damage == "bad_alpha":
        meta["runtime"]["local_sdk"]["vision"]["preprocessing"]["alpha"] = "discard"
    else:
        meta["runtime"]["manifest_version"] = 1

    def forbidden(*args, **kwargs):
        pytest.fail("Must not substitute a different contract for image metadata")

    monkeypatch.setattr(cache, "get_cached_runtime_manifest", forbidden)
    monkeypatch.setattr(cache, "fetch_runtime_manifest", forbidden)
    monkeypatch.setattr(cache, "_legacy_runtime_manifest", forbidden)
    assert cache.resolve_runtime_manifest(meta, offline=offline) is None
    assert cache.get_offline_runtime_manifest(meta) is None


def test_image_readiness_requires_both_verified_assets(manifest):
    write_program(config.get_programs_dir() / PID, metadata(manifest))
    (config.get_base_models_dir() / "base.gguf").write_bytes(MODEL)
    (config.get_base_models_dir() / "mmproj.gguf").write_bytes(PROJECTOR)
    assert cache.has_valid_program_assets(PID)
    assert not paw.is_offline_ready(PID)
    info = cache.get_cached_program_metadata(PID)
    assert info["base_model_path"] is None
    assert info["offline_ready"] is False
    info = paw.prepare_program(PID, offline=True)
    assert info["offline_ready"] is True
    assert paw.is_offline_ready(PID)
    path = cache.get_base_model_path(manifest["interpreter"], runtime_manifest=manifest, offline=True)
    assert str(path) == info["base_model_path"]
    from programasweights._vision_assets import get_cached_vision_asset_paths
    paths = get_cached_vision_asset_paths(manifest)
    paths.projector.write_bytes(b"corrupt")
    assert not paw.is_offline_ready(PID)


@pytest.mark.parametrize("prompt", ["Locate the target.", "Input: {INPUT_PLACEHOLDER}"])
def test_text_runtime_rejects_image_program_before_loading_model(
    tmp_path, manifest, monkeypatch, prompt,
):
    directory = write_program(config.get_programs_dir() / PID, metadata(manifest), prompt)

    def forbidden(*args, **kwargs):
        pytest.fail("Image program must not create a text-only model")

    fake = types.ModuleType("llama_cpp")
    fake.Llama = forbidden
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    previous = sys.modules.pop("programasweights.runtime_llamacpp", None)
    try:
        runtime = importlib.import_module("programasweights.runtime_llamacpp")
        with pytest.raises(ValueError, match="vision runtime"):
            runtime.PawFunction(directory, offline=True)
    finally:
        sys.modules.pop("programasweights.runtime_llamacpp", None)
        if previous is not None:
            sys.modules["programasweights.runtime_llamacpp"] = previous


def test_validation_does_not_import_vision_or_native_libraries(manifest):
    code = '''
import sys, importlib.abc, json
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('PIL', 'llama_cpp', 'torch', 'numpy'):
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, Block())
from programasweights import cache
manifest = json.loads(sys.argv[1])
assert cache._normalize_runtime_manifest(manifest) == manifest
'''
    result = subprocess.run([sys.executable, "-c", code, json.dumps(manifest)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_builtin_vision_base_contract_is_complete_and_defensively_copied():
    original = cache.get_base_runtime_manifest("Qwen/Qwen3.5-0.8B")
    assert original["base_inference"] == {"contract_version": 1, "format": "chat_messages"}
    assert cache._normalize_runtime_manifest(original) == original
    clone = cache.get_base_runtime_manifest("Qwen/Qwen3.5-0.8B")
    clone["local_sdk"]["vision"]["preprocessing"]["image_max_tokens"] = 64
    assert cache.get_base_runtime_manifest("Qwen/Qwen3.5-0.8B") == original
    with pytest.raises(ValueError, match="vision runtime"):
        cache.get_base_prompt_template(original)


@pytest.mark.parametrize("which", ["base_model", "projector"])
@pytest.mark.parametrize("field,value", [
    ("sha256", "a" * 64), ("size_bytes", 12345),
    ("url", "https://example.invalid/substituted.gguf"),
])
def test_named_builtin_pins_both_asset_identities(which, field, value):
    manifest = cache.get_base_runtime_manifest("Qwen/Qwen3.5-0.8B")
    local = manifest["local_sdk"]
    asset = local["base_model"] if which == "base_model" else local["vision"]["projector"]
    asset[field] = value
    assert cache._normalize_runtime_manifest(manifest) is None


@pytest.mark.parametrize("contract", [
    {"contract_version": 1, "format": "chat_messages"},
    None,
])
def test_vision_base_contract_optional_for_compiled_artifacts(manifest, contract):
    manifest["base_inference"] = contract
    assert cache._normalize_runtime_manifest(manifest) == manifest


@pytest.mark.parametrize("contract", [
    [], "chat_messages", {},
    {"contract_version": True, "format": "chat_messages"},
    {"contract_version": 2, "format": "chat_messages"},
    {"contract_version": 1, "format": "rendered_text"},
    {"contract_version": 1, "format": "chat_messages", "system_prompt": "extra"},
])
def test_unsupported_vision_base_contract_rejected(manifest, contract):
    manifest["base_inference"] = contract
    assert cache._normalize_runtime_manifest(manifest) is None
