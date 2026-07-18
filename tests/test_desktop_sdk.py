from __future__ import annotations

import hashlib
import io
import json
import sys
import types
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

import programasweights as paw
from programasweights import cache, config
from programasweights.client import PAWClient
import programasweights.client as client_module


PROGRAM_ID = "a" * 20
OTHER_PROGRAM_ID = "b" * 20
FAKE_MODEL_BYTES = b"GGUF" + (b"M" * 4092)
FAKE_ADAPTER_BYTES = b"GGUF" + (
    b"A" * (cache.MIN_ADAPTER_GGUF_SIZE - 4)
)


def _runtime(
    runtime_id: str = "test-runtime",
    file_name: str = "test-model.gguf",
) -> dict:
    return {
        "runtime_id": runtime_id,
        "manifest_version": 1,
        "display_name": "Test runtime",
        "interpreter": "test-interpreter",
        "adapter_format": "gguf_lora",
        "prompt_template": {
            "format": "rendered_text",
            "placeholder": cache.INPUT_PLACEHOLDER,
        },
        "program_assets": {"adapter_filename": "adapter.gguf"},
        "local_sdk": {
            "supported": True,
            "base_model": {
                "provider": "huggingface",
                "repo": "example/test-model",
                "file": file_name,
                "url": f"https://example.test/{file_name}",
                "size_bytes": len(FAKE_MODEL_BYTES),
                "sha256": hashlib.sha256(FAKE_MODEL_BYTES).hexdigest(),
            },
            "n_ctx": 2048,
        },
        "js_sdk": {"supported": False, "base_model": None},
    }


def _meta(program_id: str, runtime: dict, *, embed_runtime: bool = True) -> dict:
    return {
        "version": 4,
        "program_id": program_id,
        "spec": "Classify the input as yes or no.",
        "compiler_snapshot": "paw-test-compiler",
        "interpreter": runtime["interpreter"],
        "runtime_id": runtime["runtime_id"],
        "runtime_manifest_version": runtime["manifest_version"],
        "runtime": runtime if embed_runtime else {},
        "created_at": "2026-07-17T00:00:00Z",
    }


def _write_program(
    program_id: str,
    runtime: dict,
    *,
    embed_runtime: bool = True,
    include_adapter: bool = True,
    include_template: bool = True,
    meta_program_id: str | None = None,
) -> Path:
    program_dir = config.get_programs_dir() / program_id
    program_dir.mkdir(parents=True, exist_ok=True)
    if include_adapter:
        (program_dir / "adapter.gguf").write_bytes(FAKE_ADAPTER_BYTES)
    if include_template:
        (program_dir / "prompt_template.txt").write_text(
            "Input: {INPUT_PLACEHOLDER}"
        )
    meta = _meta(meta_program_id or program_id, runtime, embed_runtime=embed_runtime)
    (program_dir / "meta.json").write_text(json.dumps(meta))
    return program_dir


def _write_base_model(runtime: dict) -> Path:
    path = config.get_base_models_dir() / runtime["local_sdk"]["base_model"]["file"]
    path.write_bytes(FAKE_MODEL_BYTES)
    return path


def _paw_archive(program_id: str, runtime: dict) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("adapter.gguf", FAKE_ADAPTER_BYTES)
        archive.writestr(
            "prompt_template.txt",
            "Input: {INPUT_PLACEHOLDER}",
        )
        archive.writestr(
            "meta.json",
            json.dumps(_meta(program_id, runtime)),
        )
    return out.getvalue()


class _Response:
    def __init__(
        self,
        data: dict,
        *,
        status_code: int = 200,
        content: bytes = b"",
        chunks: list[bytes] | None = None,
        headers: dict[str, str] | None = None,
    ):
        self._data = data
        self.status_code = status_code
        self.content = content
        self._chunks = chunks
        self.headers = headers or {}
        self.text = json.dumps(data)
        self.enter_count = 0
        self.exit_count = 0
        self.iterated_bytes = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.exit_count += 1

    def read(self) -> bytes:
        return self.content

    def iter_bytes(self, chunk_size: int = 8192):
        chunks = self._chunks
        if chunks is None:
            chunks = [
                self.content[index : index + chunk_size]
                for index in range(0, len(self.content), chunk_size)
            ]
        for chunk in chunks:
            self.iterated_bytes += len(chunk)
            yield chunk

    def json(self) -> dict:
        return self._data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://api.test")
            response = httpx.Response(
                self.status_code,
                request=request,
                json=self._data,
            )
            raise httpx.HTTPStatusError(
                "mock HTTP error",
                request=request,
                response=response,
            )


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    config_dir = tmp_path / "config"
    monkeypatch.setenv("PAW_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(config, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(config, "_CONFIG_FILE", config_dir / "config.json")
    monkeypatch.delenv("PAW_API_KEY", raising=False)
    monkeypatch.delenv("PAW_API_URL", raising=False)
    monkeypatch.delenv("PAW_OFFLINE", raising=False)


def test_precheck_compile_posts_expected_request(monkeypatch):
    seen = {}
    payload = {
        "cached": False,
        "program_id": None,
        "compiler_snapshot": "paw-ft-test",
        "compiler_kind": "finetune_lora",
        "queue_length": 2,
        "estimated_wait_s": 90.0,
    }

    def fake_post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return _Response(payload)

    monkeypatch.setattr(client_module.httpx, "post", fake_post)
    result = PAWClient(
        api_url="https://api.test/",
        api_key="paw_sk_test",
    ).precheck_compile("A sufficiently long test spec.", compiler="paw-ft-test")

    assert result == payload
    assert seen["url"] == "https://api.test/api/v1/compile/precheck"
    assert seen["json"] == {
        "spec": "A sufficiently long test spec.",
        "compiler": "paw-ft-test",
    }
    assert seen["headers"]["X-API-Key"] == "paw_sk_test"
    assert seen["timeout"] == 10.0


def test_compile_async_matches_compile_request_fields(monkeypatch):
    seen = {}
    payload = {
        "job_id": "job-123",
        "status": "queued",
        "program_id": None,
    }

    def fake_post(url, **kwargs):
        seen.update(url=url, **kwargs)
        return _Response(payload, status_code=202)

    monkeypatch.setattr(client_module.httpx, "post", fake_post)
    result = PAWClient(api_url="https://api.test").compile_async(
        "A sufficiently long async compilation spec.",
        compiler="paw-ft-test",
        name="Test program",
        tags=["desktop", "test"],
        public=False,
        slug="test-program",
        ephemeral=True,
    )

    assert result == payload
    assert seen["url"] == "https://api.test/api/v1/compile/async"
    assert seen["json"] == {
        "spec": "A sufficiently long async compilation spec.",
        "compiler": "paw-ft-test",
        "ephemeral": True,
        "name": "Test program",
        "tags": ["desktop", "test"],
        "slug": "test-program",
        "public": False,
    }
    assert seen["timeout"] == 30.0


def test_compile_status_and_cancel_use_job_endpoint(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(("get", url, kwargs))
        return _Response({"job_id": "job-123", "status": "compiling"})

    def fake_delete(url, **kwargs):
        calls.append(("delete", url, kwargs))
        return _Response({"job_id": "job-123", "status": "cancelled"})

    monkeypatch.setattr(client_module.httpx, "get", fake_get)
    monkeypatch.setattr(client_module.httpx, "delete", fake_delete)
    client = PAWClient(api_url="https://api.test")

    assert client.get_compile_status("job-123")["status"] == "compiling"
    assert client.cancel_compile("job-123")["status"] == "cancelled"
    assert calls == [
        (
            "get",
            "https://api.test/api/v1/compile/job-123",
            {"headers": {"Content-Type": "application/json"}, "timeout": 10.0},
        ),
        (
            "delete",
            "https://api.test/api/v1/compile/job-123",
            {"headers": {"Content-Type": "application/json"}, "timeout": 10.0},
        ),
    ]


def test_compile_status_type_matches_ready_server_fields():
    fields = client_module.CompileStatus.__annotations__
    for field in ("cached", "slug", "version", "version_action"):
        assert field in fields


def test_top_level_compile_wrappers_are_public(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, api_url=None, api_key=None):
            calls.append(("init", api_url, api_key))

        def precheck_compile(self, spec, compiler=None):
            return {"cached": True, "program_id": PROGRAM_ID}

        def compile_async(self, spec, **kwargs):
            return {"job_id": "job-123", "status": "queued"}

        def get_compile_status(self, job_id):
            return {"job_id": job_id, "status": "ready"}

        def cancel_compile(self, job_id):
            return {"job_id": job_id, "status": "cancelled"}

    monkeypatch.setattr(client_module, "PAWClient", FakeClient)

    assert paw.precheck_compile("A sufficiently long spec.")["cached"] is True
    assert paw.compile_async(
        "A sufficiently long spec.",
        compiler="paw-ft-test",
    )["job_id"] == "job-123"
    assert paw.get_compile_status("job-123")["status"] == "ready"
    assert paw.cancel_compile("job-123")["status"] == "cancelled"
    for name in (
        "precheck_compile",
        "compile_async",
        "get_compile_status",
        "cancel_compile",
    ):
        assert name in paw.__all__


def test_is_offline_ready_requires_runtime_and_base_model(monkeypatch):
    runtime = _runtime()
    _write_program(PROGRAM_ID, runtime)
    cache.save_slug_mapping("owner/test-program", PROGRAM_ID)

    def network_forbidden(*args, **kwargs):
        raise AssertionError("offline readiness must not use the network")

    monkeypatch.setattr(client_module.httpx, "get", network_forbidden)
    monkeypatch.setattr(client_module.httpx, "post", network_forbidden)
    monkeypatch.setattr(client_module.httpx, "delete", network_forbidden)
    monkeypatch.setattr(client_module.httpx, "stream", network_forbidden)

    assert paw.is_offline_ready(PROGRAM_ID) is False
    assert paw.is_offline_ready("owner/test-program") is False

    model_path = _write_base_model(runtime)
    assert paw.is_offline_ready(PROGRAM_ID) is True
    assert paw.is_offline_ready("owner/test-program") is True
    model_path.write_bytes(b"x")
    assert paw.is_offline_ready(PROGRAM_ID) is False
    model_path.write_bytes(FAKE_MODEL_BYTES)
    assert paw.is_offline_ready(PROGRAM_ID) is True
    assert paw.is_offline_ready("owner/not-cached") is False


def test_offline_readiness_uses_versioned_runtime_manifest_cache():
    runtime = _runtime()
    _write_program(PROGRAM_ID, runtime, embed_runtime=False)
    _write_base_model(runtime)

    assert paw.is_offline_ready(PROGRAM_ID) is False
    cache.save_runtime_manifest(runtime)
    assert paw.is_offline_ready(PROGRAM_ID) is True
    assert cache._runtime_manifest_path(runtime["runtime_id"], 1).exists()
    assert cache._runtime_manifest_path(runtime["runtime_id"]).exists()
    assert cache.get_cached_runtime_manifest(
        runtime["runtime_id"],
        1,
    ) == runtime

    wrong_version = dict(runtime)
    wrong_version["manifest_version"] = 2
    cache._atomic_write_json(
        cache._runtime_manifest_path(runtime["runtime_id"]),
        wrong_version,
    )
    assert cache.get_cached_runtime_manifest(runtime["runtime_id"]) == runtime
    assert paw.is_offline_ready(PROGRAM_ID) is True
    with pytest.raises(ValueError, match="Invalid runtime manifest"):
        cache.save_runtime_manifest(wrong_version)


def test_list_cached_programs_excludes_corrupt_partial_entries():
    ready_runtime = _runtime()
    cold_runtime = _runtime("cold-runtime", "cold-model.gguf")
    _write_program(PROGRAM_ID, ready_runtime)
    _write_base_model(ready_runtime)
    _write_program(OTHER_PROGRAM_ID, cold_runtime)
    cache.save_slug_mapping("owner/ready", PROGRAM_ID)

    partial_id = "c" * 20
    partial_dir = config.get_programs_dir() / partial_id
    partial_dir.mkdir()
    (partial_dir / "adapter.gguf").write_bytes(b"adapter")

    mismatched_id = "d" * 20
    _write_program(mismatched_id, ready_runtime, meta_program_id="e" * 20)

    malformed_id = "f" * 20
    malformed_dir = config.get_programs_dir() / malformed_id
    malformed_dir.mkdir()
    (malformed_dir / "adapter.gguf").write_bytes(b"adapter")
    (malformed_dir / "prompt_template.txt").write_text("template")
    (malformed_dir / "meta.json").write_text("{not-json")

    programs = paw.list_cached_programs()

    assert [item["program_id"] for item in programs] == [
        PROGRAM_ID,
        OTHER_PROGRAM_ID,
    ]
    assert programs[0]["slugs"] == ["owner/ready"]
    assert programs[0]["offline_ready"] is True
    assert programs[1]["offline_ready"] is False


def test_prepare_program_resolves_downloads_and_reports_progress(
    monkeypatch,
    capsys,
):
    runtime = _runtime()
    archive = _paw_archive(PROGRAM_ID, runtime)
    requested_urls = []

    def fake_get(url, **kwargs):
        requested_urls.append(url)
        if "/resolve/" in url:
            return _Response({"program_id": PROGRAM_ID})
        raise AssertionError(f"unexpected URL: {url}")

    download_response = _Response({}, content=archive)

    def fake_stream(method, url, **kwargs):
        assert method == "GET"
        requested_urls.append(url)
        return download_response

    def fake_download(url, dest, progress=None):
        assert url == "https://example.test/test-model.gguf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(FAKE_MODEL_BYTES)
        if progress is not None:
            progress(
                {
                    "stage": "base_model",
                    "status": "downloading",
                    "path": str(dest),
                    "downloaded_bytes": len(FAKE_MODEL_BYTES),
                    "total_bytes": len(FAKE_MODEL_BYTES),
                }
            )

    monkeypatch.setattr(client_module.httpx, "get", fake_get)
    monkeypatch.setattr(client_module.httpx, "stream", fake_stream)
    monkeypatch.setattr(cache, "_download_file", fake_download)
    events = []

    prepared = paw.prepare_program("owner/test-program", progress=events.append)

    assert prepared["program_id"] == PROGRAM_ID
    assert prepared["offline_ready"] is True
    assert (
        Path(prepared["base_model_path"]).read_bytes()
        == FAKE_MODEL_BYTES
    )
    assert requested_urls == [
        "https://programasweights.com/api/v1/programs/resolve/owner/test-program",
        f"https://programasweights.com/api/v1/programs/{PROGRAM_ID}/download",
    ]
    assert download_response.enter_count == 1
    assert download_response.exit_count == 1
    assert [event["stage"] for event in events] == [
        "resolve",
        "resolve",
        "program",
        "program",
        "runtime",
        "base_model",
        "base_model",
        "base_model",
        "prepare",
    ]
    assert capsys.readouterr().err == ""


def test_prepare_program_offline_never_downloads(monkeypatch):
    runtime = _runtime()
    _write_program(PROGRAM_ID, runtime)

    def network_forbidden(*args, **kwargs):
        raise AssertionError("offline preparation must not use the network")

    monkeypatch.setattr(client_module.httpx, "get", network_forbidden)
    monkeypatch.setattr(client_module.httpx, "post", network_forbidden)
    monkeypatch.setattr(client_module.httpx, "stream", network_forbidden)

    with pytest.raises(RuntimeError, match="Base model"):
        paw.prepare_program(PROGRAM_ID, offline=True)

    _write_base_model(runtime)
    prepared = paw.prepare_program(PROGRAM_ID, offline=True)
    assert prepared["offline_ready"] is True


def test_program_object_resolution_prefers_immutable_id():
    runtime = _runtime()
    _write_program(PROGRAM_ID, runtime)
    _write_base_model(runtime)
    cache.save_slug_mapping("owner/stale", OTHER_PROGRAM_ID)
    program = paw.Program(
        id=PROGRAM_ID,
        slug="owner/stale",
        status="ready",
    )

    assert paw.is_offline_ready(program) is True


def test_compile_async_requires_explicit_nonempty_compiler():
    spec = "A sufficiently long asynchronous compilation spec."
    client = PAWClient(api_url="https://api.test")

    with pytest.raises(TypeError):
        client.compile_async(spec)
    with pytest.raises(ValueError, match="explicit finetune compiler"):
        client.compile_async(spec, compiler="")
    with pytest.raises(TypeError):
        paw.compile_async(spec)


def test_program_asset_validation_is_strict():
    runtime = _runtime()
    program_dir = _write_program(PROGRAM_ID, runtime)
    assert cache.has_valid_program_assets(PROGRAM_ID)

    (program_dir / "prompt_template.txt").write_text("no placeholder")
    assert cache.has_valid_program_assets(PROGRAM_ID) is False

    (program_dir / "prompt_template.txt").write_text(
        "{INPUT_PLACEHOLDER}{INPUT_PLACEHOLDER}"
    )
    assert cache.has_valid_program_assets(PROGRAM_ID) is False

    (program_dir / "prompt_template.txt").write_text(
        "{INPUT_PLACEHOLDER}"
    )
    (program_dir / "adapter.gguf").write_bytes(b"x")
    assert cache.has_valid_program_assets(PROGRAM_ID) is False

    (program_dir / "adapter.gguf").write_bytes(
        b"NOPE" + (b"A" * (cache.MIN_ADAPTER_GGUF_SIZE - 4))
    )
    assert cache.has_valid_program_assets(PROGRAM_ID) is False


def test_function_does_not_accept_partial_stale_assets(
    monkeypatch,
):
    program_dir = config.get_programs_dir() / PROGRAM_ID
    program_dir.mkdir()
    (program_dir / "adapter.gguf").write_bytes(b"stale")
    (program_dir / "prompt_template.txt").write_text(
        "{INPUT_PLACEHOLDER}"
    )
    calls = []

    class FakeClient:
        def __init__(self, api_url=None, api_key=None):
            pass

        def download_paw(self, program_id):
            calls.append(program_id)
            return program_dir

    runtime_module = types.ModuleType("programasweights.runtime_llamacpp")

    class ForbiddenPawFunction:
        def __init__(self, *args, **kwargs):
            raise AssertionError("invalid assets must not reach the runtime")

    runtime_module.PawFunction = ForbiddenPawFunction
    monkeypatch.setitem(
        sys.modules,
        "programasweights.runtime_llamacpp",
        runtime_module,
    )
    monkeypatch.setattr(client_module, "PAWClient", FakeClient)

    with pytest.raises(RuntimeError, match="missing valid compiled assets"):
        paw.function(PROGRAM_ID)
    assert calls == [PROGRAM_ID]


def test_runtime_and_slug_cache_reject_invalid_json_shapes():
    runtime = _runtime()
    runtime_path = (
        config.get_cache_dir()
        / "runtimes"
        / f"{runtime['runtime_id']}.json"
    )
    runtime_path.parent.mkdir(parents=True)
    runtime_path.write_text(
        json.dumps(
            {
                "runtime_id": runtime["runtime_id"],
                "manifest_version": "one",
                "local_sdk": {},
            }
        )
    )
    assert cache.get_cached_runtime_manifest(runtime["runtime_id"]) is None

    with pytest.raises(ValueError, match="Invalid runtime manifest"):
        cache.save_runtime_manifest(
            {
                "runtime_id": runtime["runtime_id"],
                "manifest_version": 1,
            }
        )

    slug_path = config.get_cache_dir() / "slug_cache.json"
    slug_path.write_text(
        json.dumps(
            {
                "owner/valid": PROGRAM_ID,
                "owner/invalid": 123,
            }
        )
    )
    assert cache._load_slug_mappings() == {
        "owner/valid": PROGRAM_ID,
    }
    cache.save_slug_mapping("owner/new", OTHER_PROGRAM_ID)
    assert cache._load_slug_mappings() == {
        "owner/valid": PROGRAM_ID,
        "owner/new": OTHER_PROGRAM_ID,
    }


@pytest.mark.parametrize(
    ("field", "value", "error_match"),
    [
        ("runtime_id", "different-runtime", "returned runtime ID"),
        ("manifest_version", 2, "invalid runtime manifest"),
        ("adapter_format", "raw_lora", "invalid runtime manifest"),
    ],
)
def test_fetch_runtime_manifest_rejects_untrusted_contract(
    monkeypatch,
    field,
    value,
    error_match,
):
    requested_id = "test-runtime"
    runtime = _runtime(requested_id)
    runtime[field] = value
    monkeypatch.setattr(
        cache.httpx,
        "get",
        lambda *args, **kwargs: _Response(runtime),
    )

    with pytest.raises(ValueError, match=error_match):
        cache.fetch_runtime_manifest(
            requested_id,
            api_url="https://api.test",
        )

    assert cache.get_cached_runtime_manifest(requested_id, 1) is None
    assert not cache._runtime_manifest_path(requested_id).exists()


def test_fetch_known_runtime_overlays_canonical_integrity(monkeypatch):
    canonical = cache.get_base_runtime_manifest("gpt2")
    server_manifest = json.loads(json.dumps(canonical))
    server_base = server_manifest["local_sdk"]["base_model"]
    server_base.pop("size_bytes")
    server_base["sha256"] = None
    server_manifest.pop("base_inference", None)
    monkeypatch.setattr(
        cache.httpx,
        "get",
        lambda *args, **kwargs: _Response(server_manifest),
    )

    fetched = cache.fetch_runtime_manifest(
        "gpt2-q8_0",
        api_url="https://api.test",
    )

    canonical_base = canonical["local_sdk"]["base_model"]
    assert fetched["local_sdk"]["base_model"]["size_bytes"] == canonical_base[
        "size_bytes"
    ]
    assert fetched["local_sdk"]["base_model"]["sha256"] == canonical_base[
        "sha256"
    ]
    assert fetched["base_inference"] == canonical["base_inference"]
    assert cache.get_cached_runtime_manifest(
        "gpt2-q8_0",
        1,
    ) == fetched


def test_unknown_runtime_keeps_unpinned_integrity_contract():
    runtime = _runtime("third-party-runtime", "third-party.gguf")
    base_model = runtime["local_sdk"]["base_model"]
    base_model.pop("size_bytes")
    base_model["sha256"] = None
    cache.save_runtime_manifest(runtime)

    cached = cache.get_cached_runtime_manifest("third-party-runtime", 1)
    assert cached is not None
    assert "size_bytes" not in cached["local_sdk"]["base_model"]
    assert cached["local_sdk"]["base_model"]["sha256"] is None

    model_path = config.get_base_models_dir() / "third-party.gguf"
    model_path.write_bytes(FAKE_MODEL_BYTES)
    assert cache.get_cached_base_model_path(cached) == model_path


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("prompt_template", "format", "raw_text"),
        ("prompt_template", "placeholder", "{OTHER}"),
        ("program_assets", "adapter_filename", "adapter.bin"),
    ],
)
def test_runtime_manifest_rejects_invalid_optional_contracts(
    section,
    field,
    value,
):
    runtime = _runtime()
    runtime[section][field] = value
    with pytest.raises(ValueError, match="Invalid runtime manifest"):
        cache.save_runtime_manifest(runtime)


def test_slug_updates_are_serialized_and_metadata_is_consistent():
    runtime = _runtime()
    _write_program(PROGRAM_ID, runtime)
    slugs = [f"owner/program-{index}" for index in range(20)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(
            executor.map(
                lambda slug: cache.save_slug_mapping(slug, PROGRAM_ID),
                slugs,
            )
        )

    assert cache._load_slug_mappings() == {
        slug: PROGRAM_ID for slug in slugs
    }
    metadata = cache.get_cached_program_metadata(PROGRAM_ID)
    assert metadata is not None
    assert metadata["slugs"] == sorted(slugs)
    listed = cache.list_cached_programs()
    assert listed[0]["slugs"] == sorted(slugs)


def test_windows_lock_retries_nonblocking_until_acquired():
    class FakeLockFile:
        def __init__(self):
            self.positions = []

        def seek(self, position, *args):
            self.positions.append(position)

        def fileno(self):
            return 17

    class FakeMsvcrt:
        LK_NBLCK = 2

        def __init__(self):
            self.calls = 0

        def locking(self, fd, mode, count):
            assert (fd, mode, count) == (17, self.LK_NBLCK, 1)
            self.calls += 1
            if self.calls < 3:
                raise OSError("busy")

    now = [0.0]
    sleeps = []

    def monotonic():
        return now[0]

    def sleep(duration):
        sleeps.append(duration)
        now[0] += duration

    fake_file = FakeLockFile()
    fake_msvcrt = FakeMsvcrt()
    cache._acquire_windows_file_lock(
        fake_file,
        fake_msvcrt,
        timeout_s=10.0,
        retry_s=0.25,
        monotonic=monotonic,
        sleep=sleep,
    )

    assert fake_msvcrt.calls == 3
    assert fake_file.positions == [0, 0, 0]
    assert sleeps == [0.25, 0.25]
    assert cache.WINDOWS_LOCK_TIMEOUT_S >= 60 * 60


def test_windows_lock_timeout_is_bounded():
    class AlwaysBusyMsvcrt:
        LK_NBLCK = 2

        @staticmethod
        def locking(fd, mode, count):
            raise OSError("busy")

    fake_file = types.SimpleNamespace(
        seek=lambda *args: None,
        fileno=lambda: 17,
    )
    now = [0.0]

    def sleep(duration):
        now[0] += duration

    with pytest.raises(TimeoutError, match="Timed out"):
        cache._acquire_windows_file_lock(
            fake_file,
            AlwaysBusyMsvcrt,
            timeout_s=0.5,
            retry_s=0.2,
            monotonic=lambda: now[0],
            sleep=sleep,
        )


def test_base_model_cache_validates_declared_sha256():
    runtime = _runtime()
    expected = b"GGUF" + (b"E" * 124)
    base_model = runtime["local_sdk"]["base_model"]
    base_model["size_bytes"] = len(expected)
    base_model["sha256"] = hashlib.sha256(expected).hexdigest()
    path = config.get_base_models_dir() / "test-model.gguf"
    path.write_bytes(b"GGUF" + (b"X" * 124))

    assert cache.get_cached_base_model_path(runtime) is None
    with pytest.raises(RuntimeError, match="offline mode prohibits"):
        cache.get_base_model_path(
            runtime["interpreter"],
            runtime_manifest=runtime,
            offline=True,
        )

    path.write_bytes(expected)
    assert cache.get_cached_base_model_path(runtime) == path


def test_builtin_models_pin_verified_lfs_integrity():
    expected = {
        "gpt2": (
            139804832,
            "0aa260efb2cce9def922e0546b88ad731cf1a68554db73fa2d4a0949cfa958c5",
        ),
        "Qwen/Qwen3-0.6B": (
            622733120,
            "9a16ed5cacba959e63b62e2b6840c3eca2b51c3c3e51d31367ef8e4aafeae33c",
        ),
    }
    for interpreter, (size_bytes, sha256) in expected.items():
        base_model = cache.get_base_runtime_manifest(interpreter)[
            "local_sdk"
        ]["base_model"]
        assert base_model["size_bytes"] == size_bytes
        assert base_model["sha256"] == sha256


def test_known_historical_manifest_cannot_bypass_canonical_integrity(
    monkeypatch,
):
    canonical = cache.LEGACY_RUNTIME_MANIFESTS["gpt2-q8_0"]
    canonical_base = canonical["local_sdk"]["base_model"]
    fake_sha256 = hashlib.sha256(FAKE_MODEL_BYTES).hexdigest()
    monkeypatch.setitem(
        canonical_base,
        "size_bytes",
        len(FAKE_MODEL_BYTES),
    )
    monkeypatch.setitem(canonical_base, "sha256", fake_sha256)

    historical = json.loads(json.dumps(canonical))
    historical_base = historical["local_sdk"]["base_model"]
    historical_base.pop("size_bytes")
    historical_base["sha256"] = None
    historical.pop("base_inference", None)
    program_dir = _write_program(PROGRAM_ID, historical)
    model_path = (
        config.get_base_models_dir()
        / historical_base["file"]
    )
    model_path.write_bytes(b"GGUF")

    assert cache.has_valid_program_assets(PROGRAM_ID) is True
    assert cache.get_cached_base_model_path(historical) is None
    assert paw.is_offline_ready(PROGRAM_ID) is False
    with pytest.raises(RuntimeError, match="offline mode prohibits"):
        cache.get_base_model_path(
            "gpt2",
            runtime_manifest=historical,
            offline=True,
        )

    normalized = cache.get_offline_runtime_manifest(
        json.loads((program_dir / "meta.json").read_text())
    )
    assert normalized is not None
    assert normalized["local_sdk"]["base_model"]["size_bytes"] == len(
        FAKE_MODEL_BYTES
    )
    assert normalized["local_sdk"]["base_model"]["sha256"] == fake_sha256

    model_path.write_bytes(FAKE_MODEL_BYTES)
    assert cache.get_cached_base_model_path(historical) == model_path
    assert paw.is_offline_ready(PROGRAM_ID) is True
    assert cache.get_base_model_path(
        "gpt2",
        runtime_manifest=historical,
        offline=True,
    ) == model_path


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("interpreter",), "other"),
        (("manifest_version",), 2),
        (("adapter_format",), "raw_lora"),
        (("local_sdk", "base_model", "file"), "other.gguf"),
        (("local_sdk", "n_ctx"), 4096),
        (("program_assets", "prefix_cache_required"), False),
        (("local_sdk", "base_model", "size_bytes"), 1),
        (("local_sdk", "base_model", "sha256"), "0" * 64),
    ],
)
def test_known_runtime_rejects_noncanonical_contract(path, value):
    runtime = cache.get_base_runtime_manifest("gpt2")
    target = runtime
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="Invalid runtime manifest"):
        cache.save_runtime_manifest(runtime)


def test_download_paw_replaces_stale_directory_from_staging(monkeypatch):
    runtime = _runtime()
    archive = _paw_archive(PROGRAM_ID, runtime)
    program_dir = config.get_programs_dir() / PROGRAM_ID
    program_dir.mkdir()
    (program_dir / "adapter.gguf").write_bytes(b"stale")
    (program_dir / "stale.txt").write_text("must not survive")

    monkeypatch.setattr(
        client_module.httpx,
        "stream",
        lambda *args, **kwargs: _Response({}, content=archive),
    )

    result = PAWClient(api_url="https://api.test").download_paw(PROGRAM_ID)

    assert result == program_dir
    assert cache.has_valid_program_assets(PROGRAM_ID)
    assert not (program_dir / "stale.txt").exists()
    staging = config.get_programs_dir() / ".staging"
    assert list(staging.iterdir()) == []


@pytest.mark.parametrize(
    ("retry_response", "expected_sleep"),
    [
        (
            _Response(
                {},
                status_code=202,
                headers={"Retry-After": "2"},
            ),
            2,
        ),
        (
            _Response(
                {"detail": "assets still generating"},
                status_code=404,
            ),
            3,
        ),
    ],
)
def test_download_retries_close_stream_contexts(
    monkeypatch,
    retry_response,
    expected_sleep,
):
    success_response = _Response(
        {},
        content=_paw_archive(PROGRAM_ID, _runtime()),
    )
    responses = iter([retry_response, success_response])
    sleeps = []
    monkeypatch.setattr(
        client_module.httpx,
        "stream",
        lambda *args, **kwargs: next(responses),
    )
    monkeypatch.setattr(client_module.time, "sleep", sleeps.append)

    result = PAWClient(api_url="https://api.test").download_paw(PROGRAM_ID)

    assert result == config.get_programs_dir() / PROGRAM_ID
    assert sleeps == [expected_sleep]
    assert retry_response.exit_count == 1
    assert success_response.exit_count == 1


def test_download_not_found_closes_stream_context(monkeypatch):
    response = _Response(
        {"detail": "program not found"},
        status_code=404,
    )
    monkeypatch.setattr(
        client_module.httpx,
        "stream",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(RuntimeError, match="not found"):
        PAWClient(api_url="https://api.test").download_paw(PROGRAM_ID)
    assert response.exit_count == 1


def test_download_paw_rejects_mismatched_staged_metadata_and_cleans(
    monkeypatch,
):
    runtime = _runtime()
    archive = _paw_archive(OTHER_PROGRAM_ID, runtime)
    monkeypatch.setattr(
        client_module.httpx,
        "stream",
        lambda *args, **kwargs: _Response({}, content=archive),
    )

    with pytest.raises(RuntimeError, match="mismatched staged assets"):
        PAWClient(api_url="https://api.test").download_paw(PROGRAM_ID)

    assert not config.get_programs_dir().joinpath(PROGRAM_ID).exists()
    staging = config.get_programs_dir() / ".staging"
    assert list(staging.iterdir()) == []


@pytest.mark.parametrize(
    "member",
    ("../escape", "..\\escape", "C:/escape"),
)
def test_download_paw_rejects_archive_traversal_and_cleans(
    monkeypatch,
    member,
):
    archive_file = io.BytesIO()
    with zipfile.ZipFile(archive_file, "w") as archive:
        archive.writestr(member, b"bad")
    monkeypatch.setattr(
        client_module.httpx,
        "stream",
        lambda *args, **kwargs: _Response(
            {},
            content=archive_file.getvalue(),
        ),
    )

    with pytest.raises(ValueError, match="Unsafe path"):
        PAWClient(api_url="https://api.test").download_paw(PROGRAM_ID)

    assert not config.get_programs_dir().joinpath(PROGRAM_ID).exists()
    assert not config.get_cache_dir().joinpath("escape").exists()
    staging = config.get_programs_dir() / ".staging"
    assert list(staging.iterdir()) == []


def test_download_paw_rejects_oversized_response(monkeypatch):
    archive = _paw_archive(PROGRAM_ID, _runtime())
    monkeypatch.setattr(
        client_module,
        "MAX_PAW_ARCHIVE_BYTES",
        len(archive) - 1,
    )
    monkeypatch.setattr(
        client_module.httpx,
        "stream",
        lambda *args, **kwargs: _Response({}, content=archive),
    )

    with pytest.raises(ValueError, match="too large"):
        PAWClient(api_url="https://api.test").download_paw(PROGRAM_ID)
    assert not config.get_programs_dir().joinpath(PROGRAM_ID).exists()


def test_download_paw_rejects_oversized_content_length(monkeypatch):
    archive = _paw_archive(PROGRAM_ID, _runtime())
    response = _Response(
        {},
        content=archive,
        headers={
            "content-length": str(
                client_module.MAX_PAW_ARCHIVE_BYTES + 1
            )
        },
    )
    monkeypatch.setattr(
        client_module.httpx,
        "stream",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(ValueError, match="too large"):
        PAWClient(api_url="https://api.test").download_paw(PROGRAM_ID)
    assert response.iterated_bytes == 0
    assert response.exit_count == 1


def test_download_paw_aborts_oversized_chunked_stream(monkeypatch):
    monkeypatch.setattr(client_module, "MAX_PAW_ARCHIVE_BYTES", 10)
    response = _Response(
        {},
        chunks=[b"12345678", b"ABCDEFGH", b"not-consumed"],
    )
    monkeypatch.setattr(
        client_module.httpx,
        "stream",
        lambda *args, **kwargs: response,
    )

    with pytest.raises(ValueError, match="too large"):
        PAWClient(api_url="https://api.test").download_paw(PROGRAM_ID)

    assert response.iterated_bytes == 16
    assert response.exit_count == 1
    staging = config.get_programs_dir() / ".staging"
    assert list(staging.iterdir()) == []


def test_safe_extract_rejects_excessive_member_count(
    monkeypatch,
    tmp_path,
):
    paw_path = tmp_path / "many.paw"
    with zipfile.ZipFile(paw_path, "w") as archive:
        archive.writestr("one", b"1")
        archive.writestr("two", b"2")
        archive.writestr("three", b"3")
    destination = tmp_path / "out"
    destination.mkdir()
    monkeypatch.setattr(client_module, "MAX_PAW_ARCHIVE_MEMBERS", 2)

    with pytest.raises(ValueError, match="members"):
        PAWClient._safe_extract_paw(paw_path, destination)
    assert list(destination.iterdir()) == []


def test_safe_extract_rejects_excessive_expanded_size(
    monkeypatch,
    tmp_path,
):
    paw_path = tmp_path / "expanded.paw"
    with zipfile.ZipFile(
        paw_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("large", b"x" * 4096)
    destination = tmp_path / "out"
    destination.mkdir()
    monkeypatch.setattr(client_module, "MAX_PAW_EXPANDED_BYTES", 1024)

    with pytest.raises(ValueError, match="expands"):
        PAWClient._safe_extract_paw(paw_path, destination)
    assert list(destination.iterdir()) == []


def test_download_paw_serializes_same_program_install(monkeypatch):
    runtime = _runtime()
    archive = _paw_archive(PROGRAM_ID, runtime)
    calls = []

    def fake_stream(method, url, **kwargs):
        assert method == "GET"
        calls.append(url)
        return _Response({}, content=archive)

    monkeypatch.setattr(client_module.httpx, "stream", fake_stream)
    clients = [
        PAWClient(api_url="https://api.test"),
        PAWClient(api_url="https://api.test"),
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda client: client.download_paw(PROGRAM_ID),
                clients,
            )
        )

    assert results == [
        config.get_programs_dir() / PROGRAM_ID,
        config.get_programs_dir() / PROGRAM_ID,
    ]
    assert len(calls) == 1
    assert cache.has_valid_program_assets(PROGRAM_ID)
