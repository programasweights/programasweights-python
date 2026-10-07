"""Verified two-asset caching, with no real HTTP or model initialization."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import httpx
import pytest

from programasweights import config
from programasweights import _vision_assets as assets
from test_vision_metadata import MODEL, PROJECTOR, asset, manifest  # shared inert fixture


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("PAW_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PAW_QUIET", "1")
    monkeypatch.delenv("PAW_OFFLINE", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected HTTP request")

    monkeypatch.setattr(httpx, "stream", forbidden)


@pytest.fixture
def serve(monkeypatch, manifest):
    calls = []
    payloads = {
        manifest["local_sdk"]["base_model"]["url"]: MODEL,
        manifest["local_sdk"]["vision"]["projector"]["url"]: PROJECTOR,
    }

    def install(handler=None):
        def respond(request):
            calls.append(str(request.url))
            if handler is not None:
                response = handler(request)
                if response is not None:
                    return response
            return httpx.Response(200, content=payloads[str(request.url)])

        @contextmanager
        def stream(method, url, **kwargs):
            with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                with client.stream(method, url, **kwargs) as response:
                    yield response

        monkeypatch.setattr(httpx, "stream", stream)
        return calls, payloads
    return install


def location(data):
    return config.get_base_models_dir() / "sha256" / (hashlib.sha256(data).hexdigest() + ".gguf")


def assert_no_temporary_files():
    assert not list(config.get_base_models_dir().rglob("*.tmp"))


def test_download_then_offline_reuse_and_progress(manifest, serve):
    calls, _ = serve()
    events = []
    assert assets.get_cached_vision_asset_paths(manifest) is None
    paths = assets.get_vision_asset_paths(manifest, progress=events.append)
    assert paths == (location(MODEL), location(PROJECTOR))
    assert paths.base_model.read_bytes() == MODEL
    assert paths.projector.read_bytes() == PROJECTOR
    assert len(calls) == 2
    assert assets.get_vision_asset_paths(manifest, offline=True) == paths
    assert assets.get_cached_vision_asset_paths(manifest) == paths
    assert len(calls) == 2
    for stage in ("base_model", "vision_projector"):
        selected = [e for e in events if e["stage"] == stage]
        assert selected[0]["status"] == "downloading"
        assert selected[-1]["status"] == "ready"
        assert selected[-2]["downloaded_bytes"] == 4096
        assert selected[-2]["total_bytes"] == 4096
        assert all(e["runtime_id"] == manifest["runtime_id"] for e in selected)
        assert all(e["path"] == str(paths.base_model if stage == "base_model" else paths.projector)
                   for e in selected)
    assert_no_temporary_files()


@pytest.mark.parametrize("environment", [None, "1", " true ", "YES"])
def test_offline_never_downloads_missing_assets(manifest, monkeypatch, environment):
    if environment is not None:
        monkeypatch.setenv("PAW_OFFLINE", environment)
    with pytest.raises(RuntimeError, match="base_model.*offline mode"):
        assets.get_vision_asset_paths(manifest, offline=environment is None)
    assert assets.get_cached_vision_asset_paths(manifest) is None
    assert_no_temporary_files()


def test_offline_can_adopt_legacy_files_without_deleting_or_duplicating_them(manifest):
    directory = config.get_base_models_dir()
    base, projector = directory / "base.gguf", directory / "mmproj.gguf"
    base.write_bytes(MODEL)
    projector.write_bytes(PROJECTOR)
    paths = assets.get_vision_asset_paths(manifest, offline=True)
    assert paths.base_model.read_bytes() == MODEL
    assert paths.projector.read_bytes() == PROJECTOR
    assert base.read_bytes() == MODEL
    assert projector.read_bytes() == PROJECTOR
    assert os.path.samefile(base, paths.base_model)
    assert os.path.samefile(projector, paths.projector)
    # A legacy cache replacement must not change the content-addressed file.
    replacement = directory / "new.gguf"
    replacement.write_bytes(b"different legacy version")
    os.replace(str(replacement), str(base))
    assert paths.base_model.read_bytes() == MODEL


def test_adoption_falls_back_to_verified_copy_when_links_unavailable(manifest, monkeypatch):
    directory = config.get_base_models_dir()
    (directory / "base.gguf").write_bytes(MODEL)
    (directory / "mmproj.gguf").write_bytes(PROJECTOR)

    def no_link(*args):
        raise OSError("Hard links unavailable")
    monkeypatch.setattr(assets.os, "link", no_link)
    paths = assets.get_vision_asset_paths(manifest, offline=True)
    assert paths.base_model.read_bytes() == MODEL
    assert not os.path.samefile(directory / "base.gguf", paths.base_model)
    assert paths.projector.read_bytes() == PROJECTOR
    assert_no_temporary_files()


def test_changed_legacy_source_is_not_published(manifest, monkeypatch):
    source = config.get_base_models_dir() / "base.gguf"
    source.write_bytes(MODEL)

    def changed_link(source_path, target):
        Path(target).write_bytes(b"changed")
    monkeypatch.setattr(assets.os, "link", changed_link)
    with pytest.raises(RuntimeError, match="changed while being adopted"):
        assets.get_vision_asset_paths(manifest, offline=True)
    assert not location(MODEL).exists()
    assert source.read_bytes() == MODEL
    assert_no_temporary_files()


def test_same_filenames_different_hashes_remain_independent(manifest, serve):
    calls, payloads = serve()
    first = assets.get_vision_asset_paths(manifest)
    other = copy.deepcopy(manifest)
    second_model = b"GGUF" + b"N" * 4092
    other["runtime_id"] = "other-runtime"
    other["local_sdk"]["base_model"] = asset("base.gguf", second_model)
    payloads[other["local_sdk"]["base_model"]["url"]] = second_model
    second = assets.get_vision_asset_paths(other)
    assert first.base_model != second.base_model
    assert first.base_model.read_bytes() == MODEL
    assert second.base_model.read_bytes() == second_model
    assert first.projector == second.projector
    assert len(calls) == 3


def test_different_filenames_same_content_share_cache(manifest, serve):
    calls, _ = serve()
    original = assets.get_vision_asset_paths(manifest)
    other = copy.deepcopy(manifest)
    other["runtime_id"] = "renamed-runtime"
    for descriptor in [other["local_sdk"]["base_model"], other["local_sdk"]["vision"]["projector"]]:
        descriptor["file"] = "renamed-" + descriptor["file"]
        descriptor["url"] = "https://unused.invalid/" + descriptor["file"]
        descriptor["sha256"] = descriptor["sha256"].upper()
    assert assets.get_vision_asset_paths(other, offline=True) == original
    assert len(calls) == 2


def test_missing_projector_does_not_redownload_cached_base(manifest, serve):
    calls, _ = serve()
    paths = assets.get_vision_asset_paths(manifest)
    paths.projector.unlink()
    with pytest.raises(RuntimeError, match="vision_projector.*offline mode"):
        assets.get_vision_asset_paths(manifest, offline=True)
    assert len(calls) == 2
    assert assets.get_cached_vision_asset_paths(manifest) is None
    assert assets.get_vision_asset_paths(manifest) == paths
    assert len(calls) == 3


@pytest.mark.parametrize("failure", ["hash", "short", "oversize", "magic", "http", "interrupt"])
def test_failed_download_never_publishes_and_can_retry(manifest, serve, failure):
    candidate = copy.deepcopy(manifest)
    if failure == "magic":
        bad = b"NOPE" + b"M" * 4092
        candidate["local_sdk"]["base_model"] = asset("base.gguf", bad)
    else:
        bad = {"hash": b"GGUF" + b"X" * 4092, "short": MODEL[:-1],
               "oversize": MODEL + b"extra"}.get(failure, MODEL)
    closed = []

    class Broken(httpx.SyncByteStream):
        def __iter__(self):
            yield MODEL[:100]
            raise httpx.ReadError("Connection lost")
        def close(self):
            closed.append(True)

    def handler(request):
        if failure == "http":
            return httpx.Response(503)
        if failure == "interrupt":
            return httpx.Response(200, stream=Broken())
        return httpx.Response(200, content=bad)

    calls, _ = serve(handler)
    with pytest.raises((RuntimeError, httpx.HTTPError)):
        assets.get_vision_asset_paths(candidate)
    target = config.get_base_models_dir() / "sha256" / (
        candidate["local_sdk"]["base_model"]["sha256"] + ".gguf")
    assert not target.exists()
    assert assets.get_cached_vision_asset_paths(manifest) is None
    assert_no_temporary_files()
    if failure == "interrupt":
        assert closed
    # The failed attempt releases its lock and cleans up its own temporary file.
    serve()
    assert assets.get_vision_asset_paths(manifest).base_model.read_bytes() == MODEL


def test_projector_failure_retains_completed_base(manifest, serve):
    projector_url = manifest["local_sdk"]["vision"]["projector"]["url"]
    calls, _ = serve(lambda request: httpx.Response(503) if str(request.url) == projector_url else None)
    with pytest.raises(httpx.HTTPStatusError):
        assets.get_vision_asset_paths(manifest)
    assert location(MODEL).read_bytes() == MODEL
    assert not location(PROJECTOR).exists()
    assert_no_temporary_files()
    serve()
    paths = assets.get_vision_asset_paths(manifest)
    assert paths.projector.read_bytes() == PROJECTOR
    assert calls.count(manifest["local_sdk"]["base_model"]["url"]) == 1
    assert calls.count(projector_url) == 2


def test_corrupt_cache_is_never_reused_and_failed_repair_does_not_delete_it(manifest, serve):
    serve()
    paths = assets.get_vision_asset_paths(manifest)
    corrupted = b"GGUF" + b"X" * 4092
    paths.base_model.write_bytes(corrupted)
    assert assets.get_cached_vision_asset_paths(manifest) is None
    with pytest.raises(RuntimeError, match="offline mode"):
        assets.get_vision_asset_paths(manifest, offline=True)
    serve(lambda request: httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        assets.get_vision_asset_paths(manifest)
    assert paths.base_model.read_bytes() == corrupted
    assert paths.projector.read_bytes() == PROJECTOR
    serve()
    assert assets.get_vision_asset_paths(manifest).base_model.read_bytes() == MODEL
    assert_no_temporary_files()


def test_progress_exception_cleans_up_before_publish(manifest, serve):
    serve()
    def callback(event):
        if event.get("downloaded_bytes"):
            assert not Path(event["path"]).exists()
            raise RuntimeError("Caller cancelled")
    with pytest.raises(RuntimeError, match="Caller cancelled"):
        assets.get_vision_asset_paths(manifest, progress=callback)
    assert not location(MODEL).exists()
    assert_no_temporary_files()


def test_http_content_length_does_not_override_declared_size(manifest, serve):
    payloads = {"base.gguf": MODEL, "mmproj.gguf": PROJECTOR}
    serve(lambda request: httpx.Response(
        200, content=payloads[request.url.path.rsplit("/", 1)[-1]],
        headers={"content-length": "1"},
    ))
    paths = assets.get_vision_asset_paths(manifest)
    assert paths.base_model.read_bytes() == MODEL
    assert paths.projector.read_bytes() == PROJECTOR


def test_threads_share_one_download_per_asset(manifest, serve):
    entered = threading.Event()
    release = threading.Event()
    def handler(request):
        entered.set()
        assert release.wait(5)
    calls, _ = serve(handler)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(assets.get_vision_asset_paths, manifest) for _ in range(4)]
        assert entered.wait(5)
        release.set()
        paths = [f.result(timeout=10) for f in futures]
    assert len(calls) == 2
    assert paths == [paths[0]] * 4
    assert_no_temporary_files()


def test_two_processes_share_one_download_per_asset(tmp_path, manifest):
    script = r'''
import sys, os, json, time
from contextlib import contextmanager
from pathlib import Path
import httpx
from programasweights._vision_assets import get_vision_asset_paths
manifest = json.loads(sys.argv[1])
ready, go, requests = map(Path, sys.argv[2:])
def respond(request):
    with requests.open("a") as log:
        log.write(str(request.url) + "\n")
    time.sleep(0.1)
    data = b"GGUF" + (b"M" if request.url.path.endswith("base.gguf") else b"P") * 4092
    return httpx.Response(200, content=data)
@contextmanager
def stream(method, url, **kwargs):
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with client.stream(method, url, **kwargs) as response:
            yield response
httpx.stream = stream
ready.write_text("ready")
deadline = time.monotonic() + 15
while not go.exists():
    if time.monotonic() > deadline:
        raise RuntimeError("Parent did not release start barrier")
    time.sleep(0.01)
result = get_vision_asset_paths(manifest)
assert not any(name.split(".")[0] in ("PIL", "llama_cpp", "torch", "numpy") for name in sys.modules)
print(json.dumps([str(p) for p in result]))
'''
    go, requests = tmp_path / "go", tmp_path / "requests"
    ready = [tmp_path / ("ready-" + str(i)) for i in range(2)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    workers = [subprocess.Popen(
        [sys.executable, "-c", script, json.dumps(manifest), str(path), str(go), str(requests)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    ) for path in ready]
    try:
        deadline = time.monotonic() + 15
        while not all(path.exists() for path in ready):
            assert time.monotonic() < deadline
            assert all(worker.poll() is None for worker in workers)
            time.sleep(0.01)
        go.write_text("go")
        results = []
        for worker in workers:
            stdout, stderr = worker.communicate(timeout=20)
            assert worker.returncode == 0, stderr
            results.append(json.loads(stdout))
        assert results[0] == results[1]
        assert len(requests.read_text().splitlines()) == 2
    finally:
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
                worker.wait(timeout=5)


@pytest.mark.parametrize("damage", ["legacy", "unsupported", "missing_hash"])
def test_invalid_manifest_fails_before_cache_or_network(manifest, tmp_path, damage):
    if damage == "legacy":
        manifest["manifest_version"] = 1
    elif damage == "unsupported":
        manifest["local_sdk"]["supported"] = False
    else:
        del manifest["local_sdk"]["vision"]["projector"]["sha256"]
    with pytest.raises(ValueError, match="vision runtime manifest"):
        assets.get_vision_asset_paths(manifest)
    assert not (tmp_path / "cache").exists()


def test_huggingface_repo_source_resolves_using_existing_convention(manifest, serve):
    calls, payloads = serve()
    for descriptor, data in [(manifest["local_sdk"]["base_model"], MODEL),
                             (manifest["local_sdk"]["vision"]["projector"], PROJECTOR)]:
        del descriptor["url"]
        descriptor.update(provider="huggingface", repo="example/model")
        payloads["https://huggingface.co/example/model/resolve/main/" + descriptor["file"]] = data
    paths = assets.get_vision_asset_paths(manifest)
    assert paths.base_model.read_bytes() == MODEL
    assert all(url.startswith("https://huggingface.co/example/model/resolve/main/") for url in calls)

