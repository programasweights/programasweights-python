"""Hermetic native-boundary tests; real-model qualification is separate."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import copy
import ctypes
import importlib
import json
from pathlib import Path
import sys
import threading
import types
import zipfile

import httpx
from PIL import Image as PILImage
import pytest

import programasweights as paw
from programasweights import cache, config
from test_vision_metadata import ADAPTER, MODEL, PROJECTOR, PID, manifest, metadata, write_program


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("PAW_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PAW_OFFLINE", "1")
    monkeypatch.setenv("PAW_QUIET", "1")
    def forbidden(*args, **kwargs):
        pytest.fail("Runtime test attempted network access")
    monkeypatch.setattr(httpx, "stream", forbidden)
    (config.get_base_models_dir() / "base.gguf").write_bytes(MODEL)
    (config.get_base_models_dir() / "mmproj.gguf").write_bytes(PROJECTOR)
    state = types.SimpleNamespace(
        models=[], calls=[], events=[], selected=None, handle=0,
        support=True, init_fail=False, completion=None, create_error=None,
    )
    lib = types.ModuleType("llama_cpp")
    lib.__version__ = "0.3.35"
    lib.LLAMA_FLASH_ATTN_TYPE_ENABLED = 1
    lib.llama_adapter_lora_p_ctypes = ctypes.c_void_p
    lib.LogitsProcessorList = list
    def event(name, *args):
        state.events.append((name, *args))
    lib.llama_get_memory = lambda ctx: ctx
    lib.llama_memory_clear = lambda ctx, full: event("memory_clear", full)
    def select(ctx, handles, count, scales):
        state.selected = handles[0] if count else None
        event("select", state.selected, scales[0] if count else None)
        return 0
    lib.llama_set_adapters_lora = select
    def load(model, path):
        state.handle += 1
        event("load", state.handle, path)
        return state.handle
    lib.llama_adapter_lora_init = load
    def free(handle):
        assert handle != state.selected, "Freed an attached adapter"
        event("free", handle)
    lib.llama_adapter_lora_free = free
    class Handler:
        def __init__(self, clip_model_path, **kwargs):
            self.clip_model_path = clip_model_path
            self.mtmd_ctx = None
            self._mtmd_cpp = types.SimpleNamespace(
                mtmd_context_params_default=types.SimpleNamespace,
                mtmd_init_from_file=self.init,
                mtmd_free=lambda ctx: event("projector_free"),
                mtmd_support_vision=lambda ctx: state.support,
            )
        def init(self, path, model, params):
            event("projector_init", vars(params))
            return None if state.init_fail else object()
        def __call__(self, **kwargs):
            state.calls.append(kwargs)
            if state.completion:
                return state.completion(kwargs)
            return {"choices": [{"message": {"content": " result "}}]}
    class Llama:
        def __init__(self, **kwargs):
            if state.create_error:
                raise state.create_error
            self.options = kwargs
            self.verbose = kwargs["verbose"]
            self.n_threads = kwargs["n_threads"]
            self.handler = kwargs["chat_handler"]
            self.ctx = self
            self.model = object()
            self._stack = ExitStack()
            self.closed = False
            state.models.append(self)
        def reset(self):
            event("reset")
        def create_chat_completion(self, **kwargs):
            assert not self.closed
            return self.handler(llama=self, **kwargs)
        def token_eos(self):
            return 1
        def close(self):
            assert not self.closed, "Double close"
            self._stack.close()
            self.closed = True
            event("model_free")
    lib.Llama = Llama
    chat = types.ModuleType("llama_cpp.llama_chat_format")
    chat.MTMDChatHandler = Handler
    monkeypatch.setitem(sys.modules, "llama_cpp", lib)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", chat)
    for name in ("programasweights.runtime_llamacpp", "programasweights._runtime_vision"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    runtime = importlib.import_module("programasweights._runtime_vision")
    yield state, lib, runtime
    assert not runtime._POOL, "Test leaked a shared runtime"
    sys.modules.pop("programasweights.runtime_llamacpp", None)
    sys.modules.pop("programasweights._runtime_vision", None)


@pytest.fixture
def program(backend, manifest):
    return write_program(config.get_programs_dir() / PID, metadata(manifest),
                         "Literal {INPUT_PLACEHOLDER}; do not substitute.")


def second_program(program, manifest):
    meta = metadata(copy.deepcopy(manifest))
    meta["program_id"] = "d" * 20
    directory = write_program(program.parent / meta["program_id"], meta, "Different system prompt")
    (directory / "adapter.gguf").write_bytes(ADAPTER[:-1] + b"B")
    return directory


@pytest.mark.parametrize("entrypoint", ["cached", "local"])
def test_public_ordered_parts_preserve_literal_prompt_and_native_dimensions(program, backend, tmp_path, entrypoint):
    state, _, _ = backend
    reference = PID
    if entrypoint == "local":
        reference = tmp_path / "images.paw"
        with zipfile.ZipFile(reference, "w") as output:
            for path in program.iterdir():
                output.write(path, path.name)
    image = paw.Image(PILImage.new("RGB", (31, 27), "red"))
    with paw.function(reference, offline=True) as fn:
        assert fn("one", image, "", "two", image, max_tokens=32) == "result"
        call = state.calls[-1]
        assert call["messages"][0]["content"] == "Literal {INPUT_PLACEHOLDER}; do not substitute."
        parts = call["messages"][1]["content"]
        assert [part["type"] for part in parts] == ["text", "image_url", "text", "text", "image_url"]
        assert [part["text"] for part in parts if part["type"] == "text"] == ["one", "", "two"]
        import base64, io
        with PILImage.open(io.BytesIO(base64.b64decode(parts[1]["image_url"]["url"].split(",", 1)[1]))) as actual:
            assert actual.size == (31, 27)
        assert call["enable_thinking"] is False
        assert fn.interpreter == "Qwen/Qwen3.5-0.8B"
    assert state.models[0].closed


def test_aba_shares_model_resets_memory_and_detaches_before_switch(program, backend, manifest):
    state, _, module = backend
    other = second_program(program, manifest)
    first = paw.function(PID, offline=True)
    second = paw.function(other.name, offline=True)
    try:
        assert first._runtime is second._runtime
        assert len(state.models) == 1
        for fn in (first, second, first):
            fn("read")
            assert state.selected is None
        selected = [e for e in state.events if e[0] == "select" and e[1] is not None]
        assert selected == [("select", 1, 1.0), ("select", 2, 1.0), ("select", 1, 1.0)]
        assert len([e for e in state.events if e[0] == "memory_clear"]) == 6
        first.close()
        assert not state.models[0].closed
        assert second("still open") == "result"
    finally:
        first.close()
        second.close()
    assert len([e for e in state.events if e[0] == "projector_free"]) == 1
    assert not module._POOL


@pytest.mark.parametrize("setting,value", [("n_ctx", 4096), ("n_gpu_layers", 0), ("verbose", True)])
def test_different_runtime_settings_do_not_share(program, backend, setting, value):
    state, _, _ = backend
    with paw.function(PID, offline=True) as one, paw.function(PID, offline=True, **{setting: value}) as two:
        assert one._runtime is not two._runtime
        assert len(state.models) == 2


def test_token_budget_and_gpu_settings_reach_projector(program, backend):
    state, _, _ = backend
    with paw.function(PID, offline=True, n_gpu_layers=0):
        params = next(e[1] for e in state.events if e[0] == "projector_init")
        assert params["image_min_tokens"] == 64
        assert params["image_max_tokens"] == 512
        assert params["use_gpu"] is False


def test_different_image_limits_do_not_share(program, backend, manifest):
    other = second_program(program, manifest)
    meta = json.loads((other / "meta.json").read_text())
    meta["runtime"]["local_sdk"]["vision"]["preprocessing"]["image_max_tokens"] = 128
    (other / "meta.json").write_text(json.dumps(meta))
    with paw.function(PID, offline=True) as one, paw.function(other.name, offline=True) as two:
        assert one._runtime is not two._runtime


def test_bounded_adapter_eviction_and_reload(program, backend, manifest, monkeypatch):
    state, _, module = backend
    monkeypatch.setattr(module, "_MAX_ADAPTERS", 1)
    other = second_program(program, manifest)
    with paw.function(PID, offline=True) as one, paw.function(other.name, offline=True) as two:
        for fn in (one, two, one):
            fn("read")
            assert len(fn._runtime.adapters) == 1
        assert len([e for e in state.events if e[0] == "load"]) == 3
        assert len([e for e in state.events if e[0] == "free"]) == 2


def test_changed_adapter_is_rejected_before_native_load(program, backend):
    state, _, _ = backend
    with paw.function(PID, offline=True) as fn:
        (program / "adapter.gguf").write_bytes(ADAPTER[:-1] + b"X")
        with pytest.raises(ValueError, match="changed"):
            fn("read")
        assert not [e for e in state.events if e[0] == "load"]


@pytest.mark.parametrize("parts,error", [((), ValueError), ((["bad"],), TypeError), (("ok", 32), TypeError)])
def test_invalid_content_does_not_touch_native_state(program, backend, parts, error):
    state, _, _ = backend
    with paw.function(PID, offline=True) as fn:
        before = list(state.events)
        with pytest.raises(error):
            fn(*parts)
        assert state.events == before


@pytest.mark.parametrize("tokens", [-1, True, 1.5])
def test_invalid_max_tokens(program, tokens):
    with paw.function(PID, offline=True) as fn:
        with pytest.raises(ValueError, match="max_tokens"):
            fn("read", max_tokens=tokens)


@pytest.mark.parametrize("temperature", [-1, float("nan"), float("inf"), "warm"])
def test_invalid_temperature(program, temperature):
    with paw.function(PID, offline=True) as fn:
        with pytest.raises(ValueError, match="temperature"):
            fn("read", temperature=temperature)


def test_zero_tokens_does_not_mean_unlimited_generation(program, backend):
    state, _, _ = backend
    with paw.function(PID, offline=True) as fn:
        assert fn("read", max_tokens=0) == ""
        assert not state.calls


def test_response_format_and_generation_options_pass_through(program, backend):
    state, _, _ = backend
    schema = {"type": "json_object", "schema": {"type": "object"}}
    with paw.function(PID, offline=True) as fn:
        fn("read", temperature=.25, max_tokens=16, response_format=schema)
        assert state.calls[-1]["response_format"] == schema
        assert state.calls[-1]["max_tokens"] == 16
        assert state.calls[-1]["temperature"] == .25


@pytest.mark.parametrize("failure", [ValueError("context overflow"), KeyboardInterrupt()])
def test_call_failure_cleans_state_and_next_call_works(program, backend, failure):
    state, _, _ = backend
    def fail(kwargs):
        raise failure
    with paw.function(PID, offline=True) as fn:
        state.completion = fail
        with pytest.raises(type(failure)):
            fn("read")
        assert state.selected is None
        assert state.events[-2:] == [("memory_clear", True), ("select", None, None)]
        state.completion = None
        assert fn("retry") == "result"


def test_logit_callback_exception_is_not_swallowed(program, backend):
    state, _, _ = backend
    failure = ValueError("bad user callback")
    def processor(ids, scores):
        raise failure
    def completion(kwargs):
        fallback = kwargs["logits_processor"][0]([], [0., 0., 0.])
        assert fallback == [0., 1e10, 0.]
        return {"choices": [{"message": {"content": "discarded"}}]}
    state.completion = completion
    with paw.function(PID, offline=True) as fn:
        with pytest.raises(ValueError, match="bad user callback") as caught:
            fn("read", logits_processor=[processor])
        assert caught.value is failure


@pytest.mark.parametrize("operation", ["call", "close", "load"])
def test_reentrant_callback_fails_without_corrupting_model(program, backend, operation):
    state, _, _ = backend
    with paw.function(PID, offline=True) as fn:
        def completion(kwargs):
            if operation == "call":
                return fn("nested")
            if operation == "close":
                return fn.close()
            return paw.function(PID, offline=True)
        state.completion = completion
        with pytest.raises(RuntimeError, match="inside a vision callback"):
            fn("read")
        state.completion = None
        assert fn("retry") == "result"


def test_concurrent_functions_serialize_and_close_waits(program, backend, manifest):
    state, _, _ = backend
    other = second_program(program, manifest)
    one, two = paw.function(PID, offline=True), paw.function(other.name, offline=True)
    started, release, attempted = threading.Event(), threading.Event(), threading.Event()
    def completion(kwargs):
        started.set()
        assert release.wait(5)
        return {"choices": [{"message": {"content": "ok"}}]}
    state.completion = completion
    def second_call():
        attempted.set()
        return two("two")
    try:
        with ThreadPoolExecutor(2) as executor:
            first = executor.submit(one, "one")
            assert started.wait(5)
            second = executor.submit(second_call)
            assert attempted.wait(5)
            assert len(state.calls) == 1
            release.set()
            assert first.result(5) == second.result(5) == "ok"
        started.clear(); release.clear(); attempted.clear()
        with ThreadPoolExecutor(2) as executor:
            first = executor.submit(one, "one")
            assert started.wait(5)
            def close_other():
                attempted.set()
                two.close()
            closing = executor.submit(close_other)
            assert attempted.wait(5)
            assert not closing.done()
            release.set()
            first.result(5); closing.result(5)
        assert not state.models[0].closed
    finally:
        release.set()
        one.close(); two.close()


@pytest.mark.parametrize("kind", ["projector", "unsupported", "model"])
def test_initialization_failure_is_not_pooled(program, backend, kind):
    state, _, module = backend
    state.init_fail = kind == "projector"
    state.support = kind != "unsupported"
    state.create_error = RuntimeError("model init failed") if kind == "model" else None
    with pytest.raises(RuntimeError):
        paw.function(PID, offline=True)
    assert not module._POOL
    assert all(model.closed for model in state.models)
    state.init_fail, state.support, state.create_error = False, True, None
    with paw.function(PID, offline=True) as fn:
        assert fn("retry") == "result"


def test_old_backend_fails_before_download(program, backend, monkeypatch):
    state, lib, module = backend
    lib.__version__ = "0.3.34"
    def forbidden(*args, **kwargs):
        pytest.fail("Should check version before asset resolution")
    monkeypatch.setattr(module, "get_vision_asset_paths", forbidden)
    with pytest.raises(ImportError, match="0.3.35"):
        paw.function(PID, offline=True)
    assert not state.models


def test_closed_function_rejects_calls(program):
    fn = paw.function(PID, offline=True)
    fn.close(); fn.close()
    with pytest.raises(RuntimeError, match="closed"):
        fn("read")


def test_partial_output_returns_text_but_nontext_output_errors(program, backend):
    state, _, _ = backend
    with paw.function(PID, offline=True) as fn:
        state.completion = lambda _: {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}
        assert fn("read", max_tokens=1) == "partial"
        state.completion = lambda _: {"choices": [{"message": {"content": None}}]}
        with pytest.raises(RuntimeError, match="no text content"):
            fn("read")


def test_failed_state_cleanup_retires_runtime_and_preserves_call_error(program, backend, monkeypatch):
    state, lib, module = backend
    failure = ValueError("original inference failure")
    original_clear = lib.llama_memory_clear
    count = 0
    def clear(ctx, full):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("cleanup failed")
        return original_clear(ctx, full)
    def completion(kwargs):
        raise failure
    with paw.function(PID, offline=True) as old:
        monkeypatch.setattr(lib, "llama_memory_clear", clear)
        state.completion = completion
        with pytest.raises(ValueError) as caught:
            old("read")
        assert caught.value is failure
        assert old._runtime.closed
        state.completion = None
        with paw.function(PID, offline=True) as new:
            assert new._runtime is not old._runtime
            old.close()
            assert new("recovered") == "result"
            assert len(module._POOL) == 1


def test_forked_process_rejected_before_acquiring_inherited_locks(program, backend, monkeypatch):
    _, _, module = backend
    with paw.function(PID, offline=True) as fn:
        with monkeypatch.context() as child:
            child.setattr(module, "_PROCESS_ID", -1)
            with pytest.raises(RuntimeError, match="spawned"):
                fn("read")
            with pytest.raises(RuntimeError, match="spawned"):
                paw.function(PID, offline=True)


@pytest.fixture
def vision_base(backend, manifest, monkeypatch):
    builtin = copy.deepcopy(manifest)
    builtin["runtime_id"] = "qwen3.5-0.8b-q8_0"
    builtin["base_inference"] = {"contract_version": 1, "format": "chat_messages"}
    monkeypatch.setitem(cache.BUILTIN_VISION_RUNTIMES, builtin["runtime_id"], builtin)
    def forbidden(*args, **kwargs):
        pytest.fail("Base interpreter must not use program or Hub resolution")
    monkeypatch.setattr(paw, "_resolve_program_id", forbidden)
    monkeypatch.setattr(cache, "get_program_dir", forbidden)
    monkeypatch.setattr(cache, "fetch_runtime_manifest", forbidden)
    return builtin


def test_base_vision_accepts_ordered_parts_without_system_or_adapter(vision_base, backend):
    state, _, _ = backend
    with paw.function(None, interpreter="Qwen/Qwen3.5-0.8B", offline=True) as fn:
        assert fn("first", paw.Image(PILImage.new("RGB", (31, 27))), "last") == "result"
        messages = state.calls[-1]["messages"]
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert [p["type"] for p in messages[0]["content"]] == ["text", "image_url", "text"]
        assert fn._adapter is None and fn.spec == ""
        assert "base interpreter" in repr(fn)
        assert not [e for e in state.events if e[0] == "load"]
    assert not (config.get_cache_dir() / "programs").exists()


def test_base_vision_string_calls_remain_stateless(vision_base, backend):
    state, _, _ = backend
    with paw.function(None, interpreter="Qwen/Qwen3.5-0.8B", offline=True) as fn:
        fn("old prompt")
        fn("new prompt", max_tokens=3)
        assert state.calls[-1]["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "new prompt"}]},
        ]
        assert state.calls[-1]["enable_thinking"] is False


def test_base_and_compiled_share_and_cannot_leak_lora(program, vision_base, backend):
    state, _, module = backend
    # Call the internal constructor for the already validated compiled directory:
    # vision_base deliberately forbids public Hub/program resolution.
    with module.VisionFunction(program, offline=True) as compiled:
        with paw.function(None, interpreter="Qwen/Qwen3.5-0.8B", offline=True) as base:
            assert compiled._runtime is base._runtime
            selected = []
            def completion(kwargs):
                selected.append(state.selected)
                return {"choices": [{"message": {"content": "ok"}}]}
            state.completion = completion
            for fn in (compiled, base, compiled, base):
                fn("read")
            assert selected == [1, None, 1, None]
            compiled.close()
            assert base("still open") == "ok"
    assert len(state.models) == 1 and state.models[0].closed


@pytest.mark.parametrize("missing", ["base.gguf", "mmproj.gguf"])
def test_base_vision_offline_missing_asset_fails_without_model_load(vision_base, backend, missing):
    state, _, _ = backend
    (config.get_base_models_dir() / missing).unlink()
    with pytest.raises(RuntimeError, match="offline mode"):
        paw.function(None, interpreter="Qwen/Qwen3.5-0.8B")
    assert not state.models


@pytest.mark.parametrize("n_ctx", [0, -1, True, 1.5])
def test_base_vision_bad_context_fails_before_loading(vision_base, backend, n_ctx):
    state, _, _ = backend
    with pytest.raises(ValueError, match="n_ctx"):
        paw.function(None, interpreter="Qwen/Qwen3.5-0.8B", n_ctx=n_ctx)
    assert not state.models


def test_base_vision_failed_init_does_not_leak(vision_base, backend):
    state, _, module = backend
    state.init_fail = True
    with pytest.raises(RuntimeError, match="projector"):
        paw.function(None, interpreter="Qwen/Qwen3.5-0.8B")
    assert not module._POOL and state.models[0].closed


def test_base_vision_old_backend_fails_before_asset_resolution(vision_base, backend, monkeypatch):
    state, lib, module = backend
    lib.__version__ = "0.3.34"
    def forbidden(*args, **kwargs):
        pytest.fail("Should reject old backend before asset resolution")
    monkeypatch.setattr(module, "get_vision_asset_paths", forbidden)
    with pytest.raises(ImportError, match="0.3.35"):
        paw.function(None, interpreter="Qwen/Qwen3.5-0.8B")
    assert not state.models
