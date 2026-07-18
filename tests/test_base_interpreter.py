from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import programasweights as paw
from programasweights import cache, config

FAKE_MODEL_BYTES = b"GGUF" + (b"M" * 4092)
FAKE_ADAPTER_BYTES = b"GGUF" + (
    b"A" * (cache.MIN_ADAPTER_GGUF_SIZE - 4)
)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("PAW_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("PAW_OFFLINE", raising=False)
    monkeypatch.delenv("PAW_API_KEY", raising=False)
    monkeypatch.delenv("PAW_API_URL", raising=False)
    for runtime_manifest in cache.LEGACY_RUNTIME_MANIFESTS.values():
        base_model = runtime_manifest["local_sdk"]["base_model"]
        monkeypatch.setitem(
            base_model,
            "size_bytes",
            len(FAKE_MODEL_BYTES),
        )
        monkeypatch.setitem(
            base_model,
            "sha256",
            hashlib.sha256(FAKE_MODEL_BYTES).hexdigest(),
        )


@pytest.fixture
def fake_runtime(monkeypatch):
    state = SimpleNamespace(
        instances=[],
        adapter_init_calls=[],
        adapter_apply_calls=[],
        adapter_free_calls=[],
        fail_adapter=False,
        adapter_apply_result=None,
        tokenize_result=None,
        llama_init_error=None,
        prefix_load_result=0,
        prefix_loaded_tokens=None,
        prefix_save_result=0,
        prefix_save_bytes=b"prefix state",
        prefix_save_error=None,
        prefix_save_paths=[],
    )
    module = types.ModuleType("llama_cpp")

    class FakeLlama:
        def __init__(self, **kwargs):
            if state.llama_init_error is not None:
                raise state.llama_init_error
            self.kwargs = kwargs
            self.model = object()
            self.ctx = object()
            self.n_tokens = 0
            self.input_ids = [0] * 4096
            self.tokenize_calls = []
            self.eval_calls = []
            self.reset_calls = 0
            self.closed = False
            self._sample_index = 0
            state.instances.append(self)

        def tokenize(self, data, *, add_bos, special):
            self.tokenize_calls.append(
                {
                    "data": data,
                    "add_bos": add_bos,
                    "special": special,
                }
            )
            if state.tokenize_result is not None:
                return list(state.tokenize_result)
            return list(data)

        def eval(self, tokens):
            copied = list(tokens)
            self.eval_calls.append(copied)
            self.n_tokens += len(copied)

        def sample(self, *, temp):
            token = ord("A") if self._sample_index == 0 else 0
            self._sample_index += 1
            return token

        def token_eos(self):
            return 0

        def detokenize(self, tokens):
            return bytes(tokens)

        def reset(self):
            self.reset_calls += 1
            self.n_tokens = 0
            self._sample_index = 0

        def close(self):
            self.closed = True

    def adapter_init(model, path):
        state.adapter_init_calls.append((model, path))
        if state.fail_adapter:
            return None
        return object()

    def adapter_apply(ctx, adapter, scale):
        state.adapter_apply_calls.append((ctx, adapter, scale))
        return state.adapter_apply_result

    def adapter_free(adapter):
        state.adapter_free_calls.append(adapter)

    def load_prefix(_ctx, _path, _seq_id, tokens_out, capacity, count_out):
        loaded = list(state.prefix_loaded_tokens or [])
        for index, token in enumerate(loaded[:capacity]):
            tokens_out[index] = token
        count_out._obj.value = len(loaded)
        return state.prefix_load_result

    def save_prefix(_ctx, path, _seq_id, _tokens, _count):
        decoded_path = Path(path.decode("utf-8"))
        state.prefix_save_paths.append(decoded_path)
        if state.prefix_save_error is not None:
            raise state.prefix_save_error
        if state.prefix_save_result:
            decoded_path.write_bytes(state.prefix_save_bytes)
        return state.prefix_save_result

    module.Llama = FakeLlama
    module.llama_adapter_lora_init = adapter_init
    module.llama_set_adapter_lora = adapter_apply
    module.llama_adapter_lora_free = adapter_free
    module.llama_token = ctypes.c_int
    module.llama_state_seq_load_file = load_prefix
    module.llama_state_seq_save_file = save_prefix

    monkeypatch.setitem(sys.modules, "llama_cpp", module)
    sys.modules.pop("programasweights.runtime_llamacpp", None)
    runtime = importlib.import_module("programasweights.runtime_llamacpp")
    yield runtime, state
    sys.modules.pop("programasweights.runtime_llamacpp", None)


def _write_base_model(interpreter: str) -> Path:
    manifest = cache.get_base_runtime_manifest(interpreter)
    file_name = manifest["local_sdk"]["base_model"]["file"]
    model_path = config.get_base_models_dir() / file_name
    model_path.write_bytes(FAKE_MODEL_BYTES)
    return model_path


def _write_compiled_program(tmp_path: Path, *, include_adapter: bool = True):
    interpreter = "gpt2"
    runtime = cache.get_base_runtime_manifest(interpreter)
    program_dir = tmp_path / "compiled"
    program_dir.mkdir(parents=True)
    if include_adapter:
        (program_dir / "adapter.gguf").write_bytes(FAKE_ADAPTER_BYTES)
    (program_dir / "prompt_template.txt").write_text(
        "Prefix:{INPUT_PLACEHOLDER}:Suffix",
        encoding="utf-8",
    )
    (program_dir / "meta.json").write_text(
        json.dumps(
            {
                "program_id": "a" * 20,
                "spec": "Test compiled behavior.",
                "interpreter": interpreter,
                "runtime_id": runtime["runtime_id"],
                "runtime_manifest_version": runtime["manifest_version"],
                "runtime": runtime,
            }
        ),
        encoding="utf-8",
    )
    _write_base_model(interpreter)
    return program_dir


def _forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(cache.httpx, "get", forbidden)
    monkeypatch.setattr(cache.httpx, "stream", forbidden)

    import programasweights.client as client_module

    monkeypatch.setattr(client_module.httpx, "get", forbidden)
    monkeypatch.setattr(client_module.httpx, "post", forbidden)
    monkeypatch.setattr(client_module.httpx, "stream", forbidden)


def test_function_base_argument_contract(fake_runtime):
    _write_base_model("gpt2")

    with pytest.raises(TypeError):
        paw.function()
    with pytest.raises(ValueError, match="requires an explicit interpreter"):
        paw.function(None)
    with pytest.raises(ValueError, match="Unknown interpreter"):
        paw.function(None, interpreter="GPT-2")
    with pytest.raises(ValueError, match="explicit None"):
        paw.function("")
    with pytest.raises(ValueError, match="cannot be combined"):
        paw.function("a" * 20, interpreter="gpt2")
    with pytest.raises(TypeError):
        paw.function(None, 2048, -1, False, False, "gpt2")

    fn = paw.function(None, interpreter="gpt2")
    assert fn.interpreter == "gpt2"
    fn.close()


@pytest.mark.parametrize(
    ("interpreter", "input_text", "expected"),
    [
        (
            "Qwen/Qwen3-0.6B",
            "raw <bytes>",
            (
                b"<|im_start|>user\nraw <bytes><|im_end|>\n"
                b"<|im_start|>assistant\n<think>\n\n</think>\n\n"
            ),
        ),
        ("gpt2", "raw <bytes>", b"raw <bytes>"),
    ],
)
def test_base_uses_exact_prompt_bytes_and_one_tokenize_call(
    fake_runtime,
    interpreter,
    input_text,
    expected,
):
    _, state = fake_runtime
    _write_base_model(interpreter)

    fn = paw.function(None, interpreter=interpreter)
    instance = state.instances[-1]
    assert instance.tokenize_calls == []

    assert fn(input_text, max_tokens=1) == "A"
    assert instance.tokenize_calls == [
        {
            "data": expected,
            "add_bos": False,
            "special": True,
        }
    ]
    assert instance.eval_calls[0] == list(expected)
    fn.close()


def test_base_has_no_adapter_network_or_prefix_cache(
    fake_runtime,
    monkeypatch,
):
    runtime, state = fake_runtime
    _write_base_model("gpt2")
    _forbid_network(monkeypatch)

    def prefix_forbidden(self):
        raise AssertionError("base mode must not use prefix caching")

    monkeypatch.setattr(
        runtime.PawFunction,
        "_load_or_eval_prefix",
        prefix_forbidden,
    )
    fn = paw.function(None, interpreter="gpt2", offline=True)
    assert fn("hello", max_tokens=1) == "A"
    assert state.adapter_init_calls == []
    assert list(config.get_programs_dir().glob("**/prefix_kv_cache.bin")) == []
    fn.close()


def test_base_resets_state_for_every_invocation(fake_runtime):
    _, state = fake_runtime
    _write_base_model("gpt2")
    fn = paw.function(None, interpreter="gpt2")
    instance = state.instances[-1]

    assert fn("one", max_tokens=2) == "A"
    assert fn("two", max_tokens=2) == "A"
    assert instance.reset_calls == 2
    assert [call["data"] for call in instance.tokenize_calls] == [
        b"one",
        b"two",
    ]
    fn.close()


@pytest.mark.parametrize("use_environment", [False, True])
def test_missing_base_model_offline_never_uses_network(
    fake_runtime,
    monkeypatch,
    use_environment,
):
    _forbid_network(monkeypatch)
    if use_environment:
        monkeypatch.setenv("PAW_OFFLINE", "TRUE")

    with pytest.raises(RuntimeError, match="offline mode prohibits"):
        paw.function(
            None,
            interpreter="gpt2",
            offline=not use_environment,
        )


def test_online_base_downloads_only_builtin_gguf(
    fake_runtime,
    monkeypatch,
):
    _, state = fake_runtime
    downloads = []

    def fake_download(url, destination, progress=None):
        downloads.append((url, destination))
        destination.write_bytes(FAKE_MODEL_BYTES)

    _forbid_network(monkeypatch)
    monkeypatch.setattr(cache, "_download_file", fake_download)

    fn = paw.function(None, interpreter="gpt2")
    assert len(downloads) == 1
    assert downloads[0][0] == cache.BASE_MODEL_URLS["gpt2-q8_0"]
    assert downloads[0][1].name == "gpt2-q8_0.gguf"
    assert state.adapter_init_calls == []
    fn.close()


def test_base_rejects_zero_tokens_and_context_overflow(fake_runtime):
    _, state = fake_runtime
    _write_base_model("gpt2")
    fn = paw.function(None, interpreter="gpt2", n_ctx=3)
    instance = state.instances[-1]

    state.tokenize_result = []
    with pytest.raises(ValueError, match="zero tokens"):
        fn("hello")
    assert instance.reset_calls == 1
    assert instance.eval_calls == []

    state.tokenize_result = [1, 2, 3]
    with pytest.raises(ValueError, match="Context overflow"):
        fn("hello")
    assert instance.reset_calls == 2
    assert instance.eval_calls == []
    fn.close()


def test_compiled_adapter_failure_cleans_partial_model(
    fake_runtime,
    tmp_path,
):
    runtime, state = fake_runtime
    program_dir = _write_compiled_program(tmp_path)
    state.fail_adapter = True

    with pytest.raises(RuntimeError, match="Failed to load LoRA adapter"):
        runtime.PawFunction(program_dir, offline=True)

    assert state.adapter_init_calls
    assert state.instances[-1].closed is True


def test_compiled_adapter_apply_failure_cleans_adapter_and_model(
    fake_runtime,
    tmp_path,
):
    runtime, state = fake_runtime
    program_dir = _write_compiled_program(tmp_path)
    state.adapter_apply_result = 1

    with pytest.raises(RuntimeError, match="Failed to apply LoRA adapter"):
        runtime.PawFunction(program_dir, offline=True)

    assert len(state.adapter_free_calls) == 1
    assert state.instances[-1].closed is True


def test_compiled_missing_model_offline_never_uses_network(
    fake_runtime,
    tmp_path,
    monkeypatch,
):
    runtime, state = fake_runtime
    program_dir = _write_compiled_program(tmp_path)
    model_path = (
        config.get_base_models_dir()
        / cache.get_base_runtime_manifest("gpt2")["local_sdk"]["base_model"][
            "file"
        ]
    )
    model_path.unlink()
    _forbid_network(monkeypatch)

    with pytest.raises(RuntimeError, match="offline mode prohibits"):
        runtime.PawFunction(program_dir, offline=True)
    assert state.instances == []
    assert state.adapter_init_calls == []


def test_compiled_mode_still_requires_and_applies_adapter(
    fake_runtime,
    tmp_path,
):
    runtime, state = fake_runtime
    missing_adapter = _write_compiled_program(
        tmp_path / "missing",
        include_adapter=False,
    )
    with pytest.raises(FileNotFoundError, match="LoRA adapter"):
        runtime.PawFunction(missing_adapter, offline=True)
    assert state.instances == []

    invalid_adapter = _write_compiled_program(tmp_path / "invalid")
    (invalid_adapter / "adapter.gguf").write_bytes(b"GGUF")
    with pytest.raises(ValueError, match="valid GGUF"):
        runtime.PawFunction(invalid_adapter, offline=True)
    assert state.instances == []

    program_dir = _write_compiled_program(tmp_path / "ready")
    fn = runtime.PawFunction(program_dir, offline=True)
    assert len(state.adapter_init_calls) == 1
    assert len(state.adapter_apply_calls) == 1
    assert state.instances[-1].reset_calls == 0
    assert fn("payload", max_tokens=1) == "A"
    fn.close()
    assert len(state.adapter_free_calls) == 1
    fn.close()
    assert len(state.adapter_free_calls) == 1


def test_compiled_prefix_cache_requires_exact_token_identity(
    fake_runtime,
    tmp_path,
):
    runtime, state = fake_runtime
    program_dir = _write_compiled_program(tmp_path)
    cache_path = program_dir / "prefix_kv_cache.bin"
    cache_path.write_bytes(b"stale state")
    expected_prefix = list(b"Prefix:")

    state.prefix_load_result = 1
    state.prefix_loaded_tokens = [999] * len(expected_prefix)
    fn = runtime.PawFunction(program_dir, offline=True)
    instance = state.instances[-1]
    assert instance.eval_calls[0] == expected_prefix
    assert instance.reset_calls >= 1
    assert not cache_path.exists()
    fn.close()

    cache_path.write_bytes(b"matching state")
    state.prefix_loaded_tokens = expected_prefix
    fn = runtime.PawFunction(program_dir, offline=True)
    instance = state.instances[-1]
    assert instance.eval_calls == []
    assert instance.n_tokens == len(expected_prefix)
    fn.close()
    fn.close()
    assert len(state.adapter_free_calls) == 2


def test_compiled_prefix_cache_save_is_atomic(
    fake_runtime,
    tmp_path,
):
    runtime, state = fake_runtime
    program_dir = _write_compiled_program(tmp_path)
    cache_path = program_dir / "prefix_kv_cache.bin"
    state.prefix_save_result = 1

    fn = runtime.PawFunction(program_dir, offline=True)

    assert cache_path.read_bytes() == state.prefix_save_bytes
    assert len(state.prefix_save_paths) == 1
    assert state.prefix_save_paths[0].parent == program_dir
    assert state.prefix_save_paths[0] != cache_path
    assert list(program_dir.glob(".prefix_kv_cache.bin.*.tmp")) == []
    fn.close()


def test_compiled_prefix_cache_rechecks_after_lock(
    fake_runtime,
    tmp_path,
    monkeypatch,
):
    runtime, state = fake_runtime
    program_dir = _write_compiled_program(tmp_path)
    cache_path = program_dir / "prefix_kv_cache.bin"
    expected_prefix = list(b"Prefix:")

    @contextmanager
    def create_cache_while_waiting(_program_dir):
        cache_path.write_bytes(b"winner state")
        state.prefix_load_result = 1
        state.prefix_loaded_tokens = expected_prefix
        yield

    monkeypatch.setattr(
        cache,
        "prefix_cache_lock",
        create_cache_while_waiting,
    )
    fn = runtime.PawFunction(program_dir, offline=True)

    assert state.instances[-1].eval_calls == []
    assert state.instances[-1].n_tokens == len(expected_prefix)
    fn.close()


def test_prefix_cache_readonly_failures_do_not_abort_inference(
    fake_runtime,
    tmp_path,
    monkeypatch,
):
    runtime, state = fake_runtime
    program_dir = _write_compiled_program(tmp_path)
    cache_path = program_dir / "prefix_kv_cache.bin"
    cache_path.write_bytes(b"stale")
    expected_prefix = list(b"Prefix:")
    state.prefix_load_result = 1
    state.prefix_loaded_tokens = [999] * len(expected_prefix)

    original_unlink = Path.unlink

    def fail_stale_unlink(path, *args, **kwargs):
        if path == cache_path:
            raise PermissionError("read-only cache")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_stale_unlink)
    monkeypatch.setattr(
        runtime,
        "tempfile",
        SimpleNamespace(
            mkstemp=lambda *args, **kwargs: (_ for _ in ()).throw(
                PermissionError("read-only directory")
            )
        ),
    )

    fn = runtime.PawFunction(program_dir, offline=True)
    assert state.instances[-1].eval_calls[0] == expected_prefix
    fn.close()


def test_prefix_cache_lock_failure_does_not_abort_inference(
    fake_runtime,
    tmp_path,
    monkeypatch,
):
    runtime, state = fake_runtime
    program_dir = _write_compiled_program(tmp_path)
    expected_prefix = list(b"Prefix:")

    @contextmanager
    def fail_lock(_program_dir):
        raise PermissionError("read-only lock directory")
        yield

    monkeypatch.setattr(cache, "prefix_cache_lock", fail_lock)
    fn = runtime.PawFunction(program_dir, offline=True)
    assert state.instances[-1].eval_calls[0] == expected_prefix
    fn.close()


def test_native_stderr_redirection_is_serialized(
    fake_runtime,
    monkeypatch,
):
    runtime, _ = fake_runtime
    operations = []
    monkeypatch.setattr(
        runtime,
        "sys",
        SimpleNamespace(stderr=SimpleNamespace(fileno=lambda: 2)),
    )
    monkeypatch.setattr(
        runtime,
        "os",
        SimpleNamespace(
            devnull="/dev/null",
            O_WRONLY=1,
            open=lambda *args, **kwargs: operations.append("open") or 10,
            dup=lambda fd: 11,
            dup2=lambda source, dest: None,
            close=lambda fd: None,
        ),
    )

    ready = threading.Barrier(3)
    first_entered = threading.Event()
    release_first = threading.Event()
    active = 0
    max_active = 0
    state_lock = threading.Lock()

    def worker():
        nonlocal active, max_active
        ready.wait()
        with runtime._suppress_native_stderr(True):
            with state_lock:
                active += 1
                max_active = max(max_active, active)
                is_first = not first_entered.is_set()
                first_entered.set()
            if is_first:
                assert release_first.wait(timeout=2)
            with state_lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker) for _ in range(2)]
        ready.wait()
        assert first_entered.wait(timeout=2)
        assert operations == ["open"]
        release_first.set()
        for future in futures:
            future.result(timeout=2)

    assert max_active == 1
    assert operations == ["open", "open"]
