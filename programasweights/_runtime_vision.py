"""Local ordered-content inference; shared resources are an internal detail.

Live functions with matching model/projector hashes and inference settings
share one serialized runtime. Closing the last function releases it. Up to four
LoRAs are retained per runtime; eviction never frees an attached adapter.
There is no prompt, image, or recurrent-state reuse between calls.
"""

from __future__ import annotations

from collections import OrderedDict
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading

from . import cache
from ._image_content import _to_chat_content
from ._vision_assets import get_vision_asset_paths
from ._vision_contract import VISION_BASE_INFERENCE, declares_vision

_POOL_LOCK = threading.RLock()
_POOL = {}
_MAX_ADAPTERS = 4
_ACTIVE = threading.local()
_PROCESS_ID = os.getpid()


def _outside_call():
    if os.getpid() != _PROCESS_ID:
        raise RuntimeError("Vision inference cannot be inherited through fork; use a spawned process.")
    if getattr(_ACTIVE, "running", False):
        raise RuntimeError("Cannot load, call, or close vision functions inside a vision callback.")


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _backend():
    import llama_cpp
    version = re.match(r"^(\d+)\.(\d+)\.(\d+)", llama_cpp.__version__)
    if version is None or tuple(map(int, version.groups())) < (0, 3, 35):
        raise ImportError(
            "Qwen3.5 image functions require llama-cpp-python>=0.3.35. "
            "Install programasweights[vision]."
        )
    from llama_cpp.llama_chat_format import MTMDChatHandler
    return llama_cpp, MTMDChatHandler


class _Runtime:
    def __init__(self, paths, preprocessing, *, n_ctx, n_gpu_layers, verbose):
        self.lock = threading.RLock()
        self.running = False
        self.closed = False
        self.references = 0
        self.adapters = OrderedDict()
        self.llm = None
        self.lib, handler_type = _backend()
        from .runtime_llamacpp import _suppress_native_stderr

        class Handler(handler_type):
            def __call__(handler, **kwargs):
                kwargs["enable_thinking"] = False
                return super().__call__(**kwargs)

            def _init_mtmd_context(handler, model):
                if handler.mtmd_ctx is not None:
                    return
                with _suppress_native_stderr(not verbose):
                    params = handler._mtmd_cpp.mtmd_context_params_default()
                    params.use_gpu = n_gpu_layers != 0
                    params.print_timings = verbose
                    params.n_threads = model.n_threads
                    params.image_min_tokens = preprocessing["image_min_tokens"]
                    params.image_max_tokens = preprocessing["image_max_tokens"]
                    params.flash_attn_type = self.lib.LLAMA_FLASH_ATTN_TYPE_ENABLED
                    handler.mtmd_ctx = handler._mtmd_cpp.mtmd_init_from_file(
                        os.fsencode(handler.clip_model_path), model.model, params,
                    )
                    if not handler.mtmd_ctx:
                        handler.mtmd_ctx = None
                        raise RuntimeError("Failed to initialize the vision projector.")

                    def free_projector():
                        if handler.mtmd_ctx is not None:
                            handler._mtmd_cpp.mtmd_free(handler.mtmd_ctx)
                            handler.mtmd_ctx = None

                    # Register before checking capabilities, including failures.
                    model._stack.callback(free_projector)
                    if not handler._mtmd_cpp.mtmd_support_vision(handler.mtmd_ctx):
                        raise RuntimeError("This projector does not support images.")

        try:
            handler = Handler(
                clip_model_path=str(paths.projector), verbose=verbose,
                use_gpu=n_gpu_layers != 0,
            )
            with _suppress_native_stderr(not verbose):
                self.llm = self.lib.Llama(
                    model_path=str(paths.base_model), chat_handler=handler,
                    n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=verbose,
                    n_batch=128, n_threads=2, n_threads_batch=2,
                    flash_attn=True,
                )
                handler._init_mtmd_context(self.llm)
        except BaseException:
            try:
                self.close()
            except Exception:
                pass  # Keep the initialization failure as the primary error.
            raise

    def _reset(self):
        self.llm.reset()
        # Llama.reset() alone does not clear Qwen3.5's recurrent memory.
        memory = self.lib.llama_get_memory(self.llm.ctx)
        if memory:
            self.lib.llama_memory_clear(memory, True)

    def _detach(self):
        result = self.lib.llama_set_adapters_lora(self.llm.ctx, None, 0, None)
        if result not in (0, None):
            raise RuntimeError("Failed to detach the previous vision LoRA.")

    def _select(self, adapter):
        self._reset()
        self._detach()
        if adapter is None:
            return
        path, digest, size = adapter
        handle = self.adapters.get(digest)
        if handle is None:
            if not cache._valid_gguf_file(path, expected_size=size, expected_sha256=digest):
                raise ValueError("Vision adapter changed after the function was loaded.")
            while len(self.adapters) >= _MAX_ADAPTERS:
                _, old = self.adapters.popitem(last=False)
                self.lib.llama_adapter_lora_free(old)
            handle = self.lib.llama_adapter_lora_init(self.llm.model, os.fsencode(path))
            if not handle:
                raise RuntimeError("Failed to load the vision LoRA adapter.")
            self.adapters[digest] = handle
        self.adapters.move_to_end(digest)
        handles = (self.lib.llama_adapter_lora_p_ctypes * 1)(handle)
        # The adapter GGUF already contains its training alpha/rank scaling.
        scales = (ctypes.c_float * 1)(1.0)
        result = self.lib.llama_set_adapters_lora(self.llm.ctx, handles, 1, scales)
        if result not in (0, None):
            raise RuntimeError("Failed to select the vision LoRA adapter.")

    def run(self, adapter, messages, *, max_tokens, temperature,
            logits_processor=None, response_format=None):
        with self.lock:
            if self.closed:
                raise RuntimeError("This vision runtime has been closed.")
            if self.running:
                raise RuntimeError("Reentrant calls to a shared vision runtime are unsupported.")
            self.running = True
            _ACTIVE.running = True
            error = None
            try:
                self._select(adapter)
                processor_error = None
                options = {}
                if logits_processor is not None:
                    def guarded_processor(input_ids, scores):
                        nonlocal processor_error
                        if processor_error is None:
                            try:
                                for processor in logits_processor:
                                    scores[:] = processor(input_ids, scores)
                                return scores
                            except BaseException as exc:
                                processor_error = exc
                        # ctypes swallows callback exceptions. Force EOS, then
                        # re-raise the original error on the Python side.
                        fallback = [0.0] * len(scores)
                        fallback[self.llm.token_eos()] = 1e10
                        return fallback
                    options["logits_processor"] = self.lib.LogitsProcessorList([guarded_processor])
                if response_format is not None:
                    options["response_format"] = response_format
                response = self.llm.create_chat_completion(
                    messages=messages, max_tokens=max_tokens,
                    temperature=temperature, **options,
                )
                if processor_error is not None:
                    raise processor_error
                content = response["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise RuntimeError("The vision function returned no text content.")
                return content.strip()
            except BaseException as exc:
                error = exc
                raise
            finally:
                try:
                    self._reset()
                    self._detach()
                except BaseException:
                    # Never reuse a runtime whose state could not be cleared.
                    try:
                        self.close()
                    except Exception:
                        pass
                    if error is None:
                        raise
                finally:
                    self.running = False
                    _ACTIVE.running = False

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            if self.llm is not None:
                try:
                    self._detach()
                finally:
                    try:
                        for handle in self.adapters.values():
                            self.lib.llama_adapter_lora_free(handle)
                        self.adapters.clear()
                    finally:
                        self.llm.close()


class VisionFunction:
    """A compiled image program or base interpreter, with ordered parts.

    Generation options are keyword-only. Calls return text (including partial
    text when max_tokens is exhausted); callers validate structured outputs.
    Close functions, or use a context manager, to release shared model memory.
    """

    def __init__(self, program_dir, n_ctx=2048, n_gpu_layers=-1, verbose=False,
                 api_url=None, api_key=None, offline=False):
        self._initialize(n_ctx)
        directory = Path(program_dir)
        self._meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        if not cache.validate_program_assets_dir(directory, self._meta.get("program_id")):
            raise ValueError("Invalid compiled image-program assets.")
        if not declares_vision(self._meta):
            raise ValueError("VisionFunction requires an image program.")
        manifest = cache.resolve_runtime_manifest(self._meta, offline=offline)
        self._prompt = (directory / "prompt_template.txt").read_text(encoding="utf-8")
        path = directory / "adapter.gguf"
        self._adapter = (path, _digest(path), path.stat().st_size)
        self._acquire_runtime(manifest, n_ctx, n_gpu_layers, verbose, offline)

    @classmethod
    def from_base(cls, interpreter, *, n_ctx=2048, n_gpu_layers=-1,
                  verbose=False, offline=False):
        self = cls.__new__(cls)
        self._initialize(n_ctx)
        manifest = cache.get_base_runtime_manifest(interpreter)
        if (
            not declares_vision({"runtime": manifest})
            or manifest.get("base_inference") != VISION_BASE_INFERENCE
        ):
            raise ValueError("This interpreter has no adapter-free vision contract.")
        self._meta = {"mode": "base", "interpreter": interpreter}
        self._prompt = None
        self._adapter = None
        self._acquire_runtime(manifest, n_ctx, n_gpu_layers, verbose, offline)
        return self

    def _initialize(self, n_ctx):
        _outside_call()
        self._runtime = None
        self._lock = threading.RLock()
        self._pid = os.getpid()
        if type(n_ctx) is not int or n_ctx <= 0:
            raise ValueError("n_ctx must be a positive integer.")

    def _acquire_runtime(self, manifest, n_ctx, n_gpu_layers, verbose, offline):
        _backend()  # Fail before downloading assets on an older text-only backend.
        paths = get_vision_asset_paths(manifest, offline=offline)
        local = manifest["local_sdk"]
        preprocessing = local["vision"]["preprocessing"]
        self._key = (
            self._pid, local["base_model"]["sha256"].lower(),
            local["vision"]["projector"]["sha256"].lower(),
            json.dumps(preprocessing, sort_keys=True), n_ctx, n_gpu_layers, verbose,
        )
        with _POOL_LOCK:
            runtime = _POOL.get(self._key)
            if runtime is None or runtime.closed:
                runtime = _Runtime(paths, preprocessing, n_ctx=n_ctx,
                                   n_gpu_layers=n_gpu_layers, verbose=verbose)
                _POOL[self._key] = runtime
            runtime.references += 1
            self._runtime = runtime

    def __call__(self, *parts, max_tokens=None, temperature=0.0,
                 logits_processor=None, response_format=None) -> str:
        _outside_call()
        if os.getpid() != self._pid:
            raise RuntimeError("Load a new vision function in each child process.")
        with self._lock:
            if self._runtime is None:
                raise RuntimeError("This vision function has been closed.")
            if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 0):
                raise ValueError("max_tokens must be None or a non-negative integer.")
            if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
                raise ValueError("temperature must be a finite non-negative number.")
            content = _to_chat_content(*parts)
            if max_tokens == 0:
                return ""
            messages = []
            if self._prompt is not None:
                messages.append({"role": "system", "content": self._prompt})
            messages.append({"role": "user", "content": content})
            return self._runtime.run(
                self._adapter, messages, max_tokens=max_tokens, temperature=temperature,
                logits_processor=logits_processor, response_format=response_format,
            )

    def close(self):
        if os.getpid() != self._pid:
            return  # Never free inherited native resources in a forked child.
        _outside_call()
        with self._lock, _POOL_LOCK:
            runtime = self._runtime
            if runtime is None:
                return
            with runtime.lock:
                self._runtime = None
                runtime.references -= 1
                if runtime.references == 0:
                    if _POOL.get(self._key) is runtime:
                        del _POOL[self._key]
                    runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @property
    def spec(self):
        return self._meta.get("spec", "")

    @property
    def interpreter(self):
        return self._meta["interpreter"]

    def __repr__(self):
        if self._meta.get("mode") == "base":
            return f"VisionFunction(base interpreter={self.interpreter!r})"
        return f"VisionFunction(interpreter={self.interpreter!r})"
