"""
llama.cpp runtime for local inference with LoRA adapters.

Loads a base GGUF model (Q6_K for Qwen3, Q8_0 for GPT-2) and applies a
Q4_0 LoRA adapter per-program.
Uses the pre-rendered prompt template from the .paw bundle.

Prefix KV cache is saved to disk after the first call and reloaded on
subsequent runs, eliminating the ~2-3s cold-start prefix evaluation.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import llama_cpp
from llama_cpp import Llama

from . import cache

_NATIVE_STDERR_LOCK = threading.RLock()


@contextmanager
def _suppress_native_stderr(enabled: bool) -> Iterator[None]:
    """Temporarily silence llama.cpp's native stderr output."""
    if not enabled:
        with _NATIVE_STDERR_LOCK:
            yield
        return

    with _NATIVE_STDERR_LOCK:
        devnull = None
        old_stderr = None
        try:
            stderr_fd = sys.stderr.fileno()
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stderr = os.dup(stderr_fd)
        except (AttributeError, OSError, ValueError):
            if devnull is not None:
                os.close(devnull)
            if old_stderr is not None:
                os.close(old_stderr)
            yield
            return

        try:
            os.dup2(devnull, stderr_fd)
            yield
        finally:
            try:
                os.dup2(old_stderr, stderr_fd)
            finally:
                os.close(devnull)
                os.close(old_stderr)


class PawFunction:
    """A compiled neural program that runs locally via llama.cpp.

    Usage:
        fn = PawFunction(program_dir)
        result = fn("some input text")
    """

    def __init__(
        self,
        program_dir: str | Path,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        api_url: str | None = None,
        api_key: str | None = None,
        offline: bool = False,
    ):
        self._llm = None
        self._adapter = None
        self._closed = False
        self._mode = "compiled"
        self._program_dir = Path(program_dir)
        self._verbose = verbose
        self._n_ctx = n_ctx
        if not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx <= 0:
            raise ValueError("n_ctx must be a positive integer.")

        program_dir = self._program_dir
        try:
            meta_path = program_dir / "meta.json"
            template_path = program_dir / "prompt_template.txt"
            adapter_path = program_dir / "adapter.gguf"
            for path, label in (
                (meta_path, "meta.json"),
                (template_path, "prompt_template.txt"),
                (adapter_path, "LoRA adapter"),
            ):
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.stat().st_size <= 0
                ):
                    raise FileNotFoundError(
                        f"{label} not found or invalid: {path}"
                    )
            if not cache.is_valid_adapter_file(adapter_path):
                raise ValueError(
                    f"LoRA adapter is not a valid GGUF file: {adapter_path}"
                )

            with open(meta_path, encoding="utf-8") as handle:
                self._meta = json.load(handle)
            if not isinstance(self._meta, dict):
                raise ValueError(f"Invalid program metadata: {meta_path}")
            interpreter = self._meta.get("interpreter")
            if not isinstance(interpreter, str) or not interpreter:
                raise ValueError(
                    f"Program metadata has no valid interpreter: {meta_path}"
                )
            self._interpreter = interpreter

            self._template = template_path.read_text(encoding="utf-8")
            if self._template.count(cache.INPUT_PLACEHOLDER) != 1:
                raise ValueError(
                    "Compiled prompt template must contain exactly one "
                    f"{cache.INPUT_PLACEHOLDER} placeholder."
                )

            runtime_manifest = cache.resolve_runtime_manifest(
                self._meta,
                api_url=api_url,
                api_key=api_key,
                offline=offline,
            )
            if runtime_manifest is None:
                raise RuntimeError(
                    f"No usable runtime manifest for compiled program in "
                    f"{program_dir}."
                )
            self._meta["runtime"] = runtime_manifest
            self._meta.setdefault(
                "runtime_id",
                runtime_manifest.get("runtime_id"),
            )
            self._meta.setdefault(
                "runtime_manifest_version",
                runtime_manifest.get("manifest_version"),
            )
            base_model_path = cache.get_base_model_path(
                interpreter,
                runtime_manifest=runtime_manifest,
                offline=offline,
            )

            from ._output import status

            status("Loading interpreter...")
            with _suppress_native_stderr(not verbose):
                self._llm = Llama(
                    model_path=str(base_model_path),
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=verbose,
                )
                self._adapter = llama_cpp.llama_adapter_lora_init(
                    self._llm.model,
                    str(adapter_path).encode("utf-8"),
                )
                if self._adapter is None:
                    raise RuntimeError(
                        f"Failed to load LoRA adapter: {adapter_path}"
                    )
                self._apply_adapter(1.0)

            prefix_text, self._suffix_text = self._template.split(
                cache.INPUT_PLACEHOLDER,
            )
            self._prefix_tokens = self._llm.tokenize(
                prefix_text.encode("utf-8"),
                add_bos=False,
                special=True,
            )
            self._n_prefix = len(self._prefix_tokens)
            self._load_or_eval_prefix()
            status("Ready.")
        except BaseException:
            self._cleanup_resources()
            raise

    @classmethod
    def from_base(
        cls,
        interpreter: str,
        *,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        verbose: bool = False,
        offline: bool = False,
    ) -> "PawFunction":
        """Create an adapter-free base interpreter with a built-in prompt."""
        self = cls.__new__(cls)
        self._llm = None
        self._adapter = None
        self._closed = False
        self._mode = "base"
        self._program_dir = None
        self._verbose = verbose
        self._n_ctx = n_ctx
        if not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx <= 0:
            raise ValueError("n_ctx must be a positive integer.")

        try:
            runtime_manifest = cache.get_base_runtime_manifest(interpreter)
            self._template = cache.get_base_prompt_template(runtime_manifest)
            self._interpreter = interpreter
            self._meta = {
                "mode": "base",
                "interpreter": interpreter,
                "runtime_id": runtime_manifest["runtime_id"],
                "runtime_manifest_version": runtime_manifest[
                    "manifest_version"
                ],
                "runtime": runtime_manifest,
            }
            base_model_path = cache.get_base_model_path(
                interpreter,
                runtime_manifest=runtime_manifest,
                offline=offline,
            )

            from ._output import status

            status("Loading interpreter...")
            with _suppress_native_stderr(not verbose):
                self._llm = Llama(
                    model_path=str(base_model_path),
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu_layers,
                    verbose=verbose,
                )
            status("Ready.")
            return self
        except BaseException:
            self._cleanup_resources()
            raise

    def _apply_adapter(self, scale: float):
        """Apply LoRA adapter, compatible with both old and new llama.cpp API."""
        import ctypes
        if hasattr(llama_cpp, "llama_set_adapters_lora"):
            adapters = (llama_cpp.llama_adapter_lora_p_ctypes * 1)(self._adapter)
            scales = (ctypes.c_float * 1)(scale)
            result = llama_cpp.llama_set_adapters_lora(
                self._llm.ctx,
                adapters,
                1,
                scales,
            )
        else:
            result = llama_cpp.llama_set_adapter_lora(
                self._llm.ctx,
                self._adapter,
                scale,
            )
        if result not in (None, 0):
            raise RuntimeError("Failed to apply LoRA adapter to llama.cpp.")

    def _load_or_eval_prefix(self):
        """Load prefix KV state from disk cache, or evaluate and save it."""
        if self._n_prefix == 0:
            return
        if self._program_dir is None:
            raise RuntimeError("Prefix caching requires a compiled program.")
        cache_path = self._program_dir / "prefix_kv_cache.bin"

        if self._try_load_prefix_cache(cache_path, remove_stale=False):
            return

        lock_context = cache.prefix_cache_lock(self._program_dir)
        try:
            lock_context.__enter__()
        except OSError:
            self._llm.eval(self._prefix_tokens)
            return
        try:
            if self._try_load_prefix_cache(cache_path, remove_stale=True):
                return
            self._llm.eval(self._prefix_tokens)
            self._save_prefix_cache_atomically(cache_path)
        finally:
            try:
                lock_context.__exit__(None, None, None)
            except OSError:
                pass

    def _try_load_prefix_cache(
        self,
        cache_path: Path,
        *,
        remove_stale: bool,
    ) -> bool:
        if not cache_path.exists() and not cache_path.is_symlink():
            return False
        attempted_load = False
        if not cache_path.is_symlink():
            attempted_load = True
            try:
                import ctypes
                token_array = (llama_cpp.llama_token * self._n_prefix)(*self._prefix_tokens)
                n_token_count = ctypes.c_size_t(0)
                n_loaded = llama_cpp.llama_state_seq_load_file(
                    self._llm.ctx,
                    str(cache_path).encode("utf-8"),
                    0,
                    token_array,
                    self._n_prefix,
                    ctypes.byref(n_token_count),
                )
                loaded_count = int(n_token_count.value)
                loaded_tokens = [
                    int(token_array[index])
                    for index in range(min(loaded_count, self._n_prefix))
                ]
                if (
                    n_loaded > 0
                    and loaded_count == self._n_prefix
                    and loaded_tokens == self._prefix_tokens
                ):
                    self._llm.n_tokens = self._n_prefix
                    self._llm.input_ids[:self._n_prefix] = self._prefix_tokens
                    if self._verbose:
                        print(f"Loaded prefix KV cache ({self._n_prefix} tokens) from disk")
                    return True
            except Exception:
                pass
        if attempted_load:
            reset = getattr(self._llm, "reset", None)
            if callable(reset):
                reset()
            else:
                self._llm.n_tokens = 0
        if remove_stale:
            try:
                cache_path.unlink(missing_ok=True)
            except OSError:
                pass
        return False

    def _save_prefix_cache_atomically(self, cache_path: Path) -> None:
        tmp_path = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{cache_path.name}.",
                suffix=".tmp",
                dir=str(cache_path.parent),
            )
            os.close(fd)
            tmp_path = Path(tmp_name)
            token_array = (llama_cpp.llama_token * self._n_prefix)(*self._prefix_tokens)
            result = llama_cpp.llama_state_seq_save_file(
                self._llm.ctx,
                str(tmp_path).encode("utf-8"),
                0,
                token_array,
                self._n_prefix,
            )
            if (
                not result
                or not tmp_path.is_file()
                or tmp_path.stat().st_size <= 0
            ):
                return
            os.replace(str(tmp_path), str(cache_path))
            tmp_path = None
            if self._verbose:
                size_mb = cache_path.stat().st_size / (1024 * 1024)
                print(
                    f"Saved prefix KV cache "
                    f"({self._n_prefix} tokens, {size_mb:.1f} MB)"
                )
        except Exception:
            pass
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def __call__(
        self,
        input_text: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Run the program on an input.

        Args:
            input_text: The input to process.
            max_tokens: Maximum output tokens. None = use all remaining context.
            temperature: Sampling temperature (0 = greedy).

        Returns:
            The program's output as a string.
        """
        if self._closed or self._llm is None:
            raise RuntimeError("This PawFunction has been closed.")
        if not isinstance(input_text, str):
            raise TypeError("input_text must be a string.")
        if max_tokens is not None and (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens < 0
        ):
            raise ValueError("max_tokens must be None or a non-negative integer.")

        if self._mode == "base":
            reset = getattr(self._llm, "reset", None)
            if not callable(reset):
                raise RuntimeError(
                    "The installed llama-cpp runtime cannot reset model state."
                )
            reset()
            rendered_prompt = self._template.replace(
                cache.INPUT_PLACEHOLDER,
                input_text,
                1,
            )
            prompt_tokens = self._llm.tokenize(
                rendered_prompt.encode("utf-8"),
                add_bos=False,
                special=True,
            )
            if not prompt_tokens:
                raise ValueError(
                    "The rendered base prompt tokenized to zero tokens."
                )
            return self._generate(
                prompt_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                token_description="prompt",
            )

        self._llm.n_tokens = self._n_prefix

        input_with_suffix = input_text + self._suffix_text
        input_tokens = self._llm.tokenize(
            input_with_suffix.encode("utf-8"),
            add_bos=False,
            special=True,
        )
        return self._generate(
            input_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            prior_tokens=self._n_prefix,
            token_description="input",
        )

    def _generate(
        self,
        prompt_tokens: list[int],
        *,
        max_tokens: int | None,
        temperature: float,
        prior_tokens: int = 0,
        token_description: str,
    ) -> str:
        tokens_used = prior_tokens + len(prompt_tokens)
        remaining = self._n_ctx - tokens_used
        if remaining <= 0:
            raise ValueError(
                f"Context overflow: {tokens_used} tokens used "
                f"(prior={prior_tokens}, {token_description}="
                f"{len(prompt_tokens)}), context window={self._n_ctx}. "
                "Shorten the input or increase n_ctx."
            )

        gen_limit = remaining if max_tokens is None else min(max_tokens, remaining)

        if prompt_tokens:
            self._llm.eval(prompt_tokens)

        output_tokens = []
        for _ in range(gen_limit):
            token = self._llm.sample(
                temp=temperature if temperature > 0 else 0,
            )

            if token == self._llm.token_eos():
                break

            output_tokens.append(token)
            self._llm.eval([token])

        output_bytes = self._llm.detokenize(output_tokens)
        return output_bytes.decode("utf-8", errors="replace").strip()

    def _cleanup_resources(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True

        adapter = getattr(self, "_adapter", None)
        if adapter is not None:
            free_adapter = getattr(
                llama_cpp,
                "llama_adapter_lora_free",
                None,
            )
            if not callable(free_adapter):
                free_adapter = getattr(
                    llama_cpp,
                    "llama_free_adapter_lora",
                    None,
                )
            if callable(free_adapter):
                try:
                    free_adapter(adapter)
                except Exception:
                    pass
            self._adapter = None

        llm = getattr(self, "_llm", None)
        if llm is not None:
            close = getattr(llm, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            self._llm = None

    def close(self) -> None:
        """Release llama.cpp model and adapter resources."""
        self._cleanup_resources()

    def __enter__(self) -> "PawFunction":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self._cleanup_resources()
        except Exception:
            pass

    @property
    def spec(self) -> str:
        return self._meta.get("spec", "")

    @property
    def interpreter(self) -> str:
        return self._interpreter

    def __repr__(self) -> str:
        if self._mode == "base":
            return f"PawFunction(base interpreter={self.interpreter!r})"
        spec_preview = self.spec[:60] + "..." if len(self.spec) > 60 else self.spec
        return f"PawFunction('{spec_preview}')"
