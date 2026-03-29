"""
llama.cpp runtime for local inference with LoRA adapters.

Loads a base GGUF model (Q6_K) and applies a Q4_0 LoRA adapter per-program.
Uses the pre-rendered prompt template from the .paw bundle.

Prefix KV cache is saved to disk after the first call and reloaded on
subsequent runs, eliminating the ~2-3s cold-start prefix evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import llama_cpp
from llama_cpp import Llama

from . import cache


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
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        program_dir = Path(program_dir)
        self._program_dir = program_dir
        self._verbose = verbose

        meta_path = program_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)
        else:
            self._meta = {}

        template_path = program_dir / "prompt_template.txt"
        if not template_path.exists():
            raise FileNotFoundError(f"No prompt_template.txt in {program_dir}")
        self._template = template_path.read_text()

        interpreter = self._meta.get("interpreter", "Qwen/Qwen3-0.6B")
        base_model_path = cache.get_base_model_path(interpreter)

        self._llm = Llama(
            model_path=str(base_model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

        adapter_path = program_dir / "adapter.gguf"
        if adapter_path.exists():
            self._adapter = llama_cpp.llama_adapter_lora_init(
                self._llm.model, str(adapter_path).encode("utf-8"),
            )
            if self._adapter:
                llama_cpp.llama_set_adapter_lora(self._llm.ctx, self._adapter, 1.0)
            else:
                raise RuntimeError(f"Failed to load LoRA adapter: {adapter_path}")
        else:
            self._adapter = None

        placeholder = "{INPUT_PLACEHOLDER}"
        self._use_special = interpreter not in ("gpt2",)

        if placeholder in self._template:
            prefix_text = self._template.split(placeholder)[0]
            suffix_text = self._template.split(placeholder)[1]
        else:
            prefix_text = self._template
            suffix_text = ""

        self._prefix_tokens = self._llm.tokenize(
            prefix_text.encode("utf-8"),
            add_bos=not self._use_special,
            special=self._use_special,
        )
        self._suffix_text = suffix_text
        self._n_prefix = len(self._prefix_tokens)

        self._load_or_eval_prefix()

    def _load_or_eval_prefix(self):
        """Load prefix KV state from disk cache, or evaluate and save it."""
        cache_path = self._program_dir / "prefix_kv_cache.bin"

        if cache_path.exists():
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
                if n_loaded > 0:
                    self._llm.n_tokens = self._n_prefix
                    self._llm.input_ids[:self._n_prefix] = self._prefix_tokens
                    if self._verbose:
                        print(f"Loaded prefix KV cache ({self._n_prefix} tokens) from disk")
                    return
            except Exception:
                pass

        self._llm.eval(self._prefix_tokens)

        try:
            token_array = (llama_cpp.llama_token * self._n_prefix)(*self._prefix_tokens)
            result = llama_cpp.llama_state_seq_save_file(
                self._llm.ctx,
                str(cache_path).encode("utf-8"),
                0,
                token_array,
                self._n_prefix,
            )
            if result and self._verbose:
                size_mb = cache_path.stat().st_size / (1024 * 1024)
                print(f"Saved prefix KV cache ({self._n_prefix} tokens, {size_mb:.1f} MB)")
        except Exception:
            pass

    def __call__(
        self,
        input_text: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Run the program on an input.

        Args:
            input_text: The input to process.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature (0 = greedy).

        Returns:
            The program's output as a string.
        """
        # Reset to prefix state: clear everything after the prefix
        self._llm.n_tokens = self._n_prefix

        input_with_suffix = input_text + self._suffix_text
        input_tokens = self._llm.tokenize(
            input_with_suffix.encode("utf-8"),
            add_bos=False,
            special=self._use_special,
        )

        self._llm.eval(input_tokens)

        output_tokens = []
        for _ in range(max_tokens):
            token = self._llm.sample(
                temp=temperature if temperature > 0 else 0,
            )

            if token == self._llm.token_eos():
                break

            output_tokens.append(token)
            self._llm.eval([token])

        output_bytes = self._llm.detokenize(output_tokens)
        return output_bytes.decode("utf-8", errors="replace").strip()

    def __del__(self):
        if hasattr(self, "_adapter") and self._adapter:
            try:
                llama_cpp.llama_rm_adapter_lora(self._llm.ctx, self._adapter)
                llama_cpp.llama_adapter_lora_free(self._adapter)
            except Exception:
                pass

    @property
    def spec(self) -> str:
        return self._meta.get("spec", "")

    @property
    def interpreter(self) -> str:
        return self._meta.get("interpreter", "Qwen/Qwen3-0.6B")

    def __repr__(self) -> str:
        spec_preview = self.spec[:60] + "..." if len(self.spec) > 60 else self.spec
        return f"PawFunction('{spec_preview}')"
