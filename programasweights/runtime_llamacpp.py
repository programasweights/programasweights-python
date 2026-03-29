"""
llama.cpp runtime for local inference with LoRA adapters.

Loads a base GGUF model (Q6_K) and applies a Q4_0 LoRA adapter per-program.
Uses the pre-rendered prompt template from the .paw bundle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

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
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        program_dir = Path(program_dir)
        self._program_dir = program_dir

        # Load metadata
        meta_path = program_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)
        else:
            self._meta = {}

        # Load prompt template
        template_path = program_dir / "prompt_template.txt"
        if not template_path.exists():
            raise FileNotFoundError(f"No prompt_template.txt in {program_dir}")
        self._template = template_path.read_text()

        # Get base model GGUF
        interpreter = self._meta.get("interpreter", "Qwen/Qwen3-0.6B")
        base_model_path = cache.get_base_model_path(interpreter)

        # Load base model
        self._llm = Llama(
            model_path=str(base_model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

        # Load LoRA adapter
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
        rendered = self._template.replace("{INPUT_PLACEHOLDER}", input_text)

        use_special = self._meta.get("interpreter", "") not in ("gpt2",)
        token_ids = self._llm.tokenize(
            rendered.encode("utf-8"), add_bos=not use_special, special=use_special,
        )

        out = self._llm.create_completion(
            prompt=token_ids,
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0,
            echo=False,
        )

        return out["choices"][0]["text"].strip()

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
