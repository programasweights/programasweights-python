# ADR-001: Use llama.cpp instead of PyTorch for SDK runtime

## Status: Accepted (2026-03-21)

## Context

The SDK currently depends on torch + transformers (~2GB install). Users expect a lightweight package they can `pip install` in seconds. Most users run on CPU-only machines (laptops, desktops). The .paw format v2 stores raw safetensors LoRA weights that require PyTorch to apply.

## Decision

Replace the PyTorch runtime with llama-cpp-python (~80MB install). Use GGUF model format for the base interpreter and Q4_0 quantized GGUF LoRA adapters. Pre-render chat templates server-side so the client needs no tokenizer library.

## Consequences

- Install size drops from ~2GB to ~80MB (25x reduction)
- Users no longer need CUDA, PyTorch, or transformers
- Inference uses Metal (Mac), CPU (Linux/Windows) — no GPU required
- Must pre-render chat templates server-side (no transformers tokenizer on client)
- .paw format must change from v2 (safetensors) to v3 (GGUF adapter)
- Base model is downloaded once (~594 MB for Q6_K) and shared across all functions
- Per-function adapter download is ~23MB (Q4_0, confirmed lossless at 4096-scale eval)
