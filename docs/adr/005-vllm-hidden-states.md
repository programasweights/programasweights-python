# ADR-005: Use vLLM extract_hidden_states for LoRA extraction

## Status: Accepted (2026-03-21)

## Context

Step 2 of compilation requires extracting per-layer hidden states from the trained compiler model at specific prefix token positions. Options: (A) vLLM `extract_hidden_states`, (B) SGLang `return_hidden_states`, (C) plain HuggingFace `output_hidden_states=True`.

## Decision

Use vLLM's `extract_hidden_states` speculative decoding API for production. Fall back to HuggingFace during development.

## Rationale

- vLLM provides native continuous batching for high throughput
- Output shape `[num_hidden_layers, prompt_len, hidden_size]` — extracting prefix positions is a trivial slice: `hidden_states[:, -prefix_steps:, :]`
- SGLang only returns last-layer hidden states by default (our LoRA mapper needs per-layer)
- HuggingFace works but lacks continuous batching (still fine for low-volume launch)

## Consequences

- Requires vLLM with speculative decoding config
- Hidden states saved as safetensors, then processed by LoRA mapper
- If vLLM setup proves problematic, HuggingFace is a drop-in fallback
