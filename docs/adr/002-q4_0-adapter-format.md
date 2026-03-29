# ADR-002: Use Q4_0 quantization for LoRA adapters

## Status: Accepted (2026-03-21)

## Context

LoRA adapters can be stored at various precision levels. fp32 is 162MB per adapter; users download one adapter per compiled function. We need to minimize download size without losing accuracy.

## Decision

Use Q4_0 quantized GGUF adapters (23MB, 7x smaller than fp32).

## Evidence

Empirical evaluation on 4096 test_clean examples (Qwen3 0.6B, PyTorch bf16 base):

| Adapter format | Size | EM (4096 examples) |
|---------------|------|-------------------|
| fp32 | 162 MB | 0.6580 |
| fp16 | 81 MB | 0.6577 |
| Q8_0 | 43 MB | 0.6584 |
| Q5_0 | 28 MB | 0.6580 |
| Q4_0 | 23 MB | 0.6584 |

All within noise (+-2 examples). Q4_0 adapters confirmed lossless. Combined base quantization + Q4_0 adapter also shows no compounding error.

## Consequences

- 7x smaller per-function downloads (23MB vs 162MB)
- No accuracy loss confirmed at scale
- Uses legacy GGUF quant format (block_size=32) which fits rank=64 adapter tensors
- K-quant formats (block_size=256) cannot be used for adapters due to rank dimension
