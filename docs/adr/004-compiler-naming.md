# ADR-004: Two-level compiler naming (alias + dated snapshot)

## Status: Accepted (2026-03-21)

## Context

The compilation pipeline uses two models: an untrained model for pseudo-program generation and a trained compiler for LoRA extraction. Users should not need to know about this split. We also need version pinning for reproducibility.

## Decision

Two-level naming following OpenAI's convention:

- **Pretty alias** (always points to latest): `paw-4b-qwen3-0.6b`
- **Exact snapshot** (immutable, dated): `paw-4b-qwen3-0.6b-20260325`

The backend maintains a registry mapping each name to the full pipeline config (pseudo-gen model, trained compiler path, LoRA mapper path, compatible interpreters, etc.).

## Format

```
paw-{compiler_size}-{interpreter_model}-{date}
     4b               qwen3-0.6b        20260325
```

## Consequences

- Users see a simple name; the two-model split is hidden
- Alias updates silently when we train better models
- Power users can pin to a snapshot for reproducibility
- .paw files always store the exact snapshot name
- Backend validates compiler-interpreter compatibility at request time
- Adding new compiler/interpreter combos is just a registry entry
