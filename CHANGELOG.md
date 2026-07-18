# Changelog

## 0.4.4 (2026-07-18)

- Add desktop preparation and cache inspection APIs with structured progress:
  `prepare_program`, `is_offline_ready`, and `list_cached_programs`.
- Make program/runtime/model caching strict and race-safe: immutable IDs win
  over slugs, streamed bundles are bounded and validated before atomic
  installation, runtime manifests are versioned, prefix state is atomically
  locked, and known GGUF runtimes enforce canonical size/SHA-256 metadata.
- Enforce true offline behavior through `offline=True` or `PAW_OFFLINE=1`;
  missing assets now fail clearly without a network call.
- Add precheck, explicit-finetune `compile_async`, status, and cancellation
  helpers for long-running compiles. Async compilation now requires a compiler
  and exposes ready slug/version metadata in its typed responses.
- Extend `paw run` with mutually exclusive compiled/base routing, `--offline`,
  and mode/program/interpreter fields in JSON output.
- Add an advanced, explicit adapter-free path with
  `paw.function(None, interpreter=...)` for Qwen3-0.6B and GPT-2. It uses
  versioned built-in prompts and never silently replaces compiled execution.
- Expand hermetic cache/runtime/CLI tests, add a fake-llama base-runtime suite,
  and test Python 3.9 through 3.13 in CI.

## 0.4.3 (2026-07-06)

- Fix `paw info` / `paw rename` crashing with `AttributeError` when run without
  `--api-url`/`--api-key` (they read the removed `paw.api_url`/`paw.api_key`
  module attributes; now resolved via the function-based config API).
- `--api-url` / `--api-key` now take effect on `compile`, `run`, and `login`
  (previously accepted but silently ignored on those commands).
- Docs: reference `paw.get_api_url()` / `paw.get_api_key()` instead of the
  removed module attributes.
- Add hermetic CLI auth test suite (`tests/test_cli_auth.py`) and a GitHub
  Actions CI workflow running it on Python 3.9-3.12.

## 0.2.4 (2026-04-01)

- Download reliability: asset endpoints return 202 Retry-After when program is still generating, eliminating 404 race conditions
- Guard local cleanup on HF upload success: files persist locally if upload fails
- Stderr suppression now covers LoRA adapter loading (fixes CPU_REPACK warnings on 0.2.3)
- Browser: fix LoRA switching bug (free old adapter + clear KV before loading new program)

## 0.2.3 (2026-04-01)

- Python 3.9 compatibility fix (`from __future__ import annotations`)

## 0.2.2 (2026-03-19)

- Add `paw.compile_and_load()` convenience method (compile + load in one call)
- `paw.function()` now accepts a `Program` object directly (not just string ID)
- Add `paw.list_programs()` for authenticated users to list their programs
- Suppress `llama.cpp` stderr noise by default; use `verbose=True` to enable
- Rewritten AGENTS.md with spec-writing tips, full API reference, common errors, and performance notes

## 0.2.0 (2026-04-01)

First public release.

- Compile natural language specs into neural programs via `paw.compile()`
- Load and run programs locally via `paw.function()` (llama.cpp backend)
- HuggingFace-style slug naming: `paw.function("da03/my-classifier")`
- Offline support: slug cache + program cache, no internet after first use
- CLI: `paw compile`, `paw run`, `paw info`, `paw rename`, `paw login`
- Pre-built wheels via pypi.programasweights.com for fast install
- Privacy enforcement: private programs only accessible by owner
