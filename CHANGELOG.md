# Changelog

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
