# Changelog

## 0.2.0 (2026-04-01)

First public release.

- Compile natural language specs into neural programs via `paw.compile()`
- Load and run programs locally via `paw.function()` (llama.cpp backend)
- HuggingFace-style slug naming: `paw.function("da03/my-classifier")`
- Offline support: slug cache + program cache, no internet after first use
- CLI: `paw compile`, `paw run`, `paw info`, `paw rename`, `paw login`
- Pre-built wheels via pypi.programasweights.com for fast install
- Privacy enforcement: private programs only accessible by owner
