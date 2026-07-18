# Python SDK Reference

The `programasweights` package compiles natural language specs into neural programs that run locally.

## Install

```bash
pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
```

## Import

```python
import programasweights as paw
```

## `paw.function`

```python
fn = paw.function(
    program_id,
    n_ctx=2048,
    n_gpu_layers=None,
    verbose=False,
    offline=False,
    *,
    interpreter=None,
)
```

Loads a compiled program and returns a callable. Downloads the program and base model on first use; cached locally after that. Works offline after first download.

| Parameter | Description |
|-----------|-------------|
| `program_id` | Required. A `Program` object, hash ID (e.g. `a6b454023d41ac9ca845`), slug (e.g. `da03/my-classifier`), or official shorthand (e.g. `email-triage`). A `Program` resolves by immutable `id`, not its mutable slug. |
| `n_ctx` | Context length for the local runtime (default `2048`). |
| `n_gpu_layers` | GPU layers to offload (`0` = CPU-only, `-1` = all). The default is `-1`, or `PAW_GPU_LAYERS` when set. |
| `verbose` | Enable verbose logging (default `False`). |
| `offline` | Require all program/runtime/model assets to already be cached and make zero network calls. `PAW_OFFLINE=1` has the same effect. |
| `interpreter` | Advanced adapter-free mode only. Must be passed by keyword and only when `program_id` is explicitly `None`. Supported values are `Qwen/Qwen3-0.6B` and `gpt2`. |

The returned callable:

```python
output: str = fn(input_text, max_tokens=None, temperature=0.0)
```

| Parameter | Description |
|-----------|-------------|
| `input_text` | Input string for the program. |
| `max_tokens` | Maximum tokens to generate. `None` (default) = use all remaining context window. |
| `temperature` | Sampling temperature (default `0.0`). |

**Context limits:** Spec + input + output share a ~2048 token window. Inputs that exceed it will error. `max_tokens` defaults to `None`: generation runs until EOS or the context limit.

Compiled mode is strict: the adapter, prompt template, matching metadata,
runtime manifest, and runtime-compatible base-model file must all validate. Version 0.4.4
accepts runtime manifest version 1 with `adapter_format="gguf_lora"`.
Built-in models are checked against pinned size/SHA-256 metadata and GGUF
magic. Historical manifests for those known runtime IDs are normalized to the
same canonical integrity metadata, so missing server-side checksum fields
cannot weaken validation. Missing or failed adapters raise an error; the SDK
never silently falls back to an unadapted base model.

### Advanced: adapter-free base interpreter

Pass explicit `None` plus an interpreter to run the supported base GGUF without a compiled PAW program:

```python
base = paw.function(None, interpreter="gpt2")
output = base("raw prompt text")
```

This mode is intentionally explicit:

- `paw.function()` still requires the `program_id` argument.
- `program_id=None` without `interpreter` raises `ValueError`.
- `program_id=""` raises `ValueError` and explains that base mode requires explicit `None`.
- A non-empty program reference together with `interpreter` raises `ValueError`.
- No PAW API, slug lookup, program download, adapter load, or disk prefix cache is used.
- Online mode may download only the selected base GGUF from its built-in runtime manifest. Offline mode never downloads.
- Every invocation resets model state, renders the complete prompt, and tokenizes that complete rendered prompt in one call.

The built-in prompt contract is versioned with each runtime manifest and must contain exactly one `{INPUT_PLACEHOLDER}`:

```text
# Qwen/Qwen3-0.6B
<|im_start|>user
{INPUT_PLACEHOLDER}<|im_end|>
<|im_start|>assistant
<think>

</think>


# gpt2
{INPUT_PLACEHOLDER}
```

The Qwen bytes are the exact raw-user rendering of
`apply_chat_template(add_generation_prompt=True, enable_thinking=False)`.
Zero-token prompts and prompts that consume the full context window raise
`ValueError`.

## Preparing programs for offline use

```python
prepared = paw.prepare_program("da03/my-classifier")
assert prepared["offline_ready"]

ready = paw.is_offline_ready("da03/my-classifier")  # local check; no network
cached = paw.list_cached_programs()
```

`prepare_program` resolves and downloads the program, runtime manifest, and shared base model without retaining a loaded `PawFunction`. Pass `offline=True` to require an already complete local cache and prohibit network access.

Desktop applications can receive structured progress without parsing stderr:

```python
paw.prepare_program(
    "da03/my-classifier",
    progress=lambda event: print(event["stage"], event["status"]),
)
```

Without a callback, downloads keep using the existing CLI-style status output on stderr.

## `paw.compile`

```python
program = paw.compile(
    spec,
    compiler="paw-4b-qwen3-0.6b",
    name=None,
    tags=None,
    public=True,
    slug=None,
)
```

Compiles a natural language spec on the server. Returns a `Program` object.

| Parameter | Description |
|-----------|-------------|
| `spec` | Natural language specification (10-16000 chars). |
| `compiler` | Compiler name: `paw-4b-qwen3-0.6b` (Standard) or `paw-4b-gpt2` (Compact). |
| `name` | Display title for the hub (auto-generated if omitted). |
| `tags` | Tags for discovery (list of strings, max 10). |
| `public` | Whether to list on the public hub (default `True`). |
| `slug` | URL-safe handle (e.g. `my-classifier`). Creates a `username/slug` alias. Requires authentication. |

**Return value** -- `Program` object:

| Attribute | Description |
|-----------|-------------|
| `id` | Hash-based program identifier. Use with `paw.function(program.id)`. |
| `slug` | Full slug handle (e.g. `da03/my-classifier`) if one was created, `None` otherwise. |
| `status` | `"ready"` on success, `"failed"` on error. |
| `compiler_snapshot` | Exact compiler version used. |
| `timings` | Timing metadata from the server. |
| `error` | Error message when compilation fails. |

## Long-running compile jobs

The asynchronous compile endpoint is available through both `PAWClient` and top-level helpers:

```python
check = paw.precheck_compile(SPEC, compiler="paw-ft-bs48")
job = paw.compile_async(
    SPEC,
    compiler="paw-ft-bs48",
    public=False,
)

status = paw.get_compile_status(job["job_id"])
if status["status"] == "queued":
    paw.cancel_compile(job["job_id"])
```

`compile_async` requires an explicit finetune compiler. It submits the request synchronously and returns the queued job metadata immediately; mapper compilers must use `compile`. Poll `get_compile_status` for `queued`, `compiling`, `ready`, `failed`, or `cancelled`. Ready status data includes the immutable program ID and, when naming was requested, `slug`, `version`, and `version_action`.

Status and cancellation requests must use the same authenticated account as
submission. Anonymous jobs are bound to the validated client IP that submitted
them.

## `paw.compile_and_load`

```python
fn = paw.compile_and_load(spec, compiler="paw-4b-qwen3-0.6b", **kwargs)
```

Convenience method that compiles a spec and immediately loads the result for local inference. Equivalent to `paw.function(paw.compile(spec, ...).id)`. Returns a callable.

Accepts all the same parameters as `paw.compile`.

## `paw.list_programs`

```python
result = paw.list_programs(sort="recent", per_page=20)
```

Returns a dict with the authenticated user's compiled programs. Requires authentication.

| Parameter | Description |
|-----------|-------------|
| `sort` | Sort order: `"recent"` (default), `"votes"`, `"recommended"`. |
| `per_page` | Number of results per page (default `20`). |

**Return value** -- dict:

| Key | Description |
|-----|-------------|
| `programs` | List of program dicts with `id`, `spec`, `name`, `compiler`, etc. |
| `total` | Total number of programs. |

## `paw.login`

```python
paw.login(key=None)
```

Saves an API key for authenticated requests. If `key` is provided, saves it directly. If omitted, opens the browser to generate a key at `programasweights.com/settings`.

Keys are stored in `~/.config/programasweights/config.json` and loaded automatically on subsequent imports.

You can also set the `PAW_API_KEY` environment variable instead:

```bash
export PAW_API_KEY=paw_sk_...
```

## Configuration

| Name | Description |
|------|-------------|
| `paw.get_api_url()` | Base URL for API requests. Default: `https://programasweights.com`. Override with `PAW_API_URL` env var. |
| `paw.get_api_key()` | API key for authenticated calls. Set via `paw.login()` or `PAW_API_KEY` env var. |
| `paw.__version__` | Installed package version string. |

## Related

- [CLI Reference](cli.md)
- [REST API Reference](rest-api.md)
- [Naming Programs](../getting-started/naming-programs.md)
