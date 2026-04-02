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
fn = paw.function(name_or_id, n_ctx=2048, n_gpu_layers=0, verbose=False)
```

Loads a compiled program and returns a callable. Downloads the program and base model on first use; cached locally after that. Works offline after first download.

| Parameter | Description |
|-----------|-------------|
| `name_or_id` | A `Program` object, hash ID (e.g. `a6b454023d41ac9ca845`), slug (e.g. `da03/my-classifier`), or official shorthand (e.g. `email-triage`). |
| `n_ctx` | Context length for the local runtime (default `2048`). |
| `n_gpu_layers` | GPU layers to offload (`0` = CPU-only, `-1` = all). Set `PAW_GPU_LAYERS` env var as default. |
| `verbose` | Enable verbose logging (default `False`). |

The returned callable:

```python
output: str = fn(input_text, max_tokens=512, temperature=0.0)
```

| Parameter | Description |
|-----------|-------------|
| `input_text` | Input string for the program. |
| `max_tokens` | Maximum tokens to generate (default `512`). |
| `temperature` | Sampling temperature (default `0.0`). |

**Context limits:** Spec + input + output share a ~2048 token window. Inputs exceeding the window will error (not silently truncated). Setting `max_tokens` high is safe -- generation stops at EOS or when the window is full.

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
| `spec` | Natural language specification (10-8000 chars). |
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
| `paw.api_url` | Base URL for API requests. Default: `https://programasweights.com`. Override with `PAW_API_URL` env var. |
| `paw.api_key` | API key for authenticated calls. Set via `paw.login()` or `PAW_API_KEY` env var. |
| `paw.__version__` | Installed package version string. |

## Related

- [CLI Reference](cli.md)
- [REST API Reference](rest-api.md)
- [Naming Programs](../getting-started/naming-programs.md)
