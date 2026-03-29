# Python SDK Reference

The `programasweights` package exposes a small surface for compiling specifications on the server and running compiled programs locally.

## Import

```python
import programasweights as paw
```

Use the package name `programasweights`; the conventional alias is `paw`.

## `paw.function`

```python
fn = paw.function(name_or_id, n_ctx=2048, n_gpu_layers=0, verbose=False)
```

Loads a compiled program and returns a callable `PawFunction`.

| Parameter | Description |
|-----------|-------------|
| `name_or_id` | Program hash ID, a namespaced slug (for example `programasweights/email-triage`), or an official shorthand such as `email-triage`. |
| `n_ctx` | Context length passed to the local runtime (default `2048`). |
| `n_gpu_layers` | Number of layers to offload to GPU in the local runtime (default `0`, CPU-only). |
| `verbose` | Enable verbose logging from the loader/runtime (default `False`). |

The returned callable has this signature:

```python
output: str = fn(input_text, max_tokens=512, temperature=0.0)
```

| Parameter | Description |
|-----------|-------------|
| `input_text` | Input string for the program. |
| `max_tokens` | Maximum tokens to generate (default `512`). |
| `temperature` | Sampling temperature (default `0.0`). |

## `paw.compile`

```python
result = paw.compile(spec, compiler="paw-4b-qwen3-0.6b")
```

Sends `spec` to the server for compilation. Returns a `CompileResult` with at least:

| Attribute | Description |
|-----------|-------------|
| `program_id` | Identifier usable with `paw.function` or the CLI. |
| `status` | Compilation status. |
| `pseudo_program` | Human-readable pseudo-program representation when available. |
| `timings` | Timing metadata from the server. |
| `error` | Error information when compilation fails. |

## Configuration and metadata

| Name | Description |
|------|-------------|
| `paw.api_url` | Base URL for API requests. Default: `https://programasweights.com`. |
| `paw.api_key` | API key for authenticated calls. Set via `paw.login()` or the `PAW_API_KEY` environment variable. |
| `paw.__version__` | Installed package version string. |

## `paw.login`

```python
paw.login(email)
```

Authenticates the SDK and stores credentials under `~/.config/programasweights/` for subsequent requests.

## Related

- [CLI Reference](cli.md)
- [REST API Reference](rest-api.md)
