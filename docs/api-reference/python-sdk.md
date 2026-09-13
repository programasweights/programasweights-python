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

Loads a compiled program and returns a callable. Hub references download the
program and base model on first use; local `.paw` files supply the program
bundle directly. Required runtime metadata and base models are cached for reuse.

| Parameter | Description |
|-----------|-------------|
| `program_id` | Required. A `Program` object, hash ID (e.g. `a6b454023d41ac9ca845`), slug (e.g. `da03/my-classifier`), official shorthand (e.g. `email-triage`), or local `.paw` path (see below). A `Program` resolves by immutable `id`, not its mutable slug. |
| `n_ctx` | Context length for the local runtime (default `2048`). |
| `n_gpu_layers` | GPU layers to offload (`0` = CPU-only, `-1` = all). The default is `-1`, or `PAW_GPU_LAYERS` when set. |
| `verbose` | Enable verbose logging (default `False`). |
| `offline` | Use only local files/cache and make zero network calls; fail if required validated assets are missing. `PAW_OFFLINE=1` has the same effect. |
| `interpreter` | Advanced adapter-free mode only. Must be passed by keyword and only when `program_id` is explicitly `None`. Supported values are `Qwen/Qwen3-0.6B` and `gpt2`. |

The returned callable:

```python
output: str = fn(input_text, max_tokens=None, temperature=0.0, logits_processor=None)
```

| Parameter | Description |
|-----------|-------------|
| `input_text` | Input string for the program. |
| `max_tokens` | Maximum tokens to generate. `None` (default) = use all remaining context window. |
| `temperature` | Sampling temperature (default `0.0`). |
| `logits_processor` | SDK 0.4.6+. Optional `llama_cpp.LogitsProcessorList` of caller-supplied processors, applied at every generation step. `None` (default) keeps sampling unchanged. |

This advanced hook is not built-in regex or JSON-schema validation. Each processor takes `(input_ids, scores)` and returns modified scores. Its token history includes the full prompt (including any compiled prefix and suffix or base-model template) plus generated tokens. Create or reset stateful processors for each call; processor exceptions propagate to the caller. Token limits and the usual output whitespace trimming still apply, so validate the returned result.

**Context limits:** Spec + input + output share a ~2048 token window. Inputs that exceed it will error. `max_tokens` defaults to `None`: generation runs until EOS or the context limit.

Compiled mode is strict: the adapter, prompt template, matching metadata,
runtime manifest, and runtime-compatible base-model file must all validate. Version 0.4.5
accepts runtime manifest version 1 with `adapter_format="gguf_lora"`.
Built-in models are checked against pinned size/SHA-256 metadata and GGUF
magic. Historical manifests for those known runtime IDs are normalized to the
same canonical integrity metadata, so missing server-side checksum fields
cannot weaken validation. Missing or failed adapters raise an error; the SDK
never silently falls back to an unadapted base model.

### Loading a local `.paw` file

Version 0.4.5 adds local-file inputs to `paw.function`:

```python
from pathlib import Path

fn = paw.function(Path("classifier.paw"))
# With the required runtime metadata and base model already available locally:
fn = paw.function("./classifier.paw", offline=True)
```

Use a current GGUF ZIP `.paw` bundle, such as one downloaded from a hosted
compile. It must contain `meta.json`, `adapter.gguf`, and `prompt_template.txt`,
with only `pseudo_program.txt` allowed as an optional extra; serialized native
prefix state is not accepted from archives. Local inputs are selected
deterministically: a `Path`/`os.PathLike`
object, an explicit path such as `./classifier.paw` or an absolute path, or a
string ending in `.paw` (case-insensitive). Ordinary IDs and slugs such as
`da03/my-classifier` keep their existing Hub behavior even if a matching local
file exists. Use `Path(...)` or an explicit path for a filename without the
`.paw` suffix. URL inputs are unsupported.

The bundle is validated and imported under
`PAW_CACHE_DIR/local_programs/<archive-sha256>` (default cache root:
`~/.cache/programasweights`). Its source file is unchanged, and its metadata
cannot replace a Hub program-ID or slug cache. Missing or invalid local files
raise an error without falling back to a Hub lookup or program download.

A local program does not necessarily make the first load fully offline:
the existing runtime policy may fetch required runtime metadata from PAW and
download the shared base model. Pass `offline=True` or set `PAW_OFFLINE=1`
to prohibit all network access. Historical `PAW\x02` tensor containers,
including output from the legacy `convert_peft_to_paw` module, are not supported
by this loader; it does not convert PEFT tensors to GGUF.

Only `paw.function` gains local-file inputs. `prepare_program` and
`is_offline_ready` continue to accept Hub program references.

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
| `status` | Status returned by the server, normally `"ready"` on success. HTTP errors raise `APIError` instead of returning a failed `Program`. |
| `compiler_snapshot` | Exact compiler version used. |
| `timings` | Timing metadata from the server. |
| `error` | Error message when compilation fails. |

### Compile timeouts

Synchronous `compile` uses `httpx.Timeout(120.0, read=2400.0)`: connect, write,
and pool waits remain 120 seconds; the read timeout is 2,400 seconds. This
allows for the origin's 1,900-second provider wait plus up to 330 seconds of
artifact finalization. It is a timeout while waiting for response data, **not a
40-minute total deadline or guarantee**; upstream services may fail earlier.
The same setting applies to the compile step of `compile_and_load`.

Async submission retains a 30-second timeout. Precheck, status polling, and
cancellation each retain a 10-second timeout. For long finetunes, prefer the
explicit async workflow below so you retain a job ID for later status checks.

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

### Compile API errors

`compile`, `precheck_compile`, `compile_async`, `get_compile_status`, and
`cancel_compile` raise `paw.APIError` for HTTP 4xx/5xx responses. It is a subclass
of `httpx.HTTPStatusError`, so existing handlers continue to work. When supplied
by the server, `code`, `message`, and `request_id` are available as attributes
and included in the exception text. Missing fields are `None`; the original
`request` and `response` remain available, including response headers and body.

```python
try:
    job = paw.compile_async(SPEC, compiler="paw-ft-bs48")
except paw.APIError as error:
    print(error.code, error.message, error.request_id)
    # error.response.status_code and error.response.headers are unchanged.
    raise
```

For example, a `durable_queue_unavailable` 503 reports that durable Redis must
be healthy before async compilation can proceed. That rejection occurs before
the job is accepted; the caller can submit again after service recovery.
The SDK does not automatically retry compilation requests: other failures may
occur after a job has already been recorded. Invalid/non-JSON error bodies
retain the ordinary HTTP error description rather than displaying raw content.

Transport errors such as `httpx.ReadTimeout` propagate unchanged, rather than
becoming `APIError`. A timeout does not prove the server rejected or cancelled
the work, so the SDK does not automatically resubmit it. Both `paw.compile`
and `paw.compile_and_load` propagate these errors; `compile_and_load` does not
attempt to load a function when compilation raises.

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
