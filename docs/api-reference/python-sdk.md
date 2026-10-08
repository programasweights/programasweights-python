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
    remote=False,
    interpreter=None,
    prompt_template=None,
)
```

Returns a callable for a compiled program. Local inference is the default.
Hub references download the
program and base model on first use; local `.paw` files supply the program
bundle directly. Required runtime metadata and base models are cached for reuse.

| Parameter | Description |
|-----------|-------------|
| `program_id` | Required. A `Program` object, hash ID (e.g. `a6b454023d41ac9ca845`), slug (e.g. `da03/my-classifier`), official shorthand (e.g. `email-triage`), or local `.paw` path (see below). A `Program` resolves by immutable `id`, not its mutable slug. |
| `n_ctx` | Context length for the local runtime (default `2048`). |
| `n_gpu_layers` | GPU layers to offload (`0` = CPU-only, `-1` = all). The default is `-1`, or `PAW_GPU_LAYERS` when set. |
| `verbose` | Enable verbose logging (default `False`). |
| `offline` | Use only local files/cache and make zero network calls; fail if required validated assets are missing. `PAW_OFFLINE=1` has the same effect. |
| `remote` | Run hosted inference without downloading model assets (default `False`). Accepts a `Program` object, ID, or slug. Cannot be combined with offline mode, local file paths, `interpreter`, or non-default local runtime options. |
| `interpreter` | Advanced adapter-free mode only. Must be passed by keyword and only when `program_id` is explicitly `None`. Supported values are `Qwen/Qwen3-0.6B`, `gpt2`, and `Qwen/Qwen3.5-0.8B`. |
| `prompt_template` | Unreleased. Optional complete numbered template for a local base interpreter. Pass by keyword with `program_id=None`. Works with text-only and image-capable models. Cannot override a compiled bundle or be used with `remote=True`. |

For existing local text-only programs and text base interpreters without an
explicit `prompt_template`, the returned callable accepts:

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

### Remote inference

```python
with paw.function("email-triage", remote=True) as remote_fn:
    output = remote_fn("Urgent: the server is down!")
```

The callable returns a string and accepts optional `max_tokens` and `temperature`.
Omitting either argument, or passing `None`, uses the server's default for that
setting. `logits_processor` is supported only for local inference.

Slugs are resolved once when the function is loaded. The `with` block closes
the HTTP client; otherwise, call `remote_fn.close()` when finished.

Remote calls use the existing [SDK configuration](#configuration) for the API
URL and API key. HTTP 4xx/5xx responses raise `paw.APIError`, preserving the
response status, headers, and structured error details. Transport errors
propagate unchanged.

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

Model files download on first use. Pass `offline=True` to use cached files only.
Each call is independent.

The text interpreters format prompts as follows:

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

The Qwen3 interpreter uses its chat template with thinking disabled. GPT-2 uses
the input as a raw prompt. These defaults are unchanged when `prompt_template`
is omitted.

#### Complete numbered templates (unreleased)

For a local base model, pass `prompt_template=` when loading the function to
specify the complete prompt. This option works for both text-only and
image-capable interpreters:

```python
prompt = (
    "<|im_start|>system\n{INPUT_0}<|im_end|>\n"
    "<|im_start|>user\n{INPUT_1}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
answer = paw.function(
    None, interpreter="Qwen/Qwen3-0.6B", prompt_template=prompt,
)
output = answer(
    "Answer using only the facts in the question.",
    "Alice owns three cats. How many cats does Alice own?",
)
```

The template controls role delimiters, examples, whitespace, thinking markers
and the assistant prefix. The SDK adds no chat wrapper. Use the format expected
by the selected model. Text is tokenized with special tokens recognized and
without an automatically added BOS prefix. Image-capable models insert the
required vision tokens and embeddings at image slots.

Numbered templates share these rules across text and image models:

- `{INPUT_0}`, `{INPUT_1}`, etc. refer to positional arguments. Used indices must
  start at zero without gaps; `{INPUT_1}/{INPUT_0}/{INPUT_1}` is valid.
- Pass exactly the number of distinct indices. Repeating a slot repeats its
  content at that position. A nonempty constant template may have no slots and
  be called with `fn()`.
- Input strings are inserted literally and are not parsed again for slots.
  Only canonical numbered placeholders are interpreted. Other brace content is
  literal; this is not Python `str.format`. Doubled braces do not escape a slot:
  `{{INPUT_0}}` surrounds the inserted value with literal braces.
- Text-only models accept strings. Image-capable models accept strings or
  `paw.Image` in any slot, including calls containing only strings.
- Generation options are keyword-only for numbered-template calls:
  `fn("first", "second", max_tokens=32)`.

For an already rendered text prompt, use `prompt_template="{INPUT_0}"`.
Qwen3.5's default base template is this identity template: it accepts one
argument unchanged and supplies no conversation wrapper. GPT-2 and Qwen3-0.6B
keep the default text templates shown above when the option is omitted.

`prompt_template=` is available only with `program_id=None` and local inference.
Compiled programs use their bundled `prompt_template.txt`; passing an override
with a compiled program or `remote=True` raises an error.

Existing compiled text programs still require exactly one
`{INPUT_PLACEHOLDER}`. Their literal text, tokenization boundaries and prefix
cache behavior are unchanged. Existing text calls such as `fn(input_text="x")`
and `fn("x", 32)` remain valid; explicit numbered templates use the positional
inputs and keyword generation options described above.

### Local text/image calls

Install `programasweights[vision]` to use Qwen3.5-0.8B. With the unreleased
numbered-template API, order images and text through the complete template:

```python
prompt = (
    "<|im_start|>user\n"
    "Before: {INPUT_1}\nAfter: {INPUT_2}\n{INPUT_0}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
compare = paw.function(
    None, interpreter="Qwen/Qwen3.5-0.8B", prompt_template=prompt,
)
answer = compare(
    "What changed?", paw.Image("before.png"), paw.Image("after.png"),
)
print(answer)
```

Here the question is argument zero, but the model receives both images before
that question. Repeating `{INPUT_1}` would place the first image at each
occurrence. There is no separate public image marker to insert manually.

For a compiled image program, `prompt_template.txt` contains the complete
prompt, for example:

```text
<|im_start|>system
Locate the requested object.<|im_end|>
<|im_start|>user
{INPUT_0}
{INPUT_1}<|im_end|>
<|im_start|>assistant
<think>

</think>

```

With that template, the call is:

```python
fn = paw.function("./locator.paw")
answer = fn("Find the red cup.", paw.Image("scene.png"))
```

The image runtime manifest declares the template contract as:

```json
"prompt_template": {
  "format": "rendered_text",
  "placeholder": "{INPUT_N}"
}
```

This is the template field inside the full runtime manifest, not a standalone
manifest. Keep the required model, projector, preprocessing and adapter metadata.
Older experimental image bundles with `chat_messages`, `system_prompt_file`,
`chat_format` or `enable_thinking` prompt metadata are rejected. Re-export those
bundles with the new contract and a full template matching the adapter's training
prompt. Changing the metadata alone does not reconstruct the missing roles or
assistant prefix. Existing text bundles do not need this migration.

`paw.Image(source)` accepts a local file path, encoded image bytes, or a Pillow
image. For an image URL, download the file first. Plain strings are text inputs.

Image functions return a string. Their positional argument count and order
come from the template. Generation options are passed by keyword:

```python
output: str = fn(
    *parts,
    max_tokens=None,
    temperature=0.0,
    logits_processor=None,
    response_format=None,
    return_info=False,
)
```

`max_tokens`, `temperature`, and `logits_processor` work as described above for
text functions. Use `response_format={"type": "json_object"}` to request JSON
output.

Pass `return_info=True` to receive a `paw.FunctionResult` instead of a string:

```python
result = fn("Find the target:", paw.Image("scene.png"), return_info=True)
print(result.text)
print(result.finish_reason)    # e.g. "stop" or "length"; None when unavailable.
print(result.usage)            # Token counts, or None when unavailable.
print(result.elapsed_seconds)  # Total call time in seconds.
```

The result and its `usage` mapping are read-only. Use `dict(result.usage)`
for JSON serialization when token counts are available. `elapsed_seconds`
measures the full call, excluding model loading and construction of
`paw.Image` inputs.

`return_info` is available for local Qwen3.5 functions, including text-only calls
and compiled image programs. A `finish_reason` of `"length"` means generation
reached a token limit.

#### Offline use and closing functions

Model files download on first use. Once cached, pass `offline=True` to
`paw.function` to load without network access.

Call `fn.close()` when finished, or use `with paw.function(...) as fn:` to close
automatically. For multiprocessing, start workers with `spawn` and load the
function inside each worker.

## Preparing programs for offline use

```python
prepared = paw.prepare_program("da03/my-classifier")
assert prepared["offline_ready"]

ready = paw.is_offline_ready("da03/my-classifier")  # local check; no network
cached = paw.list_cached_programs()
```

`prepare_program` resolves and downloads the program, runtime manifest, and shared base model (plus the projector for image programs) without retaining a loaded function. Pass `offline=True` to require an already complete local cache and prohibit network access.

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
fn = paw.compile_and_load(spec, compiler=None, remote=False, **kwargs)
```

Compiles a spec and returns a callable. Inference runs locally by default.
Pass `remote=True` for hosted inference without downloading model assets.

Accepts the parameters of `paw.compile`, plus `n_ctx`, `n_gpu_layers`, `verbose`,
and `remote`. Local runtime options must retain their defaults with `remote=True`.

```python
with paw.compile_and_load(
    "Classify sentiment as positive or negative", remote=True,
) as remote_fn:
    output = remote_fn("I love this!")
```

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
