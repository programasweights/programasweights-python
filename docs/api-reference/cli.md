# CLI Reference

The `paw` command-line interface mirrors the Python SDK. Install from PyPI; the entry point is `paw`.

```bash
pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
```

## Global options

| Option | Description |
|--------|-------------|
| `--api-url` | Override the API base URL (default: `https://programasweights.com`). |
| `--api-key` | API key for authenticated requests. Also reads `PAW_API_KEY` env var. |
| `--json` | Machine-readable JSON output (supported by all subcommands). |

## `paw compile`

Compile a natural-language specification on the server.

```bash
paw compile --spec "Classify message urgency" [--compiler paw-4b-qwen3-0.6b] [--slug my-classifier] [--json]
```

| Option | Description |
|--------|-------------|
| `--spec` | Specification text to compile (required). |
| `--compiler` | Compiler model (default: `paw-4b-qwen3-0.6b`). |
| `--slug` | URL-safe handle (e.g. `my-classifier`). Creates a `username/slug` alias. Requires auth. |
| `--json` | JSON output with `program_id`, `slug`, `status`, `timings`. |

## `paw run`

Run inference locally against a compiled program (the normal mode), or
explicitly against a bare base interpreter (advanced mode).

```bash
paw run --program <id_or_slug> --input "your text" [--offline] [--json]

# Advanced adapter-free mode
paw run --base --interpreter gpt2 --input "raw prompt" [--offline] [--json]
```

| Option | Description |
|--------|-------------|
| `--program` | Program hash ID, slug (e.g. `da03/my-classifier`), or official name (e.g. `email-triage`). |
| `--base` | Select adapter-free base mode. Mutually exclusive with `--program` and requires `--interpreter`. |
| `--interpreter` | Base interpreter: `Qwen/Qwen3-0.6B` or `gpt2`. Only valid with `--base`. |
| `--input` | Input text for the program. |
| `--max-tokens` | Maximum tokens to generate (default: 512). |
| `--temperature` | Sampling temperature (default: 0.0). |
| `--verbose` | Print llama.cpp debug output. |
| `--offline` | Require all selected assets to already be cached and make zero network calls. |
| `--json` | JSON output with `mode`, `program`, `interpreter`, `input`, and `output`. |

Exactly one of `--program` and `--base` is required. An empty or
whitespace-only `--program` is rejected. Base mode makes no PAW API, slug,
program, adapter, or prefix-cache calls; online it may download only the
selected base GGUF. It resets state and tokenizes the complete versioned
runtime prompt on every invocation. See the
[Python SDK reference](python-sdk.md#advanced-adapter-free-base-interpreter)
for the exact prompt bytes and error semantics.

## `paw rename`

Set or change a program's slug handle. Requires authentication.

```bash
paw rename <program_id> <new-slug> [--json]
```

| Argument | Description |
|----------|-------------|
| `program_id` | Program hash ID or current slug. |
| `new-slug` | New slug name (2-50 chars, lowercase alphanumeric and hyphens). Pass `""` to remove. |

## `paw info`

Show metadata for a program.

```bash
paw info <program_id_or_slug> [--json]
```

## `paw login`

Save an API key for authenticated requests.

```bash
paw login [key]
```

If `key` is provided, saves it directly. If omitted, opens the browser to generate one at `programasweights.com/settings`.

## Agent workflow example

```bash
PROGRAM_ID=$(paw compile --spec "Classify urgency" --json | jq -r '.program_id')
paw run --program "$PROGRAM_ID" --input "Please review by EOD" --json
```

## Related

- [Python SDK Reference](python-sdk.md)
- [REST API Reference](rest-api.md)
