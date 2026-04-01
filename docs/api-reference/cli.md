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

Run inference locally against a compiled program.

```bash
paw run --program <id_or_slug> --input "your text" [--max-tokens 512] [--temperature 0.0] [--json]
```

| Option | Description |
|--------|-------------|
| `--program` | Program hash ID, slug (e.g. `da03/my-classifier`), or official name (e.g. `email-triage`). |
| `--input` | Input text for the program. |
| `--max-tokens` | Maximum tokens to generate (default: 512). |
| `--temperature` | Sampling temperature (default: 0.0). |
| `--verbose` | Print llama.cpp debug output. |
| `--json` | JSON output with `program`, `input`, `output`. |

## `paw rename`

Set or change a program's slug handle. Requires authentication.

```bash
paw rename <program_id> <new-slug> [--json]
```

| Argument | Description |
|----------|-------------|
| `program_id` | Program hash ID or current slug. |
| `new-slug` | New slug name (2-50 chars, lowercase alphanumeric and hyphens). |

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
