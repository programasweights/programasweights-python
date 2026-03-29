# CLI Reference

The `paw` command-line interface mirrors the Python SDK for compile, run, and account workflows. Install the package from PyPI; the entry point is `paw`.

## Global options

| Option | Description |
|--------|-------------|
| `--api-url` | Override the API base URL (default: production). |
| `--api-key` | API key for authenticated requests. |

## Structured output

Every subcommand supports `--json` for machine-readable output. Use this in scripts and for agents that must parse results reliably.

## `paw compile`

Compile a natural-language specification on the server.

```bash
paw compile --spec "..." [--compiler ...] [--json]
```

| Option | Description |
|--------|-------------|
| `--spec` | Specification text to compile. |
| `--compiler` | Compiler model identifier (optional; server default may apply). |
| `--json` | Emit JSON instead of human-oriented text. |

## `paw run`

Run inference locally against a loaded program.

```bash
paw run --program <name_or_id> --input "..." [--max-tokens N] [--temperature T] [--json]
```

| Option | Description |
|--------|-------------|
| `--program` | Program hash ID, namespaced slug, or official shorthand. |
| `--input` | Input text for the program. |
| `--max-tokens` | Maximum tokens to generate. |
| `--temperature` | Sampling temperature. |
| `--json` | Structured output. |

## `paw info`

Show metadata for a program.

```bash
paw info <name_or_id> [--json]
```

## `paw login`

Authenticate and store credentials for the CLI (same account flow as the SDK).

```bash
paw login [email]
```

## Agent workflow: compile then run with `jq`

Pipe compile output into `run` by extracting `program_id` (field names follow the JSON schema emitted with `--json`):

```bash
PROGRAM_ID=$(paw compile --spec "Classify urgency: immediate vs can-wait" --json | jq -r '.program_id')
paw run --program "$PROGRAM_ID" --input "Please review by EOD" --json
```

Adjust `jq` selectors if your toolchain wraps the payload; always validate against the actual `--json` shape for your installed version.

## Related

- [Python SDK Reference](python-sdk.md)
- [REST API Reference](rest-api.md)
