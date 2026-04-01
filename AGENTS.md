# ProgramAsWeights (PAW)

PAW compiles natural language specifications into tiny neural functions that run locally. Use it when you need fuzzy text processing — classification, extraction, format repair, search, triage — that regex can't handle but a full LLM is overkill for.

Website: https://programasweights.com
Docs: https://programasweights.readthedocs.io

## Install

```bash
pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
```

## Usage

```python
import programasweights as paw

# Use a pre-compiled function (downloads once, runs locally forever)
fn = paw.function("email-triage")
fn("Urgent: server is down!")  # "immediate"
fn("Newsletter: spring picnic")  # "wait"

# Compile your own from a description
program = paw.compile(
    "Fix malformed JSON: repair missing quotes and trailing commas",
    compiler="paw-4b-qwen3-0.6b"  # or "paw-4b-gpt2" for smaller/faster
)
fn = paw.function(program.id)
fn("{name: 'Alice',}")  # '{"name": "Alice"}'
```

## Two Compilers

| | Standard (Qwen3 0.6B) | Compact (GPT-2 124M) |
|---|---|---|
| Compiler name | `paw-4b-qwen3-0.6b` | `paw-4b-gpt2` |
| Accuracy | Higher | Lower |
| Base model size | 594 MB | 105 MB |
| Program size | ~22 MB | ~5 MB |
| Inference speed | ~90ms (server) | ~50ms (server) |
| Runs in browser | No | Yes |

Default is Standard (Qwen3 0.6B). Use Compact (GPT-2) when you need smaller files or browser deployment.

## Browser / JavaScript SDK

Programs compiled with GPT-2 also run entirely in the browser via WebAssembly — no server needed. Users' data never leaves their device.

```bash
npm install @programasweights/web
```

```javascript
import paw from '@programasweights/web';

const fn = await paw.function('programasweights/email-triage');
const result = await fn('Urgent: server is down!');
// result: "immediate"
```

Use this when you want to offload all computation to the user's browser: zero server costs, zero API keys, full data privacy. The 105 MB base model downloads once and is cached.

## When to Use PAW

- **Fuzzy search** — typo-tolerant matching, semantic search, near-duplicate detection
- **Format repair** — fix broken JSON, normalize dates, repair malformed inputs
- **Classification** — sentiment, urgency, categories defined in your own words
- **Extraction** — emails, names, dates from messy unstructured text
- **Log triage** — extract errors from verbose output, filter noise
- **Intent routing** — map user descriptions to the closest URL, menu item, or setting
- **Agent preprocessing** — parse tool calls, validate outputs, route tasks

## Authentication (optional)

Everything works without authentication. Sign in for higher rate limits and program naming.

```bash
export PAW_API_KEY=paw_sk_...
```

Generate API keys at https://programasweights.com/settings. The SDK reads `PAW_API_KEY` from the environment automatically.

| | Anonymous | Authenticated |
|---|---|---|
| Compile rate limit | 20/hr | 60/hr |
| Name programs (slugs) | No | Yes |

## Program Naming

Every compiled program gets an immutable hash ID (e.g. `a6b454023d41ac9ca845`). Authenticated users can also assign a human-readable slug:

```python
program = paw.compile("Classify sentiment", slug="my-classifier")
# program.id   -> "a6b454023d41ac9ca845"
# program.slug -> "da03/my-classifier"

fn = paw.function("da03/my-classifier")  # by slug (requires auth to create)
fn = paw.function("a6b454023d41ac9ca845") # by hash (always works)
```

## CLI

```bash
paw compile --spec "Classify sentiment" --json
paw run --program <program_id> --input "I love this!" --json
paw info <program_id>             # show program metadata
paw rename <program_id> my-slug   # name a program (requires auth)
paw login                         # save API key
```

`--json` gives structured output for programmatic use.

## API

```python
paw.compile(spec)                 # compile (returns Program with .id)
paw.function(program_id_or_slug)  # load for local inference
paw.login()                       # save API key
```

## Browse Programs

https://programasweights.com/hub

## Add PAW to Your Project

Copy this file into your project as AGENTS.md:
https://programasweights.com/agents
