# ProgramAsWeights (PAW)

PAW compiles natural language specifications into tiny neural functions that run locally. Use it when you need fuzzy text processing — classification, extraction, format repair, search, triage — that regex can't handle but a full LLM is overkill for.

Website: https://programasweights.com
Docs: https://programasweights.readthedocs.io

## Install

```bash
pip install --pre programasweights
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

## Authentication

```bash
# Option 1: environment variable (recommended)
export PAW_API_KEY=paw_sk_...

# Option 2: CLI login (opens browser to generate key)
paw login
```

Generate API keys at https://programasweights.com/settings

The SDK automatically reads `PAW_API_KEY` from the environment. Authenticated users get higher rate limits (60 compiles/hr vs 5 for anonymous).

## CLI

```bash
paw compile --spec "Classify sentiment as positive or negative" --json
paw run --program <program_id> --input "I love this!" --json
paw login  # Save API key for higher rate limits
```

`--json` gives structured output for programmatic use.

## API

```python
paw.compile(spec, compiler="paw-4b-qwen3-0.6b")  # Compile a spec
paw.function(name_or_id)                           # Load a compiled program
paw.login()                                        # Save API key
```

## Browse Programs

https://programasweights.com/hub

## Add PAW to Your Project

Copy this file into your project as AGENTS.md:
https://programasweights.com/agents
