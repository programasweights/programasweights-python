# ProgramAsWeights

**Compile natural language specs into tiny neural functions that run locally.**

Define what a function should do in plain English. PAW compiles it into a small neural program that runs on your machine — no API keys at runtime, no internet needed after setup, fully deterministic.

## Install

```bash
pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
```

## Quick Start

```python
import programasweights as paw

# Use a pre-compiled function (downloads once, runs locally forever)
fn = paw.function("email-triage")
fn("Urgent: the server is down!")        # "immediate"
fn("Newsletter: spring picnic")          # "wait"

# Compile your own from a description
program = paw.compile(
    "Fix malformed JSON: repair missing quotes and trailing commas",
    compiler="paw-4b-qwen3-0.6b",  # or "paw-4b-gpt2" for smaller/faster
    slug="json-fixer"              # optional: creates username/json-fixer handle
)
fn = paw.function(program.slug)    # or paw.function(program.id)
fn("{name: 'Alice',}")  # '{"name":"Alice"}'

# Or compile and load in one step
fn = paw.compile_and_load("Classify sentiment as positive or negative")
fn("I love this!")  # "positive"
```

## Two Compilers


|                 | Standard (Qwen3 0.6B) | Compact (GPT-2 124M) |
| --------------- | --------------------- | -------------------- |
| Compiler name   | `paw-4b-qwen3-0.6b`   | `paw-4b-gpt2`        |
| Accuracy        | Higher                | Lower                |
| Base model size | 594 MB                | 134 MB               |
| Program size    | ~22 MB                | ~5 MB                |
| Local inference | ~0.05-0.5s per call   | ~0.03-0.3s per call  |
| Runs in browser | No                    | Yes (WebAssembly)    |

Default is Standard (Qwen3 0.6B). Use Compact (GPT-2) when you need smaller files or browser deployment.

GPU acceleration is enabled by default (Metal on Mac, CUDA on Linux, falls back to CPU). Set `PAW_GPU_LAYERS=0` to force CPU if GPU causes issues.

## Browser SDK

Programs compiled with GPT-2 also run entirely in the browser via WebAssembly — no server needed, data never leaves the user's device.

```bash
npm install @programasweights/web
```

```javascript
import paw from '@programasweights/web';

const fn = await paw.function('programasweights/email-triage');
const result = await fn('Urgent: the server is down!');
// result: "immediate"
```

See the [browser SDK repo](https://github.com/programasweights/programasweights-js) for full documentation.

## Use with AI Agents

PAW works with Cursor, Claude, Codex, and other AI coding assistants. Paste this into your agent's chat:

> I want to use ProgramAsWeights (PAW) to create fuzzy text functions that run locally. Read the instructions at [https://programasweights.com/AGENTS.md](https://programasweights.com/AGENTS.md) and help me integrate it.

Or save `[AGENTS.md](https://programasweights.com/agents)` to your project root — agents read it automatically.

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

Generate API keys at [programasweights.com/settings](https://programasweights.com/settings). Authenticated users get higher rate limits.

## CLI

```bash
paw compile --spec "Extract error lines from logs" --json
paw run --program <program_id> --input "[ERROR] timeout" --json
paw login
```

`--json` gives structured output for programmatic use.

## Links

- **Website**: [programasweights.com](https://programasweights.com)
- **Documentation**: [programasweights.readthedocs.io](https://programasweights.readthedocs.io)
- **Python SDK**: [github.com/programasweights/programasweights-python](https://github.com/programasweights/programasweights-python)
- **Browser SDK**: [github.com/programasweights/programasweights-js](https://github.com/programasweights/programasweights-js)
- **Program Hub**: [programasweights.com/hub](https://programasweights.com/hub)

## License

MIT