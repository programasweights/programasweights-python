# ProgramAsWeights (PAW)

PAW compiles natural language specifications into tiny neural functions that run locally. Each function takes a single text input and returns a single text output. Use it when you need fuzzy text processing — classification, extraction, format repair, search, triage — that regex can't handle but a full LLM is overkill for.

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
# "email-triage" is an official pre-compiled program (slug)
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

# Or compile and load in one step
fn = paw.compile_and_load("Classify sentiment as positive or negative")
fn("I love this!")  # "positive"
```

## Two Compilers

| | Standard (Qwen3 0.6B) | Compact (GPT-2 124M) |
|---|---|---|
| Compiler name | `paw-4b-qwen3-0.6b` | `paw-4b-gpt2` |
| Accuracy | Higher | Lower |
| Base model size | 594 MB | 105 MB |
| Program size | ~22 MB | ~5 MB |
| Inference speed | ~90ms (server) | ~50ms (server) |
| Runs in browser | No | Yes (must use this compiler) |

Default is Standard. Use Compact for smaller files or browser deployment.

## When to Use PAW

- **Fuzzy search** — typo-tolerant matching, semantic search, near-duplicate detection
- **Format repair** — fix broken JSON, normalize dates, repair malformed inputs
- **Classification** — sentiment, urgency, categories defined in your own words
- **Extraction** — emails, names, dates from messy unstructured text
- **Log triage** — extract errors from verbose output, filter noise
- **Intent routing** — map user descriptions to the closest URL, menu item, or setting
- **Agent preprocessing** — parse tool calls, validate outputs, route tasks

## Writing Good Specs

Description + examples. Use `Input: ... Output: ...` format.

```python
fn = paw.compile_and_load("""
Classify user intent. Return ONLY one of: search, create, delete, other.

Input: Find the latest report
Output: search

Input: Make a new folder
Output: create

Input: Remove old backups
Output: delete
""")
```

- State output constraints explicitly if any: "Return ONLY one of: X, Y, Z"
- Each function is stateless: one input, one output. No conversation history.
- Write a few test inputs with expected outputs, then try different spec phrasings and pick the one that passes the most.

## Chaining Functions

Multiple PAW functions can be composed for multi-step tasks:

```python
classifier = paw.compile_and_load("Classify the bug type. Return ONLY one of: off-by-one, type-error, other")
fixer = paw.compile_and_load("Fix the bug described in the first line. Return only the corrected code.")

label = classifier(code_snippet)
if label != "other":
    fix = fixer(f"{label}: {code_snippet}")
```

Each function is independent -- chain them with regular Python logic.

## Browser / JavaScript SDK

Programs compiled with `paw-4b-gpt2` run in the browser via WebAssembly.

```bash
npm install @programasweights/web
```

```javascript
import paw from '@programasweights/web';

const fn = await paw.function('programasweights/email-triage');
const result = await fn('Urgent: server is down!');
// result: "immediate"
```

## Authentication (optional)

Sign in for higher rate limits and program naming. Everything works without it.

```bash
export PAW_API_KEY=paw_sk_...
```

Generate API keys at https://programasweights.com/settings.

| | Anonymous | Authenticated |
|---|---|---|
| Compile rate limit | 20/hr | 60/hr |
| Name programs (slugs) | No | Yes |

## CLI

```bash
paw compile --spec "Classify sentiment" --json
paw run --program <program_id> --input "I love this!" --json
paw info <program_id>             # show program metadata
paw rename <program_id> my-slug   # name a program (requires auth)
paw login                         # save API key
```

`--json` gives structured output. Example:

```json
{"program_id": "a6b454023d41ac9ca845", "slug": null, "status": "ready", "error": null, "timings": {"total_ms": 2800}}
```

## Full API Reference

```python
program = paw.compile(
    spec,                               # natural language specification (str)
    compiler="paw-4b-qwen3-0.6b",
    slug=None,                          # URL-safe handle (requires auth)
    public=True,                        # list on public hub
)
# Returns: Program(id, slug, status, timings, error)

fn = paw.function(program)              # accepts Program object, hash ID, or slug
fn = paw.function("a6b454023d41ac9ca845")
fn = paw.function("da03/my-classifier")

result: str = fn(input_text: str, max_tokens=512, temperature=0.0)

fn = paw.compile_and_load(spec, compiler="paw-4b-qwen3-0.6b")

programs = paw.list_programs(sort="recent", per_page=20)  # requires auth

paw.login()
```

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: assets not ready` on download | Program still generating after compile | SDK polls automatically for up to 30s. If persistent, recompile. |
| `httpx.HTTPStatusError: 422` on compile | Spec too short (<10 chars) | Adjust spec length. |
| `httpx.HTTPStatusError: 429` | Rate limit exceeded | Wait, or sign in for higher limits. |

## Performance

- **First call** ~500ms (loads base model). Subsequent calls ~50-90ms.
- **Base model shared** across functions. Each LoRA adapter adds ~22 MB.
- **Thread-safe** and **blocking**.
- **Cache**: `~/.cache/programasweights/`. Override with `PAW_CACHE_DIR`.
- **Offline** after first download.

## Limits

- Spec + input + output share a ~2048 token context window, and inputs that exceed the context window will error out (not silently truncated).
- Setting `max_tokens` high is safe -- generation stops at EOS or when the window is full.

## Browse Programs

https://programasweights.com/hub
