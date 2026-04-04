# ProgramAsWeights (PAW)

PAW compiles natural language specifications into tiny neural functions that run locally. Each function takes a single text input and returns a single text output. Use it when you need fuzzy text processing — classification, extraction, format repair, search, triage — that regex can't handle but a full LLM is overkill for.

Website: https://programasweights.com
Full documentation: https://programasweights.readthedocs.io

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
fn("Newsletter: spring picnic")  # "can wait"

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

- **Standard** (`paw-4b-qwen3-0.6b`) — higher accuracy, 594 MB base + ~22 MB/program. Default.
- **Compact** (`paw-4b-gpt2`) — smaller (105 MB base + ~5 MB/program), runs in browser via WebAssembly.

## When to Use PAW

- **Fuzzy search** — typo-tolerant matching, semantic search, near-duplicate detection
- **Format repair** — fix broken JSON, normalize dates, repair malformed inputs
- **Classification** — sentiment, urgency, categories defined in your own words
- **Extraction** — emails, names, dates from messy unstructured text
- **Log triage** — extract errors from verbose output, filter noise
- **Intent routing** — map user descriptions to the closest URL, menu item, or setting
- **Agent preprocessing** — parse tool calls, validate outputs, route tasks

## Writing Good Specs

**The #1 practice: iterate with test cases.** Do not accept low performance on the first try. Build a test suite of input/output pairs, measure accuracy, then iteratively adjust wording and formatting until performance is good enough. Treat spec writing like software engineering: test, debug specific failures, fix the wording, retest.

A good spec has a description plus `Input: ... Output: ...` examples.

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

**Spec-tuning tips:**

- Each function is stateless: one text input, one text output. No conversation history.
- **State output constraints explicitly**: "Return ONLY one of: X, Y, Z". Without this the model may produce free-form text.
- **Include examples from your actual data**: Examples outperform prose-only descriptions.
- **Debug failures before sweeping**: Look at specific failing examples and understand WHY before trying many variants.

## Chaining Functions

Multiple PAW functions can be composed for multi-step tasks:

```python
classifier = paw.compile_and_load("Classify the bug type. Return ONLY one of: off-by-one, type-error, other")
fixer = paw.compile_and_load("Fix the bug described in the first line. Return only the corrected code.")

label = classifier(code_snippet)
if label != "other":
    fix = fixer(f"{label}: {code_snippet}")
```

Chain them with regular Python logic.

## Case Studies

Production examples with iterative spec-tuning walkthroughs: [site navigation](https://programasweights.readthedocs.io/case-studies/site-navigation/) (5-program pipeline, Cmd+K helper) and [semantic search](https://programasweights.readthedocs.io/case-studies/semantic-search/) (reranking without a vector database).

## Event-Driven Monitoring

PAW functions can classify log output. Compile once with examples from your specific logs, then reuse the function locally forever:

```python
program = paw.compile("""
Classify log lines. Return ONLY one word: ALERT or QUIET.

Input: [step 100] loss=0.05 lr=0.0001
Output: QUIET

Input: [Checkpoint] Saved model at step 1000
Output: ALERT

Input: Traceback (most recent call last):
Output: ALERT

Input: Training complete. Final loss: 0.11
Output: ALERT
""")

fn = paw.function(program.id)  # reuse with saved program.id
fn("[step 200] loss=0.04")           # "QUIET"
fn("[Checkpoint] Saved model")       # "ALERT"
```

Full tool with file watching, truncation, and stall detection: [examples/paw_monitor.py](https://github.com/programasweights/programasweights-python/blob/main/examples/paw_monitor.py)

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

Commands: `paw compile --spec "..." --json`, `paw run --program <id> --input "..."`, `paw info <id>`, `paw rename <id> <slug>`, `paw login`. All support `--json` for structured output.

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

result: str = fn(input_text: str, max_tokens=None, temperature=0.0)

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

- **First call** ~5-15s (loads base model). Subsequent calls 0.5-5s depending on input length.
- **Base model shared** across functions on disk. Each LoRA adapter adds ~22 MB.
- **Cache**: `~/.cache/programasweights/`. Override with `PAW_CACHE_DIR`.
- **Offline** after first download.

## Limits

- Spec + input + output share a ~2048 token context window. Inputs that exceed it will error.
- `max_tokens` defaults to `None`: generation runs until EOS or the context limit.

