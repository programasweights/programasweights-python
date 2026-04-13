# ProgramAsWeights (PAW)

PAW compiles natural language specifications into tiny neural functions that run locally. Each function takes a single text input and returns a single text output. Use it when you need fuzzy text processing — classification, extraction, format repair, search, triage — that regex can't handle but a full LLM is overkill for.

Website: https://programasweights.com
Full documentation: https://programasweights.readthedocs.io

## When to Use PAW

- **Fuzzy search** — typo-tolerant matching, semantic search, near-duplicate detection
- **Format repair** — fix broken JSON, normalize dates, repair malformed inputs
- **Classification** — sentiment, urgency, categories defined in your own words
- **Extraction** — emails, names, dates from messy unstructured text
- **Log triage** — extract errors from verbose output, filter noise
- **Intent routing** — map user descriptions to the closest URL, menu item, or setting
- **Agent preprocessing** — parse tool calls, validate outputs, route tasks

## Install

```bash
pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
```

## Quickstart

```python
import programasweights as paw

# Use a pre-compiled function (downloads once, runs locally forever)
fn = paw.function("email-triage")
fn("Urgent: server is down!")  # "immediate"
fn("Newsletter: spring picnic")  # "wait"

# Compile your own from a description
program = paw.compile(
    "Fix malformed JSON: repair missing quotes and trailing commas"
)
fn = paw.function(program.id)
fn("{name: 'Alice',}")  # '{"name":"Alice"}'

# Or compile and load in one step
fn = paw.compile_and_load("Classify sentiment as positive or negative")
fn("I love this!")  # "positive"
```

If you want the smaller browser-compatible runtime explicitly, pass `compiler="paw-4b-gpt2"`. Otherwise, omit `compiler` and let the server default decide.

## Current Public Compilers

- **Standard** (`paw-4b-qwen3-0.6b`) — higher accuracy, 594 MB base + ~22 MB/program. This is the current server default.
- **Compact** (`paw-4b-gpt2`) — smaller (134 MB base + ~5 MB/program), runs in browser via WebAssembly.

Best practice:

- For quickstarts and reusable agent workflows, prefer `paw.compile(spec)` with no explicit compiler.
- If you need to target a specific runtime, pass `compiler="paw-4b-gpt2"` or another supported alias explicitly.
- If you need to inspect current server-supported compiler names at runtime, call `paw.list_compilers()`.

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

- **State output constraints explicitly**: "Return ONLY one of: X, Y, Z". Without this the model may produce free-form text.
- **Include examples from your actual data**: Examples outperform prose-only descriptions.
- **Debug failures before sweeping**: Look at specific failing examples and understand WHY before trying many variants.

## Constraints And Runtime Behavior

- Each PAW function is stateless: one text input, one text output. No conversation history.
- Spec + input + output share a ~2048 token context window. Inputs that exceed it will error.
- `max_tokens` defaults to `None`: generation runs until EOS or the context limit.
- Compile runs on the hosted PAW API. Inference should usually run locally through the SDK.
- **GPU acceleration** is enabled by default (`n_gpu_layers=-1`). Uses Metal on Mac, CUDA on Linux, and falls back to CPU automatically. If GPU causes issues, set `PAW_GPU_LAYERS=0` or pass `n_gpu_layers=0`.
- **First call** is usually ~1-5s because it loads the base model. Subsequent calls are typically ~0.05-0.5s depending on input length and GPU availability.
- **Base model files are shared** across programs on disk. Each Standard LoRA adapter is ~22 MB; each Compact LoRA adapter is ~5 MB.
- Cache root is `~/.cache/programasweights/`. Override with `PAW_CACHE_DIR`.
- After the first download, inference works offline.

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `RuntimeError: assets not ready` on download | Program is still generating after compile | The SDK polls automatically for up to 30s. If it still fails, retry shortly or recompile. |
| `httpx.HTTPStatusError: 422` on compile | Spec too short (<10 chars) or request validation failed | Adjust spec length or request shape. |
| `httpx.HTTPStatusError: 429` | Hosted compile API limit exceeded | Wait, or sign in for higher compile limits. |
| GPU/Metal errors on load | GPU backend not available or incompatible | Set `PAW_GPU_LAYERS=0` or pass `n_gpu_layers=0` to force CPU. |

## Browser / JavaScript SDK

Programs compiled with `paw-4b-gpt2` run in the browser via WebAssembly.

```bash
npm install @programasweights/web
```

```javascript
import paw from '@programasweights/web';

const fn = await paw.function('email-triage-browser');
const result = await fn('Urgent: server is down!');
// result: "immediate"
```

The browser SDK resolves slugs through the PAW API, then downloads browser assets from Hugging Face and runs inference client-side. If you load by program ID, browser inference stays independent of the PAW API at runtime.

## Authentication (optional)

Sign in for higher rate limits and program naming. Everything works without it.

```bash
export PAW_API_KEY=paw_sk_...
```

Generate API keys at https://programasweights.com/settings.

| | Anonymous | Authenticated |
|---|---|---|
| Compile rate limit | 20/hr | 60/hr |
| Concurrent compile requests | 1 | 2 |
| Name programs (slugs) | No | Yes |

Hosted API limits apply to compile requests. Most inference should run locally through the SDK.

## CLI

Commands: `paw compile --spec "..." --json`, `paw run --program <id> --input "..."`, `paw info <id>`, `paw rename <id> <slug>`, `paw login`. All support `--json` for structured output.

## Versioning

Slugs support version history. Recompiling with the same slug creates a new version:

```python
p1 = paw.compile("Count words v1", slug="word-counter")  # v1
p2 = paw.compile("Count words v2", slug="word-counter")  # v2 (auto-bumps)

fn = paw.function("da03/word-counter")     # resolves to main (latest)
fn = paw.function("da03/word-counter@v1")  # pinned to v1

versions = paw.list_versions("da03/word-counter")  # all versions
```

Pinned versions (`@v1`) are immutable and cached locally forever. Bare slugs always check the server for the latest main version and fall back to cache if offline.

## Full API Reference

```python
program = paw.compile(
    spec,                               # natural language specification (str)
    compiler=None,                      # omit to use the current server default (today: paw-4b-qwen3-0.6b)
    slug=None,                          # URL-safe handle (requires auth)
    public=True,                        # list on public hub
)
# Returns: Program(id, slug, status, version, version_action, timings, error)

fn = paw.function(program)              # accepts Program object, hash ID, or slug
fn = paw.function("a6b454023d41ac9ca845")
fn = paw.function("da03/my-classifier")
fn = paw.function("da03/my-classifier@v2")  # pinned version
fn = paw.function("da03/my-classifier", offline=True)  # skip server check

result: str = fn(input_text: str, max_tokens=None, temperature=0.0)

fn = paw.compile_and_load(spec)

versions = paw.list_versions("da03/my-classifier")  # version history
programs = paw.list_programs(sort="recent", per_page=20)  # requires auth
compilers = paw.list_compilers()  # discover available compilers at runtime

paw.login()
```

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

## Worked Example: Log Monitoring

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

## Case Studies

Detailed walkthroughs of building production systems with PAW, including what we tried and what we learned: [log monitoring](https://programasweights.readthedocs.io/en/latest/case-studies/log-monitoring/), [site navigation](https://programasweights.readthedocs.io/en/latest/case-studies/site-navigation/), [semantic search](https://programasweights.readthedocs.io/en/latest/case-studies/semantic-search/), [tool calling](https://programasweights.readthedocs.io/en/latest/case-studies/tool-calling/).

