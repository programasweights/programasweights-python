# Tool Calling with a 10-Function Pipeline

Large language models are increasingly expected to call tools — pick the right API, extract parameters, chain multi-step flows, and know when NOT to act. This normally requires large models (70B+). We used PAW to build a tool-calling system from 10 tiny compiled functions running on a 0.6B interpreter, and tested it on the [ToolCall-15](https://github.com/stevibe/ToolCall-15) benchmark.

The entire pipeline was built in a single Cursor session by an AI coding agent (Claude Opus 4.6) that read PAW's [AGENTS.md](https://programasweights.com/AGENTS.md) and the benchmark source.

**Result:** 93% (28/30 points) on ToolCall-15, with 14 out of 15 scenarios passing.

!!! warning "Benchmark leakage"
    The AI agent that built this pipeline had access to the benchmark source code, including the 15 scenario definitions and scoring logic. This means the specs and heuristics were likely tailored to these specific test cases. We report 93% as a proof of concept for the architecture, not a generalizable accuracy claim. A principled evaluation would test on held-out scenarios the builder has never seen.

## The benchmark: ToolCall-15

[ToolCall-15](https://github.com/stevibe/ToolCall-15) (by [stevibe](https://x.com/stevibe)) is a visual benchmark for LLM tool use. It runs 15 hand-picked scenarios through an OpenAI-compatible chat completions interface, scores each deterministically, and renders results in a live dashboard.

**12 available tools** (given to every scenario):

`web_search`, `get_weather`, `calculator`, `send_email`, `search_files`, `read_file`, `create_calendar_event`, `get_contacts`, `translate_text`, `get_stock_price`, `set_reminder`, `run_code`

**15 scenarios across 5 categories** (3 each, max 6 points per category):

| Category | Scenarios | What it tests |
|---|---|---|
| A: Tool Selection | TC-01 Weather in Berlin, TC-02 AAPL stock price, TC-03 "Let Sarah know..." | Pick the right tool, resist distractors, infer implicit chains |
| B: Parameter Precision | TC-04 Tokyo in Fahrenheit, TC-05 "Next Monday 9:30am", TC-06 Translate to both Spanish & Japanese | Pass correct arguments, parse dates, handle multi-value |
| C: Multi-Step Chains | TC-07 Find report → read → email total, TC-08 Check weather → if raining → set reminder, TC-09 Weather AND stock price | Thread data across steps, branch conditionally, run in parallel |
| D: Restraint & Refusal | TC-10 "When did WWII end?", TC-11 "15% of 200", TC-12 "Delete my emails" | Don't use tools for trivial knowledge, refuse impossible requests |
| E: Error Recovery | TC-13 Empty search results → retry, TC-14 Stock API rate limited, TC-15 Search population → calculate 2% | Handle errors gracefully, maintain data integrity |

Scoring: pass = 2 points, partial = 1, fail = 0. Final score = average of 5 category percentages.

## The approach: decompose into specialists

The key insight is the same one from our [site navigation](site-navigation.md) case study: **split the task into small, focused functions.** A single monolithic prompt trying to handle tool selection, parameter extraction, multi-step reasoning, and restraint all at once would fail. Instead, each PAW function handles one decision.

The 10 compiled functions divide into three groups:

**Routing (which tool?):**

| Function | Spec (abbreviated) | Output |
|---|---|---|
| `tc15-needs-tool` | "Does this request need an external tool, or can it be answered directly?" | YES / NO |
| `tc15-tool-router` | "Which tool should be called?" with 10 examples covering all 12 tools + NONE | One of 13 labels |
| `tc15-impossible-check` | "Is this request something no available tool can fulfill?" | IMPOSSIBLE / POSSIBLE |
| `tc15-second-tool` | "After the first tool, is a second tool needed?" | Tool name or NONE |

**Extraction (what parameters?):**

| Function | Spec (abbreviated) | Output |
|---|---|---|
| `tc15-extract-location` | "Extract the city name from a weather request" | City name |
| `tc15-extract-ticker` | "Extract the stock ticker symbol" | Ticker (e.g. AAPL) |
| `tc15-extract-units` | "Extract temperature units, default celsius" | fahrenheit / celsius |
| `tc15-extract-search-query` | "Extract the file search query" | Search string |
| `tc15-extract-person` | "Extract the person's name" | Name |
| `tc15-extract-translate` | "Extract source_lang, target_lang, text" | Pipe-separated triple |

**Not compiled (handled by code):**

Some tasks are better handled by regular code than by a neural function:

- **Date/time parsing** (regex): "next Monday at 9:30am" → `2026-03-23T09:30`
- **Direct answers** (lookup table): "15% of 200" → `30`
- **Response synthesis** (template): format tool results into natural language
- **Protocol formatting**: build OpenAI-compatible `tool_calls` JSON

This is a general design principle: **use PAW for fuzzy judgment, use code for structured logic.**

## The full specs

Here are the actual specs the agent compiled. Each is a standalone PAW function.

### tc15-tool-router

```
Classify which tool should be called for a user request.
Output EXACTLY one of these tool names:
web_search, get_weather, calculator, send_email, search_files,
read_file, create_calendar_event, get_contacts, translate_text,
get_stock_price, set_reminder, run_code, NONE.
Output NONE if the user's question can be answered from general
knowledge without any tool.

Input: What's the weather in Berlin?
Output: get_weather

Input: What is the price of AAPL stock?
Output: get_stock_price

Input: What year did WWII end?
Output: NONE

Input: What is 15% of 200?
Output: NONE

Input: Delete all my emails
Output: NONE

Input: Translate hello to Spanish
Output: translate_text

Input: Find the Q3 budget report
Output: search_files

Input: Send an email to Sarah
Output: get_contacts

Input: Schedule a meeting for Monday
Output: create_calendar_event

Input: Remind me to bring an umbrella
Output: set_reminder
```

### tc15-needs-tool

```
Determine if a user request needs an external tool or can be
answered directly. Output YES if the request needs real-time data,
file access, sending messages, scheduling, or looking up current
information. Output NO if it's basic knowledge, simple math, or
an impossible request with no available tool.

Input: What's the weather?
Output: YES

Input: What year did WWII end?
Output: NO

Input: What is 15% of 200?
Output: NO

Input: Delete all my emails
Output: NO

Input: What's AAPL stock price?
Output: YES

Input: Find the budget report
Output: YES
```

### tc15-extract-location

```
Extract the location/city name from a weather or location request.
Output just the city or location name, nothing else.

Input: What's the weather in Berlin?
Output: Berlin

Input: Temperature in Tokyo in Fahrenheit
Output: Tokyo

Input: How's the weather in Paris right now?
Output: Paris

Input: London weather forecast
Output: London
```

### tc15-extract-ticker

```
Extract the stock ticker symbol from a stock price request.
Output just the uppercase ticker symbol.

Input: What is the price of AAPL stock?
Output: AAPL

Input: MSFT stock price
Output: MSFT

Input: How much is Apple stock?
Output: AAPL

Input: Microsoft stock price?
Output: MSFT
```

### tc15-extract-units

```
Extract temperature units from a weather request.
Output 'fahrenheit' if Fahrenheit is mentioned, otherwise
output 'celsius'.

Input: Temperature in Tokyo in Fahrenheit
Output: fahrenheit

Input: Weather in Berlin
Output: celsius

Input: Paris weather in F
Output: fahrenheit

Input: London temperature celsius
Output: celsius
```

### tc15-extract-search-query

```
Extract the file search query from a user request about finding
documents or files. Output a concise search query string.

Input: Find the Q3 budget report
Output: Q3 budget report

Input: Find the Johnson proposal document
Output: Johnson proposal

Input: Search for the annual review
Output: annual review
```

### tc15-extract-person

```
Extract the person's name from a request about contacting or
looking up someone. Output just the person's name.

Input: I need to let Sarah know about the meeting
Output: Sarah

Input: Send an email to John
Output: John

Input: Contact my manager
Output: manager

Input: Email the report to Jamie
Output: Jamie
```

### tc15-extract-translate

```
Extract translation parameters from a translate request.
Output in the format: source_lang|target_lang|text.
If multiple target languages, output one line per target language.

Input: Translate hello to Spanish
Output: English|Spanish|hello

Input: Translate Where is the hospital from English to Japanese
Output: English|Japanese|Where is the hospital
```

### tc15-impossible-check

```
Determine if a request asks for something that cannot be done
with standard tools (web search, weather, calculator, email,
file search/read, calendar, contacts, translate, stocks,
reminders, code execution).
Output IMPOSSIBLE if no tool can fulfill it (like deleting
emails, modifying files, accessing databases, etc).
Output POSSIBLE otherwise.

Input: Delete all my emails from last month
Output: IMPOSSIBLE

Input: What's the weather?
Output: POSSIBLE

Input: Hack into the server
Output: IMPOSSIBLE

Input: Send an email
Output: POSSIBLE
```

### tc15-second-tool

```
Given a user request that may need multiple tools, identify if
a SECOND tool is needed after the first. Output the second tool
name or NONE.

Input: Weather in London and MSFT stock price
Output: get_stock_price

Input: Find the report and email it to my manager
Output: read_file

Input: Check weather, if raining remind me about umbrella
Output: set_reminder

Input: What's the weather in Berlin?
Output: NONE

Input: Let Sarah know the meeting moved to 3pm
Output: send_email
```

## Pipeline architecture

The 10 PAW functions are orchestrated by a FastAPI proxy server that speaks the OpenAI `/v1/chat/completions` protocol. Here's the flow:

```
User message arrives at proxy
│
├─ detect_parallel_tools (regex)
│  "weather AND stock price" → call both tools in parallel
│  └─ if match: extract params for each → return tool_calls → done
│
├─ tc15-needs-tool → "Does this need a tool?"
│  └─ NO → tc15-impossible-check → explain refusal or answer directly
│
├─ tc15-tool-router → "Which tool?"
│  └─ NONE → answer from knowledge (lookup table or simple math)
│
├─ extract_params_for_tool(tool_name):
│  ├─ get_weather    → tc15-extract-location + tc15-extract-units
│  ├─ get_stock_price → tc15-extract-ticker
│  ├─ search_files   → tc15-extract-search-query
│  ├─ get_contacts   → tc15-extract-person
│  ├─ translate_text  → tc15-extract-translate (may produce 2 calls)
│  ├─ calendar_event  → regex date/time parsing
│  └─ set_reminder   → regex date/time parsing
│
└─ Return tool_calls in OpenAI JSON format

──── On follow-up turns (tool results come back) ────

├─ decide_follow_up_tool:
│  ├─ Error result? → explain to user
│  ├─ Empty results? → retry with broader query
│  ├─ tc15-second-tool → "Is another tool needed?"
│  │  └─ Chain logic: search → read → contacts → send_email
│  └─ No more tools → synthesize_response from all results
│
└─ Return either more tool_calls or final text response
```

The proxy handles multi-turn conversations by tracking which tools have already been called and threading data between steps (e.g., using a `file_id` from a search result in a subsequent `read_file` call).

## How it was built

A single prompt to Cursor (Claude Opus 4.6):

> "Build a PAW-powered proxy for the ToolCall-15 benchmark. Read AGENTS.md for PAW usage. Analyze the benchmark scenarios and create specialized PAW functions for tool routing and parameter extraction. Wire them into an OpenAI-compatible endpoint."

The agent:

1. Read the benchmark methodology and all 15 scenario definitions
2. Identified which decisions need fuzzy judgment (PAW) vs. structured logic (code)
3. Designed and compiled 10 PAW functions with specs tailored to the scenarios
4. Built a FastAPI proxy server (~700 lines) with conversation state management
5. Ran the benchmark, debugged failures, and iterated

### Iteration 1: 70% (21/30)

The first run had several failures:

- **TC-01, TC-02 (Tool Selection): Infinite loop.** After calling the correct tool and receiving results, the proxy called the same tool again instead of synthesizing a response. Root cause: the conversation state handler didn't properly detect "I already have results, stop calling tools."
- **TC-08 (Conditional Branching): Wrong first tool.** The router sent "check weather, if raining set reminder" to `create_calendar_event` instead of `get_weather`. The tool-router spec didn't have an example for conditional chains.
- **TC-15 (Data Integrity): Wrong second tool.** After `web_search` returned Iceland's population, the proxy called `get_stock_price("Iceland")` instead of `calculator`. The `tc15-second-tool` function didn't have enough examples for search-then-calculate chains.
- **TC-07 (Multi-Step Chain): Partial.** The 4-step chain (search → read → contacts → email) partially worked but didn't thread data correctly between all steps.

### Iteration 2: fix the orchestration

The agent fixed the proxy code (not the PAW specs):

- Added proper conversation state tracking to detect when tool results are present
- Added routing overrides for common patterns (conditional weather checks, search-then-calculate)
- Fixed data threading in multi-step chains (extracting `file_id` from search results, email from contacts)
- Recompiled 2 functions that had HuggingFace download failures

### Final result: 93% (28/30)

| Category | Score | Details |
|---|---|---|
| A: Tool Selection | 100% (6/6) | TC-01 pass, TC-02 pass, TC-03 pass |
| B: Parameter Precision | 100% (6/6) | TC-04 pass, TC-05 pass, TC-06 pass |
| C: Multi-Step Chains | 100% (6/6) | TC-07 pass, TC-08 pass, TC-09 pass |
| D: Restraint & Refusal | 100% (6/6) | TC-10 pass, TC-11 pass, TC-12 pass |
| E: Error Recovery | 67% (4/6) | TC-13 **fail**, TC-14 pass, TC-15 pass |

**14 out of 15 scenarios passed.** The single failure:

### Why TC-13 failed

**TC-13 (Empty Results Retry):** The user asks "Find the Johnson proposal document." The proxy correctly calls `search_files(query="Johnson proposal")`, gets empty results, but instead of retrying with a broader query like `"Johnson"`, it immediately responds "No results were found."

**Root cause:** The proxy has loop-prevention logic that blocks re-calling a tool already in `tools_called`. This prevents infinite loops (which was a real problem in iteration 1) but also prevents the legitimate retry pattern where you call the same tool with different parameters. The agent's fix for the infinite loop was too aggressive — it stopped all same-tool retries.

This is an **architectural limitation**, not a PAW function limitation. The `tc15-extract-search-query` function works fine. The issue is in the orchestration code's retry policy.

## Contamination disclosure

The AI agent that built this pipeline had full access to:

- The 15 scenario definitions (user messages, expected tools, scoring logic)
- The mocked tool responses
- The benchmark source code

This means:

1. The specs were tuned for these 15 specific scenarios (e.g., the tool-router examples map closely to the actual test inputs)
2. The routing overrides in the proxy code target specific scenario patterns
3. The direct-answer lookup table contains exact answers for TC-10 and TC-11

**What this proves:** The architecture works — 10 small PAW specialists + Python glue can implement a tool-calling system that scores 93% on a multi-category benchmark.

**What this does NOT prove:** That these exact specs would generalize to unseen tool-calling scenarios. A principled evaluation would compile specs without seeing the test cases, then evaluate on held-out scenarios.

## Full code

### compile_functions.py

The compilation script that creates all 10 functions:

```python
import json, sys
import programasweights as paw

COMPILER = "paw-4b-qwen3-0.6b"

FUNCTIONS_TO_COMPILE = [
    {"slug": "tc15-tool-router",         "spec": "Classify which tool..."},
    {"slug": "tc15-needs-tool",          "spec": "Determine if request needs tool..."},
    {"slug": "tc15-extract-location",    "spec": "Extract city name..."},
    {"slug": "tc15-extract-ticker",      "spec": "Extract stock ticker..."},
    {"slug": "tc15-extract-units",       "spec": "Extract temperature units..."},
    {"slug": "tc15-extract-search-query","spec": "Extract file search query..."},
    {"slug": "tc15-extract-person",      "spec": "Extract person's name..."},
    {"slug": "tc15-extract-translate",   "spec": "Extract translation parameters..."},
    {"slug": "tc15-impossible-check",    "spec": "Is this request impossible?..."},
    {"slug": "tc15-second-tool",         "spec": "Is a second tool needed?..."},
]

for fn_def in FUNCTIONS_TO_COMPILE:
    program = paw.compile(fn_def["spec"], compiler=COMPILER)
    print(f"{fn_def['slug']}: {program.id}")
```

### server.py (key routing logic)

The proxy server's main decision flow:

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]
    user_msg = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")

    # If we already have tool results, decide: more tools or synthesize?
    if any(m.get("role") == "tool" for m in messages):
        follow_up = decide_follow_up_tool(user_msg, messages)
        if follow_up:
            return make_response(tool_calls=follow_up)
        return make_response(content=synthesize_response(user_msg, messages))

    # First turn: should we use tools at all?
    tool_calls = build_tool_calls_for_user(user_msg, messages)
    if tool_calls:
        return make_response(tool_calls=tool_calls)

    # No tools needed — answer directly
    return make_response(content=generate_direct_answer(user_msg))
```

The full source code is available in the [paw-proxy directory](https://github.com/stevibe/ToolCall-15/tree/main/paw-proxy) of the ToolCall-15 repository.

## Takeaways

- **Decomposition works.** 10 small specialists beat one monolithic prompt. Each function handles one decision — "which tool?", "what city?", "is this impossible?" — and does it well.
- **PAW + code hybrid.** Fuzzy judgment (tool routing, entity extraction, intent classification) goes to PAW. Structured logic (date parsing, JSON formatting, conversation state) stays in Python. Neither alone would work.
- **An AI agent can build the whole thing.** From analyzing the benchmark to compiling functions to writing the proxy server — one Cursor session, one prompt.
- **Loop prevention vs. retry is hard.** The single failure (TC-13) came from the tension between preventing infinite tool loops and allowing legitimate retries with different parameters. This is an orchestration problem, not a model problem.
- **0.6B is enough for focused tasks.** Tool routing across 12 options, entity extraction, yes/no classification — none of these need a large model when the spec is precise and examples are clear.
- **Always evaluate on unseen data.** Our 93% includes benchmark leakage. The architecture is sound, but the specific accuracy is optimistic.
