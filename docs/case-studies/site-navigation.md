# Natural Language Site Navigation

Every website with more than a handful of pages forces users to click through menus and navigation layers to find what they need. A Cmd+K / "Quick Find" helper that understands natural language can route users instantly — but normally requires a full LLM backend. PAW lets you build this with a pipeline of tiny compiled functions.

**Try it live:** Press Cmd+K (or Ctrl+K) on [programasweights.com](https://programasweights.com) and type something like "how do I run in the browser" or "where do I get an API key."

## How we built it

### Attempt 1: Generate the URL directly

The simplest idea: compile one program with all your pages listed in the spec, ask it to output the matching URL.

```
Given a user query, return the URL of the most relevant page.
Pages: /playground, /hub, /browser, /docs, /settings, /about
```

**Result:** The model hallucinated plausible-sounding but wrong paths (`/programs`, `/SDK`, `/api`). Small models are unreliable at generating precise strings from memory.

**Lesson:** Don't ask the model to generate structured output like URLs. It will hallucinate.

### Attempt 2: Reframe as classification

Instead of generating URLs, output a short semantic label such as `playground`, `docs`, or `feedback`. The frontend maps those labels to actual destinations.

```
Classify the user's intent. Return ONLY a single label.
playground = Create or compile a new program
hub = Browse or search existing programs
browser = Run a program in the browser
...
none = None of the above
```

**Result:** Accuracy jumped from ~67% to ~88%. Classification is far more reliable than generation for routing.

**Lesson:** Reframe generation tasks as classification whenever possible. Output a short label, map it to the real value in your code.

### Attempt 3: Add Q&A support

Users don't only navigate — they also ask questions ("is PAW free?", "what languages are supported?"). We tried adding Q&A to the same program.

**Result:** Cramming routing + Q&A into one spec degraded both tasks. The model couldn't handle the complexity.

**Lesson:** One program, one job. When accuracy drops, split the task.

### Attempt 4: Multi-program pipeline

Split into specialized programs, each doing one thing well:

1. **Page classifier** — routes to a page or says "this is a question"
2. **Question-type classifier** — is it a yes/no question or a how/what question?
3. **Yes/no answerer** — answers factual yes/no questions (facts baked into the spec)
4. **How/what answerer** — answers descriptive questions
5. **Answer validator** — checks "does this answer actually address the question?"

The validator was the key final addition. Without it, the pipeline would sometimes produce answers like "yes" for "what is the license?" — grammatically fine but useless. The validator catches these.

**Result:** The pipeline handles navigation, FAQ, and edge cases reliably. Each program compiles in seconds and runs in milliseconds.

## The solution

```python
import programasweights as paw

router = paw.function("my-page-router")
q_type = paw.function("my-question-type")
yes_no = paw.function("my-yes-no-answerer")
howto  = paw.function("my-howto-answerer")
validator = paw.function("my-answer-validator")

def handle_query(user_query: str):
    destination = router(user_query)
    if destination != "none":
        return {"action": "navigate", "page": PAGE_MAP[destination]}

    category = q_type(user_query)
    if category == "yes_no":
        answer = yes_no(user_query)
    elif category == "how_what":
        answer = howto(user_query)
    else:
        return {"action": "fallback", "message": "Try browsing the docs."}

    if validator(f"Q: {user_query}\nA: {answer}") != "valid":
        return {"action": "fallback", "message": "I'm not sure. Try the docs."}

    return {"action": "answer", "text": answer}
```

Each program has a focused spec. For example, the page classifier:

```python
router = paw.compile("""
Classify the user's intent. Return ONLY a single label.
playground = Create or compile something new
hub = Browse or search existing items
browser = Run something in the browser
docs = Read documentation
settings = Manage account or API keys
none = None of the above (likely a question)

Input: how do I get started
Output: docs

Input: browse community programs
Output: hub

Input: is it free?
Output: none
""")
```

## Adapting this for your site

1. **List your pages** with short descriptions of what users do there
2. **Compile a classifier** that maps intents to short labels
3. **Test with 20-30 real queries** your users would type — iterate on the spec wording
4. If users also ask questions, add Q&A programs and a validator
5. Each program compiles once and is cached forever — the pipeline runs locally with no API calls

## Takeaways

- **Classification beats generation** for routing. Output a short label, map it in code.
- **Multiple small programs beat one complex program.** When accuracy drops, split.
- **A validator catches failures** the other programs miss — cheap insurance.
- **Iterate with real queries.** Build a small test set, measure, adjust wording, repeat.
