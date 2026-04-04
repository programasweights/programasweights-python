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

Instead of generating URLs, output a digit: 1 = Playground, 2 = Hub, 3 = Browser, etc. The frontend maps digits to actual routes.

```
Classify the user's intent. Return ONLY a single digit.
1 = Create or compile a new program
2 = Browse or search existing programs
3 = Run a program in the browser
...
0 = None of the above
```

**Result:** Accuracy jumped from ~67% to ~88%. Classification is far more reliable than generation for routing.

**Lesson:** Reframe generation tasks as classification whenever possible. Output a label or digit, map it to the real value in your code.

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
    if destination != "0":
        return {"action": "navigate", "page": PAGES[int(destination)]}

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
Classify the user's intent. Return ONLY a single digit.
1 = Create or compile something new
2 = Browse or search existing items
3 = Run something in the browser
4 = Read documentation
5 = Manage account or API keys
0 = None of the above (likely a question)

Input: how do I get started
Output: 4

Input: browse community programs
Output: 2

Input: is it free?
Output: 0
""")
```

## Adapting this for your site

1. **List your pages** with short descriptions of what users do there
2. **Compile a classifier** that maps intents to page numbers
3. **Test with 20-30 real queries** your users would type — iterate on the spec wording
4. If users also ask questions, add Q&A programs and a validator
5. Each program compiles once and is cached forever — the pipeline runs locally with no API calls

## Takeaways

- **Classification beats generation** for routing. Output a label, map it in code.
- **Multiple small programs beat one complex program.** When accuracy drops, split.
- **A validator catches failures** the other programs miss — cheap insurance.
- **Iterate with real queries.** Build a small test set, measure, adjust wording, repeat.
