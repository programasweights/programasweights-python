# Writing Good Specs

A specification is a **self-contained** natural-language description of what the function should do. There is no separate “examples” API field: everything the compiler sees must live in that single spec string.

## What to include

Write clearly enough that someone unfamiliar with your project could implement the behavior from the text alone.

### Good practices

- **Output shape:** Say how results should look. Examples: “return a JSON list of strings”, “respond with only `positive` or `negative`”, “output valid CSV with a header row”.
- **Edge cases:** State behavior for empty input, missing data, or failure modes. Example: “if no emails are found, return an empty list”.
- **Ambiguity:** Explain how to handle unclear inputs (e.g., multiple interpretations, partial data).
- **Length:** Keep the spec under **8000 characters**; the API enforces this limit.

### Examples inside the spec

You may embed example pairs directly in the spec text, for example:

```text
Examples:
Input: hello
Output: greeting
```

Use a consistent pattern so the compiler can treat them as demonstrations of desired behavior.

## What to avoid

- **Vague goals:** Phrases like “do something useful” or “be smart about it” do not constrain behavior.
- **Excessive length:** Long essays dilute the task definition and hit limits unnecessarily.
- **Contradictions:** Conflicting instructions produce unreliable pseudo-programs and weaker adapters.

## How the compiler uses your spec

The compiler **generates a pseudo-program** from your spec. That artifact often **rephrases and expands** the spec with structured examples. You can inspect what was produced in the playground under **“View compiled program internals”** to verify alignment with your intent.
