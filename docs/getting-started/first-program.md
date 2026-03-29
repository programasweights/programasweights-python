# Your First Program

This guide walks through loading an official program, compiling your own, testing it, and (optionally) naming it on the hub.

## Step 1: Try a pre-built program

Official programs are referenced by short name. Load one and call it like any Python function:

```python
import programasweights as paw

fn = paw.function("email-triage")

result = fn("Thesis defense committee needs your signature by EOD")
print(result)

result = fn("Department newsletter: spring picnic next Friday")
print(result)
```

The first call may download the program and runtime assets; later calls use the local cache.

## Step 2: Compile your own program

Describe the behavior you want in natural language, compile it, then load the result by `program_id`:

```python
import programasweights as paw

program = paw.compile(
    "Fix malformed JSON: repair missing quotes and trailing commas"
)

fn = paw.function(program.id)

output = fn("{name: 'Alice', age: 30,}")
print(output)
```

`program.id` is the content-addressable identifier for the compiled artifact you can reuse in code or scripts.

## Step 3: Test with different inputs

Exercise the same function on several inputs to confirm behavior:

```python
import programasweights as paw

program = paw.compile(
    "Fix malformed JSON: repair missing quotes and trailing commas"
)
fn = paw.function(program.id)

samples = [
    '{"ok": true}',
    "{broken: true}",
    '{"nested": {inner: 1}}',
]

for text in samples:
    print("IN :", text)
    print("OUT:", fn(text))
    print()
```

Adjust the spec or inputs until the outputs match what you need for your pipeline.

## Step 4: Name it (signed in)

Human-readable names are managed on the hub. To assign an alias:

1. Open [programasweights.com](https://programasweights.com).
2. Sign in with GitHub.
3. Compile your specification through the site (or use a program you already compiled).
4. Name the program so you can load it later with `paw.function("your-alias")` instead of only by hash.

Naming requires authentication; anonymous compiles remain addressable by `program_id` only.
