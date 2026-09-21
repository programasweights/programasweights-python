# Your First Program

Compile your own function, test it, save its ID for reuse, and optionally name it on the hub.

## Step 1: Compile your own function

Describe what the function should do, then compile and load it:

```python
import programasweights as paw

program = paw.compile(
    "Classify if a message needs immediate attention or can wait. "
    "Return only 'immediate' or 'wait'."
)
fn = paw.function(program.id)

result = fn("Thesis defense committee needs your signature by EOD")
print(result)

result = fn("Department newsletter: spring picnic next Friday")
print(result)
```

Compilation runs on PAW's server. Loading downloads the required assets on first use; calls to `fn` run locally.

### Optional: remote inference

For fast inference without downloading model assets, pass `remote=True`:

```python
with paw.function(program.id, remote=True) as remote_fn:
    print(remote_fn("Urgent: the server is down!"))
```

For direct HTTP calls, see the [REST API reference](../api-reference/rest-api.md#post-infer).

## Step 2: Test different inputs

Try the same function on several inputs:

```python
samples = [
    "Urgent: production database is down",
    "Newsletter: team picnic next Friday",
    "Please approve this request by the end of today",
    "FYI: new parking policy starts next month",
]

for text in samples:
    print("IN :", text)
    print("OUT:", fn(text))
    print()
```

If the outputs do not match what you need, refine the specification and compile again.

## Step 3: Save and reload the program

Print the program ID and save it for future runs:

```python
print(program.id)
```

In a new Python session, load your saved ID:

```python
import programasweights as paw

saved_program_id = "PASTE_YOUR_PROGRAM_ID_HERE"
fn = paw.function(saved_program_id)
print(fn("Urgent: the server is down!"))
```

Load the function once during application setup and reuse it across calls. Keep compilation outside request handlers.

## Step 4: Name it (optional)

Human-readable names are managed on the hub. To assign an alias:

1. Open `https://programasweights.com/hub/YOUR_PROGRAM_ID`, replacing `YOUR_PROGRAM_ID` with your saved ID.
2. Sign in with GitHub.
3. Name the existing program `message-triage`.
4. Load it with `paw.function("your-username/message-triage")`, replacing `your-username` with your account username.

You can keep using the saved program ID without naming it. See [Naming Programs](naming-programs.md).
