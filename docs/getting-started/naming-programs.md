# Naming Programs

ProgramAsWeights uses two ways to refer to compiled programs: immutable content hashes and optional human-readable slugs.

## Content-addressable IDs

Every compiled program has a stable identifier derived from its contents, similar to a Git commit hash:

```text
9b57fd6fccf77885400e
```

You can load a program by this ID with `paw.function("<id>")` wherever you have the exact string. Hash IDs never change and always work.

## Slugs (human-readable handles)

Authenticated users can optionally assign a slug at compile time:

```python
program = paw.compile(
    "Classify message urgency as immediate or wait",
    slug="message-classifier"
)
# Creates da03/message-classifier (username/slug)
```

The slug creates a `username/slug` handle that works everywhere a hash ID works:

```python
fn = paw.function("da03/message-classifier")
```

Slugs are **never auto-generated**. If you don't provide one, your program is
identified by hash only. This keeps the namespace clean for intentional naming.

### Adding or changing a slug later

You can add a slug to an existing program, or rename it:

```bash
paw rename a6b454023d41ac9ca845 message-classifier
```

Or via the API:

```python
import httpx
httpx.patch(
    "https://programasweights.com/api/v1/programs/a6b454023d41ac9ca845",
    json={"slug": "message-classifier"},
    headers={"X-API-Key": "paw_sk_..."}
)
```

## Official namespace

Programs published under the `programasweights/` namespace are the official catalog. For these, the prefix may be omitted:

- `paw.function("email-triage")` resolves to `programasweights/email-triage`.

## Community programs

Programs published under a user prefix must include that prefix:

```python
paw.function("da03/message-classifier")
```

## Authentication

Creating or managing slugs requires signing in with GitHub on [programasweights.com](https://programasweights.com).

## Convention

This mirrors HuggingFace, Docker Hub, and npm:

- Short names like `email-triage` refer to official programs (`programasweights/email-triage`).
- `username/name` is for community programs, like `Qwen/Qwen3-0.6B` on HuggingFace.
