# Naming Programs

ProgramAsWeights uses two ways to refer to compiled programs: immutable content hashes and optional human-readable aliases.

## Content-addressable IDs

Every compiled program has a stable identifier derived from its contents, similar to a Git commit hash:

```text
9b57fd6fccf77885400e
```

You can load a program by this ID with `paw.function("<id>")` wherever you have the exact string.

## User aliases

Signed-in users may register aliases in `username/name` form, for example:

```text
yourname/my-classifier
```

Aliases are unique per user: you cannot register two different programs both named `yourname/foo`.

Multiple aliases may point to the same underlying program (same content hash).

## Official namespace

Programs published under the `programasweights/` namespace are the official catalog. For these, the registry prefix may be omitted:

- `paw.function("email-triage")` resolves to `programasweights/email-triage`.

## Community programs

Programs published under a user or organization prefix must include that prefix when loading:

```python
paw.function("yourname/custom-filter")
```

Omitting the prefix is only valid for official names in the `programasweights/` namespace.

## Authentication

Creating or managing aliases requires signing in with GitHub on [programasweights.com](https://programasweights.com).

## Convention

This mirrors common package registries (for example Docker Hub and npm):

- Short names such as `nginx` refer to official images; `myuser/myapp` refers to community-published artifacts.
- In PAW, `email-triage` behaves like an official short name; `username/name` behaves like a scoped community name.
