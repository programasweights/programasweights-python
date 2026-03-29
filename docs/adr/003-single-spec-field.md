# ADR-003: Single spec text field in compile API (no separate examples)

## Status: Accepted (2026-03-21)

## Context

The compile endpoint needs to accept a specification describing the function to compile. Users may want to include input/output examples. Should the API have separate fields for spec and examples?

## Decision

The API takes a single `spec` string field. No separate `examples` field. Users include examples naturally in the spec text if they want. The website frontend provides structured UI (spec box + examples editor) for UX convenience, but concatenates them client-side before calling the API.

## Rationale

- The compiler was trained on specs that include examples as text
- The compile function internally takes a single spec string
- Separate fields create ambiguity (override? append? format?)
- Simpler API, simpler logging (one field = one column)
- Training data pipeline is cleaner (each row = one spec string)

## Consequences

- API is simpler and matches the underlying model interface
- Website does the structured -> flat conversion in frontend JS
- Logging stores the merged spec string
- Power users can format examples however they want in the spec text
