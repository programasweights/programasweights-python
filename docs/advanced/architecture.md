# Architecture

This page summarizes how the ProgramAsWeights production system is structured end to end.

## Components

| Layer | Technology | Role |
|--------|------------|------|
| Frontend | React, Vite | Web UI; static assets served by nginx |
| API | FastAPI, uvicorn | REST endpoints, orchestration, auth integration |
| Database | PostgreSQL | Users, programs, aliases, votes, cases, operational logs |
| GPU services | Three vLLM instances | Pseudo-program generation, compiler (including hidden-state work), inference |
| Storage | Hugging Face, local disk | `.paw` bundles on Hugging Face; PEFT adapter artifacts on server disk |
| Auth | GitHub OAuth | Sign-in; session backed by HTTP cookies |

### GPU layout

Typical allocation:

- **GPU 0** — pseudo-program generation
- **GPU 1** — compiler workload (including pooling / hidden-state extraction used in the pipeline)
- **GPU 2** — multi-LoRA inference

Exact mapping may vary by deployment; the important split is dedicated vLLM roles per stage.

## Compile pipeline

High-level flow:

1. **Pseudo-generation** (vLLM) — turn the natural-language spec into a pseudo-program representation.
2. **LoRA extraction** — derive adapter weights from vLLM hidden states / pooling as implemented in the compiler stack.
3. **Quantization** — convert adapters to **Q4_0 GGUF** for the bundle format used by the runtime.
4. **Bundle** — assemble the **`.paw`** package with metadata and weights.
5. **Upload** — publish the `.paw` artifact to Hugging Face for CDN-backed distribution.

## Caching

The system uses **two-level caching**:

1. **Pseudo-generation cache** — avoid recomputing pseudo-programs for identical or equivalent spec inputs where the cache key applies.
2. **Program-level disk cache** — reuse compiled artifacts and intermediate state on the server when the same content-addressed program is requested again.

Together these reduce redundant GPU work and speed up repeat compiles.

## Downloads and the SDK

The Python SDK **downloads `.paw` files from the Hugging Face CDN** (or equivalent object storage fronted as a CDN). Programs are **not** served as large binary payloads from the ProgramAsWeights API host, which keeps the API focused on metadata, auth, and orchestration.
