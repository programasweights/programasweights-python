# How It Works

ProgramAsWeights (PAW) compiles natural language specifications into **neural programs**: small, local functions that combine a textual program with learned adapter weights.

## Overview

The system turns a written spec into a runnable artifact. Compilation is a fixed pipeline; at runtime the SDK loads a shared base model, applies a program-specific adapter, and runs inference through llama.cpp.

## The compilation pipeline

Compilation has three stages.

### 1. Pseudo-program generation

An untrained 4B-parameter instruction model (Qwen3-4B-Instruct) generates a **pseudo-program**: structured text that includes a task description and illustrative examples derived from your spec.

### 2. LoRA extraction

A trained 4B **compiler** model reads the original spec together with the pseudo-program and produces internal representations. A **LoRA mapper** maps those hidden states into LoRA adapter weights that encode the desired behavior.

### 3. Bundling

The LoRA weights are quantized to **Q4_0** GGUF format (on the order of **23 MB**) and packaged with the pseudo-program into a **`.paw`** file. That bundle is what the SDK downloads and caches.

## Runtime behavior

When you run a program locally:

- The SDK loads a **Q6_K** base model (Qwen3-0.6B, about **594 MB**, downloaded once).
- It applies the **Q4_0** LoRA adapter from the bundle.
- The pseudo-program is prepended as a **prompt prefix**; user input follows.
- Inference uses **llama.cpp** (CPU or GPU backends as configured).

## Discrete plus continuous

Two mechanisms work together:

- The **pseudo-program** supplies discrete instructions and structure (what the task is, how examples look).
- The **LoRA adapter** supplies continuous behavioral tuning aligned to that task.

Either part alone is weaker than the combination; PAW is designed around this joint design.

## Deterministic identity and caching

**Content-addressable IDs:** For a given specification and compiler version, the resulting **program ID** is deterministic. The same inputs yield the same identifier.

**Caching:** Repeated compiles of the same spec resolve quickly: the service can skip redundant pseudo-program generation and reuse cached program artifacts where applicable, so you do not pay full compilation cost on every identical request.
