# Architecture Decision Records

Concise records of major technical choices. Full ADR files may live elsewhere in the repository; this page is the canonical summary for documentation.

---

## ADR 001: llama.cpp instead of PyTorch for the SDK runtime

**Decision:** Ship local inference via **llama-cpp-python** (llama.cpp), not a PyTorch stack.

**Context:** A PyTorch install typically exceeds **2 GB** of dependencies. llama.cpp bindings add on the order of **80 MB**, which keeps the SDK viable as a lightweight dependency.

**Consequence:** Adapter weights must be converted to **GGUF** and loaded through the llama.cpp path. Training and server-side tooling may still use PyTorch where needed; the **end-user runtime** is GGUF-centric.

---

## ADR 002: Quantization levels for base models and adapters

**Decision:** Use **Q4_0** for adapters. For base models, use **Q6_K** for Qwen3 0.6B and **Q8_0** for GPT-2.

**Context:** Empirical evaluation on **4096** held-out examples across quantization settings informed the trade-off.

**Consequence:**

- **Q6_K base (Qwen3 0.6B)** — quality is preserved while the footprint is roughly **60% smaller** than fp16. 4096-sample eval shows no accuracy loss vs fp16.
- **Q8_0 base (GPT-2)** — Q6_K caused ~3.5% accuracy loss on GPT-2 (4096-sample eval). Q8_0 closes the gap at only ~29 MB additional cost (134 MB vs 105 MB).
- **Q4_0 adapter** — quality loss is negligible for both models; adapter size drops to about **23 MB** (Qwen3) / **5 MB** (GPT-2) versus **78 MB** / **19 MB** at fp16.

---

## ADR 003: Single specification field in the API

**Decision:** The compile API accepts **one text field** for the specification. There is **no separate “examples” field** in the contract.

**Context:** The compiler was trained on specs that naturally embed examples inside the same prose block.

**Consequence:** The web UI may offer structured fields for examples or hints, but the client **merges** them into a single string before calling the API.

---

## ADR 004: Two-level compiler naming

**Decision:** Expose **pretty aliases** (for example `paw-4b-qwen3-0.6b`) that resolve to **dated snapshots** (for example `paw-4b-qwen3-0.6b-20260325`).

**Context:** Mirrors patterns such as OpenAI’s `gpt-4o` mapping to dated model IDs like `gpt-4o-2024-11-20`.

**Consequence:** Users get stable marketing names while the platform can roll forward immutable snapshots without breaking references that pin the dated id.

---

## ADR 005: vLLM for GPU services

**Decision:** Run **all** GPU-heavy paths (pseudo-program generation, hidden-state extraction / pooling, multi-LoRA inference) on **vLLM** instead of hand-rolled Hugging Face inference loops in production.

**Context:** vLLM improves **throughput**, **batching**, and **memory efficiency** for serving and batched extraction workloads.

**Consequence:** Operations that might classically be expressed as “run HuggingFace model X” are implemented as vLLM-managed models and schedules in the deployed stack.

---

## ADR 006: GitHub OAuth for authentication

**Decision:** Use **GitHub OAuth** rather than email verification or password accounts for primary sign-in.

**Context:** The target audience already maintains GitHub accounts; OAuth removes friction for naming programs, voting, and submitting cases.

**Consequence:** Identity is tied to GitHub; users without GitHub need an alternative path if one is offered separately.
