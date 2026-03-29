# PAW Architecture

## System Overview

Programs-as-Weights (PAW) compiles natural language specifications into neural functions that run locally via llama.cpp. The system has three main components:

1. **API Backend** — FastAPI server handling compilation, inference, auth, and the program hub
2. **GPU Services** — vLLM instances for pseudo-program generation, hidden state extraction, and multi-LoRA inference
3. **SDK** — `pip install programasweights` — lightweight Python client (~80MB) using llama-cpp-python

## Data Flow

```
User spec → API → GPU0 (vLLM pseudo-gen) → GPU1 (hidden state extraction)
         → LoRA mapper → Q4_0 GGUF adapter → .paw bundle → CDN
         
User SDK → download .paw → load base GGUF (once) + adapter → llama.cpp inference
```

## Compilation Pipeline

1. **Pseudo-program generation** (GPU 0): vLLM serves the untrained Qwen3-4B-Instruct model. Given a spec, generates a discrete pseudo-program (~200-500 tokens).
2. **Hidden state extraction** (GPU 1): vLLM `extract_hidden_states` API. The trained compiler processes `[spec_prompt | pseudo_program | EOS | prefix_tokens]` and outputs hidden states at prefix positions.
3. **LoRA mapper** (CPU): Projects hidden states into LoRA A/B matrices for each target module.
4. **Adapter conversion** (CPU): Quantizes LoRA matrices to Q4_0 GGUF format (~23MB).
5. **Bundle** (CPU): Packages adapter + pseudo-program + pre-rendered prompts + metadata into `.paw` ZIP.

## Inference (Server-side "Try It")

vLLM multi-LoRA serving on GPU 2 (or GPU 1 in 2-GPU config). Each compiled program has a LoRA adapter registered. vLLM batches requests across different adapters.

## Inference (Client-side SDK)

llama-cpp-python loads the base GGUF model (623MB, downloaded once) and hot-swaps Q4_0 adapters per function call. No PyTorch or transformers required.

## GPU Assignment

See `docs/adr/` for the rationale behind each decision.
