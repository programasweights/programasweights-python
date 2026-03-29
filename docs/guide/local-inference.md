# Local Inference

After the first-time downloads, PAW runs **entirely on your machine**. No ongoing network access is required for inference once assets are cached.

## First run

The initial execution typically fetches:

- The **Q6_K** base model (about **594 MB**) from Hugging Face.
- Your program’s **`.paw`** bundle (LoRA and pseudo-program; on the order of **23 MB**).

Subsequent runs use the local cache and do not repeat those downloads unless you clear storage or change versions.

## Performance

On typical **CPU** hardware (for example Apple Silicon or modern x86), a single call often completes in roughly **100–500 ms**, depending on sequence length, hardware, and backend settings.

## Configuration

Common options exposed by the runtime include:

| Option | Role |
|--------|------|
| `n_ctx` | Context window size (default often **2048**). |
| `n_gpu_layers` | Offload layers to **Metal** or **CUDA** when available. |
| `verbose` | Enable detailed logging from the inference backend. |

Exact names and defaults follow the SDK; refer to the package API for the authoritative list.

## Cache location

Models and program bundles are stored under:

`~/.cache/programasweights/`

Manage disk usage by removing cached files there if you need to reclaim space or force a fresh download.

## Backend

Inference uses **llama.cpp**. Platform-specific optimizations (Metal on Apple GPUs, CUDA on NVIDIA, SIMD such as AVX on CPU) are selected by the build you install; you do not configure those low-level details separately.
