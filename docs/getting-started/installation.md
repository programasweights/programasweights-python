# Installation

Install the ProgramAsWeights SDK from PyPI:

```bash
pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
```

The extra index provides prebuilt binaries for fast installation. If no matching binary is available, pip builds the runtime from source.

## Requirements

- **Python:** 3.8 through 3.14.
- **Local runtime:** `llama-cpp-python` is installed with the package.

GPU offload defaults to all available layers (`n_gpu_layers=-1`). Set
`PAW_GPU_LAYERS=0` to force CPU-only execution.

The same installation supports [remote inference](../api-reference/python-sdk.md#remote-inference)
with `paw.function(..., remote=True)`, without downloading local model assets.

## Apple Silicon Macs

Use ARM64 Python for native performance and Metal acceleration.
Check your Python architecture:

```bash
python -c "import platform; print(platform.machine())"
```

It should print `arm64`. If it prints `x86_64`, Python is running
under Rosetta. With Conda, create a native environment:

```bash
conda create -n paw-arm64 --platform osx-arm64 python=3.12 pip
conda activate paw-arm64
python -m pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
```

## Anaconda on Linux: OpenMP / libgomp errors

If the build fails with `libgomp`-related errors, disable OpenMP:

```bash
CMAKE_ARGS="-DGGML_OPENMP=OFF" pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
```

## Cache and configuration

| Location | Purpose |
|----------|---------|
| `~/.cache/programasweights/` | Downloaded models and compiled programs |
| `~/.config/programasweights/` | Local configuration |

Program bundles, runtime manifests, slug mappings, and base models are
validated before use and installed with atomic cache updates. Built-in base
models must match their pinned Hugging Face LFS byte size and SHA-256 digest
and have GGUF magic; adapters must also be nontrivial GGUF files. To prohibit
all network access, pass `offline=True` to
`paw.function`/`paw.prepare_program` or set:

```bash
export PAW_OFFLINE=1
```

Offline mode fails clearly when any required validated bundle, runtime
manifest, adapter, or base-model file is missing.

## Verify the install

```bash
python -c "import programasweights as paw; print(paw.__version__)"
```

You should see the installed package version printed with no errors.
