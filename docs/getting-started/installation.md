# Installation

Install the ProgramAsWeights SDK from PyPI:

```bash
pip install --pre programasweights
```

## Requirements

- **Python:** 3.9 through 3.12. Python 3.13 is not supported yet because `llama-cpp-python` does not support it.

## First-time build

`llama-cpp-python` may compile from source on first install. Expect roughly five minutes; this is normal.

## Anaconda on Linux: OpenMP / libgomp errors

If the build fails with `libgomp`-related errors, disable OpenMP for the GGML build:

```bash
CMAKE_ARGS="-DGGML_OPENMP=OFF" pip install --pre programasweights
```

## Cache and configuration

| Location | Purpose |
|----------|---------|
| `~/.cache/programasweights/` | Downloaded models and compiled programs |
| `~/.config/programasweights/` | Local configuration |

## Verify the install

```bash
python -c "import programasweights as paw; print(paw.__version__)"
```

You should see the installed package version printed with no errors.
