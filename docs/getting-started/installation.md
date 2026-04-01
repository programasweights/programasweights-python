# Installation

Install the ProgramAsWeights SDK from PyPI:

```bash
pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
```

The `--extra-index-url` flag provides pre-built binaries for `llama-cpp-python`, making installation fast (~10 seconds). Without it, the C++ backend compiles from source (~5 minutes).

## Requirements

- **Python:** 3.9 through 3.13.

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

## Verify the install

```bash
python -c "import programasweights as paw; print(paw.__version__)"
```

You should see the installed package version printed with no errors.
