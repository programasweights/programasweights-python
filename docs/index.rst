ProgramAsWeights Documentation
==============================

**Compile natural language specifications into neural programs that run locally.**

ProgramAsWeights (PAW) is a new programming paradigm where programs are represented
as compact neural weight blobs (.paw files) that execute locally via a small interpreter
model. No API calls, no internet required at runtime.

Quick Start
-----------

.. code-block:: python

   import programasweights as paw
   
   # Load and run a compiled program
   fn = paw.function("c098443b9db9")
   result = fn("Contact alice@company.com or bob@example.org")
   print(result)  # ["alice@company.com", "bob@example.org"]

Or compile through the web interface at `programasweights.com <https://programasweights.com>`_.

Key Features
------------

- **Local execution** – No API calls, no internet at runtime
- **Deterministic** – Same input always gives same output
- **Lightweight** – Programs are 1-50MB .paw files
- **LoRA support** – Fine-tune and share custom adapters
- **Program Hub** – Browse and share community programs
- **Multimodal** – Supports text and image inputs

How It Works
------------

1. **Compile**: Natural language spec → pseudo-program + KV cache prefix
2. **Save**: Everything stored in a single .paw file
3. **Run**: Small interpreter model reads .paw file and executes

.. code-block:: text

   Spec: "Extract emails from text"
         ↓ (Compiler)
   .paw file: [pseudo-program + KV cache + metadata]
         ↓ (Interpreter: Qwen3-0.6B)
   fn("Contact alice@test.com") → ["alice@test.com"]

Documentation
=============

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   using-pretrained
   api-reference

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/evaluation-tasks

Links
-----

- **Website**: `programasweights.com <https://programasweights.com>`_
- **Program Hub**: `programasweights.com/hub <https://programasweights.com/hub>`_
- **GitHub**: `github.com/programasweights <https://github.com/programasweights/programasweights>`_
- **PyPI**: `pypi.org/project/programasweights <https://pypi.org/project/programasweights/>`_
