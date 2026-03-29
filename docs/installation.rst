Installation
============

ProgramAsWeights requires Python 3.9+ and PyTorch.

Quick Install
-------------

.. code-block:: bash

   pip install programasweights

This installs the core package with PyTorch, transformers, and safetensors.

Optional Dependencies
---------------------

.. code-block:: bash

   # For image/multimodal programs
   pip install programasweights[images]
   
   # For ONNX runtime (experimental, lighter alternative)
   pip install programasweights[onnx]

   # For training your own models
   pip install programasweights[train]
   
   # For generating synthetic datasets
   pip install programasweights[data]

System Requirements
-------------------

**Memory:**

- **CPU:** 4GB+ RAM recommended
- **GPU:** 8GB+ VRAM for larger models (optional, falls back to CPU)

**Storage:**

- ~2GB for interpreter model (downloaded once, cached)
- ~1-50MB per compiled program (.paw file)

**Network:**

- Internet for initial model download
- Offline execution after download

Verify Installation
-------------------

.. code-block:: python

   import programasweights as paw
   print(paw.__version__)  # Should print 0.2.0

GPU Setup (Optional)
--------------------

ProgramAsWeights automatically uses GPU if available. To force CPU:

.. code-block:: bash

   export PROGRAMASWEIGHTS_DEVICE=cpu
