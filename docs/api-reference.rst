API Reference
=============

Core Functions
--------------

paw.function()
~~~~~~~~~~~~~~

Load a .paw program and return a callable function.

.. code-block:: python

   paw.function(path, *, name=None, interpreter_name="Qwen/Qwen3-0.6B",
                max_new_tokens=512)

**Parameters:**

- **path** (*str*) – Path to .paw file, program ID, or URL
- **name** (*str, optional*) – Custom name for the program
- **interpreter_name** (*str*) – HuggingFace model ID or local path for interpreter
- **max_new_tokens** (*int*) – Maximum tokens to generate (default: 512)

**Returns:** Callable that accepts string inputs and returns string output

**Example:**

.. code-block:: python

   fn = paw.function("c098443b9db9")
   result = fn("Extract emails from: alice@test.com")
   
   # With local .paw file
   fn = paw.function("./my_program.paw")
   
   # With custom interpreter
   fn = paw.function("program.paw", interpreter_name="Qwen/Qwen3-0.6B")

paw.compile()
~~~~~~~~~~~~~

Compile a specification into a .paw neural program.

.. code-block:: python

   paw.compile(out_path, *, spec, checkpoint_dir, prefix_steps=64,
               interpreter_model_name="Qwen/Qwen3-0.6B",
               compiler_prompt_style="minimal",
               compiler_max_new_tokens=512)

**Parameters:**

- **out_path** (*str*) – Output .paw file path
- **spec** (*str*) – Natural language specification
- **checkpoint_dir** (*str*) – Path to trained compiler model directory
- **prefix_steps** (*int*) – Number of prefix steps for KV cache (default: 64)
- **interpreter_model_name** (*str*) – Interpreter model (default: Qwen/Qwen3-0.6B)
- **compiler_prompt_style** (*str*) – Prompt style: "minimal", "primitives", "freeform", "examples"
- **compiler_max_new_tokens** (*int*) – Max tokens for pseudo-program generation

**Returns:** Path to the saved .paw file

**Example:**

.. code-block:: python

   paw.compile(
       "email_extractor.paw",
       spec="Extract all email addresses from text",
       checkpoint_dir="train_runs/compiler",
   )

LoRA Functions
--------------

paw.save_lora_to_paw()
~~~~~~~~~~~~~~~~~~~~~~~

Convert a LoRA adapter to .paw format for sharing.

.. code-block:: python

   paw.save_lora_to_paw(
       out_path, lora_weights, lora_config,
       interpreter_model="Qwen/Qwen3-0.6B",
       spec="", description="", author="",
       kv_layers=None, pseudo_program="",
       prefix_steps=0, generation_config=None,
       tags=None, examples=None,
   )

**Parameters:**

- **out_path** (*str*) – Output .paw file path
- **lora_weights** (*dict*) – Dict of LoRA tensors ``{name: tensor}``
- **lora_config** (*dict*) – LoRA config ``{"rank": int, "alpha": float, "target_modules": list}``
- **interpreter_model** (*str*) – HuggingFace model ID
- **spec** (*str*) – Program specification
- **description** (*str*) – Human-readable description
- **kv_layers** (*list, optional*) – KV cache prefix layers
- **pseudo_program** (*str, optional*) – Discrete pseudo-program text
- **generation_config** (*dict, optional*) – Default generation parameters
- **tags** (*list, optional*) – Tags for hub discovery
- **examples** (*list, optional*) – Input/output examples

**Example:**

.. code-block:: python

   paw.save_lora_to_paw(
       "sentiment.paw",
       lora_weights={"q_proj.lora_A": tensor_a, "q_proj.lora_B": tensor_b},
       lora_config={"rank": 16, "alpha": 32, "target_modules": ["q_proj", "v_proj"]},
       spec="Classify sentiment",
       tags=["sentiment"],
   )

paw.load_paw_lora()
~~~~~~~~~~~~~~~~~~~~

Load LoRA weights from a .paw file.

.. code-block:: python

   lora_weights, lora_config = paw.load_paw_lora("sentiment.paw")

**Returns:** Tuple of (lora_weights_dict, lora_config_dict)

Configuration
-------------

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Variable
     - Description
     - Default
   * - ``PROGRAMASWEIGHTS_DEVICE``
     - Force device (``cpu`` or ``cuda``)
     - Auto-detect
   * - ``PROGRAMASWEIGHTS_CACHE_DIR``
     - Cache directory for downloaded programs
     - ``~/.cache/programasweights``
   * - ``PROGRAMASWEIGHTS_RUNTIME``
     - Runtime backend (``pytorch`` or ``onnx``)
     - ``pytorch``

.paw File Format v2
--------------------

The ``.paw`` binary format stores neural programs:

::

   [4 bytes] Magic: "PAW\x02"
   [4 bytes] Version: uint32 (2)
   [4 bytes] Metadata length: uint32
   [N bytes] Metadata (JSON, UTF-8)
   [M bytes] Tensors (safetensors format)

**Metadata fields:**

- ``format_version`` – Format version (2)
- ``interpreter_model`` – HuggingFace model ID
- ``spec`` – Program specification
- ``pseudo_program`` – Discrete pseudo-program text
- ``prefix_type`` – "kv_cache" or "embeddings"
- ``prefix_steps`` – Number of prefix steps
- ``has_lora`` – Whether file contains LoRA weights
- ``lora_config`` – LoRA configuration
- ``generation_config`` – Default generation parameters
- ``description``, ``author``, ``tags``, ``examples`` – Hub metadata

**Tensor keys:**

- ``layer_{i}_key``, ``layer_{i}_value`` – KV cache prefix
- ``lora_{name}`` – LoRA adapter weights
