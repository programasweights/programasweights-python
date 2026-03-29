Quick Start Guide
=================

Get started with ProgramAsWeights in minutes.

Run a Program
-------------

.. code-block:: python

   import programasweights as paw
   
   # Load a program (by ID or local .paw file)
   fn = paw.function("c098443b9db9")
   
   # Call it like any Python function
   result = fn("Contact alice@company.com or bob@example.org for help")
   print(result)  # ["alice@company.com", "bob@example.org"]

Compile a Program
-----------------

Compile a natural language specification into a neural program:

.. code-block:: python

   import programasweights as paw
   
   paw.compile(
       "email_extractor.paw",
       spec="Extract all email addresses from text and return as JSON list",
       checkpoint_dir="path/to/trained/compiler",
   )
   
   # Use it
   fn = paw.function("email_extractor.paw")
   print(fn("Email me at test@example.com"))

Or compile through the web interface at `programasweights.com <https://programasweights.com>`_.

Create a LoRA Program
---------------------

If you already train LoRA adapters with PEFT, converting to .paw is one line:

.. code-block:: python

   # Standard PEFT training workflow (what you already do):
   from peft import get_peft_model, LoraConfig
   from transformers import AutoModelForCausalLM, Trainer
   
   base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
   lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
   model = get_peft_model(base_model, lora_config)
   
   # ... train with Trainer or your own loop ...
   model.save_pretrained("my_adapter/")

   # Convert to .paw (one line!):
   import programasweights as paw
   
   paw.from_peft(
       "my_adapter/",           # Your PEFT checkpoint directory
       "sentiment.paw",         # Output .paw file
       spec="Classify sentiment as positive or negative",
       tags=["sentiment", "classification"],
       examples=[
           {"input": "Great movie!", "output": "positive"},
           {"input": "Terrible.", "output": "negative"},
       ],
   )
   
   # Use it:
   fn = paw.function("sentiment.paw")
   print(fn("This film is amazing!"))  # → "positive"

The ``.paw`` file is self-contained and can be shared on the Hub or distributed to users.

Share on the Hub
----------------

Programs compiled through the website are automatically published to the
`Program Hub <https://programasweights.com/hub>`_. You can also upload .paw files
directly through the Hub interface.

Browse community programs, upvote the best ones, and download for local use.

.paw File Format
----------------

A ``.paw`` file is a self-contained neural program:

.. list-table::
   :header-rows: 1

   * - Component
     - Description
     - Required
   * - KV cache prefix
     - Continuous program weights
     - Optional
   * - Pseudo-program
     - Discrete text instructions
     - Optional
   * - LoRA adapter
     - Fine-tuned adapter weights
     - Optional
   * - Generation config
     - Temperature, top_p, max_tokens
     - Optional
   * - Metadata
     - Interpreter model, spec, author, tags
     - Required

Next Steps
----------

- :doc:`api-reference` - Complete API documentation
- :doc:`using-pretrained` - Browse available programs
- `Program Hub <https://programasweights.com/hub>`_ - Community programs
