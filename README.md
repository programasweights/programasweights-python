# ProgramAsWeights

**Compile natural language specifications into neural programs (.paw files) that run locally.**

Programs are stored as weight blobs (KV cache prefix + optional LoRA adapters) interpreted by a small fixed model. No API calls needed at runtime — fully deterministic, local execution.

## Installation

```bash
pip install programasweights
```

## Quick Start

### Run a Program

```python
import programasweights as paw

# Load and run a compiled program
fn = paw.function("program_id_or_path.paw")
result = fn("Contact alice@company.com or bob@example.org")
print(result)  # ["alice@company.com", "bob@example.org"]
```

### Compile a Program

```python
import programasweights as paw

# Compile from natural language specification
paw.compile(
    "output.paw",
    spec="Extract all email addresses from text and return as JSON list",
    checkpoint_dir="path/to/trained/compiler",
)
```

## LoRA Support (PEFT Compatible)

Already using PEFT for LoRA training? Convert to .paw in one line:

```python
import programasweights as paw

# Standard PEFT workflow:
# model = get_peft_model(base_model, LoraConfig(r=16, target_modules=["q_proj", "v_proj"]))
# trainer.train()
# model.save_pretrained("my_adapter/")

# Convert to .paw:
paw.from_peft(
    "my_adapter/",       # Your PEFT checkpoint
    "sentiment.paw",     # Output .paw file
    spec="Classify sentiment as positive or negative",
    tags=["sentiment", "classification"],
    examples=[
        {"input": "Great movie!", "output": "positive"},
        {"input": "Terrible film.", "output": "negative"},
    ],
)

# Use it:
fn = paw.function("sentiment.paw")
print(fn("This is amazing!"))  # → "positive"
```

Load LoRA from a .paw file:

```python
lora_weights, lora_config = paw.load_paw_lora("sentiment.paw")
print(lora_config)  # {"rank": 16, "alpha": 32, ...}
```

Or use `save_lora_to_paw()` directly if you have raw tensors instead of a PEFT checkpoint.

## .paw File Format v2

A `.paw` file is a self-contained neural program that includes:

| Component | Description | Required |
|-----------|-------------|----------|
| KV cache prefix | Continuous program (prefix weights) | Optional |
| Pseudo-program | Discrete text instructions | Optional |
| LoRA adapter | Fine-tuned adapter weights | Optional |
| Generation config | Temperature, top_p, max_tokens | Optional |
| Metadata | Interpreter model, spec, author, tags | Required |

## Program Hub

Browse and share programs at [hub.programasweights.com](https://hub.programasweights.com)

## Links

- **Website**: [programasweights.com](https://programasweights.com)
- **Documentation**: [programasweights.readthedocs.io](https://programasweights.readthedocs.io)
- **GitHub**: [github.com/programasweights/programasweights](https://github.com/programasweights/programasweights)
- **Program Hub**: [hub.programasweights.com](https://hub.programasweights.com)

## License

MIT
