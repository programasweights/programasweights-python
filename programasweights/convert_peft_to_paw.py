"""
Convert a PEFT/LoRA checkpoint to .paw format.

Supports the standard PEFT workflow:
  1. Train with HuggingFace PEFT library
  2. Save adapter: model.save_pretrained("my_adapter/")
  3. Convert to .paw: paw.from_peft("my_adapter/", "output.paw", spec="...")

The resulting .paw file can be:
  - Shared on the Program Hub
  - Loaded with paw.function("output.paw")
  - Distributed as a single file
"""

import json
from pathlib import Path
from typing import Optional, List, Dict

import torch
from safetensors.torch import load_file as load_safetensors

from .paw_format import save_paw_program


def from_peft(
    adapter_path: str,
    out_path: str,
    spec: str = "",
    description: str = "",
    author: str = "",
    tags: Optional[List[str]] = None,
    examples: Optional[List[Dict[str, str]]] = None,
    generation_config: Optional[Dict] = None,
    interpreter_model: Optional[str] = None,
) -> str:
    """
    Convert a PEFT/LoRA adapter to .paw format.
    
    Args:
        adapter_path: Path to PEFT adapter directory (contains adapter_config.json + adapter_model.safetensors)
        out_path: Output .paw file path
        spec: What the program does (natural language)
        description: Longer description for the Hub
        author: Author name
        tags: Tags for discovery
        examples: List of {"input": ..., "output": ...} demos
        generation_config: Override default generation params
        interpreter_model: Override base model (auto-detected from adapter config)
    
    Returns:
        Path to saved .paw file
    
    Example:
        >>> import programasweights as paw
        >>>
        >>> # After training with PEFT:
        >>> # from peft import get_peft_model, LoraConfig
        >>> # model = get_peft_model(base_model, lora_config)
        >>> # trainer.train()
        >>> # model.save_pretrained("my_adapter/")
        >>>
        >>> # Convert to .paw:
        >>> paw.from_peft(
        ...     "my_adapter/",
        ...     "sentiment.paw",
        ...     spec="Classify text sentiment as positive, negative, or neutral",
        ...     tags=["sentiment", "classification"],
        ...     examples=[
        ...         {"input": "Love this product!", "output": "positive"},
        ...         {"input": "Worst experience ever", "output": "negative"},
        ...     ],
        ... )
    """
    adapter_dir = Path(adapter_path)
    
    # 1. Load adapter config
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"adapter_config.json not found in {adapter_dir}.\n"
            f"Make sure this is a valid PEFT adapter directory.\n"
            f"Expected files: adapter_config.json, adapter_model.safetensors (or .bin)"
        )
    
    with open(config_path) as f:
        adapter_config = json.load(f)
    
    # Extract LoRA config
    lora_rank = adapter_config.get("r", adapter_config.get("rank", 0))
    lora_alpha = adapter_config.get("lora_alpha", lora_rank)
    target_modules = adapter_config.get("target_modules", [])
    base_model_name = adapter_config.get("base_model_name_or_path", "")
    
    if not interpreter_model:
        interpreter_model = base_model_name
    
    if not interpreter_model:
        raise ValueError(
            "Could not determine base model. "
            "Set interpreter_model explicitly or ensure adapter_config.json has base_model_name_or_path."
        )
    
    lora_config = {
        "rank": lora_rank,
        "alpha": lora_alpha,
        "target_modules": target_modules,
        "peft_type": adapter_config.get("peft_type", "LORA"),
        "task_type": adapter_config.get("task_type", ""),
    }
    
    print(f"📋 PEFT adapter config:")
    print(f"   Base model: {interpreter_model}")
    print(f"   LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"   Target modules: {target_modules}")
    
    # 2. Load adapter weights
    weights_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    
    if weights_path.exists():
        print(f"📦 Loading weights from {weights_path}")
        raw_weights = load_safetensors(str(weights_path))
    elif bin_path.exists():
        print(f"📦 Loading weights from {bin_path}")
        raw_weights = torch.load(str(bin_path), map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No adapter weights found in {adapter_dir}.\n"
            f"Expected: adapter_model.safetensors or adapter_model.bin"
        )
    
    # 3. Normalize weight names for .paw format
    # PEFT uses names like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    # We normalize to: layers.0.self_attn.q_proj.lora_A
    lora_weights = {}
    for name, tensor in raw_weights.items():
        # Strip common PEFT prefixes
        clean_name = name
        for prefix in ["base_model.model.", "base_model."]:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
        
        # Strip .weight suffix (PEFT convention)
        if clean_name.endswith(".weight"):
            clean_name = clean_name[:-7]
        
        lora_weights[clean_name] = tensor
    
    print(f"   Loaded {len(lora_weights)} LoRA tensors")
    
    # Calculate total LoRA params
    total_params = sum(t.numel() for t in lora_weights.values())
    print(f"   Total LoRA parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB in fp32)")
    
    # 4. Save as .paw
    if not out_path.endswith('.paw'):
        out_path = out_path + '.paw'
    
    save_paw_program(
        filepath=out_path,
        kv_layers=None,
        spec=spec,
        base_model=interpreter_model,
        prefix_steps=0,
        pseudo_program="",
        lora_weights=lora_weights,
        lora_config=lora_config,
        generation_config=generation_config,
        description=description,
        author=author,
        tags=tags,
        examples=examples,
        source="peft",
        source_info={
            "training_method": "peft",
            "peft_type": lora_config.get("peft_type", "LORA"),
            "base_model": interpreter_model,
            "adapter_path": str(adapter_dir),
            "forkable": False,
        },
    )
    
    file_size = Path(out_path).stat().st_size
    print(f"\n✅ Saved {out_path} ({file_size / 1024 / 1024:.1f} MB)")
    print(f"   Share on Hub: https://programasweights.com/hub/upload")
    
    return out_path
