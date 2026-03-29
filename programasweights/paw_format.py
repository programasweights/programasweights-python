"""
.paw format v2 for neural programs.

Supports:
- KV cache prefix (continuous program)
- Discrete pseudo-program text
- Prompt token IDs
- LoRA adapter weights
- Generation config defaults
- Hub metadata (author, tags, examples, description)

Format:
  [4 bytes] Magic: b"PAW\x02"
  [4 bytes] Version: uint32
  [4 bytes] Metadata length: uint32
  [N bytes] Metadata (JSON, UTF-8)
  [M bytes] Tensors (safetensors)
"""

from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
from safetensors.torch import save_file, load_file


class PAWFormat:
    """Handler for .paw binary format files."""
    
    MAGIC = b"PAW\x02"
    VERSION = 2
    
    @staticmethod
    def save(
        filepath: str,
        tensors_dict: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Save tensors and metadata to a .paw file.
        """
        metadata_bytes = json.dumps(metadata, ensure_ascii=False).encode('utf-8')
        
        # Save tensors to temp file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        save_file(tensors_dict, temp_path)
        
        with open(temp_path, 'rb') as f:
            tensors_data = f.read()
        
        # Write .paw file
        with open(filepath, 'wb') as f:
            f.write(PAWFormat.MAGIC)
            f.write(struct.pack('<I', PAWFormat.VERSION))
            f.write(struct.pack('<I', len(metadata_bytes)))
            f.write(metadata_bytes)
            f.write(tensors_data)
        
        Path(temp_path).unlink()
    
    @staticmethod
    def load(filepath: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load tensors and metadata from a .paw file.
        
        Returns:
            Tuple of (tensors_dict, metadata)
        """
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            if magic != PAWFormat.MAGIC:
                raise ValueError(f"Invalid .paw file: wrong magic bytes")
            
            version = struct.unpack('<I', f.read(4))[0]
            
            metadata_len = struct.unpack('<I', f.read(4))[0]
            metadata_bytes = f.read(metadata_len)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            tensors_data = f.read()
        
        # Load tensors via temp file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(tensors_data)
            temp_path = temp_file.name
        
        tensors_dict = load_file(temp_path)
        Path(temp_path).unlink()
        
        return tensors_dict, metadata
    
    @staticmethod
    def is_paw_file(filepath: str) -> bool:
        """Check if a file is a valid .paw file."""
        try:
            with open(filepath, 'rb') as f:
                magic = f.read(4)
                return magic == PAWFormat.MAGIC
        except Exception:
            return False


# ============================================================
# Convenience functions
# ============================================================

def save_paw_program(
    filepath: str,
    kv_layers: Optional[List[Tuple]] = None,
    spec: str = "",
    base_model: str = "",
    prefix_steps: int = 0,
    pseudo_program: str = "",
    prompt_token_ids: Optional[List[int]] = None,
    prefix_type: str = "kv_cache",  # "kv_cache" or "embeddings"
    lora_weights: Optional[Dict[str, Any]] = None,
    lora_config: Optional[Dict[str, Any]] = None,
    generation_config: Optional[Dict[str, Any]] = None,
    description: str = "",
    author: str = "",
    tags: Optional[List[str]] = None,
    examples: Optional[List[Dict[str, str]]] = None,
    # Source tracking
    source: str = "compiled",  # "compiled" | "finetuned" | "peft" | "custom"
    source_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a neural program as .paw v2 file.
    
    Args:
        filepath: Output file path
        kv_layers: KV cache layers [(k, v), ...] (optional)
        spec: Program specification text
        base_model: Interpreter model name on HuggingFace
        prefix_steps: Number of prefix steps
        pseudo_program: Discrete pseudo-program text
        prompt_token_ids: Token IDs for discrete prompt (optional)
        prefix_type: Type of prefix ("kv_cache" or "embeddings")
        lora_weights: Dict of LoRA weight tensors (optional)
        lora_config: LoRA configuration (rank, alpha, target_modules)
        generation_config: Default generation parameters
        description: Human-readable description
        author: Author name
        tags: List of tags
        examples: List of input/output example dicts
    """
    # Build metadata
    metadata = {
        "format_version": 2,
        "kind": "neural_program",
        "interpreter_model": base_model,
        "spec": spec,
        "pseudo_program": pseudo_program,
        "prefix_type": prefix_type,
        "prefix_steps": prefix_steps,
        "num_layers": len(kv_layers) if kv_layers else 0,
        "has_lora": lora_weights is not None and len(lora_weights) > 0,
        "description": description,
        "author": author,
        "tags": tags or [],
        "examples": examples or [],
        # Backward compat
        "base_model": base_model,
    }
    
    # Source tracking
    metadata["source"] = source  # "compiled", "finetuned", "peft", "custom"
    metadata["source_info"] = source_info or {}
    
    # Auto-populate source_info based on source type
    if source == "compiled" and not source_info:
        metadata["source_info"] = {
            "compiler_model": "",  # Filled by compile()
            "compiler_version": "",
            "forkable": True,  # Users can fork and edit the spec
        }
    elif source in ("finetuned", "peft") and not source_info:
        metadata["source_info"] = {
            "training_method": source,
            "forkable": False,  # Can't edit training from the UI
        }
    
    if prompt_token_ids is not None:
        metadata["prompt_token_ids"] = prompt_token_ids
    
    if lora_config:
        metadata["lora_config"] = lora_config
    
    if generation_config:
        metadata["generation_config"] = generation_config
    else:
        metadata["generation_config"] = {
            "max_new_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 50
        }
    
    # Build tensors dict
    tensors_dict = {}
    
    # KV cache layers
    if kv_layers:
        for i, (k, v) in enumerate(kv_layers):
            tensors_dict[f"layer_{i}_key"] = k
            tensors_dict[f"layer_{i}_value"] = v
    
    # LoRA weights
    if lora_weights:
        for name, tensor in lora_weights.items():
            tensors_dict[f"lora_{name}"] = tensor
    
    PAWFormat.save(filepath, tensors_dict, metadata)


def load_paw_program(filepath: str) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    Load a neural program from .paw file.
    
    Returns:
        Tuple of (kv_layers, metadata)
        kv_layers: list of (key_tensor, value_tensor) tuples
        metadata: dict with all program metadata
    """
    tensors_dict, metadata = PAWFormat.load(filepath)
    
    # Reconstruct KV layers
    kv_layers = []
    layer_count = len([k for k in tensors_dict.keys() if k.endswith('_key')])
    
    for i in range(layer_count):
        key_tensor = tensors_dict[f"layer_{i}_key"]
        value_tensor = tensors_dict[f"layer_{i}_value"]
        kv_layers.append((key_tensor, value_tensor))
    
    return kv_layers, metadata


def load_paw_lora(filepath: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load LoRA weights from a .paw file.
    
    Returns:
        Tuple of (lora_weights_dict, lora_config)
    """
    tensors_dict, metadata = PAWFormat.load(filepath)
    
    # Extract LoRA weights
    lora_weights = {}
    for name, tensor in tensors_dict.items():
        if name.startswith("lora_"):
            lora_weights[name[5:]] = tensor  # Strip "lora_" prefix
    
    lora_config = metadata.get("lora_config", {})
    
    return lora_weights, lora_config


def validate_paw_file(filepath: str) -> Tuple[bool, List[str]]:
    """
    Validate a .paw file for hub upload.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check file exists
    if not Path(filepath).exists():
        return False, ["File not found"]
    
    # Check file size (max 500MB)
    file_size = Path(filepath).stat().st_size
    if file_size > 500 * 1024 * 1024:
        errors.append(f"File too large: {file_size / (1024*1024):.1f}MB (max 500MB)")
    
    # Try loading
    try:
        tensors_dict, metadata = PAWFormat.load(filepath)
    except Exception as e:
        return False, [f"Failed to load .paw file: {e}"]
    
    # Check format version
    version = metadata.get("format_version", 2)
    if version != PAWFormat.VERSION:
        errors.append(f"Unsupported format version: {version}")
    
    # Check interpreter model
    interpreter_model = metadata.get("interpreter_model") or metadata.get("base_model")
    if not interpreter_model:
        errors.append("Missing interpreter_model")
    
    # Check prefix_steps
    prefix_steps = metadata.get("prefix_steps", 0)
    if prefix_steps > 256:
        errors.append(f"prefix_steps too large: {prefix_steps} (max 256)")
    
    # Check LoRA config
    if metadata.get("has_lora"):
        lora_config = metadata.get("lora_config", {})
        lora_rank = lora_config.get("rank", 0)
        if lora_rank > 128:
            errors.append(f"LoRA rank too large: {lora_rank} (max 128)")
        
        # Verify LoRA tensors exist
        lora_tensor_count = len([k for k in tensors_dict.keys() if k.startswith("lora_")])
        if lora_tensor_count == 0:
            errors.append("has_lora=True but no LoRA tensors found")
    
    # Check KV layers consistency
    num_layers = metadata.get("num_layers", 0)
    actual_layers = len([k for k in tensors_dict.keys() if k.endswith("_key")])
    if num_layers != actual_layers:
        errors.append(f"Metadata says {num_layers} layers but found {actual_layers}")
    
    # Check no executable code in metadata (basic check)
    metadata_str = json.dumps(metadata)
    dangerous_patterns = ["__import__", "eval(", "exec(", "os.system", "subprocess"]
    for pattern in dangerous_patterns:
        if pattern in metadata_str:
            errors.append(f"Suspicious content in metadata: {pattern}")
    
    return len(errors) == 0, errors
