from __future__ import annotations

import os
import threading
import hashlib
import urllib.request
import urllib.parse
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

import torch
import transformers as _tf

from programasweights.paw_format import load_paw_program, PAWFormat


# Global lock to guard generate() calls for simple thread-safety
_GLOBAL_GENERATE_LOCK = threading.RLock()


def get_cache_dir() -> Path:
    """Get the global cache directory for .paw files."""
    # Respect PROGRAMASWEIGHTS_CACHE_DIR environment variable
    import os
    cache_dir_env = os.getenv('PROGRAMASWEIGHTS_CACHE_DIR')
    cache_dir = Path(cache_dir_env) if cache_dir_env else (Path.home() / ".cache" / "programasweights")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def resolve_program_path(path: str) -> str:
    """
    Resolve a program path to a local .paw file, downloading if necessary.
    
    Handles:
    - Local files: "./program.paw" -> "./program.paw"
    - Program IDs: "abc123" -> downloads to ~/.cache/programasweights/abc123.paw
    - URLs: "https://..." -> downloads to cache with URL hash
    """
    # If it's already a local file that exists, use it
    if os.path.exists(path):
        if path.endswith('.paw') or PAWFormat.is_paw_file(path):
            return path
        else:
            raise ValueError(f"File {path} is not a valid .paw file")
    
    # If it looks like a URL, download it
    if path.startswith(('http://', 'https://')):
        return download_from_url(path)
    
    # Treat as program ID or slug (e.g., "yuntian-deng/email-extractor")
    return download_program_id(path)


def download_program_id(program_id: str) -> str:
    """
    Download a program by ID or slug from programasweights.com.
    
    Supports:
    - Hash IDs: "c098443b9db9"
    - Slugs: "yuntian-deng/email-extractor"
    """
    clean_id = program_id.replace('.paw', '')
    
    # Use slug-safe filename for cache
    cache_name = clean_id.replace('/', '_')
    cache_dir = get_cache_dir()
    cached_file = cache_dir / f"{cache_name}.paw"
    
    if cached_file.exists():
        print(f"Using cached program: {cached_file}")
        return str(cached_file)
    
    # Try downloading - slug uses different endpoint
    if '/' in clean_id:
        url = f"https://programasweights.com/api/hub/programs/{clean_id}/download"
    else:
        url = f"https://programasweights.com/api/hub/programs/{clean_id}/download"
    
    print(f"Downloading program {clean_id}...")
    
    try:
        urllib.request.urlretrieve(url, cached_file)
        print(f"Downloaded and cached: {cached_file}")
        return str(cached_file)
    except Exception as e:
        raise RuntimeError(f"Failed to download program {clean_id}: {e}")


def download_from_url(url: str) -> str:
    """Download a program from a URL and cache it."""
    # Create cache filename from URL hash
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    cache_dir = get_cache_dir()
    cached_file = cache_dir / f"url_{url_hash}.paw"
    
    # Return cached file if it exists
    if cached_file.exists():
        print(f"Using cached program from URL: {cached_file}")
        return str(cached_file)
    
    # Download and cache
    print(f"Downloading program from {url}...")
    try:
        urllib.request.urlretrieve(url, cached_file)
        print(f"Downloaded and cached: {cached_file}")
        return str(cached_file)
    except Exception as e:
        raise RuntimeError(f"Failed to download program from {url}: {e}")


@dataclass
class RegisteredProgram:
    name: str
    path: str
    kv_layers: List[Tuple[torch.Tensor, torch.Tensor]]
    metadata: Dict[str, str]
    lora_weights: Optional[Dict[str, torch.Tensor]] = None
    lora_config: Optional[Dict[str, any]] = None


def _ensure_list(x: Union[str, List[str]]) -> Tuple[bool, List[str]]:
    if isinstance(x, list):
        return True, x
    if isinstance(x, str):
        return False, [x]
    raise TypeError("Input must be str or List[str]")


class _Interpreter:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self.model_name = model_name
        self.device = device
        self.tokenizer = _tf.AutoTokenizer.from_pretrained(model_name)
        self.model = _tf.AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Store checkpoint path for loading image processor weights
        self.checkpoint_path = model_name if os.path.isdir(model_name) else None
        
        # Lazy loading: image processor will be initialized when first image is encountered
        self.image_processor = None
        self._image_processor_initialized = False
        self._programs: Dict[str, RegisteredProgram] = {}

    def _ensure_image_processor(self):
        """Lazily initialize the image processor when first image is encountered."""
        if not self._image_processor_initialized:
            print("🖼️  First image detected - initializing CLIP ViT-B/16 model...")
            
            # Use the same ImageProcessor class as training for consistency
            from training.loops.prefix_tuning_sft import ImageProcessor
            interpreter_hidden_size = self.model.config.hidden_size
            
            # Try to load configuration from checkpoint first
            config = self._load_image_processor_config()
            if config:
                self.image_processor = ImageProcessor(
                    target_hidden_size=config["target_hidden_size"],
                    image_size=config["image_size"],
                    model_name=config["model_name"]
                )
            else:
                print("⚠️  Image processor config not found - using defaults")
                # Fallback to default values
                self.image_processor = ImageProcessor(
                    target_hidden_size=interpreter_hidden_size,
                    image_size=224,
                    model_name="openai/clip-vit-base-patch16"
                )
            self.image_processor.to(self.device)
            
            # Try to load trained weights from checkpoint if available
            self._load_image_processor_from_checkpoint()
            
            self._image_processor_initialized = True
            print("✅ Image processor initialized successfully!")
    
    def _load_image_processor_config(self):
        """Load image processor configuration from checkpoint."""
        if self.checkpoint_path is None:
            print("🔍 DEBUG: checkpoint_path is None")
            return None
        
        print(f"🔍 DEBUG: checkpoint_path = {self.checkpoint_path}")
        # checkpoint_path points to the interpreter directory, so we need to go up one level
        checkpoint_root = os.path.dirname(self.checkpoint_path)
        print(f"🔍 DEBUG: checkpoint_root = {checkpoint_root}")
        interpreter_image_processor_dir = os.path.join(checkpoint_root, "interpreter_image_processor")
        print(f"🔍 DEBUG: interpreter_image_processor_dir = {interpreter_image_processor_dir}")
        config_path = os.path.join(interpreter_image_processor_dir, "config.json")
        print(f"🔍 DEBUG: config_path = {config_path}")
        print(f"🔍 DEBUG: config_path exists = {os.path.exists(config_path)}")
        
        if os.path.exists(config_path):
            import json
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"✅ Loaded image processor config: {config}")
            return config
        else:
            print("⚠️  Image processor config not found - using defaults")
            return None
    
    def _load_image_processor_from_checkpoint(self):
        """Try to load trained image processor weights from checkpoint."""
        if self.checkpoint_path is None:
            print("⚠️  No checkpoint path available - using random image processor weights")
            return
        
        # Try to load interpreter image processor weights
        # checkpoint_path points to the interpreter directory, so we need to go up one level
        checkpoint_root = os.path.dirname(self.checkpoint_path)
        interpreter_image_processor_dir = os.path.join(checkpoint_root, "interpreter_image_processor")
        if os.path.exists(interpreter_image_processor_dir):
            config_path = os.path.join(interpreter_image_processor_dir, "config.json")
            model_path = os.path.join(interpreter_image_processor_dir, "pytorch_model.bin")
            
            if os.path.exists(config_path) and os.path.exists(model_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                # Load the trained weights
                self.image_processor.load_state_dict(torch.load(model_path, map_location=self.device))
                print("✅ Loaded trained image processor weights from checkpoint")
            else:
                print("⚠️  Image processor checkpoint files not found - using random weights")
        else:
            print("⚠️  Image processor checkpoint directory not found - using random weights")

    def register_program(self, path: str, name: Optional[str]) -> RegisteredProgram:
        program_name = name if name is not None else os.path.basename(path)
        
        # Resolve path (download if necessary)
        resolved_path = resolve_program_path(path)
        
        # Load .paw file
        kv_layers, metadata = load_paw_program(resolved_path)
        # Move tensors to device
        kv_layers = [(k.to(self.device), v.to(self.device)) for k, v in kv_layers]
        
        # Load LoRA weights if present
        lora_weights = None
        lora_config = None
        if metadata.get("has_lora"):
            from programasweights.paw_format import load_paw_lora
            lora_weights_raw, lora_config = load_paw_lora(resolved_path)
            lora_weights = {k: v.to(self.device) for k, v in lora_weights_raw.items()}
            print(f"🔧 Loaded LoRA adapter: rank={lora_config.get('rank')}, modules={lora_config.get('target_modules')}")

        program = RegisteredProgram(
            name=program_name, 
            path=resolved_path, 
            kv_layers=kv_layers,
            metadata=metadata,
            lora_weights=lora_weights,
            lora_config=lora_config,
        )
        self._programs[program_name] = program
        
        # Apply LoRA if present
        if lora_weights:
            self._apply_lora(program)
        
        return program
    
    def _apply_lora(self, program: RegisteredProgram):
        """Apply LoRA weights to the interpreter model."""
        if not program.lora_weights or not program.lora_config:
            return
        
        config = program.lora_config
        scaling = config.get('alpha', 16) / config.get('rank', 16)
        
        # Store original weights for cleanup
        if not hasattr(self, '_original_weights'):
            self._original_weights = {}
        
        applied = 0
        for name, param in self.model.named_parameters():
            # Check if this parameter has a LoRA adapter
            # PEFT names: model.layers.0.self_attn.q_proj.weight
            # LoRA keys: model.layers.0.self_attn.q_proj.lora_A, .lora_B
            lora_a_key = name.replace('.weight', '.lora_A')
            lora_b_key = name.replace('.weight', '.lora_B')
            
            if lora_a_key in program.lora_weights and lora_b_key in program.lora_weights:
                lora_a = program.lora_weights[lora_a_key]  # [rank, in_features]
                lora_b = program.lora_weights[lora_b_key]  # [out_features, rank]
                
                # Save original weight
                self._original_weights[name] = param.data.clone()
                
                # Apply: W' = W + scaling * B @ A
                with torch.no_grad():
                    param.data += scaling * (lora_b @ lora_a).to(param.dtype)
                applied += 1
        
        print(f"   Applied LoRA to {applied} parameters (scaling={scaling:.2f})")
    
    def _remove_lora(self):
        """Remove LoRA modifications, restore original weights."""
        if hasattr(self, '_original_weights'):
            for name, original in self._original_weights.items():
                param = dict(self.model.named_parameters())[name]
                param.data.copy_(original)
            self._original_weights.clear()

    def get_callable(self, program_name: str, max_new_tokens: int) -> Callable[..., Union[str, List[str]]]:
        if program_name not in self._programs:
            raise ValueError(f"Program '{program_name}' is not registered")

        def _call(*args) -> Union[str, List[str]]:
            # Handle variable number of arguments
            if len(args) == 0:
                raise ValueError("At least one input argument is required")
            
            # Convert all arguments to a list of inputs
            inputs = []
            for arg in args:
                if isinstance(arg, str):
                    inputs.append(arg)
                elif PIL_AVAILABLE and isinstance(arg, Image.Image):
                    inputs.append(arg)
                elif isinstance(arg, list):
                    # Flatten nested lists
                    inputs.extend(arg)
                else:
                    raise TypeError(f"Input must be str, Image.Image, or List[Union[str, Image.Image]], got {type(arg)}")
            
            outputs = self._generate(program_name, inputs, max_new_tokens=max_new_tokens)
            # Return single output if only one input, otherwise return list
            return outputs[0] if len(outputs) == 1 else outputs

        return _call

    def _generate(self, program_name: str, inputs: List[Union[str, 'Image.Image']], *, max_new_tokens: int) -> List[str]:
        """
        Generate output using the new architecture:
        1. Build interpreter prompt from pseudo_program + user input
        2. Generate with prefix KV cache
        """
        program = self._programs[program_name]
        kv_prefix = program.kv_layers
        pseudo_program = program.metadata.get("pseudo_program", "")
        
        device = next(self.model.parameters()).device
        
        # Build interpreter prompt (matches eval script)
        # For text inputs, concatenate all text args
        text_inputs = []
        for inp in inputs:
            if isinstance(inp, str):
                text_inputs.append(inp)
            elif PIL_AVAILABLE and isinstance(inp, Image.Image):
                # Image support: embed image and prepend to text
                self._ensure_image_processor()
                # For now, skip images in prompt (TODO: multimodal interpreter prompt)
                text_inputs.append("[image]")
            else:
                raise ValueError(f"Unsupported input type: {type(inp)}")
        
        task_input = "\n".join(text_inputs) if len(text_inputs) > 1 else text_inputs[0]
        
        # Build interpreter prompt (minimal style, matching eval)
        prompt = f"""{pseudo_program.strip()}

[INPUT]
{task_input}
[END_INPUT]""".strip()
        
        # Apply chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                rendered = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
                )
            except TypeError:
                rendered = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
        else:
            rendered = prompt
        
        # Tokenize
        toks = self.tokenizer(rendered, return_tensors="pt").to(device)
        input_ids = toks["input_ids"]
        attention_mask = toks["attention_mask"]
        prompt_len = input_ids.shape[1]
        bsz = 1
        
        # Expand KV prefix for batch size
        expanded_kv: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for (k, v) in kv_prefix:
            if k.dim() == 3:
                k = k.unsqueeze(0).expand(bsz, -1, -1, -1).contiguous()
                v = v.unsqueeze(0).expand(bsz, -1, -1, -1).contiguous()
            expanded_kv.append((k, v))
        
        prefix_len = expanded_kv[0][0].size(2)
        
        # Prepend dummy input_ids for cached portion
        prefix_dummy_ids = torch.zeros((bsz, prefix_len), device=device, dtype=input_ids.dtype)
        full_input_ids = torch.cat([prefix_dummy_ids, input_ids], dim=1)
        
        # Prepend ones to attention mask for cached portion
        prefix_mask = torch.ones((bsz, prefix_len), device=device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        
        full_input_len = prefix_len + prompt_len
        
        # Convert to DynamicCache
        from transformers.cache_utils import DynamicCache
        prefix_cache = DynamicCache.from_legacy_cache(tuple(expanded_kv))
        
        # Generate
        with _GLOBAL_GENERATE_LOCK:
            outputs = self.model.generate(
                input_ids=full_input_ids,
                attention_mask=full_attention_mask,
                past_key_values=prefix_cache,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        # Decode generated tokens
        gen_ids = outputs[0, full_input_len:]
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            eos_pos = (gen_ids == eos_id).nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                gen_ids = gen_ids[:eos_pos[0].item()]
        
        generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return [generated_text]


_INTERPRETER_SINGLETON: Optional[_Interpreter] = None


def _select_device() -> torch.device:
    override = os.environ.get("PROGRAMASWEIGHTS_DEVICE", "").strip().lower()
    if override == "cpu":
        return torch.device("cpu")
    if override == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_interpreter(model_name: str) -> _Interpreter:
    global _INTERPRETER_SINGLETON
    if _INTERPRETER_SINGLETON is None:
        device = _select_device()
        _INTERPRETER_SINGLETON = _Interpreter(model_name=model_name, device=device)
    return _INTERPRETER_SINGLETON


def function(
    path: str,
    *,
    name: Optional[str] = None,
    interpreter_name: str = "programasweights/interpreter",
    max_new_tokens: int = 512,
) -> Callable[..., Union[str, List[str]]]:
    """
    Register a .paw program file and return a callable that runs it.
    """
    interp = _get_interpreter(model_name=interpreter_name)
    program = interp.register_program(path=path, name=name)
    return interp.get_callable(program_name=program.name, max_new_tokens=max_new_tokens) 
