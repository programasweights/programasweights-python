"""
ONNX-based runtime for programasweights.

Lightweight alternative to PyTorch runtime:
- ~100MB install (vs ~2.5GB for PyTorch)
- Faster inference
- Lower memory usage

The .paw file format remains unchanged - only the interpreter is ONNX.
"""
import os
import numpy as np
from pathlib import Path
from typing import Union, List, Callable
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None  # Will raise helpful error if user tries to use images


def resolve_program_path(path: str) -> str:
    """
    Resolve a program path to a local .paw file, downloading if necessary.
    Duplicate of logic from interpreter.py to avoid torch import.
    """
    import urllib.request
    
    # If it's already a local file that exists, use it
    if os.path.exists(path):
        return path
    
    # If it looks like a URL, download it
    if path.startswith(('http://', 'https://')):
        # Download to cache with URL hash
        import hashlib
        url_hash = hashlib.sha256(path.encode()).hexdigest()[:16]
        cache_dir = get_onnx_models_cache_dir().parent
        cached_file = cache_dir / f"url_{url_hash}.paw"
        
        if not cached_file.exists():
            urllib.request.urlretrieve(path, cached_file)
        
        return str(cached_file)
    
    # Otherwise treat as program ID and download from programasweights.com
    clean_id = path.replace('.paw', '')
    cache_dir = get_onnx_models_cache_dir().parent
    cached_file = cache_dir / f"{clean_id}.paw"
    
    if cached_file.exists():
        print(f"Using cached program: {cached_file}")
        return str(cached_file)
    
    # Download from programasweights.com
    url = f"https://programasweights.com/programs/{clean_id}.paw"
    print(f"Downloading program {clean_id} from {url}...")
    
    try:
        urllib.request.urlretrieve(url, cached_file)
        print(f"Downloaded and cached: {cached_file}")
        return str(cached_file)
    except Exception as e:
        raise FileNotFoundError(f"Failed to download program {clean_id}: {e}")


def get_onnx_models_cache_dir() -> Path:
    """Get cache directory for ONNX interpreter models."""
    cache_dir_env = os.getenv('PROGRAMASWEIGHTS_CACHE_DIR')
    if cache_dir_env:
        cache_dir = Path(cache_dir_env) / "onnx_models"
    else:
        cache_dir = Path.home() / ".cache" / "programasweights" / "onnx_models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_onnx_models(model_name: str, download_image_encoder: bool = False) -> dict:
    """
    Download ONNX interpreter models from HuggingFace Hub or load from local path.
    Downloads only required files (lazy loading for optional components).
    
    Args:
        model_name: HuggingFace repo ID (e.g., "yuntian-deng/paw-interpreter-onnx")
                    OR local path to ONNX models directory
        download_image_encoder: If True, also download image encoder (default: False, lazy)
    
    Returns:
        dict with paths to downloaded ONNX models
    """
    # Check if it's a local path
    if os.path.exists(model_name):
        # Local directory path (e.g., "../../outputs_onnx/")
        local_dir = Path(model_name)
        if not local_dir.exists():
            raise FileNotFoundError(f"Local ONNX directory not found: {local_dir}")
        
        print(f"📂 Using local ONNX models from: {local_dir}")
        
        return {
            'text_embeddings': local_dir / "text_embeddings.onnx",
            'interpreter': local_dir / "interpreter.onnx",
            'image_encoder': local_dir / "image_encoder.onnx" if (local_dir / "image_encoder.onnx").exists() else None,
            'tokenizer_path': local_dir / "tokenizer" / "tokenizer.json",
            'metadata_path': local_dir / "model_config.json"
        }
    
    # HuggingFace repo ID
    from huggingface_hub import hf_hub_download
    
    cache_dir = get_onnx_models_cache_dir()
    
    try:
        # Check if models are already cached (skip first-time message)
        from huggingface_hub import try_to_load_from_cache
        
        already_cached = try_to_load_from_cache(
            repo_id=model_name,
            filename="interpreter.onnx.data",
            cache_dir=cache_dir
        )
        
        if already_cached is None:
            # First time - show helpful message
            print()
            print("="*70)
            print("🚀 FIRST-TIME SETUP")
            print("="*70)
            print(f"Downloading interpreter from HuggingFace: {model_name}")
            print()
            print("⏱️  The first time setup will take some time")
            print("💡 Future calls will be instant")
            print("="*70)
            print()
        
        # Download metadata to check if embeddings are tied
        metadata_path = hf_hub_download(
            repo_id=model_name,
            filename="model_config.json",
            cache_dir=cache_dir
        )
        
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        tie_word_embeddings = metadata.get('tie_word_embeddings', False)
        
        # Download REQUIRED files
        required_files = [
            "interpreter.onnx",
            "interpreter.onnx.data",
            "tokenizer/tokenizer.json",
        ]
        
        # Add embeddings.npy ONLY if NOT tied
        if not tie_word_embeddings:
            required_files.append("embeddings.npy")
            print(f"   📦 Downloading embeddings.npy (untied)")
        else:
            print(f"   🔗 Tied embeddings: will extract from interpreter (no download needed!)")
        
        paths = {'metadata_path': Path(metadata_path)}
        
        for filename in required_files:
            print(f"   Downloading {filename}...")
            local_path = hf_hub_download(
                repo_id=model_name,
                filename=filename,
                cache_dir=cache_dir
            )
            
            # Store in results
            if filename == "tokenizer/tokenizer.json":
                paths['tokenizer_path'] = Path(local_path)
            elif filename == "interpreter.onnx":
                paths['interpreter'] = Path(local_path)
            elif filename == "model_config.json":
                paths['metadata_path'] = Path(local_path)
            elif filename == "embeddings.npy":
                paths['embeddings_npy'] = Path(local_path)
        
        print(f"✅ Core models downloaded")
        
        # Lazy download image encoder (only if requested)
        if download_image_encoder:
            print(f"   Downloading image encoder (optional)...")
            try:
                img_enc_path = hf_hub_download(
                    repo_id=model_name,
                    filename="image_encoder.onnx",
                    cache_dir=cache_dir
                )
                hf_hub_download(
                    repo_id=model_name,
                    filename="image_encoder.onnx.data",
                    cache_dir=cache_dir
                )
                paths['image_encoder'] = Path(img_enc_path)
                print(f"   ✅ Image encoder downloaded")
            except Exception as e:
                print(f"   ⚠️  Image encoder not available: {e}")
                paths['image_encoder'] = None
        else:
            paths['image_encoder'] = None
        
        return paths
        
    except Exception as e:
        raise RuntimeError(
            f"Failed to download ONNX models from HuggingFace.\n"
            f"Error: {e}\n\n"
            f"Make sure the model exists at: https://huggingface.co/{model_name}\n"
            f"Or install PyTorch runtime: pip install programasweights[pytorch]"
        )


class ONNXInterpreter:
    """
    ONNX-based interpreter for .paw programs.
    Lightweight alternative to PyTorch runtime.
    """
    
    def __init__(self, model_name: str = "programasweights/interpreter-onnx"):
        """
        Initialize ONNX interpreter.
        
        Args:
            model_name: Name of interpreter model to use
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Install with:\n"
                "  pip install onnxruntime\n"
                "Or install PyTorch runtime:\n"
                "  pip install programasweights[pytorch]"
            )
        
        self.model_name = model_name
        
        # Download/load ONNX models
        model_paths = download_onnx_models(model_name)
        
        print(f"📦 Loading ONNX models...")
        
        # Load metadata
        import json
        with open(model_paths['metadata_path'], 'r') as f:
            metadata = json.load(f)
        
        tie_word_embeddings = metadata.get('tie_word_embeddings', False)
        
        # Load embeddings (always numpy!)
        if 'embeddings_npy' in model_paths and model_paths['embeddings_npy']:
            # Untied: load from embeddings.npy
            self.embedding_weights = np.load(model_paths['embeddings_npy'])
            print(f"   ✅ Loaded embeddings.npy: {self.embedding_weights.shape}")
            
            # Load interpreter normally
            self.interpreter_session = ort.InferenceSession(str(model_paths['interpreter']))
            
        elif tie_word_embeddings:
            # Tied: extract lm_head from interpreter.onnx.data using offset
            print(f"   🔗 Extracting tied embeddings from interpreter...")
            
            lm_head_info = metadata.get('lm_head_tensor')
            if not lm_head_info:
                raise RuntimeError("Tied embeddings but no lm_head_tensor in metadata")
            
            # Read tensor directly from .data file using offset
            data_path = model_paths['interpreter'].parent / (model_paths['interpreter'].name + ".data")
            
            offset = lm_head_info['offset']
            length = lm_head_info['length']
            shape = tuple(lm_head_info['shape'])
            dtype = np.dtype(lm_head_info['dtype'])  # Get dtype from metadata
            
            with open(data_path, "rb") as f:
                f.seek(offset)
                lm_head_weights = np.frombuffer(f.read(length), dtype=dtype).reshape(shape)
            
            print(f"      Read lm_head at offset {offset}: {lm_head_weights.shape}")
            
            # Transpose: (hidden, vocab) -> (vocab, hidden)
            self.embedding_weights = lm_head_weights.T
            print(f"   ✅ Embeddings extracted: {self.embedding_weights.shape}")
            
            # Load interpreter
            self.interpreter_session = ort.InferenceSession(str(model_paths['interpreter']))
            
        else:
            raise RuntimeError("No embeddings available (not tied and no embeddings.npy)")
        
        # Image encoder is loaded lazily (only when first image is used)
        self.image_enc_session = None
        self._image_encoder_downloaded = False
        
        # Load tokenizer (using lightweight tokenizers library, not transformers!)
        from tokenizers import Tokenizer
        tokenizer_path = model_paths['tokenizer_path']
        
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        # Load special token IDs from metadata (same as training)
        import json
        metadata_path = model_paths['metadata_path']
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.separator_token_id = metadata['separator_token_id']
        self.eos_token_id = metadata['eos_token_id']
        
        print(f"✅ ONNX interpreter ready")
        print(f"   Separator token ID: {self.separator_token_id}")
        print(f"   EOS token ID: {self.eos_token_id}")
    
    def __call__(self, paw_file: str, *inputs, max_new_tokens: int = 128) -> str:
        """
        Run a .paw program with given inputs.
        
        Args:
            paw_file: Path to .paw file (or program ID for auto-download)
            *inputs: Variable arguments (text strings or PIL Images)
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Generated text output
        """
        # 1. Load .paw file (KV cache)
        # Handle both file paths and program IDs (uses local resolve_program_path)
        resolved_path = resolve_program_path(paw_file)
        
        from programasweights.paw_format import load_paw_program
        kv_layers, metadata = load_paw_program(resolved_path)
        
        # Convert to flat tensor format: [2*num_layers, 1, num_heads, prefix_len, head_dim]
        kv_flat_list = []
        for k, v in kv_layers:
            # Add batch dimension if needed
            if k.ndim == 3:
                k = np.expand_dims(k, axis=0)
                v = np.expand_dims(v, axis=0)
            
            kv_flat_list.append(k)
            kv_flat_list.append(v)
        
        past_kv = np.stack(kv_flat_list, axis=0)  # [2*num_layers, 1, num_heads, prefix_len, head_dim]
        
        # 2. Process inputs into embeddings
        input_embeddings_list = []
        
        for inp in inputs:
            if isinstance(inp, str):
                # Text input → tokenize → embed
                encoding = self.tokenizer.encode(inp, add_special_tokens=False)
                input_ids = np.array(encoding.ids, dtype=np.int64)  # [seq_len]
                
                # Always use numpy indexing (simple and fast!)
                text_emb = self.embedding_weights[input_ids]  # [seq_len, hidden]
                text_emb = np.expand_dims(text_emb, axis=0)  # [1, seq_len, hidden]
                input_embeddings_list.append(text_emb)
            
            elif PIL_AVAILABLE and isinstance(inp, Image.Image):
                # Image input → lazy load encoder if needed
                if self.image_enc_session is None and not self._image_encoder_downloaded:
                    if os.path.exists(self.model_name):
                        # Local path - check for image encoder directly
                        img_enc_path = Path(self.model_name) / "image_encoder.onnx"
                        if img_enc_path.exists():
                            print()
                            print("="*70)
                            print("🖼️  FIRST IMAGE DETECTED")
                            print("="*70)
                            print(f"Loading image encoder from {img_enc_path}")
                            print("="*70)
                            print()
                            
                            import onnxruntime as ort
                            self.image_enc_session = ort.InferenceSession(str(img_enc_path))
                            self._image_encoder_downloaded = True
                            
                            print("   ✅ Image encoder ready!")
                            print()
                        else:
                            print(f"   ⚠️  Image encoder not found at {img_enc_path}")
                            raise ValueError("Image encoder not available for this model")
                    else:
                    # Check if already cached
                        from huggingface_hub import hf_hub_download, try_to_load_from_cache
                        
                        cache_dir = get_onnx_models_cache_dir()
                        cached_path = try_to_load_from_cache(
                            repo_id=self.model_name,
                            filename="image_encoder.onnx.data",
                            cache_dir=cache_dir
                        )
                        
                        if cached_path is None:
                            # First time - show message
                            print()
                            print("="*70)
                            print("🖼️  FIRST IMAGE DETECTED")
                            print("="*70)
                            print("Downloading image encoder")
                            print("⏱️  This will take some time")
                            print("💡 Future image calls will be instant")
                            print("="*70)
                            print()
                        
                        # Download image encoder (instant if cached)
                        try:
                            img_enc_path = hf_hub_download(
                                repo_id=self.model_name,
                                filename="image_encoder.onnx",
                                cache_dir=cache_dir
                            )
                            hf_hub_download(
                                repo_id=self.model_name,
                                filename="image_encoder.onnx.data",
                                cache_dir=cache_dir
                            )
                            
                            import onnxruntime as ort
                            self.image_enc_session = ort.InferenceSession(img_enc_path)
                            self._image_encoder_downloaded = True
                            
                            if cached_path is None:
                                print()
                                print("   ✅ Image encoder ready!")
                                print()
                        except Exception as e:
                            print(f"   ❌ Failed to download image encoder: {e}")
                            raise ValueError("Image encoder not available for this model")
                
                if self.image_enc_session is None:
                    raise ValueError("Image encoder not available for this model")
                
                # Preprocess image (resize, normalize)
                from transformers import CLIPImageProcessor
                processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
                pixel_values = processor(inp, return_tensors="np")["pixel_values"]
                
                image_emb = self.image_enc_session.run(
                    ['image_embeddings'],
                    {'pixel_values': pixel_values}
                )[0]
                input_embeddings_list.append(image_emb)
            
            else:
                # Check if it's an image but Pillow not installed
                if not PIL_AVAILABLE and str(type(inp).__name__) in ['Image', 'JpegImageFile', 'PngImageFile']:
                    raise ImportError(
                        "Image input detected but Pillow not installed.\n"
                        "Install with: pip install programasweights[images]"
                    )
                raise ValueError(f"Unsupported input type: {type(inp)}")
        
        # Concatenate all input embeddings
        input_embeddings = np.concatenate(input_embeddings_list, axis=1)  # [1, total_seq, hidden]
        
        # Add separator token embedding (between input and output)
        separator_emb = self.embedding_weights[self.separator_token_id]  # [hidden]
        separator_emb = np.expand_dims(np.expand_dims(separator_emb, 0), 0)  # [1, 1, hidden]
        input_embeddings = np.concatenate([input_embeddings, separator_emb], axis=1)
        
        # 3. Generate output tokens autoregressively
        generated_ids = []
        
        for _ in range(max_new_tokens):
            # Run interpreter
            logits, past_kv = self.interpreter_session.run(
                ['logits', 'updated_kv'],
                {
                    'embeddings': input_embeddings,
                    'past_key_values': past_kv
                }
            )
            
            # Get next token (greedy decoding)
            next_token_id = np.argmax(logits[0, -1, :])
            
            # Check for EOS
            if next_token_id == self.eos_token_id:
                break
            
            generated_ids.append(int(next_token_id))
            
            # Prepare next input (embed the generated token)
            next_emb = self.embedding_weights[next_token_id]  # [hidden]
            input_embeddings = np.expand_dims(np.expand_dims(next_emb, 0), 0)  # [1, 1, hidden]
        
        # 4. Decode output
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output_text


# Global singleton
_onnx_interpreter = None

def function(
    path: str,
    *,
    max_new_tokens: int = 128,
    interpreter_name: str = "programasweights/interpreter-onnx",
    **kwargs  # Ignore other kwargs for compatibility
) -> Callable:
    """
    Load a .paw program and return a callable function (ONNX runtime).
    
    Args:
        path: Path to .paw file or program ID
        max_new_tokens: Maximum tokens to generate
        interpreter_name: Name of ONNX interpreter model
    
    Returns:
        Callable function that runs the program
    """
    global _onnx_interpreter
    
    # Lazy load interpreter (shared across all programs)
    if _onnx_interpreter is None:
        _onnx_interpreter = ONNXInterpreter(interpreter_name)
    
    def fn(*inputs):
        return _onnx_interpreter(path, *inputs, max_new_tokens=max_new_tokens)
    
    return fn

