from __future__ import annotations

import json
import os
import sys
from typing import Optional, List, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from ..paw_format import save_paw_program

# Add project root for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils import MLPProgramPrefixMapper


def _unwrap(model):
    if hasattr(model, "module"):
        return model.module
    return model


def compiler_prompt(spec: str, style: str = "minimal") -> str:
    """Build compiler prompt (from eval script)."""
    if style == "minimal":
        return f"""[SPEC]
{spec}
[END_SPEC]

[PSEUDO_PROGRAM]""".strip()
    raise ValueError(f"Unknown compiler prompt style: {style}")


def interpreter_prompt(pseudo_program: str, task_input: str, style: str = "minimal") -> str:
    """Build interpreter prompt (from eval script)."""
    if style == "minimal":
        return f"""{pseudo_program.strip()}

[INPUT]
{task_input}
[END_INPUT]""".strip()
    raise ValueError(f"Unknown interpreter prompt style: {style}")


def compile(
    out_path: str,
    *,
    spec: Union[str, List],
    checkpoint_dir: str = "train_runs_stage2/compiler",
    prefix_steps: int = 64,
    compiler_prompt_style: str = "minimal",
    compiler_max_new_tokens: int = 512,
    prefix_mapper_fp32: bool = True,
    teacher_layer_align_type: str = "depth_ratio",
    interpreter_model_name: str = "Qwen/Qwen3-0.6B",
    input_images: Optional[List[str]] = None,
) -> str:
    """
    Compile a spec into a .paw neural program file.

    Two-phase compilation:
    1. Generate pseudo-program from spec using trained compiler
    2. Extract prefix KV cache from compiler hidden states via prefix mapper

    Args:
        out_path: Output path for the .paw file
        spec: Program specification text
        checkpoint_dir: Path to trained compiler directory (contains model + ../prefix_mapper.pt)
        prefix_steps: Number of prefix steps for KV cache
        compiler_prompt_style: Prompt style for compiler
        compiler_max_new_tokens: Max tokens for pseudo-program generation
        prefix_mapper_fp32: Use fp32 for prefix mapper (recommended)
        teacher_layer_align_type: How to align teacher/student layers
        interpreter_model_name: Interpreter model name (for metadata)
        input_images: Unused for now
    """
    if not out_path.endswith('.paw'):
        out_path = out_path + '.paw'

    # Device and dtype
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        dtype = torch.float32

    # =========================================================================
    # Phase 1: Generate pseudo-program
    # =========================================================================
    print(f"📝 Phase 1: Generating pseudo-program from spec...")

    # Load compiler model
    comp_tok = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    comp_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if device.type != "cuda":
        comp_model.to(device)
    comp_model.eval()

    if comp_tok.pad_token_id is None:
        comp_tok.pad_token = comp_tok.eos_token
    comp_tok.padding_side = "left"

    # Generate pseudo-program
    prompt = compiler_prompt(spec, style=compiler_prompt_style)
    messages = [{"role": "user", "content": prompt}]
    try:
        rendered = comp_tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        rendered = comp_tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    toks = comp_tok(rendered, return_tensors="pt").to(device)
    prompt_len = toks["input_ids"].shape[1]

    with torch.no_grad():
        outputs = comp_model.generate(
            **toks,
            max_new_tokens=compiler_max_new_tokens,
            do_sample=False,
            temperature=1.0,
            eos_token_id=comp_tok.eos_token_id,
            pad_token_id=comp_tok.pad_token_id or comp_tok.eos_token_id,
        )

    gen_ids = outputs[0, prompt_len:]
    eos_pos = (gen_ids == comp_tok.eos_token_id).nonzero(as_tuple=False)
    if eos_pos.numel() > 0:
        gen_ids = gen_ids[:eos_pos[0].item()]

    pseudo_program = comp_tok.decode(gen_ids, skip_special_tokens=True).strip()
    print(f"   Generated pseudo-program ({len(pseudo_program)} chars)")

    # =========================================================================
    # Phase 2: Extract prefix KV cache
    # =========================================================================
    print(f"🔧 Phase 2: Extracting prefix KV cache...")

    # Add prefix tokens
    prefix_tokens = [f"<prefix_{i}>" for i in range(1, prefix_steps + 1)]
    added = comp_tok.add_special_tokens({"additional_special_tokens": prefix_tokens})
    if added > 0:
        _unwrap(comp_model).resize_token_embeddings(len(comp_tok))

    prefix_token_ids = comp_tok.convert_tokens_to_ids(prefix_tokens)
    prefix_token_ids_t = torch.tensor(prefix_token_ids, device=device, dtype=torch.long)

    # Load interpreter config (needed for prefix mapper dimensions)
    int_tok = AutoTokenizer.from_pretrained(interpreter_model_name, trust_remote_code=True)
    int_model = AutoModelForCausalLM.from_pretrained(
        interpreter_model_name,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    int_cfg = _unwrap(int_model).config
    num_layers = int(int_cfg.num_hidden_layers)
    num_kv_heads = int(getattr(int_cfg, "num_key_value_heads", getattr(int_cfg, "num_attention_heads")))
    num_attn_heads = int(getattr(int_cfg, "num_attention_heads"))
    head_dim = int(getattr(int_cfg, "head_dim", int(int_cfg.hidden_size // num_attn_heads)))

    # Free interpreter model (only needed config)
    del int_model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Load prefix mapper
    parent_dir = os.path.dirname(checkpoint_dir.rstrip("/"))
    pm_path = os.path.join(parent_dir, "prefix_mapper.pt")

    if not os.path.isfile(pm_path):
        raise FileNotFoundError(f"prefix_mapper.pt not found at {pm_path}")

    prefix_mapper_dtype = torch.float32 if prefix_mapper_fp32 else dtype
    prefix_mapper = MLPProgramPrefixMapper(
        teacher_hidden_size=int(_unwrap(comp_model).config.hidden_size),
        student_num_layers=num_layers,
        student_num_kv_heads=num_kv_heads,
        student_head_dim=head_dim,
        prefix_steps=prefix_steps,
        use_kv_norm=True,
    ).to(device=device, dtype=prefix_mapper_dtype)

    pm_state = torch.load(pm_path, map_location=device, weights_only=True)
    prefix_mapper.load_state_dict(pm_state)
    prefix_mapper.eval()

    print(f"   Loaded prefix_mapper from {pm_path}")

    # Get prefix hidden states (from eval script's get_prefix_hidden_states)
    from main_no_spec_direct_ans_mix_continuous_sampleref_shorterprompt_vllm import get_prefix_hidden_states

    prefix_hidden = get_prefix_hidden_states(
        model=comp_model,
        tokenizer=comp_tok,
        prompts=[compiler_prompt(spec, style=compiler_prompt_style)],
        completions=[pseudo_program],
        prefix_token_ids=prefix_token_ids_t,
        prefix_steps=prefix_steps,
        device=device,
        dtype=dtype,
        teacher_layer_align_type=teacher_layer_align_type,
        num_student_layers=num_layers,
        use_chat_template=True,
    )

    # Run prefix mapper
    if prefix_mapper_fp32 and dtype != torch.float32:
        prefix_hidden_fp32 = [h.to(torch.float32) for h in prefix_hidden]
        kv_pairs = prefix_mapper(prefix_hidden_fp32)
        kv_pairs = [(k.to(dtype), v.to(dtype)) for k, v in kv_pairs]
    else:
        kv_pairs = prefix_mapper(prefix_hidden)

    # Convert to CPU for saving
    kv_layers = [(k.squeeze(0).cpu(), v.squeeze(0).cpu()) for k, v in kv_pairs]

    print(f"   KV cache: {len(kv_layers)} layers, prefix_len={kv_layers[0][0].shape[1]}")

    # =========================================================================
    # Save .paw file
    # =========================================================================
    save_paw_program(
        filepath=out_path,
        kv_layers=kv_layers,
        spec=spec,
        base_model=interpreter_model_name,
        prefix_steps=prefix_steps,
        pseudo_program=pseudo_program,
        source="compiled",
        source_info={
            "compiler_model": checkpoint_dir,
            "compiler_prompt_style": compiler_prompt_style,
            "prefix_steps": prefix_steps,
            "teacher_layer_align_type": teacher_layer_align_type,
            "forkable": True,
        },
    )

    print(f"✅ Compiled to {out_path}")
    return out_path


def compile_dummy(*args, **kwargs):
    """Placeholder for compile_dummy."""
    from .dummy import compile_dummy as _compile_dummy
    return _compile_dummy(*args, **kwargs)
