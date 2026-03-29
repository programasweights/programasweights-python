from __future__ import annotations

import random
from typing import Optional, List


def compile_dummy(out_path: str, spec: Optional[str] = None, *, seed: Optional[int] = None, num_tokens: int = 32, input_images: Optional[List[str]] = None) -> str:
    """
    Dummy compiler that writes a random textual prefix to out_path.

    - Ignores `spec` (placeholder for future real compiler)
    - Ignores `input_images` (placeholder for future multimodal support)
    - If `seed` is provided, generation is deterministic
    - Resulting file is a UTF-8 text file that the runtime uses as a prompt/prefix
    """
    # TODO: Process input_images for multimodal compilation (placeholder)
    if input_images:
        print(f"Dummy compiler: Ignoring {len(input_images)} input images")
    rng = random.Random(seed)
    tokens = [f"[P{rng.randint(0, 9999):04d}]" for _ in range(num_tokens)]
    prefix = " ".join(tokens)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(prefix)
    return out_path 