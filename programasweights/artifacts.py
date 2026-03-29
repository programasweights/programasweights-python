from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ProgramArtifact:
    kind: str  # e.g., "prefix_text", "prefix_tuning", "lora", "prefix_kv"
    base_model: Optional[str]
    path: str  # directory for manifest-based artifacts; file path for text prefix
    metadata: dict


def load_artifact(path: str) -> ProgramArtifact:
    """
    Load a program artifact which can be either:
    - A directory containing a manifest 'program.json' (preferred)
    - A single text file treated as a prompt/prefix (MVP fallback)
    """
    if os.path.isdir(path):
        manifest_path = os.path.join(path, "program.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Missing program.json in {path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        kind = m.get("kind", "prefix_text")
        base_model = m.get("base_model")
        return ProgramArtifact(kind=kind, base_model=base_model, path=path, metadata=m)

    if os.path.isfile(path):
        # If there is a sibling manifest, prefer it
        manifest_path = os.path.join(os.path.dirname(path), "program.json")
        base_model = None
        metadata = {}
        if os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    base_model = m.get("base_model")
                    metadata = m
                    kind = m.get("kind", "prefix_text")
                    return ProgramArtifact(kind=kind, base_model=base_model, path=os.path.dirname(path), metadata=metadata)
            except Exception:
                pass
        # Fallback: treat the file itself as a text prefix
        return ProgramArtifact(kind="prefix_text", base_model=base_model, path=path, metadata=metadata)

    raise FileNotFoundError(path) 