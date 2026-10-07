"""Per-call inference results without mutable state on a function."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional


@dataclass(frozen=True)
class FunctionResult:
    """Text and optional backend metadata from one successful image-runtime call.

    ``usage`` is a copied, read-only mapping of backend token counts, or None
    when unavailable. ``elapsed_seconds`` covers the call through native cleanup,
    including input preparation and lock waits, but not function loading.
    """

    text: str
    finish_reason: Optional[str]
    usage: Optional[Mapping[str, int]]
    elapsed_seconds: float

    def __post_init__(self):
        if self.usage is not None:
            object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))
