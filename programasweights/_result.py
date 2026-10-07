"""Per-call inference results without mutable state on a function."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional


@dataclass(frozen=True)
class FunctionResult:
    """Text, token counts, and timing returned with ``return_info=True``.

    ``usage`` is a read-only mapping of token counts, or None when unavailable.
    ``elapsed_seconds`` is the total call duration, excluding model loading.
    """

    text: str
    finish_reason: Optional[str]
    usage: Optional[Mapping[str, int]]
    elapsed_seconds: float

    def __post_init__(self):
        if self.usage is not None:
            object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))

    def __reduce__(self):
        return (
            type(self),
            (
                self.text,
                self.finish_reason,
                None if self.usage is None else dict(self.usage),
                self.elapsed_seconds,
            ),
        )
