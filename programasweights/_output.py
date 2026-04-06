"""User-facing status messages for the PAW SDK.

All messages go to stderr so they don't pollute stdout when users pipe output.
Set PAW_QUIET=1 to suppress all status messages.
"""
from __future__ import annotations

import os
import sys


def _quiet() -> bool:
    return os.environ.get("PAW_QUIET", "").strip() in ("1", "true", "yes")


def status(msg: str, **kwargs) -> None:
    """Print a status message to stderr."""
    if _quiet():
        return
    print(msg, file=sys.stderr, flush=True, **kwargs)


def status_inline(msg: str) -> None:
    """Print a status message inline (no newline) for progress updates."""
    if _quiet():
        return
    print(f"\r{msg}", end="", file=sys.stderr, flush=True)


def status_end() -> None:
    """End an inline status line."""
    if _quiet():
        return
    print(file=sys.stderr, flush=True)
