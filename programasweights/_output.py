"""User-facing status messages for the PAW SDK.

All messages go to stderr so they don't pollute stdout when users pipe output.
Set PAW_QUIET=1 to suppress all status messages.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, TypedDict


class ProgressEvent(TypedDict, total=False):
    """Structured progress update emitted by preparation/download APIs."""

    stage: str
    status: str
    message: str
    program_id: str
    runtime_id: str
    path: str
    downloaded_bytes: int
    total_bytes: int


ProgressCallback = Callable[[ProgressEvent], None]


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


def report_progress(
    progress: ProgressCallback | None,
    event: ProgressEvent,
    fallback_message: str | None = None,
) -> None:
    """Send a structured event, or preserve the existing stderr fallback."""
    if progress is not None:
        progress(event)
    elif fallback_message is not None:
        status(fallback_message)
