"""Typed input snapshots, independent of model loading and preprocessing."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
import stat
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from PIL.Image import Image as _PILImage


def _read_file(path: Union[str, os.PathLike]) -> bytes:
    # Nonblocking open lets us reject FIFOs without waiting for a writer.
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Image source must be a regular local file.")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            return source.read()
    finally:
        os.close(descriptor)


def _copy_pixels(source):
    snapshot = source.copy()
    # Pillow's copy() copies pixels and palette, but only shallow-copies info.
    snapshot.info = deepcopy(source.info)
    return snapshot


@dataclass(frozen=True, init=False, repr=False, eq=False)
class Image:
    """An image to pass to a local image function.

    Accepts a file path, encoded image bytes, or a Pillow image. Pass it as a
    positional argument to the function's numbered prompt template. For example,
    when the template uses {INPUT_0} for the question and {INPUT_1} for an image:
    ``fn("Describe this image.", paw.Image("photo.png"))``.
    """

    _source: Union[bytes, _PILImage]

    def __init__(self, source: Union[str, os.PathLike, bytes, _PILImage]):
        if isinstance(source, bytes):
            snapshot = bytes(source)
        elif isinstance(source, (str, os.PathLike)):
            path = os.fspath(source)
            if not isinstance(path, str):
                raise TypeError("Image paths must resolve to str; bytes are encoded image data.")
            snapshot = _read_file(path)
        else:
            # A PIL object normally means Pillow is already imported. Keep it
            # optional for text users and for callers supplying paths or bytes.
            try:
                from PIL.Image import Image as PILImage
            except ModuleNotFoundError as error:
                if error.name not in ("PIL", "PIL.Image"):
                    raise
                PILImage = None
            if PILImage is None or not isinstance(source, PILImage):
                raise TypeError(
                    "Image source must be a local path, encoded bytes, or a PIL.Image.Image."
                )
            snapshot = _copy_pixels(source)
        object.__setattr__(self, "_source", snapshot)

    def _copy_source(self):
        """Give preprocessing its own copy without exposing mutable pixels."""
        if isinstance(self._source, bytes):
            return self._source
        return _copy_pixels(self._source)

    def __repr__(self) -> str:
        kind = "encoded bytes" if isinstance(self._source, bytes) else "PIL image"
        return f"Image(<{kind} snapshot>)"


def _normalize_parts(*parts: Union[str, Image]) -> Tuple[Union[str, Image], ...]:
    """Validate positional content without merging text or interpreting paths."""
    if not parts:
        raise ValueError("At least one input part is required.")
    for index, part in enumerate(parts, start=1):
        if not isinstance(part, (str, Image)):
            raise TypeError(
                f"Input part {index} must be str or paw.Image, got {type(part).__name__}. "
                "Wrap image sources in paw.Image(...); unpack a list with *parts."
            )
    return parts
