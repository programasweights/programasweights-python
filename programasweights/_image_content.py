"""Convert typed input snapshots to local llama.cpp chat content."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict, List, Union

from ._inputs import Image, _normalize_parts

# Restrict byte decoding to raster formats; other Pillow plugins may invoke
# external programs. PIL objects have already been decoded by the caller.
_RASTER_FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "TIFF", "BMP", "ICO")
_PIXEL_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr", "HSV"})


def _rgb_on_white(pixels):
    from PIL import Image as PILImage

    palette_has_alpha = pixels.palette is not None and pixels.palette.mode == "RGBA"
    if ("A" not in pixels.getbands() and "transparency" not in pixels.info
            and not palette_has_alpha):
        return pixels.convert("RGB")
    # Follow qwen-vl-utils' white-background compositing for RGBA, also
    # normalizing grayscale and palette/color-key transparency to RGBA first.
    with pixels.convert("RGBA") as rgba, rgba.getchannel("A") as alpha:
        background = PILImage.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=alpha)
        return background


def _png_url(pixels) -> str:
    if pixels.width < 1 or pixels.height < 1:
        raise ValueError("Image dimensions must both be positive.")
    if pixels.mode not in _PIXEL_MODES:
        raise ValueError(
            f"Unsupported image mode {pixels.mode!r}; explicitly convert to an "
            "8-bit image before wrapping it in paw.Image."
        )
    # Resolve transparency before encoding the model's three RGB channels.
    with _rgb_on_white(pixels) as rgb, BytesIO() as output:
        # Model input contains pixels only. In particular, don't attach EXIF
        # orientation or caller metadata to the newly encoded pixel grid.
        rgb.info.clear()
        rgb.save(output, format="PNG")
        payload = base64.b64encode(output.getvalue()).decode("ascii")
    return "data:image/png;base64," + payload


def _image_url(image: Image, index: int) -> str:
    try:
        from PIL import Image as PILImage
    except ModuleNotFoundError as error:
        if error.name not in ("PIL", "PIL.Image"):
            raise
        raise ImportError(
            "Image conversion requires Pillow; install programasweights[vision]."
        ) from error

    source = image._copy_source()
    try:
        if isinstance(source, bytes):
            # verify() checks container integrity (including PNG checksums).
            # Reopen afterwards because verification consumes the decoder.
            with BytesIO(source) as stream:
                with PILImage.open(stream, formats=_RASTER_FORMATS) as pixels:
                    pixels.verify()
            with BytesIO(source) as stream:
                with PILImage.open(stream, formats=_RASTER_FORMATS) as pixels:
                    return _png_url(pixels)
        with source:
            return _png_url(source)
    except (OSError, ValueError, SyntaxError, PILImage.DecompressionBombError) as error:
        raise ValueError(
            f"Input part {index}: image could not be decoded or encoded. {error}"
        ) from error


def _to_chat_content(*parts: Union[str, Image]) -> List[Dict[str, Any]]:
    """Prepare ordered local text/image parts without loading an inference backend.

    Text is passed through exactly, including empty strings and adjacent text
    parts. Images become RGB PNG data URLs at their native dimensions.
    Transparency is composited onto white, including alpha channels and
    palette/color-key transparency. No EXIF rotation or color-profile transform
    is applied. Metadata is omitted from the generated PNG.

    Encoded PNG, JPEG, WebP, GIF, TIFF, BMP, and ICO inputs are decoded locally.
    Encoded animations use their first frame; PIL inputs use the frame already
    snapshotted by Image. Floating-point and high-bit-depth pixel modes must be
    explicitly converted by the caller rather than silently clipped here.

    The supplied snapshots are unchanged. Pillow is imported only when an
    image is converted. This helper does not alter text-function call behavior
    or perform model-specific resizing, tokenization, or inference.
    """
    normalized = _normalize_parts(*parts)
    content = []
    for index, part in enumerate(normalized, start=1):
        if isinstance(part, str):
            content.append({"type": "text", "text": part})
        else:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_url(part, index)},
            })
    return content
