"""Decode emitted model inputs to verify pixels, order, and isolation."""

import base64
from io import BytesIO
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from programasweights import Image
from programasweights._image_content import _to_chat_content


@pytest.fixture
def pil():
    return pytest.importorskip("PIL.Image")


def encoded(pixels, format="PNG", **options):
    with BytesIO() as stream:
        pixels.save(stream, format=format, **options)
        return stream.getvalue()


def decoded(part, pil):
    assert set(part) == {"type", "image_url"}
    assert part["type"] == "image_url"
    assert set(part["image_url"]) == {"url"}
    header, payload = part["image_url"]["url"].split(",", 1)
    assert header == "data:image/png;base64"
    with BytesIO(base64.b64decode(payload, validate=True)) as stream:
        with pil.open(stream) as pixels:
            assert pixels.format == "PNG"
            pixels.load()
            return pixels.copy()


@pytest.mark.parametrize("size", [(1, 1), (7, 5), (166, 164), (1025, 3)])
def test_preserves_native_dimensions_and_rgb_pixels(pil, size):
    source = pil.new("RGB", size, (12, 34, 56))
    source.putpixel((size[0] - 1, size[1] - 1), (210, 180, 70))
    result = decoded(_to_chat_content(Image(encoded(source)))[0], pil)
    assert result.mode == "RGB"
    assert result.size == size
    assert result.tobytes() == source.tobytes()


def test_preserves_text_and_image_order(pil):
    first = Image(pil.new("RGB", (3, 2), "red"))
    second = Image(pil.new("RGB", (2, 3), "blue"))
    result = _to_chat_content("Before:\n", first, "", "After:", "\t é\n", second, first)
    assert result[0] == {"type": "text", "text": "Before:\n"}
    assert result[2:5] == [
        {"type": "text", "text": ""},
        {"type": "text", "text": "After:"},
        {"type": "text", "text": "\t é\n"},
    ]
    for index, color, size in [(1, (255, 0, 0), (3, 2)), (5, (0, 0, 255), (2, 3)),
                               (6, (255, 0, 0), (3, 2))]:
        pixels = decoded(result[index], pil)
        assert pixels.getpixel((0, 0)) == color
        assert pixels.size == size


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP", "GIF", "TIFF", "BMP", "ICO"])
def test_encoded_raster_formats_use_decoded_pixels_without_resize(pil, format):
    source = pil.new("RGB", (32, 32), (51, 102, 153))
    source.putpixel((1, 2), (200, 50, 90))
    payload = encoded(source, format, **({"sizes": [(32, 32)]} if format == "ICO" else {}))
    with pil.open(BytesIO(payload)) as opened:
        expected = opened.convert("RGB")
    result = decoded(_to_chat_content(Image(payload))[0], pil)
    assert result.size == expected.size
    assert result.tobytes() == expected.tobytes()


@pytest.mark.parametrize("mode,color,expected", [
    ("1", 1, (255, 255, 255)),
    ("L", 17, (17, 17, 17)),
    ("LA", (17, 0), (255, 255, 255)),
    ("RGB", (12, 34, 56), (12, 34, 56)),
    ("RGBA", (12, 34, 56, 0), (255, 255, 255)),
    ("CMYK", (0, 255, 255, 0), (255, 0, 0)),
    ("YCbCr", (128, 128, 128), (128, 128, 128)),
    ("HSV", (0, 255, 255), (255, 0, 0)),
])
def test_pixel_modes_follow_explicit_rgb_and_alpha_policy(pil, mode, color, expected):
    result = decoded(_to_chat_content(Image(pil.new(mode, (7, 5), color)))[0], pil)
    assert result.mode == "RGB"
    assert result.size == (7, 5)
    assert result.getpixel((0, 0)) == expected


@pytest.mark.parametrize("kind", ["pil", "bytes", "path"])
@pytest.mark.parametrize("mode,pixels,expected", [
    ("RGBA", [(255, 0, 0, 0), (255, 0, 0, 128), (12, 34, 56, 255)],
     [(255, 255, 255), (255, 127, 127), (12, 34, 56)]),
    ("LA", [(17, 0), (17, 128), (17, 255)],
     [(255, 255, 255), (136, 136, 136), (17, 17, 17)]),
])
def test_alpha_composites_onto_white(pil, tmp_path, kind, mode, pixels, expected):
    source = pil.new(mode, (3, 1))
    source.putdata(pixels)
    original = source.tobytes()
    if kind == "pil":
        value = source
    elif kind == "bytes":
        value = encoded(source)
    else:
        value = tmp_path / "transparent.png"
        source.save(value)
    image = Image(value)
    first = decoded(_to_chat_content(image)[0], pil)
    second = decoded(_to_chat_content(image)[0], pil)
    assert first.mode == "RGB"
    assert first.size == (3, 1)
    assert [first.getpixel((x, 0)) for x in range(3)] == expected
    assert second.tobytes() == first.tobytes()
    assert first.info == {}
    assert source.mode == mode
    assert source.tobytes() == original
    if kind == "pil":
        with image._copy_source() as snapshot:
            assert snapshot.mode == mode
            assert snapshot.tobytes() == original


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("transparency,expected", [
    (0, [(255, 255, 255), (255, 0, 0), (12, 34, 56)]),
    (bytes([0, 128, 255]), [(255, 255, 255), (255, 127, 127), (12, 34, 56)]),
])
def test_palette_transparency_composites_onto_white(pil, as_bytes, transparency, expected):
    source = pil.new("P", (3, 1))
    source.putpalette([0, 255, 0, 255, 0, 0, 12, 34, 56] + [0] * 759)
    source.putdata([0, 1, 2])
    source.info["transparency"] = transparency
    image = Image(encoded(source) if as_bytes else source)
    result = decoded(_to_chat_content(image)[0], pil)
    assert [result.getpixel((x, 0)) for x in range(3)] == expected
    assert result.info == {}
    assert source.info["transparency"] == transparency
    if not as_bytes:
        with image._copy_source() as snapshot:
            assert snapshot.info["transparency"] == transparency
            assert snapshot.tobytes() == source.tobytes()


@pytest.mark.parametrize("as_bytes", [False, True])
def test_rgba_palette_composites_without_transparency_metadata(pil, as_bytes):
    source = pil.new("P", (3, 1))
    source.putpalette([0, 255, 0, 0, 255, 0, 0, 128, 12, 34, 56, 255]
                      + [0, 0, 0, 255] * 253, rawmode="RGBA")
    source.putdata([0, 1, 2])
    assert "transparency" not in source.info
    result = decoded(_to_chat_content(Image(encoded(source) if as_bytes else source))[0], pil)
    assert [result.getpixel((x, 0)) for x in range(3)] == [
        (255, 255, 255), (255, 127, 127), (12, 34, 56),
    ]


@pytest.mark.parametrize("as_bytes", [False, True])
@pytest.mark.parametrize("mode,key,opaque,expected", [
    ("RGB", (12, 34, 56), (12, 34, 57), (12, 34, 57)),
    ("L", 17, 18, (18, 18, 18)),
])
def test_color_key_transparency_composites_onto_white(pil, as_bytes, mode, key, opaque, expected):
    source = pil.new(mode, (2, 1), key)
    source.putpixel((1, 0), opaque)
    source.info["transparency"] = key
    result = decoded(_to_chat_content(Image(encoded(source) if as_bytes else source))[0], pil)
    assert result.getpixel((0, 0)) == (255, 255, 255)
    assert result.getpixel((1, 0)) == expected
    assert result.info == {}


def test_preserves_snapshot_and_omits_metadata_without_exif_rotation(pil):
    source = pil.new("RGB", (7, 5), "red")
    source.putpixel((6, 0), (0, 0, 255))
    exif = pil.Exif()
    exif[274] = 6  # Rotate 90 degrees when interpreted by an EXIF-aware viewer.
    source.info["exif"] = exif.tobytes()
    source.info["icc_profile"] = b"caller color profile"
    source.info["private"] = {"values": ["caller metadata"]}
    image = Image(source)
    expected = source.tobytes()
    source.paste("green", (0, 0, 7, 5))
    source.close()
    result = decoded(_to_chat_content(image)[0], pil)
    assert result.size == (7, 5)
    assert result.tobytes() == expected
    assert result.info == {}
    snapshot = image._copy_source()
    assert snapshot.tobytes() == expected
    assert snapshot.info["exif"] == exif.tobytes()
    assert snapshot.info["private"] == {"values": ["caller metadata"]}


def test_file_snapshot_can_be_converted_after_file_removal(pil, tmp_path):
    path = tmp_path / "source.png"
    pil.new("RGB", (7, 5), "blue").save(path)
    image = Image(path)
    path.unlink()
    assert decoded(_to_chat_content(image)[0], pil).getpixel((0, 0)) == (0, 0, 255)


def test_encoded_exif_does_not_rotate_pixels_or_survive_conversion(pil):
    source = pil.new("RGB", (7, 5), "red")
    source.putpixel((6, 0), (0, 0, 255))
    exif = pil.Exif()
    exif[274] = 6
    payload = encoded(source, exif=exif.tobytes(), icc_profile=b"caller profile")
    result = decoded(_to_chat_content(Image(payload))[0], pil)
    assert result.size == source.size
    assert result.tobytes() == source.tobytes()
    assert result.info == {}


def test_encoded_animation_uses_first_frame_and_pil_uses_snapshotted_frame(pil):
    first = pil.new("RGB", (7, 5), "red")
    payload = encoded(first, "GIF", save_all=True,
                      append_images=[pil.new("RGB", (7, 5), "blue")], duration=100)
    with pil.open(BytesIO(payload)) as source:
        source.seek(1)
        second = Image(source)
        source.seek(0)
    a, b = _to_chat_content(Image(payload), second)
    assert decoded(a, pil).getpixel((0, 0)) == (255, 0, 0)
    assert decoded(b, pil).getpixel((0, 0)) == (0, 0, 255)


@pytest.mark.parametrize("mode", ["I", "F", "I;16"])
def test_high_bit_depth_requires_explicit_conversion(pil, mode):
    with pytest.raises(ValueError, match="Input part 2:.*explicitly convert"):
        _to_chat_content("Frame:", Image(pil.new(mode, (7, 5))))


@pytest.mark.parametrize("size", [(0, 1), (1, 0), (0, 0)])
def test_empty_pixel_grid_has_clear_error(pil, size):
    with pytest.raises(ValueError, match="Input part 2:.*dimensions must both be positive"):
        _to_chat_content("Frame:", Image(pil.new("RGB", size)))


@pytest.mark.parametrize("payload", [b"", b"not an image", b"https://example.invalid/image.png"])
def test_invalid_bytes_report_part_index(pil, payload):
    with pytest.raises(ValueError, match="Input part 2: image could not be decoded or encoded"):
        _to_chat_content("Frame:", Image(payload))


@pytest.mark.parametrize("format", ["PNG", "JPEG"])
def test_truncated_image_fails_during_conversion(pil, format):
    payload = encoded(pil.new("RGB", (32, 32), "red"), format)
    with pytest.raises(ValueError, match="Input part 1: image could not be decoded or encoded"):
        _to_chat_content(Image(payload[:len(payload) // 2]))


def test_png_with_corrupt_checksum_is_rejected(pil):
    payload = bytearray(encoded(pil.new("RGB", (7, 5), "red")))
    offset = 8
    while offset < len(payload):
        length = struct.unpack_from(">I", payload, offset)[0]
        if payload[offset + 4:offset + 8] == b"IDAT":
            payload[offset + 8 + length] ^= 1
            break
        offset += 12 + length
    else:
        raise AssertionError("Fixture contains no IDAT chunk")
    with pytest.raises(ValueError, match="Input part 1: image could not be decoded or encoded"):
        _to_chat_content(Image(bytes(payload)))


def test_conversion_obeys_pillow_decompression_limit(pil, monkeypatch):
    payload = encoded(pil.new("RGB", (10, 10), "red"))
    monkeypatch.setattr(pil, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="Input part 1: image could not be decoded or encoded"):
        _to_chat_content(Image(payload))
    assert pil.MAX_IMAGE_PIXELS == 10


def test_returned_content_is_independent_between_calls(pil):
    image = Image(pil.new("RGB", (7, 5), "red"))
    first = _to_chat_content("Frame:", image)
    first[0]["text"] = "changed"
    first[1]["image_url"]["url"] = "changed"
    second = _to_chat_content("Frame:", image)
    assert second[0]["text"] == "Frame:"
    assert decoded(second[1], pil).getpixel((0, 0)) == (255, 0, 0)


def run_isolated(code):
    result = subprocess.run([sys.executable, "-c", code],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_text_and_invalid_arguments_do_not_import_pillow_or_backend():
    run_isolated(r'''
import sys, importlib.abc
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('PIL', 'llama_cpp', 'numpy', 'torch'):
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockImports())
from programasweights import Image
from programasweights._image_content import _to_chat_content
parts = ('', 'https://example.invalid/image.png', '\t é\n', 'frame.png')
assert _to_chat_content(*parts) == [{'type': 'text', 'text': p} for p in parts]
for parts in [(), (['text'],), (Image(b'invalid'), 3)]:
    try:
        _to_chat_content(*parts)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError('Invalid arguments accepted')
''')


def test_missing_pillow_has_actionable_error():
    run_isolated(r'''
import sys, importlib.abc
class NoPillow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'PIL':
            raise ModuleNotFoundError('No Pillow', name='PIL')
sys.meta_path.insert(0, NoPillow())
from programasweights import Image
from programasweights._image_content import _to_chat_content
try:
    _to_chat_content(Image(b'bytes'))
except ImportError as error:
    assert 'programasweights[vision]' in str(error)
else:
    raise AssertionError('Missing Pillow accepted')
''')


def test_broken_pillow_import_is_not_misreported_as_missing_extra():
    run_isolated(r'''
import sys, importlib.abc
class BrokenPillow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'PIL':
            raise ModuleNotFoundError('Missing codec dependency', name='broken_codec')
sys.meta_path.insert(0, BrokenPillow())
from programasweights import Image
from programasweights._image_content import _to_chat_content
try:
    _to_chat_content(Image(b'bytes'))
except ModuleNotFoundError as error:
    assert error.name == 'broken_codec'
else:
    raise AssertionError('Broken import hidden')
''')


def test_image_conversion_is_local_and_does_not_load_inference_backend(pil):
    payload = encoded(pil.new("RGB", (7, 5), "red"))
    run_isolated(r'''
import sys, importlib.abc, base64
class BlockBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'numpy':
            raise ModuleNotFoundError('NumPy is unavailable', name=fullname)
        if fullname.split('.')[0] in ('llama_cpp', 'torch'):
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockBackend())
def offline(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto',
                 'subprocess.Popen', 'os.system', 'os.posix_spawn'):
        raise AssertionError('Unexpected external operation: ' + event)
sys.addaudithook(offline)
from programasweights import Image
from programasweights._image_content import _to_chat_content
''' + f'''
payload = base64.b64decode({base64.b64encode(payload).decode('ascii')!r})
assert _to_chat_content(Image(payload))[0]['type'] == 'image_url'
# EPS must not reach Pillow's Ghostscript-backed rasterizer.
eps = b'%!PS-Adobe-3.0 EPSF-3.0\\n%%BoundingBox: 0 0 7 5\\n%%EndComments\\nshowpage\\n'
try:
    _to_chat_content(Image(eps))
except ValueError:
    pass
else:
    raise AssertionError('Unsupported vector input accepted')
assert not any(name.split('.')[0] == 'numpy' for name in sys.modules)
''')
