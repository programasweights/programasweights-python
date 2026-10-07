"""Image snapshots and ordered content do not depend on an inference backend."""

from copy import copy, deepcopy
from dataclasses import FrozenInstanceError
from io import BytesIO
import os
from pathlib import Path
import subprocess
import sys

import pytest

import programasweights as paw
from programasweights._inputs import _normalize_parts
from programasweights import _inputs


@pytest.fixture
def pil():
    return pytest.importorskip("PIL.Image")


@pytest.mark.parametrize("as_string", [False, True])
def test_file_is_snapshotted_before_replacement_and_deletion(tmp_path, as_string):
    source = tmp_path / "frame.png"
    source.write_bytes(b"original encoded data")
    image = paw.Image(str(source) if as_string else source)
    source.write_bytes(b"replacement encoded data")
    assert image._copy_source() == b"original encoded data"
    source.unlink()
    assert image._copy_source() == b"original encoded data"


def test_custom_text_pathlike_is_supported(tmp_path):
    source = tmp_path / "frame.png"
    source.write_bytes(b"image bytes")

    class SourcePath:
        def __fspath__(self):
            return str(source)

    assert paw.Image(SourcePath())._copy_source() == b"image bytes"


def test_binary_pathlike_is_not_confused_with_encoded_bytes():
    class BinaryPath:
        def __fspath__(self):
            return b"frame.png"

    with pytest.raises(TypeError, match="paths must resolve to str"):
        paw.Image(BinaryPath())


@pytest.mark.parametrize("payload", [b"", b"not decoded yet", b"frame.png"])
def test_bytes_are_retained_without_decoding_or_path_interpretation(payload):
    assert paw.Image(payload)._copy_source() == payload


def test_missing_file_fails_at_construction(tmp_path):
    with pytest.raises(FileNotFoundError):
        paw.Image(tmp_path / "missing.png")


def test_directory_is_rejected(tmp_path):
    # os.open/fdopen reject directories differently across platforms.
    with pytest.raises((OSError, ValueError)):
        paw.Image(tmp_path)


@pytest.mark.parametrize("fail_read", [False, True])
def test_file_descriptor_is_closed_on_success_and_failure(tmp_path, monkeypatch, fail_read):
    source = tmp_path / "frame.png"
    source.write_bytes(b"encoded data")
    descriptors = []
    real_open = os.open

    def record_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(_inputs.os, "open", record_open)
    if fail_read:
        def failed_stream(*args, **kwargs):
            raise OSError("Cannot create stream")

        monkeypatch.setattr(_inputs.os, "fdopen", failed_stream)
        with pytest.raises(OSError, match="Cannot create stream"):
            paw.Image(source)
    else:
        assert paw.Image(source)._copy_source() == b"encoded data"
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO test requires POSIX")
def test_fifo_is_rejected_without_waiting_for_a_writer(tmp_path):
    fifo = tmp_path / "not-an-image"
    os.mkfifo(fifo)
    result = subprocess.run(
        [sys.executable, "-c", "import programasweights as paw; paw.Image(__import__('sys').argv[1])", str(fifo)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "regular local file" in result.stderr


@pytest.mark.parametrize("mode,color", [
    ("RGB", (12, 34, 56)), ("RGBA", (12, 34, 56, 78)),
    ("L", 17), ("CMYK", (12, 34, 56, 78)), ("F", 0.5),
])
def test_pil_pixels_mode_dimensions_and_metadata_are_snapshotted(pil, mode, color):
    source = pil.new(mode, (7, 5), color)
    source.info["exif"] = b"exif snapshot"
    source.info["nested"] = {"values": [1, 2]}
    expected = source.tobytes()
    image = paw.Image(source)
    source.paste(0, (0, 0, 7, 5))
    source.info["nested"]["values"].append(3)
    source.close()
    snapshot = image._copy_source()
    assert snapshot.mode == mode
    assert snapshot.size == (7, 5)
    assert snapshot.tobytes() == expected
    assert snapshot.info == {"exif": b"exif snapshot", "nested": {"values": [1, 2]}}
    # A consumer may mutate its copy without changing subsequent calls.
    snapshot.paste(0, (0, 0, 7, 5))
    snapshot.info["nested"]["values"].clear()
    again = image._copy_source()
    assert again.tobytes() == expected
    assert again.info["nested"]["values"] == [1, 2]


def test_palette_is_snapshotted(pil):
    source = pil.new("P", (2, 2), 0)
    source.putpalette([255, 0, 0] + [0] * 765)
    image = paw.Image(source)
    source.putpalette([0, 0, 255] + [0] * 765)
    assert image._copy_source().getpalette()[:3] == [255, 0, 0]


def test_lazy_pil_image_survives_closed_input_stream(pil):
    buffer = BytesIO()
    pil.new("RGB", (7, 5), "red").save(buffer, format="PNG")
    with BytesIO(buffer.getvalue()) as stream:
        with pil.open(stream) as source:
            image = paw.Image(source)
    assert image._copy_source().getpixel((0, 0)) == (255, 0, 0)


def test_pil_snapshot_uses_current_animation_frame(pil):
    buffer = BytesIO()
    first = pil.new("RGB", (4, 4), "red")
    first.save(buffer, format="GIF", save_all=True,
               append_images=[pil.new("RGB", (4, 4), "blue")], duration=100)
    with pil.open(BytesIO(buffer.getvalue())) as source:
        source.seek(1)
        expected = source.copy()
        image = paw.Image(source)
        source.seek(0)
    snapshot = image._copy_source()
    assert snapshot.mode == expected.mode
    assert snapshot.tobytes() == expected.tobytes()


@pytest.mark.parametrize("source", [None, 42, True, bytearray(b"image"), memoryview(b"image"), [], {}, BytesIO(b"image")])
def test_unsupported_image_sources_fail_clearly(source):
    with pytest.raises(TypeError, match="Image source must be"):
        paw.Image(source)


def test_image_wrapper_cannot_be_reassigned_and_repr_omits_payload():
    image = paw.Image(b"private image content")
    with pytest.raises(FrozenInstanceError):
        image._source = b"replacement"
    assert "private image content" not in repr(image)
    assert "snapshot" in repr(image)


@pytest.mark.parametrize("copy_value", [copy, deepcopy])
def test_image_value_can_be_copied(copy_value):
    image = paw.Image(b"encoded data")
    copied = copy_value(image)
    assert copied._copy_source() == b"encoded data"
    with pytest.raises(FrozenInstanceError):
        copied._source = b"replacement"


def test_parts_preserve_interleaving_empty_strings_and_exact_text():
    before, after = paw.Image(b"before"), paw.Image(b"after")
    parts = ("Before:\n", before, "", "After:", "\t é\n", after, after)
    result = _normalize_parts(*parts)
    assert result == parts
    assert result[1] is before
    assert result[-1] is after


def test_plain_strings_are_never_opened_or_interpreted_as_images():
    parts = ("nonexistent.png", "https://example.invalid/image.png", "", "second text")
    assert _normalize_parts(*parts) == parts


def test_dynamic_input_list_must_be_unpacked():
    parts = ["Frame:", paw.Image(b"image")]
    assert _normalize_parts(*parts) == tuple(parts)
    with pytest.raises(TypeError, match=r"unpack a list with \*parts"):
        _normalize_parts(parts)


@pytest.mark.parametrize("part", [None, 1, False, b"image", Path("frame.png"), [], ["nested"], ("nested",), {}])
def test_invalid_content_reports_argument_position(part):
    with pytest.raises(TypeError, match="Input part 2 must be str or paw.Image"):
        _normalize_parts("first", part)


def test_zero_parts_are_rejected():
    with pytest.raises(ValueError, match="At least one input part"):
        _normalize_parts()


def test_import_and_encoded_inputs_do_not_import_heavy_optional_modules(tmp_path):
    source = tmp_path / "frame.png"
    source.write_bytes(b"encoded data")
    code = r'''
import sys
import importlib.abc
class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('PIL', 'numpy', 'llama_cpp', 'torch'):
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockHeavyImports())
import programasweights as paw
from programasweights._inputs import _normalize_parts
assert paw.Image(b'bytes')._copy_source() == b'bytes'
assert paw.Image(sys.argv[1])._copy_source() == b'encoded data'
assert _normalize_parts('text') == ('text',)
'''
    result = subprocess.run([sys.executable, "-c", code, str(source)],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_invalid_source_has_clear_error_without_pillow():
    code = r'''
import sys
import importlib.abc
class NoPillow(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'PIL':
            raise ModuleNotFoundError('No Pillow installed', name='PIL')
sys.meta_path.insert(0, NoPillow())
import programasweights as paw
try:
    paw.Image(object())
except TypeError as error:
    assert 'Image source must be' in str(error)
else:
    raise AssertionError('Invalid source accepted')
'''
    result = subprocess.run([sys.executable, "-c", code],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
