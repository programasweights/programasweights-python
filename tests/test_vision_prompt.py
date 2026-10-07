"""Exercise actual prompt/chunk code with a deterministic native API double."""

import ctypes
from types import SimpleNamespace as NS

import pytest

import base64
from io import BytesIO
from PIL import Image as PILImage
from programasweights._inputs import Image
from programasweights._runtime_vision import _Runtime

def evaluate_prompt(handler, model, content, lib):
    model.chat_handler = handler
    return _Runtime._eval_prompt(NS(llm=model, lib=lib), content)


def load_image(url):
    with PILImage.open(BytesIO(base64.b64decode(url.split(',', 1)[1]))) as pixels:
        return 'A' if pixels.getpixel((0, 0)) == (255, 0, 0) else 'B'


class InputText(ctypes.Structure):
    _fields_ = [("text", ctypes.c_char_p), ("text_len", ctypes.c_size_t),
                ("add_special", ctypes.c_bool), ("parse_special", ctypes.c_bool)]


class Vector(list):
    def __getitem__(self, key):
        result = super().__getitem__(key)
        return Vector(result) if isinstance(key, slice) else result

    def __setitem__(self, key, value):
        if isinstance(key, slice) and isinstance(value, int):
            value = [value] * (key.stop - key.start)
        super().__setitem__(key, value)

    def tolist(self):
        return list(self)


@pytest.fixture
def native():
    state = NS(chunks=[], flags=None, loaded=[], freed=[], chunk_freed=False,
               images=[], fail_tokenize=False, fail_image=False)
    model = NS(n_tokens=0, input_ids=Vector([0] * 256), n_ctx=lambda: 256,
               _ctx=NS(ctx=None), n_batch=128)
    def evaluate(tokens):
        model.input_ids[model.n_tokens:model.n_tokens + len(tokens)] = tokens
        model.n_tokens += len(tokens)
    model.eval = evaluate
    def tokenize(ctx, chunks, ptr, bitmaps, count):
        prompt = ptr._obj
        state.flags = (prompt.add_special, prompt.parse_special)
        state.text = prompt.text[:prompt.text_len].decode()
        state.chunks = []
        for i, piece in enumerate(state.text.split("<__media__>")):
            if i:
                state.chunks.append(NS(kind=2, tokens=[0, 0], image=bitmaps[i - 1]))
            if piece:
                state.chunks.append(NS(kind=1, tokens=list(piece.encode())))
        return 1 if state.fail_tokenize else 0
    def image_eval(ctx, model_ctx, chunk, past, seq, batch, logits_last, new):
        state.images.append((chunk.image, logits_last))
        new._obj.value = past.value + len(chunk.tokens)
        return 1 if state.fail_image else 0
    def text_tokens(chunk, count):
        count._obj.value = len(chunk.tokens)
        return chunk.tokens
    mtmd = NS(
        mtmd_default_marker=lambda: b"<__media__>", mtmd_input_text=InputText,
        mtmd_bitmap_p_ctypes=ctypes.c_void_p,
        mtmd_input_chunks_init=lambda: 1,
        mtmd_input_chunks_free=lambda chunks: setattr(state, "chunk_freed", True),
        mtmd_bitmap_free=state.freed.append, mtmd_tokenize=tokenize,
        mtmd_input_chunks_size=lambda chunks: len(state.chunks),
        mtmd_input_chunks_get=lambda chunks, i: state.chunks[i],
        mtmd_input_chunk_get_n_tokens=lambda c: len(c.tokens),
        mtmd_input_chunk_get_type=lambda c: c.kind,
        mtmd_input_chunk_get_tokens_text=text_tokens,
        mtmd_helper_eval_chunk_single=image_eval,
        MTMD_INPUT_CHUNK_TYPE_TEXT=1, MTMD_INPUT_CHUNK_TYPE_IMAGE=2,
    )
    def bitmap(data):
        state.loaded.append(data)
        return len(state.loaded)
    handler = NS(_mtmd_cpp=mtmd, mtmd_ctx=1, load_image=load_image,
                 _create_bitmap_from_bytes=bitmap)
    lib = NS(llama_pos=ctypes.c_int, llama_seq_id=ctypes.c_int)
    return state, handler, model, lib


def text(value):
    return value


def image(value):
    return Image(PILImage.new("RGB", (3, 5), "red" if value == "A" else "blue"))


def test_exact_text_and_image_order_no_wrapping_or_special_token_insertion(native):
    state, handler, model, lib = native
    prompt = evaluate_prompt(handler, model, [text(" pre"), text("fix "), image("B"),
        text("mid"), image("A"), image("B"), text("end\n ")], lib)
    assert state.text == " prefix <__media__>mid<__media__><__media__>end\n "
    assert state.flags == (False, True)
    assert state.loaded == ["B", "A", "B"]
    assert state.images == [(1, True), (2, True), (3, True)]
    assert state.freed == [1, 2, 3] and state.chunk_freed
    assert len(prompt) == len(" prefix midend\n ") + 6


def test_text_only_calls_take_the_same_path_without_bitmaps(native):
    state, handler, model, lib = native
    assert evaluate_prompt(handler, model, [text("he"), text("llo")], lib) == list(b"hello")
    assert state.loaded == [] and state.flags == (False, True)


def test_image_can_end_prompt_and_requests_logits(native):
    state, handler, model, lib = native
    assert evaluate_prompt(handler, model, [image("A")], lib) == [0, 0]
    assert state.images == [(1, True)]


@pytest.mark.parametrize("parts", [[text("<__media__>")], [text("<__"), text("media__>")]])
def test_literal_media_marker_cannot_create_unbound_images(native, parts):
    state, handler, model, lib = native
    with pytest.raises(ValueError, match="reserved"):
        evaluate_prompt(handler, model, parts, lib)
    assert not state.loaded


@pytest.mark.parametrize("failure", ["fail_tokenize", "fail_image", "overflow"])
def test_native_errors_release_all_allocations(native, failure):
    state, handler, model, lib = native
    if failure == "overflow":
        model.n_ctx = lambda: 2
    else:
        setattr(state, failure, True)
    with pytest.raises((ValueError, RuntimeError)):
        evaluate_prompt(handler, model, [image("A"), image("B")], lib)
    assert state.freed == [1, 2] and state.chunk_freed


def test_nonempty_runtime_rejected_before_allocating(native):
    state, handler, model, lib = native
    model.n_tokens = 1
    with pytest.raises(RuntimeError, match='reset runtime'):
        evaluate_prompt(handler, model, ['hello'], lib)
    assert not state.loaded and not state.chunk_freed


def test_empty_prompt_releases_chunks(native):
    state, handler, model, lib = native
    with pytest.raises(ValueError, match='no tokens'):
        evaluate_prompt(handler, model, [''], lib)
    assert state.chunk_freed


def test_partial_bitmap_allocation_releases_previous_image(native):
    state, handler, model, lib = native
    original = handler._create_bitmap_from_bytes
    handler._create_bitmap_from_bytes = lambda data: original(data) if not state.loaded else None
    with pytest.raises(RuntimeError, match='bitmap'):
        evaluate_prompt(handler, model, [image('A'), image('B')], lib)
    assert state.freed == [1] and not state.chunk_freed


def test_chunk_allocation_failure_releases_images(native):
    state, handler, model, lib = native
    handler._mtmd_cpp.mtmd_input_chunks_init = lambda: None
    with pytest.raises(RuntimeError, match='allocate'):
        evaluate_prompt(handler, model, [image('A')], lib)
    assert state.freed == [1]


@pytest.mark.parametrize('end', [0, 256])
def test_invalid_image_position_releases_resources(native, end):
    state, handler, model, lib = native
    def bad_position(*args):
        args[-1]._obj.value = end
        return 0
    handler._mtmd_cpp.mtmd_helper_eval_chunk_single = bad_position
    with pytest.raises(RuntimeError, match='end position'):
        evaluate_prompt(handler, model, [image('A')], lib)
    assert state.freed == [1] and state.chunk_freed


def test_image_ending_prompt_marks_live_logits(native):
    state, handler, model, lib = native
    model._requires_eval = True
    evaluate_prompt(handler, model, [image('A')], lib)
    assert model._requires_eval is False


from types import SimpleNamespace
from unittest.mock import Mock
import sys
import types
import pytest
from programasweights._runtime_vision import _Runtime


from contextlib import contextmanager


@contextmanager
def fake_grammar(convert):
    chat = types.ModuleType("llama_cpp.llama_chat_format")
    chat._grammar_for_response_format = convert
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setitem(sys.modules, "llama_cpp.llama_chat_format", chat)
        yield convert


def runtime():
    result = {'choices': [{'text': ' OK ', 'finish_reason': 'stop'}],
              'usage': {'prompt_tokens': 3, 'completion_tokens': 1}}
    native = Mock(return_value=result)
    return SimpleNamespace(llm=SimpleNamespace(create_completion=native),
                           _eval_prompt=Mock(return_value=[1, 2, 3])), result


def test_exact_prompt_tokens_and_generation_options_preserved():
    owner, result = runtime()
    parts = ('one', 'two')
    actual = _Runtime._complete_prompt(owner, parts, max_tokens=17, temperature=0.25)
    owner._eval_prompt.assert_called_once_with(parts)
    owner.llm.create_completion.assert_called_once_with(prompt=[1, 2, 3], max_tokens=17, temperature=0.25)
    assert actual is result


def test_logits_processor_passed_unchanged():
    owner, _ = runtime()
    processor = object()
    _Runtime._complete_prompt(owner, ('input',), max_tokens=None, temperature=0,
                              logits_processor=processor)
    assert owner.llm.create_completion.call_args.kwargs['logits_processor'] is processor


@pytest.mark.parametrize('format', [{'type': 'json_object'}, {'type': 'json_object', 'schema': {'type': 'object'}}, {'type': 'text'}])
def test_existing_backend_grammar_conversion_reused(format):
    owner, _ = runtime()
    grammar = object()
    with fake_grammar(Mock(return_value=grammar)) as convert:
        _Runtime._complete_prompt(owner, ('input',), max_tokens=8, temperature=0,
                                  response_format=format)
    convert.assert_called_once_with(format)
    assert owner.llm.create_completion.call_args.kwargs['grammar'] is grammar


def test_grammar_error_precedes_model_evaluation():
    owner, _ = runtime()
    with fake_grammar(Mock(side_effect=ValueError('bad schema'))):
        with pytest.raises(ValueError, match='bad schema'):
            _Runtime._complete_prompt(owner, ('input',), max_tokens=8, temperature=0,
                                      response_format={'type': 'json_object'})
    owner._eval_prompt.assert_not_called()
    owner.llm.create_completion.assert_not_called()


def test_evaluation_error_prevents_generation():
    owner, _ = runtime()
    owner._eval_prompt.side_effect = RuntimeError('native failure')
    with pytest.raises(RuntimeError, match='native failure'):
        _Runtime._complete_prompt(owner, ('input',), max_tokens=8, temperature=0)
    owner.llm.create_completion.assert_not_called()
