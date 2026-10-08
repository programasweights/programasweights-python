"""Numbered text templates share binding while existing text calls stay intact."""
import pytest
import programasweights as paw
from programasweights._prompt_template import bind_template, parse_template
from test_base_interpreter import fake_runtime, _isolated_cache, _write_base_model, _write_compiled_program

@pytest.mark.parametrize('model', ['gpt2', 'Qwen/Qwen3-0.6B'])
def test_exact_numbered_prompt(model, fake_runtime):
    _, state = fake_runtime
    _write_base_model(model)
    template = '<role> {INPUT_1}\n{INPUT_0} / {INPUT_1} </role>\nAnswer:'
    with paw.function(None, interpreter=model, prompt_template=template, offline=True) as fn:
        assert fn('A{INPUT_1}', 'B', max_tokens=1) == 'A'
        assert state.instances[-1].tokenize_calls == [dict(data=b'<role> B\nA{INPUT_1} / B </role>\nAnswer:', add_bos=False, special=True)]
        assert state.instances[-1].reset_calls == 1
        assert fn('C', 'D', max_tokens=0) == ''
        assert state.instances[-1].reset_calls == 2

@pytest.mark.parametrize('template,args,expected', [
    ('{INPUT_0}', ('raw <|im_start|> prompt',), 'raw <|im_start|> prompt'),
    ('Constant prompt', (), 'Constant prompt'),
    ('{{INPUT_0}}', ('x',), '{x}'),
    ('{INPUT_0}\n{INPUT_PLACEHOLDER}', ('X',), 'X\n{INPUT_PLACEHOLDER}'),
])
def test_template_forms(template, args, expected, fake_runtime):
    _, state = fake_runtime
    _write_base_model('gpt2')
    with paw.function(None, interpreter='gpt2', prompt_template=template, offline=True) as fn:
        fn(*args, max_tokens=0)
        assert state.instances[-1].tokenize_calls[0]['data'] == expected.encode()

@pytest.mark.parametrize('template,error', [('{INPUT_1}', ValueError), ('{INPUT_0}{INPUT_2}', ValueError), (12, TypeError), (False, TypeError)])
def test_invalid_template_precedes_native_load(template, error, fake_runtime):
    _, state = fake_runtime
    with pytest.raises(error):
        paw.function(None, interpreter='gpt2', prompt_template=template, offline=True)
    assert state.instances == []

@pytest.mark.parametrize('args,kwargs,error', [
    ((), {}, TypeError), (('a','b','c'), {}, TypeError),
    (('a',32), {}, TypeError), (('a','b'), {'input_text':'c'}, TypeError),
    (('a','b'), {'max_tokens':True}, ValueError),
    (('a','b'), {'max_tokens':-1}, ValueError),
])
def test_invalid_call_does_not_touch_native_state(args, kwargs, error, fake_runtime):
    _, state = fake_runtime
    _write_base_model('gpt2')
    with paw.function(None, interpreter='gpt2', prompt_template='{INPUT_0}{INPUT_1}', offline=True) as fn:
        with pytest.raises(error):
            fn(*args, **kwargs)
        assert state.instances[-1].reset_calls == 0
        assert state.instances[-1].tokenize_calls == []

@pytest.mark.parametrize('compiled', [False, True])
def test_legacy_keyword_positional_calls_and_literal_slots(compiled, fake_runtime, tmp_path):
    runtime, state = fake_runtime
    if compiled:
        directory = _write_compiled_program(tmp_path)
        (directory/'prompt_template.txt').write_text('P{INPUT_0}{{INPUT_PLACEHOLDER}}S')
        fn = runtime.PawFunction(directory, offline=True)
    else:
        _write_base_model('gpt2')
        fn = paw.function(None, interpreter='gpt2', offline=True)
    with fn:
        fn(input_text='alpha', max_tokens=0)
        fn('beta', 0, 0.0, None)
        if compiled:
            assert [c['data'] for c in state.instances[-1].tokenize_calls] == [b'P{INPUT_0}{', b'alpha}S', b'beta}S']
        else:
            assert [c['data'] for c in state.instances[-1].tokenize_calls] == [b'alpha', b'beta']

@pytest.mark.parametrize('reference,kwargs', [('compiled-program', {}), (None, {'remote':True})])
def test_override_not_allowed_for_compiled_or_remote(reference, kwargs):
    with pytest.raises(ValueError, match='local adapter-free'):
        paw.function(reference, prompt_template='{INPUT_0}', **kwargs)


@pytest.mark.parametrize("template,expected", [
    ("{INPUT_PLACEHOLDER}", ("", 0, "")),
    ("{INPUT_0}{{INPUT_PLACEHOLDER}}", ("{INPUT_0}{", 0, "}")),
])
def test_old_placeholder_parser_preserves_literal_braces(template, expected):
    assert parse_template(template, "{INPUT_PLACEHOLDER}") == expected


@pytest.mark.parametrize("template", ["", "{INPUT_0}", "{INPUT_PLACEHOLDER}{INPUT_PLACEHOLDER}"])
def test_old_placeholder_still_requires_exactly_one(template):
    with pytest.raises(ValueError, match="exactly one"):
        parse_template(template, "{INPUT_PLACEHOLDER}")


def test_numbered_binding_preserves_images_and_does_not_reparse_values():
    picture = paw.Image(b"snapshot not decoded at binding")
    segments = parse_template("{INPUT_1}|{INPUT_0}|{INPUT_1}", "{INPUT_N}")
    bound = bind_template(segments, "literal {INPUT_1}", picture)
    assert bound == ("", picture, "|", "literal {INPUT_1}", "|", picture, "")
    assert bound[1] is picture and bound[5] is picture


@pytest.mark.parametrize("template,args,error", [
    ("{INPUT_0}", (), TypeError), ("constant", ("extra",), TypeError),
    ("{INPUT_0}", (["not a string"],), TypeError),
])
def test_shared_binder_rejects_wrong_arity_and_types(template, args, error):
    with pytest.raises(error):
        bind_template(parse_template(template, "{INPUT_N}"), *args)


def test_unknown_placeholder_contract_rejected():
    with pytest.raises(ValueError, match="Unsupported placeholder"):
        parse_template("{INPUT_0}", "unknown")


def test_custom_text_template_rejects_an_image(fake_runtime):
    _write_base_model("gpt2")
    with paw.function(None, interpreter="gpt2", prompt_template="{INPUT_0}", offline=True) as fn:
        with pytest.raises(TypeError, match="Text-only"):
            fn(paw.Image(b"encoded snapshot"))


def test_closed_numbered_text_function_rejects_calls(fake_runtime):
    _write_base_model("gpt2")
    fn = paw.function(None, interpreter="gpt2", prompt_template="{INPUT_0}", offline=True)
    fn.close()
    with pytest.raises(RuntimeError, match="closed"):
        fn("input")
