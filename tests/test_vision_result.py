"""Opt-in result metadata at the native boundary; no models or network used."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import threading

import pytest
import programasweights as paw
from test_vision_metadata import PID, manifest
from test_vision_runtime import backend, program, second_program, vision_base


def completion(text=" result ", reason="stop", usage=None):
    return {"choices": [{"message": {"content": text}, "finish_reason": reason}],
            "usage": usage}


@pytest.mark.parametrize("reason", ["stop", "length", "backend-specific", None])
def test_result_is_opt_in_and_preserves_backend_metadata(program, backend, reason):
    state, _, _ = backend
    counts = {"prompt_tokens": 13, "completion_tokens": 2, "total_tokens": 15}
    state.completion = lambda _: completion(reason=reason, usage=counts)
    with paw.function(PID, offline=True) as fn:
        assert fn("read") == "result"
        assert fn("read", return_info=False) == "result"
        result = fn("read", max_tokens=2, return_info=True)
        assert isinstance(result, paw.FunctionResult)
        assert result.text == "result" and result.finish_reason == reason
        assert result.usage == counts
        assert result.elapsed_seconds >= 0
        assert "return_info" not in state.calls[-1]
        assert state.calls[-1]["max_tokens"] == 2
        counts["prompt_tokens"] = 999
        assert result.usage["prompt_tokens"] == 13
        with pytest.raises(FrozenInstanceError):
            result.text = "changed"
        with pytest.raises(TypeError):
            result.usage["prompt_tokens"] = 0
        assert "FunctionResult" in paw.__all__


def test_missing_metadata_and_empty_usage_are_distinct(program, backend):
    state, _, _ = backend
    with paw.function(PID, offline=True) as fn:
        result = fn("read", return_info=True)
        assert result.finish_reason is result.usage is None
        state.completion = lambda _: completion(usage={})
        assert dict(fn("read", return_info=True).usage) == {}


def test_zero_tokens_returns_metadata_without_generation(program, backend):
    state, _, _ = backend
    with paw.function(PID, offline=True) as fn:
        before = list(state.events)
        result = fn("read", max_tokens=0, return_info=True)
        assert result.text == ""
        assert result.finish_reason is result.usage is None
        assert result.elapsed_seconds >= 0
        assert not state.calls and state.events == before


@pytest.mark.parametrize("flag", [None, 0, 1, "yes"])
def test_flag_is_boolean_and_validation_precedes_native_state(program, backend, flag):
    state, _, _ = backend
    with paw.function(PID, offline=True) as fn:
        before = list(state.events)
        with pytest.raises(TypeError, match="return_info"):
            fn("read", return_info=flag)
        assert not state.calls and state.events == before


@pytest.mark.parametrize("kind", ["generation", "callback", "nontext", "cleanup"])
def test_opt_in_does_not_swallow_errors_or_publish_a_partial_result(program, backend, monkeypatch, kind):
    state, _, module = backend
    failure = ValueError("original failure")
    with paw.function(PID, offline=True) as fn:
        def fail(*args):
            raise failure
        if kind == "nontext":
            state.completion = lambda _: completion(text=None)
        elif kind == "callback":
            def callback_response(kwargs):
                scores = [0.0, 0.0]
                # Simulate the native callback boundary, which finishes sampling
                # before the wrapper re-raises the processor's error.
                kwargs["logits_processor"][0]([], scores)
                return completion()
            state.completion = callback_response
        elif kind == "cleanup":
            reset = fn._runtime._reset
            resets = []
            def fail_cleanup():
                resets.append(True)
                if len(resets) == 2:
                    raise failure
                reset()
            monkeypatch.setattr(fn._runtime, "_reset", fail_cleanup)
        else:
            state.completion = fail
        with pytest.raises(RuntimeError if kind == "nontext" else ValueError) as caught:
            fn("read", return_info=True,
               logits_processor=[fail] if kind == "callback" else None)
        if kind != "nontext":
            assert caught.value is failure
        if kind == "cleanup":
            assert fn._runtime.closed
        else:
            state.completion = None
            assert fn("retry", return_info=True).text == "result"


def test_elapsed_includes_preparation_both_locks_and_cleanup(program, backend, monkeypatch):
    state, _, module = backend
    clock = [100.0]
    monkeypatch.setattr(module, "perf_counter", lambda: clock[0])
    prepare = module._to_chat_content
    def prepare_input(*parts):
        clock[0] += 1
        return prepare(*parts)
    monkeypatch.setattr(module, "_to_chat_content", prepare_input)
    class TimedLock:
        def __init__(self, lock, seconds):
            self.lock, self.seconds = lock, seconds
        def __enter__(self):
            self.lock.acquire()
            clock[0] += self.seconds
        def __exit__(self, *_):
            self.lock.release()
    with paw.function(PID, offline=True) as fn:
        fn._lock = TimedLock(fn._lock, 2)
        fn._runtime.lock = TimedLock(fn._runtime.lock, 3)
        reset = fn._runtime._reset
        def reset_state():
            clock[0] += 4
            reset()
        monkeypatch.setattr(fn._runtime, "_reset", reset_state)
        def infer(kwargs):
            clock[0] += 5
            return completion()
        state.completion = infer
        result = fn("read", return_info=True)
        assert result.elapsed_seconds == 1 + 2 + 3 + 4 + 5 + 4


@pytest.mark.parametrize("same_function", [False, True])
def test_serialized_calls_keep_their_own_results(program, backend, manifest, same_function):
    state, _, _ = backend
    other = second_program(program, manifest)
    started, release, attempted = threading.Event(), threading.Event(), threading.Event()
    shared_counts = {"completion_tokens": 0}
    def infer(kwargs):
        text = kwargs["messages"][-1]["content"][0]["text"]
        if text == "first":
            started.set()
            assert release.wait(5)
            shared_counts["completion_tokens"] = 1
        else:
            shared_counts["completion_tokens"] = 2
        return completion(text=text, usage=shared_counts)
    state.completion = infer
    with paw.function(PID, offline=True) as one, paw.function(other.name, offline=True) as two:
        def second_call():
            attempted.set()
            return (one if same_function else two)("second", return_info=True)
        with ThreadPoolExecutor(2) as executor:
            first = executor.submit(one, "first", return_info=True)
            try:
                assert started.wait(5)
                second = executor.submit(second_call)
                assert attempted.wait(5)
                assert len(state.calls) == 1
            finally:
                release.set()
            a, b = first.result(5), second.result(5)
        assert (a.text, a.usage["completion_tokens"]) == ("first", 1)
        assert (b.text, b.usage["completion_tokens"]) == ("second", 2)
        assert a is not b


def test_base_interpreter_supports_result_metadata(vision_base, backend):
    with paw.function(None, interpreter="Qwen/Qwen3.5-0.8B", offline=True) as fn:
        assert fn("read", return_info=True).text == "result"
        assert fn("read") == "result"


def test_usage_is_copied_before_releasing_the_shared_runtime(program, backend, manifest, monkeypatch):
    state, _, _ = backend
    other = second_program(program, manifest)
    counts = {"completion_tokens": 0}
    first_done, second_done = threading.Event(), threading.Event()
    def infer(kwargs):
        text = kwargs["messages"][-1]["content"][0]["text"]
        counts["completion_tokens"] = 1 if text == "first" else 2
        return completion(text=text, usage=counts)
    state.completion = infer
    with paw.function(PID, offline=True) as one, paw.function(other.name, offline=True) as two:
        run = one._runtime.run
        def interleave(*args, **kwargs):
            output = run(*args, **kwargs)
            if output[0] == "first":
                first_done.set()
                assert second_done.wait(5)
            else:
                second_done.set()
            return output
        monkeypatch.setattr(one._runtime, "run", interleave)
        with ThreadPoolExecutor(2) as executor:
            first = executor.submit(one, "first", return_info=True)
            try:
                assert first_done.wait(5)
                second = executor.submit(two, "second", return_info=True)
                a, b = first.result(5), second.result(5)
            finally:
                second_done.set()
        assert a.usage["completion_tokens"] == 1
        assert b.usage["completion_tokens"] == 2
