import os
import types
import threading
from typing import Any, Dict, List

import torch
import pytest


class MockTokenizer:
    def __init__(self) -> None:
        self.decode_calls: List[List[int]] = []

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls()

    def __call__(self, texts, return_tensors=None, padding=False, truncation=False) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        # Represent each input by its character length for simplicity
        lengths = [len(t) for t in texts]
        input_ids = torch.tensor([[l] for l in lengths], dtype=torch.long)
        return {"input_ids": input_ids}

    def decode(self, seq: torch.Tensor, skip_special_tokens: bool = True) -> str:
        # Convert last token to an integer and format a deterministic string
        if isinstance(seq, torch.Tensor):
            data = seq.detach().cpu().tolist()
            # handle 1D or 2D
            if isinstance(data[0], list):
                last = data[0][-1]
            else:
                last = data[-1]
        else:
            last = int(seq)
        out = f"decoded:{last}"
        return out


class MockModel:
    def __init__(self) -> None:
        self.generate_calls: List[Dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls()

    def to(self, device):
        return self

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int, do_sample: bool, num_beams: int, temperature: float, **kwargs):
        # Record call for assertions
        self.generate_calls.append({
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "temperature": temperature,
        })
        # Deterministic: output is input_ids with +1 appended token per row
        batch, _ = input_ids.shape
        outputs = []
        for b in range(batch):
            last_val = int(input_ids[b, -1].item())
            seq = torch.tensor([last_val + 1], dtype=torch.long)
            outputs.append(seq)
        return torch.stack(outputs, dim=0)


@pytest.fixture(autouse=True)
def patch_transformers(monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", types.SimpleNamespace(from_pretrained=MockTokenizer.from_pretrained))
    monkeypatch.setattr(transformers, "AutoModelForSeq2SeqLM", types.SimpleNamespace(from_pretrained=MockModel.from_pretrained))
    yield


@pytest.fixture
def tmp_prefix_file(tmp_path):
    f = tmp_path / "prefix.txt"
    f.write_text("PREFIX_A", encoding="utf-8")
    return str(f)


def test_import_and_callable(tmp_prefix_file):
    import importlib

    programasweights = importlib.import_module("programasweights")
    fn = programasweights.function(tmp_prefix_file)
    assert callable(fn)
    out = fn("Hello")
    assert isinstance(out, str)


def test_list_input_returns_list(tmp_prefix_file):
    import programasweights

    fn = programasweights.function(tmp_prefix_file)
    outs = fn(["a", "bb", "ccc"])
    assert isinstance(outs, list)
    assert len(outs) == 3


def test_singleton_model_loader(monkeypatch, tmp_prefix_file):
    import importlib

    load_counts = {"tok": 0, "model": 0}

    def count_tok(model_name):
        load_counts["tok"] += 1
        return MockTokenizer.from_pretrained(model_name)

    def count_model(model_name):
        load_counts["model"] += 1
        return MockModel.from_pretrained(model_name)

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", types.SimpleNamespace(from_pretrained=count_tok))
    monkeypatch.setattr(transformers, "AutoModelForSeq2SeqLM", types.SimpleNamespace(from_pretrained=count_model))

    import programasweights
    # Force interpreter fresh load by reloading module namespace
    import importlib
    if "programasweights.runtime" in list(importlib.sys.modules.keys()):
        importlib.sys.modules.pop("programasweights.runtime")
    if "programasweights" in list(importlib.sys.modules.keys()):
        importlib.sys.modules.pop("programasweights")
    programasweights = importlib.import_module("programasweights")

    fn1 = programasweights.function(tmp_prefix_file)
    fn2 = programasweights.function(tmp_prefix_file)
    _ = fn1("x")
    _ = fn2("y")

    assert load_counts["tok"] == 1
    assert load_counts["model"] == 1


def test_multiple_program_registrations_switch(tmp_path):
    import programasweights

    p1 = tmp_path / "p1.txt"
    p2 = tmp_path / "p2.txt"
    p1.write_text("AAA", encoding="utf-8")
    p2.write_text("BBBBB", encoding="utf-8")

    f1 = programasweights.function(str(p1), name="prog1")
    f2 = programasweights.function(str(p2), name="prog2")

    o1 = f1("x")
    o2 = f2("x")
    # Since prefix lengths differ, decoded token should differ deterministically
    assert o1 != o2


def test_deterministic_params_exposed(monkeypatch, tmp_prefix_file):
    import programasweights

    calls = []

    class ModelWithCapture(MockModel):
        def generate(self, *args, **kwargs):
            calls.append(kwargs)
            return super().generate(*args, **kwargs)

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForSeq2SeqLM", types.SimpleNamespace(from_pretrained=ModelWithCapture.from_pretrained))

    fn = programasweights.function(tmp_prefix_file)
    _ = fn("hello")

    assert len(calls) >= 1
    kw = calls[-1]
    assert kw.get("do_sample") is False
    assert kw.get("num_beams") == 1


def test_device_override_cpu(monkeypatch, tmp_prefix_file):
    import importlib

    monkeypatch.setenv("PROGRAMASWEIGHTS_DEVICE", "cpu")
    # reload runtime to apply device selection again
    if "programasweights.runtime" in list(importlib.sys.modules.keys()):
        importlib.sys.modules.pop("programasweights.runtime")
    if "programasweights" in list(importlib.sys.modules.keys()):
        importlib.sys.modules.pop("programasweights")

    programasweights = importlib.import_module("programasweights")
    # Access internal device through constructing a function
    fn = programasweights.function(tmp_prefix_file)
    out = fn("hello")
    assert isinstance(out, str)


def test_dummy_compiler_end_to_end(tmp_path):
    import programasweights as paw

    out_path = tmp_path / "weights.safetensors"
    paw.compile_dummy(str(out_path), seed=42, num_tokens=8)

    fn = paw.function(str(out_path))
    result = fn("ping")
    assert isinstance(result, str)
    assert len(result) > 0


def test_dummy_compiler_determinism(tmp_path):
    import programasweights as paw

    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    paw.compile_dummy(str(a), seed=123, num_tokens=10)
    paw.compile_dummy(str(b), seed=123, num_tokens=10)

    txt_a = a.read_text(encoding="utf-8")
    txt_b = b.read_text(encoding="utf-8")

    assert txt_a == txt_b


def test_alias_import_and_smoke(tmp_path):
    import programasweights as paw

    path = tmp_path / "alias_demo.safetensors"
    paw.compile_dummy(str(path), seed=7, num_tokens=4)
    fn = paw.function(str(path))
    assert callable(fn)
    out = fn(["one", "two"])
    assert isinstance(out, list) and len(out) == 2


def test_version_present():
    import programasweights as paw

    assert isinstance(paw.__version__, str)
    assert len(paw.__version__) > 0 