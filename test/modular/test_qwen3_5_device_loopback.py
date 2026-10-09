"""Qwen3.5 decode with the device-side loop-back: the step's input ids come
from the sampler's slot master, the routed signal carries no tensor, and the
knob or a sampler without masters keeps the tensor path."""
from types import SimpleNamespace

import pytest
import torch

from mstar.model.qwen3_5.config import SAMPLER
from mstar.model.qwen3_5.submodules import LLMSubmodule


class _Sampler:
    def __init__(self, masters=True, tokens=None):
        self.has_slot_masters = masters
        self.tokens = tokens
        self.asked = None

    def loopback_tokens(self, request_ids):
        self.asked = list(request_ids)
        return self.tokens


def _sub(sampler):
    sub = LLMSubmodule.__new__(LLMSubmodule)
    sub.node_resources = {SAMPLER: sampler}
    return sub


def _prepare(sub, walk, inputs):
    return LLMSubmodule.prepare_inputs(sub, graph_walk=walk, fwd_info=None, inputs=inputs)


def test_decode_takes_no_input_ids_when_the_sampler_keeps_them():
    sub = _sub(_Sampler(masters=True))
    out = _prepare(sub, "decode", {})
    assert out.input_ids is None and out.input_embeds is None and out.input_seq_len == 1
    # a routed tensor is ignored on this path: the master is the source
    out = _prepare(sub, "decode", {"text_inputs": [torch.tensor([7])]})
    assert out.input_ids is None
    assert LLMSubmodule.device_loopback_signals(sub, "decode") == frozenset({"text_inputs"})
    assert LLMSubmodule.device_loopback_signals(sub, "prefill_text") == frozenset()


def test_without_masters_or_with_the_knob_off_the_tensor_path_stays(monkeypatch):
    sub = _sub(_Sampler(masters=False))
    out = _prepare(sub, "decode", {"text_inputs": [torch.tensor([7])]})
    assert out.input_ids.tolist() == [7] and out.input_seq_len == 1
    assert LLMSubmodule.device_loopback_signals(sub, "decode") == frozenset()
    with pytest.raises(KeyError):
        _prepare(sub, "decode", {})  # the tensor path needs the signal
    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "0")
    sub = _sub(_Sampler(masters=True))
    out = _prepare(sub, "decode", {"text_inputs": [torch.tensor([9])]})
    assert out.input_ids.tolist() == [9]
    assert LLMSubmodule.device_loopback_signals(sub, "decode") == frozenset()


def test_preprocess_reads_the_step_tokens_off_the_sampler():
    sampler = _Sampler(masters=True, tokens=torch.tensor([3, 4, 0, 0]))
    sub = _sub(sampler)
    rows = [_prepare(sub, "decode", {}) for _ in range(4)]
    engine_inputs = SimpleNamespace(
        resources={SAMPLER: sampler}, request_ids=["r1", "r2", "pad1", "pad2"],
    )
    seen = {}

    def _position_ids_3d(inputs):
        seen["n"] = len(inputs)
        return torch.zeros(3, len(inputs))

    sub.model = SimpleNamespace(model=SimpleNamespace(
        build_cos_sin=lambda pos, dtype: (pos, pos),
        embed_tokens=SimpleNamespace(weight=torch.zeros(1)),
    ))
    sub._position_ids_3d = _position_ids_3d
    out = LLMSubmodule.preprocess(sub, "decode", engine_inputs, rows)
    assert out["input_ids"].tolist() == [3, 4, 0, 0]
    assert sampler.asked == ["r1", "r2", "pad1", "pad2"]
    assert "input_embeds" not in out and seen["n"] == 4


def test_the_loop_back_decode_row_is_uniform(monkeypatch):
    from mstar.model.submodule_base import ARNodeInputs

    sub = _sub(_Sampler(masters=True))
    row = LLMSubmodule.uniform_row_inputs(sub, "decode")
    assert isinstance(row, ARNodeInputs) and row.input_seq_len == 1
    assert row.input_ids is None and row.input_embeds is None
    assert LLMSubmodule.uniform_row_inputs(sub, "prefill_text") is None
    assert LLMSubmodule.uniform_row_inputs(_sub(_Sampler(masters=False)), "decode") is None
    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "0")
    assert LLMSubmodule.uniform_row_inputs(sub, "decode") is None
