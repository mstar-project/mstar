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
        self.asked = 0

    def loopback_tokens(self):
        self.asked += 1
        return self.tokens


def _sub(sampler):
    sub = LLMSubmodule.__new__(LLMSubmodule)
    sub.node_resources = {SAMPLER: sampler}
    return sub


def _prepare(sub, walk, inputs, has_loopback=True):
    return LLMSubmodule.prepare_inputs(
        sub, graph_walk=walk, fwd_info=None, inputs=inputs, has_loopback=has_loopback,
    )


def test_decode_takes_no_input_ids_when_the_sampler_keeps_them():
    sub = _sub(_Sampler(masters=True))
    out = _prepare(sub, "decode", {})
    assert out.input_ids is None and out.input_embeds is None and out.input_seq_len == 1
    # a routed tensor is ignored on this path: the master is the source
    out = _prepare(sub, "decode", {"text_inputs": [torch.tensor([7])]})
    assert out.input_ids is None
    assert LLMSubmodule.device_loopback_signals(sub, "decode") == frozenset({"text_inputs"})
    assert LLMSubmodule.device_loopback_signals(sub, "prefill_text") == frozenset()


def test_a_row_without_loopback_reads_its_routed_token():
    # e.g. its prefill ran on another rank: the master does not hold its token
    sub = _sub(_Sampler(masters=True))
    out = _prepare(sub, "decode", {"text_inputs": [torch.tensor([7])]}, has_loopback=False)
    assert out.input_ids.tolist() == [7] and out.input_seq_len == 1


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


def _stub_model(sub, seen):
    def _position_ids_3d(inputs):
        seen["n"] = len(inputs)
        return torch.zeros(3, len(inputs))

    sub.model = SimpleNamespace(model=SimpleNamespace(
        build_cos_sin=lambda pos, dtype: (pos, pos),
        embed_tokens=SimpleNamespace(weight=torch.zeros(1)),
    ))
    sub._position_ids_3d = _position_ids_3d


def test_preprocess_mixes_routed_rows_into_the_sampler_tokens():
    sampler = _Sampler(masters=True, tokens=torch.tensor([3, 4, 5, 0]))
    sub = _sub(sampler)
    rows = [
        _prepare(sub, "decode", {"text_inputs": [torch.tensor([9])]}, has_loopback=False),
        _prepare(sub, "decode", {}),
        _prepare(sub, "decode", {"text_inputs": [torch.tensor([8])]}, has_loopback=False),
        _prepare(sub, "decode", {}),
    ]
    engine_inputs = SimpleNamespace(
        resources={SAMPLER: sampler}, request_ids=["r1", "r2", "r3", "pad1"],
    )
    _stub_model(sub, {})
    out = LLMSubmodule.preprocess(sub, "decode", engine_inputs, rows)
    assert out["input_ids"].tolist() == [9, 4, 8, 0]


def test_preprocess_reads_the_step_tokens_off_the_sampler():
    sampler = _Sampler(masters=True, tokens=torch.tensor([3, 4, 0, 0]))
    sub = _sub(sampler)
    rows = [_prepare(sub, "decode", {}) for _ in range(4)]
    engine_inputs = SimpleNamespace(
        resources={SAMPLER: sampler}, request_ids=["r1", "r2", "pad1", "pad2"],
    )
    seen = {}
    _stub_model(sub, seen)
    out = LLMSubmodule.preprocess(sub, "decode", engine_inputs, rows)
    assert out["input_ids"].tolist() == [3, 4, 0, 0]
    assert sampler.asked == 1
    assert "input_embeds" not in out and seen["n"] == 4


def test_the_loop_back_decode_row_is_uniform(monkeypatch):
    from mstar.model.submodule_base import ARNodeInputs

    sub = _sub(_Sampler(masters=True))
    row = LLMSubmodule.uniform_row_inputs(sub, "decode", has_loopback=True)
    assert isinstance(row, ARNodeInputs) and row.input_seq_len == 1
    assert row.input_ids is None and row.input_embeds is None
    # a row without loopback in the batch: row by row
    assert LLMSubmodule.uniform_row_inputs(sub, "decode", has_loopback=False) is None
    assert LLMSubmodule.uniform_row_inputs(sub, "prefill_text", has_loopback=True) is None
    assert LLMSubmodule.uniform_row_inputs(
        _sub(_Sampler(masters=False)), "decode", has_loopback=True,
    ) is None
    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "0")
    assert LLMSubmodule.uniform_row_inputs(sub, "decode", has_loopback=True) is None


def test_in_graph_rope_knob_drops_only_the_staged_tables(monkeypatch):
    sampler = _Sampler(masters=True, tokens=torch.tensor([3, 4]))
    sub = _sub(sampler)
    rows = [_prepare(sub, "decode", {}) for _ in range(2)]
    leased = SimpleNamespace(
        resources={SAMPLER: sampler}, request_ids=["r1", "r2"],
        step=SimpleNamespace(ctx=SimpleNamespace(slot_lease=object(), graph_walk="decode")),
    )
    sub.model = SimpleNamespace(model=SimpleNamespace(
        build_cos_sin=lambda pos, dtype: (pos, pos),
        embed_tokens=SimpleNamespace(weight=torch.zeros(1)),
    ))
    sub._position_ids_3d = lambda inputs: torch.zeros(3, len(inputs))
    monkeypatch.delenv("MSTAR_INGRAPH_DECODE_ROPE", raising=False)
    out = LLMSubmodule.preprocess(sub, "decode", leased, rows)
    assert out["input_ids"].tolist() == [3, 4] and "cos_3d" in out
    monkeypatch.setenv("MSTAR_INGRAPH_DECODE_ROPE", "1")
    out = LLMSubmodule.preprocess(sub, "decode", leased, rows)
    assert out["input_ids"].tolist() == [3, 4] and "cos_3d" not in out
    eager = SimpleNamespace(
        resources={SAMPLER: sampler}, request_ids=["r1", "r2"],
        step=SimpleNamespace(ctx=SimpleNamespace(slot_lease=None, graph_walk="decode")),
    )
    out = LLMSubmodule.preprocess(sub, "decode", eager, rows)
    assert out["input_ids"].tolist() == [3, 4] and "cos_3d" in out


def test_forward_in_graph_rope_reads_planned_positions_at_capture(monkeypatch):
    """A capture hands the region the slot's static input_ids and rotary
    tables. The ids are always the static ones; with the rope knob on, the
    forward builds the tables from the planned positions instead. The
    capture's own context carries the piecewise walk, not "decode"."""
    from mstar.model.qwen3_5.config import ATTN, ROPE

    calls = {}

    class _Embed:
        weight = torch.zeros(1)

        def __call__(self, ids):
            calls["ids"] = ids.tolist()
            return ids.float()

    class _Inner:
        embed_tokens = _Embed()

        def __call__(self, embeds, label, cos_sin):
            calls["cos"] = cos_sin[0]
            return embeds

        def build_cos_sin(self, pos, dtype):
            calls["built"] = pos
            return pos, pos

    sampler = _Sampler(masters=True, tokens=torch.tensor([3, 4]))
    sampler.sample = lambda logits: "sampled"
    sub = _sub(sampler)
    sub.node_resources[ROPE] = SimpleNamespace(pos_ids=lambda label: torch.tensor([10, 20]))
    sub.model = SimpleNamespace(model=_Inner(), lm_head=lambda h: h)
    static_ids, static_cos = torch.tensor([9, 9]), torch.full((3, 2), 7.0)

    def run(lease):
        calls.clear()
        engine_inputs = SimpleNamespace(
            resources={SAMPLER: sampler, ATTN: object()}, request_ids=["r1", "r2"],
            step=SimpleNamespace(ctx=SimpleNamespace(slot_lease=lease, graph_walk="__piecewise__")),
        )
        out = LLMSubmodule._forward(
            sub, "decode", engine_inputs, cos_3d=static_cos, sin_3d=static_cos, input_ids=static_ids,
        )
        assert out == "sampled"
        return dict(calls)

    monkeypatch.delenv("MSTAR_INGRAPH_DECODE_ROPE", raising=False)
    c = run(object())
    assert c["ids"] == [9, 9] and c["cos"] is static_cos and "built" not in c
    monkeypatch.setenv("MSTAR_INGRAPH_DECODE_ROPE", "1")
    c = run(object())
    assert c["ids"] == [9, 9] and c["built"].tolist() == [[10, 20]] * 3 and c["cos"] is not static_cos
    # no lease (an eager step): the knob does not apply, the inputs are used
    c = run(None)
    assert c["ids"] == [9, 9] and c["cos"] is static_cos and "built" not in c


def test_sampler_keeps_slot_masters_only_for_a_node_that_reads_them(monkeypatch):
    # Whisper's decoder takes its tokens on the host, so the sampler must not
    # pay the eager-step master write for it; the Qwen3.5 LLM node declares
    # the read and the engine enables the masters for its sampler.
    from mstar.engine.resources.sampler.resource import SamplerResource

    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "1")
    res = SamplerResource(vocab_size=None, enable_repetion_penalty=False, device=torch.device("cpu"))
    assert res._keep_last_token is False
    assert res.has_slot_masters is False
    res.enable_device_loopback_reader()
    assert res._keep_last_token is True

    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "0")
    off = SamplerResource(vocab_size=None, enable_repetion_penalty=False, device=torch.device("cpu"))
    off.enable_device_loopback_reader()
    assert off._keep_last_token is False


def test_the_loop_back_read_is_declared_per_submodule():
    from mstar.model.submodule_base import NodeSubmodule

    assert LLMSubmodule.reads_device_loopback is True
    assert NodeSubmodule.reads_device_loopback is False
