"""Qwen3.5 decode with the device-side loop-back: the step's input ids come
off the node's loop-back rows, and a row that brings its token routed uses it."""
from types import SimpleNamespace

import torch

from mstar.engine.resources.loopback_rows import LoopbackRows
from mstar.model.qwen3_5.config import SAMPLER
from mstar.model.qwen3_5.submodules import LLMSubmodule


def _rows(tokens: dict) -> LoopbackRows:
    """Loop-back rows holding each request's last token."""
    rows = LoopbackRows(torch.device("cpu"))
    for rid in tokens:
        rows.add(rid)
    rows.write("text_inputs", list(tokens), torch.tensor(list(tokens.values())))
    return rows


def _sub(sampler=None):
    sub = LLMSubmodule(None, SimpleNamespace(stop_token_ids=frozenset()))
    sub.node_resources = {SAMPLER: sampler}
    return sub


def _prepare(sub, walk, inputs, has_loopback=True):
    return LLMSubmodule.prepare_inputs(
        sub, graph_walk=walk, fwd_info=None, inputs=inputs, has_loopback=has_loopback,
    )


def _engine_inputs(rows, request_ids, captured=False, **resources):
    return SimpleNamespace(
        resources={SAMPLER: None, **resources}, request_ids=request_ids, captured=captured, loopback_rows=rows,
        rows_in_graph=False,
    )


def test_decode_takes_no_input_ids_when_the_rows_keep_them():
    sub = _sub()
    out = _prepare(sub, "decode", {})
    assert out.input_ids is None and out.input_embeds is None and out.input_seq_len == 1
    # a routed tensor is ignored on this path: the rows are the source
    out = _prepare(sub, "decode", {"text_inputs": [torch.tensor([7])]})
    assert out.input_ids is None


def test_a_row_without_loopback_reads_its_routed_token():
    # e.g. its prefill ran on another rank: the rows do not hold its token
    sub = _sub()
    out = _prepare(sub, "decode", {"text_inputs": [torch.tensor([7])]}, has_loopback=False)
    assert out.input_ids.tolist() == [7] and out.input_seq_len == 1


def _stub_model(sub, seen):
    def _position_ids_3d(inputs):
        seen["n"] = len(inputs)
        return torch.zeros(3, len(inputs))

    sub.model = SimpleNamespace(model=SimpleNamespace(
        build_cos_sin=lambda pos, dtype: (pos, pos),
        embed_tokens=SimpleNamespace(weight=torch.zeros(1)),
    ))
    sub._position_ids_3d = _position_ids_3d


def test_preprocess_mixes_routed_rows_into_the_kept_tokens():
    rows = _rows({"r1": 3, "r2": 4, "r3": 5})
    sub = _sub()
    batch = [
        _prepare(sub, "decode", {"text_inputs": [torch.tensor([9])]}, has_loopback=False),
        _prepare(sub, "decode", {}),
        _prepare(sub, "decode", {"text_inputs": [torch.tensor([8])]}, has_loopback=False),
        _prepare(sub, "decode", {}),
    ]
    _stub_model(sub, {})
    out = LLMSubmodule.preprocess(sub, "decode", _engine_inputs(rows, ["r1", "r2", "r3", "pad1"]), batch)
    assert out["input_ids"].tolist() == [9, 4, 8, 0]


def test_preprocess_reads_the_step_tokens_off_the_rows():
    sub = _sub()
    batch = [_prepare(sub, "decode", {}) for _ in range(4)]
    seen = {}
    _stub_model(sub, seen)
    # padding rows read the trash row
    out = LLMSubmodule.preprocess(sub, "decode", _engine_inputs(_rows({"r1": 3, "r2": 4}), ["r1", "r2", -1, -2]), batch)
    assert out["input_ids"].tolist() == [3, 4, 0, 0]
    assert "input_embeds" not in out and seen["n"] == 4


def test_the_loop_back_decode_row_is_uniform():
    from mstar.model.submodule_base import ARNodeInputs

    sub = _sub()
    row = LLMSubmodule.uniform_row_inputs(sub, "decode", has_loopback=True)
    assert isinstance(row, ARNodeInputs) and row.input_seq_len == 1
    assert row.input_ids is None and row.input_embeds is None
    # a row without loopback in the batch: row by row
    assert LLMSubmodule.uniform_row_inputs(sub, "decode", has_loopback=False) is None
    assert LLMSubmodule.uniform_row_inputs(sub, "prefill_text", has_loopback=True) is None


def test_captured_decode_drops_only_the_staged_tables():
    kept = _rows({"r1": 3, "r2": 4})
    sub = _sub()
    rows = [_prepare(sub, "decode", {}) for _ in range(2)]
    # every row routed: each request's first decode step, or the knob off
    routed = [_prepare(sub, "decode", {"text_inputs": [torch.tensor([t])]}, has_loopback=False) for t in (5, 6)]
    leased = _engine_inputs(kept, ["r1", "r2"], captured=True)
    sub.model = SimpleNamespace(model=SimpleNamespace(
        build_cos_sin=lambda pos, dtype: (pos, pos),
        embed_tokens=SimpleNamespace(weight=torch.zeros(1)),
    ))
    sub._position_ids_3d = lambda inputs: torch.zeros(3, len(inputs))
    out = LLMSubmodule.preprocess(sub, "decode", leased, rows)
    assert out["input_ids"].tolist() == [3, 4] and "cos_3d" not in out
    out = LLMSubmodule.preprocess(sub, "decode", leased, routed)
    assert out["input_ids"].tolist() == [5, 6] and "cos_3d" not in out
    eager = _engine_inputs(kept, ["r1", "r2"])
    out = LLMSubmodule.preprocess(sub, "decode", eager, rows)
    assert out["input_ids"].tolist() == [3, 4] and "cos_3d" in out


def test_forward_in_graph_rope_reads_planned_positions_at_capture():
    """A capture hands the forward the slot's static input_ids and rotary
    tables. The ids are always the static ones; the forward builds the tables
    from the planned positions instead."""
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

    sampler = SimpleNamespace(sample=lambda rids, logits: ("sampled", list(rids)))
    sub = _sub(sampler)
    sub.node_resources[ROPE] = SimpleNamespace(pos_ids=lambda label: torch.tensor([10, 20]))
    sub.model = SimpleNamespace(model=_Inner(), lm_head=lambda h: h)
    static_ids, static_cos = torch.tensor([9, 9]), torch.full((3, 2), 7.0)

    def run(captured):
        calls.clear()
        engine_inputs = SimpleNamespace(
            resources={SAMPLER: sampler, ATTN: object()}, request_ids=["r1", "r2"], captured=captured,
        )
        out = LLMSubmodule._forward(
            sub, "decode", engine_inputs, cos_3d=static_cos, sin_3d=static_cos, input_ids=static_ids,
        )
        assert out == ("sampled", ["r1", "r2"])
        return dict(calls)

    c = run(True)
    assert c["ids"] == [9, 9] and c["built"].tolist() == [[10, 20]] * 3 and c["cos"] is not static_cos
    # no lease (an eager step): the inputs are used
    c = run(False)
    assert c["ids"] == [9, 9] and c["cos"] is static_cos and "built" not in c
