"""The declared chunk mask and the request's read-keyed lengths."""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.conductor.conductor import _with_chunk_causal_layout
from mstar.conductor.request_info import CurrentForwardPassInfo, WalkStep
from mstar.engine.resources.attn.ragged.chunk_causal import dense_visibility
from mstar.engine.resources.attn.ragged.config import RaggedChunkCausalAttentionSpec
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep, Attend, ChunkCausal, ChunkCausalLayout, KVEntry, Layout, Read
from mstar.engine.state.adapters import (
    CHUNK_CAUSAL_ATTEND,
    CHUNK_CAUSAL_LAYOUT_KWARG,
    ChunkCausalAttend,
    build_specs,
    chunk_causal_layouts,
    chunk_causal_resource_layout,
    kv_layouts,
    node_handles,
    state_steps,
)


def _array() -> Array:
    return Array(
        "tokens", nodes={"dit"}, layers={0}, entry=KVEntry(heads=4, head_dim=32), max_len=0,
        reads={"attn": Read(mask=ChunkCausal(8), op=Attend(q_heads=4))},
    )


def _lengths() -> ChunkCausalLayout:
    return ChunkCausalLayout(global_len=4, noised_len=16, clean_len=8)


def _visible(q: int, k: int, layout: ChunkCausalLayout, chunk_len: int) -> bool:
    g, n = layout.global_len, layout.noised_len
    if k < g:
        return True
    if q < g:
        return False
    if q < g + n:
        cq = (q - g) // chunk_len
        if k < g + n:
            return (k - g) // chunk_len == cq
        return (k - g - n) // chunk_len < cq
    if k < g + n:
        return False
    return (k - g - n) // chunk_len <= (q - g - n) // chunk_len


def test_read_keyed_lengths_and_declaration_are_hashable():
    array = _array()
    layout = Layout(chunk_causal={"attn": _lengths()})
    assert array == _array() and hash(array) == hash(_array())
    assert layout.chunk_causal["attn"].span == 28
    assert "mask_step" not in ArrayStep.__dataclass_fields__
    assert ArrayStep(segments=(Segment("r0", "main", 28),)) == ArrayStep(
        segments=(Segment("r0", "main", 28),)
    )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_mask_refuses_invalid_fixed_chunk_len(value):
    with pytest.raises(ValueError, match="ChunkCausal.chunk_len"):
        ChunkCausal(value)


@pytest.mark.parametrize("field,value", [
    ("global_len", -1), ("global_len", True), ("noised_len", -1), ("clean_len", 1.5),
])
def test_request_layout_refuses_invalid_lengths(field, value):
    with pytest.raises(ValueError, match=f"ChunkCausalLayout.{field}"):
        ChunkCausalLayout(**{"global_len": 4, "noised_len": 16, "clean_len": 8, field: value})


def test_ordinary_attend_binding_selects_chunk_resource():
    array = _array()
    assert isinstance(CHUNK_CAUSAL_ATTEND, ChunkCausalAttend)
    specs = build_specs([array])
    assert len(specs) == 1 and isinstance(specs[0], RaggedChunkCausalAttentionSpec)
    assert specs[0].resource_key == "attn"


def test_request_layout_checks_read_key_before_dropping_within_step_kv_layout():
    array = _array()
    assert kv_layouts([array], {"tokens": {"main": Layout(chunk_causal={"attn": _lengths()})}}, {}) == {}
    with pytest.raises(RuntimeError, match="names no ChunkCausal read"):
        kv_layouts([array], {"tokens": {"main": Layout(chunk_causal={"other": _lengths()})}}, {})
    with pytest.raises(RuntimeError, match="must be a ChunkCausalLayout"):
        kv_layouts([array], {"tokens": {"main": Layout(chunk_causal={"attn": "bad"})}}, {})


def test_two_request_layouts_reach_binding_and_plan_without_cross_request_visibility():
    array = _array()
    requested = {
        "r0": ChunkCausalLayout(global_len=4, noised_len=16, clean_len=8),
        "r1": ChunkCausalLayout(global_len=2, noised_len=8, clean_len=0),
    }
    infos = {}
    for rid, lengths in requested.items():
        layout = chunk_causal_layouts([array], {"tokens": {"main": Layout(chunk_causal={"attn": lengths})}})
        (walk,) = _with_chunk_causal_layout([WalkStep("encode")], layout)
        infos[rid] = CurrentForwardPassInfo(
            request_id=rid, graph_walk="encode", fwd_index=0, random_seed=0, max_tokens=1,
            step_metadata=dict(walk.step_metadata),
        )
    class Model:
        def declare_state(self):
            return [array]

    handles = node_handles(Model(), "dit", {"attn": SimpleNamespace(run=lambda *args, **kwargs: None)})
    segments = [Segment(rid, "main", lengths.span) for rid, lengths in requested.items()]
    expanded = state_steps(
        SubmoduleStep(segments=segments, state={"tokens": ArrayStep()}), handles, per_request_info=infos,
    )
    resource_step = expanded.steps["attn"]
    assert resource_step.segments == tuple(segments)
    assert resource_step.layouts == tuple(
        chunk_causal_resource_layout(array.reads["attn"].mask, lengths) for lengths in requested.values()
    )
    (spec,) = build_specs([array])
    manager = build_resource(spec, EngineResourceInfo(device=torch.device("cpu"), kv_dtype=torch.float32))
    manager.plan(resource_step, StepContext(request_ids=[0, 1], graph_walk="encode", slot=0, capture=False))
    planned = manager._wrapper("main").block_mask
    total = sum(s.span for s in segments)
    from test.modular.test_ragged_chunk_causal import dense_from_block_mask

    actual = dense_from_block_mask(planned, total)
    expected = dense_visibility(resource_step.layouts)
    assert torch.equal(actual, expected)
    assert not actual[:segments[0].span, segments[0].span:].any()
    assert not actual[segments[0].span:, :segments[0].span].any()


def test_missing_or_misaligned_real_request_layout_is_refused():
    array = _array()
    class Model:
        def declare_state(self):
            return [array]

    handles = node_handles(Model(), "dit", {"attn": SimpleNamespace(run=lambda *args, **kwargs: None)})
    step = SubmoduleStep(segments=[Segment("r0", "main", 28)], state={"tokens": ArrayStep()})
    info = CurrentForwardPassInfo(request_id="r0", graph_walk="encode", fwd_index=0, random_seed=0, max_tokens=1)
    with pytest.raises(ValueError, match="has no state_layout chunk lengths"):
        state_steps(step, handles, per_request_info={"r0": info})
    info.step_metadata[CHUNK_CAUSAL_LAYOUT_KWARG] = {
        "tokens": {"main": {"attn": [4, 8, 8]}}
    }
    with pytest.raises(ValueError, match="spans 20, but its segment spans 28"):
        state_steps(step, handles, per_request_info={"r0": info})


@pytest.mark.parametrize("lengths", [
    ChunkCausalLayout(global_len=4, noised_len=16, clean_len=8),
    ChunkCausalLayout(global_len=1, noised_len=7, clean_len=3),
    ChunkCausalLayout(global_len=2, noised_len=0, clean_len=0),
])
def test_read_keyed_layout_matches_independent_full_mask(lengths):
    mask = _array().reads["attn"].mask
    resource_layout = chunk_causal_resource_layout(mask, lengths)
    actual = dense_visibility([resource_layout])
    expected = torch.tensor([
        [_visible(q, k, lengths, mask.chunk_len) for k in range(lengths.span)]
        for q in range(lengths.span)
    ])
    assert torch.equal(actual, expected)


def test_mask_negative_control_exposes_same_chunk_clean_leak():
    lengths = ChunkCausalLayout(global_len=4, noised_len=16, clean_len=16)
    mask = _array().reads["attn"].mask
    actual = dense_visibility([chunk_causal_resource_layout(mask, lengths)])
    noised_chunk_zero = lengths.global_len
    clean_chunk_zero = lengths.global_len + lengths.noised_len
    assert not actual[noised_chunk_zero, clean_chunk_zero]
    wrong_same_chunk_rule = torch.tensor([
        [
            _visible(q, k, lengths, mask.chunk_len)
            or (lengths.global_len <= q < clean_chunk_zero and k >= clean_chunk_zero
                and (q - lengths.global_len) // mask.chunk_len == (k - clean_chunk_zero) // mask.chunk_len)
            for k in range(lengths.span)
        ] for q in range(lengths.span)
    ])
    assert not torch.equal(actual, wrong_same_chunk_rule)


def test_chunk_mask_cannot_silently_take_block_plan():
    from mstar.engine.state.adapters import ChunkCausalQueries

    with pytest.raises(RuntimeError, match="use WithinStepSteps"):
        ChunkCausalQueries(_array()).step((Segment("r0", "main", 28),), (), False)
