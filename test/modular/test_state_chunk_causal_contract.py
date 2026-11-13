"""The declared chunk mask and the request's read-keyed lengths."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.attn.ragged.chunk_causal import dense_visibility
from mstar.engine.resources.attn.ragged.config import RaggedChunkCausalAttentionSpec
from mstar.engine.resources.step import Segment
from mstar.engine.state import Array, ArrayStep, Attend, ChunkCausal, ChunkCausalLayout, KVEntry, Layout, Read
from mstar.engine.state.adapters import (
    CHUNK_CAUSAL_ATTEND,
    ChunkCausalAttend,
    build_specs,
    chunk_causal_resource_layout,
    kv_layouts,
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


def test_chunk_mask_cannot_silently_take_block_plan():
    from mstar.engine.state.adapters import ChunkCausalQueries

    with pytest.raises(RuntimeError, match="request-layout transport is not bound"):
        ChunkCausalQueries(_array()).step((Segment("r0", "main", 28),), (), False)
