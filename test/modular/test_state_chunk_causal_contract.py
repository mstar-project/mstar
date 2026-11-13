"""The proposed declaration and per-request step layout for chunk-causal reads."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from mstar.engine.resources.step import Segment
from mstar.engine.state import (
    Array,
    ArrayStep,
    Attend,
    ChunkCausal,
    ChunkCausalLayout,
    ChunkCausalStep,
    KVEntry,
    Read,
)
from mstar.engine.state.adapters import build_specs


def _layout(**overrides) -> ChunkCausalLayout:
    return ChunkCausalLayout(**{"global_len": 4, "chunk_len": 8, "noised_len": 16, "clean_len": 8, **overrides})


def test_mask_declaration_and_step_layout_are_hashable_values():
    def declaration():
        return Array(
            "tokens", nodes={"dit"}, layers={0}, entry=KVEntry(heads=4, head_dim=32), max_len=0,
            reads={"attn": Read(mask=ChunkCausal(), op=Attend(q_heads=4))},
        )

    segment = Segment(request_id="r0", label="main", span=28)
    step = ArrayStep(segments=(segment,), mask_step=ChunkCausalStep(layouts=(_layout(),)))
    assert declaration() == declaration() and hash(declaration()) == hash(declaration())
    same_step = ArrayStep(segments=(segment,), mask_step=ChunkCausalStep(layouts=(_layout(),)))
    assert step == same_step and hash(step) == hash(same_step)
    assert step.mask_step.layouts[0].span == segment.span


@pytest.mark.parametrize("field,value", [
    ("global_len", -1), ("global_len", True),
    ("noised_len", -1), ("clean_len", 1.5),
    ("chunk_len", 0), ("chunk_len", True),
])
def test_layout_refuses_invalid_lengths(field, value):
    with pytest.raises(ValueError, match=f"ChunkCausalLayout.{field}"):
        _layout(**{field: value})


def test_step_refuses_misaligned_layouts():
    segment = Segment(request_id="r0", label="main", span=28)
    with pytest.raises(ValueError, match="1 layouts for 2 segments"):
        ArrayStep(segments=(segment, segment), mask_step=ChunkCausalStep(layouts=(_layout(),)))
    with pytest.raises(ValueError, match="layout.*span is 28, but segment span is 27"):
        ArrayStep(segments=(Segment(request_id="r0", label="main", span=27),),
                  mask_step=ChunkCausalStep(layouts=(_layout(),)))


def test_step_refuses_unknown_mask_metadata():
    with pytest.raises(ValueError, match="ArrayStep.mask_step must be a ChunkCausalStep"):
        ArrayStep(mask_step="unknown")


def test_unbound_mask_is_refused_instead_of_running_block_attention():
    array = Array(
        "tokens", nodes={"dit"}, layers={0}, entry=KVEntry(heads=4, head_dim=32), max_len=0,
        reads={"attn": Read(mask=ChunkCausal(), op=Attend(q_heads=4))},
    )
    with pytest.raises(ValueError, match="mask must be Block"):
        build_specs([array])
