"""A `BlockPrefix` read: each query sees the first entries of its own segment, as many as the step gives that segment.

Step-Audio2's Whisper tower pads a request's clips to its longest, and the reference masks each clip's keys past its
valid frames while every frame still queries. The read plans a `KeyPrefixAttentionStep` on a key-prefix resource, which
is what the model wrote by hand before; anything that would drop the mask (SDPA, a cross or segmented read) is refused.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.attn.ragged.config import (
    KeyPrefixAttentionStep,
    RaggedAttentionConfig,
    RaggedAttentionSpec,
    RaggedKeyPrefixAttentionSpec,
)
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep, Attend, Block, BlockPrefix, KVEntry, Read
from mstar.engine.state.adapters import build_specs, node_handles, state_setup, state_steps

HEADS, DIM = 4, 8


def _array(name="tokens", mask=None, **read) -> Array:
    return Array(
        name, nodes={"tower"}, layers={0, 1}, entry=KVEntry(heads=HEADS, head_dim=DIM), max_len=0,
        reads={f"{name}_attn": Read(mask=mask or BlockPrefix(), op=Attend(q_heads=HEADS), **read)},
    )


class _StubKeyPrefix:
    def __init__(self):
        self.runs = []

    def run(self, q, k=None, v=None, label=None, return_lse=False):
        self.runs.append((label, q.shape, k.shape))
        return q


def _handles(*arrays):
    specs = build_specs(arrays)
    model = SimpleNamespace(declare_state=lambda: list(arrays))
    return node_handles(model, "tower", {spec.resource_key: _StubKeyPrefix() for spec in specs})


def _ctx() -> StepContext:
    return StepContext(request_ids=[7], graph_walk="encode", slot=0, capture=False)


def _step(lens, spans=(6, 4), name="tokens") -> SubmoduleStep:
    return SubmoduleStep(
        segments=[Segment(7, "main", span) for span in spans], state={name: ArrayStep(prefix_lens=lens)},
    )


def test_a_prefix_read_builds_a_key_prefix_resource():
    (spec,) = build_specs([_array()])
    assert spec == RaggedKeyPrefixAttentionSpec(
        resource_key="tokens_attn", nodes={"tower"},
        config=RaggedAttentionConfig(num_qo_heads=HEADS, num_kv_heads=HEADS, head_dim=DIM),
    )


def test_a_prefix_read_and_a_block_read_of_one_geometry_stay_apart():
    specs = build_specs([_array("a"), _array("b", mask=Block())])
    assert [type(spec) for spec in specs] == [RaggedKeyPrefixAttentionSpec, RaggedAttentionSpec]


def test_a_step_plans_each_segments_prefix():
    expanded = state_steps(_step((3, 4)), _handles(_array()))
    assert expanded.steps == {"tokens_attn": KeyPrefixAttentionStep(
        segments=(Segment(7, "main", 6), Segment(7, "main", 4)), key_lens=(3, 4),
    )}


@pytest.mark.parametrize("lens, cause", [
    (None, r"its BlockPrefix read needs prefix_lens, one per segment of its 2, and the step gives None"),
    ((3,), r"its BlockPrefix read needs prefix_lens, one per segment of its 2, and the step gives \(3,\)"),
    ((7, 4), r"prefix_lens must be within each segment's span, and above 0 unless the span is; got \(span, prefix\) "
             r"\[\(6, 7\)\]"),
    ((0, 4), r"got \(span, prefix\) \[\(6, 0\)\]"),
])
def test_a_step_with_no_prefix_for_a_segment_is_refused(lens, cause):
    with pytest.raises(ValueError, match=r"Array 'tokens': .*" + cause):
        state_steps(_step(lens), _handles(_array()))


def test_prefix_lens_on_an_array_with_no_prefix_read_are_refused_when_the_step_opens():
    handle = _handles(_array(mask=Block()))["tokens"]
    step = _step((3, 4))
    step.set_ctx(_ctx())
    with pytest.raises(ValueError, match="Array 'tokens': a step sets prefix_lens, and the array has no BlockPrefix"):
        handle.begin_step(step)


def test_a_read_attends_through_the_resource_under_the_arrays_label():
    handles = _handles(_array())
    resources = handles["tokens"].reads
    q = k = v = torch.zeros(10, HEADS, DIM)
    handle = handles["tokens"]
    step = _step((3, 4))
    step.set_ctx(_ctx())
    handle.begin_step(step)
    handle.add(0, k=k, v=v)
    assert handle.read(0, q) is q
    assert resources["tokens_attn"].runs == [("main", q.shape, k.shape)]


@pytest.mark.parametrize("read", [{"queries": "other"}, {"packed": False, "segmented": True}])
def test_a_prefix_mask_takes_no_queries_and_no_segmented_rows(read):
    with pytest.raises(ValueError, match="queries|segmented"):
        _array(**read)


def test_sdpa_on_a_prefix_read_is_refused_at_load():
    setup, problems = state_setup({"resources": {"tokens_attn": {"backend": "sdpa"}}}, [_array()])
    assert problems == ["resources.tokens_attn: backend sdpa: Array 'tokens': no layout yet: reads['tokens_attn'] is "
                        "a BlockPrefix read, which SDPA doesn't run"]
    assert dict(setup.by_read) == {}
