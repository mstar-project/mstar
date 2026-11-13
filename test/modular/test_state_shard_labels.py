"""A node's sequence shard over several labels, and what a node that splits its steps refuses at load.

Cosmos3 splits its generation steps over SP ranks with both guidance branches packed into one plan, so its shard lists
``cond`` and ``uncond`` and brackets their combined plan. A shard that bracketed one branch would cut the packed tokens
by that branch's range, and a pool that can't run on a rank's share would attend one rank's entries with no error, so
each case is decided here before any step splits.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
from test_state_windows import _ring_array

from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttnBackend,
    BlockCausalAttentionSpec,
    CrossAttentionConfig,
    CrossAttentionSpec,
)
from mstar.engine.resources.sequence_shard import SeqShardSpec
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, Attend, Bidirectional, KVEntry, Read, SeqShard
from mstar.engine.state.adapters import build_specs
from mstar.engine.state.analysis import _shard_problems, split_read_problems
from mstar.engine.state.calls import ShardHandle

LABELS = frozenset({"cond", "uncond"})


def _kv(**overrides) -> Array:
    return Array("kv", **{
        "nodes": {"dit"}, "layers": frozenset(range(2)), "entry": KVEntry(heads=4, head_dim=8), "max_len": 64,
        "labels": LABELS, "reads": {"attn": Read(mask=Bidirectional(), op=Attend(q_heads=4))}, **overrides,
    })


def _step(segments, combined=None) -> SubmoduleStep:
    step = SubmoduleStep(segments=segments, combined_labels=combined or {})
    step.set_ctx(StepContext(request_ids=(1,), graph_walk="gen", slot=0, capture=False, slot_lease=None))
    return step


def _bracketed(step: SubmoduleStep) -> str | None:
    h = ShardHandle(SeqShard("shard", nodes={"dit"}, labels=LABELS), resource=None)
    h.begin_step(step)
    return h.label


def test_a_shard_over_several_labels_is_no_declaration_problem():
    shard = SeqShard("shard", nodes={"dit"}, labels=LABELS, walks={"gen"})
    assert _shard_problems(shard, [_kv()], {"gen": {"dit"}}, set()) == []


def test_a_shard_with_two_labels_brackets_the_step_s_label():
    both = [Segment(1, "cond", 4), Segment(1, "uncond", 4)]
    assert _bracketed(_step([Segment(1, "cond", 4)])) == "cond"
    assert _bracketed(_step(both, {("cond", "uncond"): "cond+uncond"})) == "cond+uncond", (
        "the shard would cut both branches' packed tokens by one branch's range"
    )


def test_a_shard_brackets_one_label_a_step():
    with pytest.raises(ValueError, match=r"SeqShard 'shard': no layout yet: a step opens its labels \['cond', "
                                         r"'uncond'\] in plans of their own"):
        _bracketed(_step([Segment(1, "cond", 4), Segment(1, "uncond", 4)]))


def test_a_read_with_no_op_stays_off_a_node_that_splits_it():
    reach = _kv(reads={"window": Read(mask=Bidirectional(kept_window=True), op=None)})
    assert split_read_problems([reach], (), {"dit": ("cond",)}) == [
        "Array 'kv'.reads['window']: a read with no op gathers its reach from this rank's pages, which hold one "
        "rank's share of labels ['cond', 'uncond'] where the sequence shard of ['dit'] splits them over SP ranks; run "
        "the node on one SP rank"
    ]
    assert split_read_problems([reach], (), {}) == [], "a node on one SP rank holds every entry"


@pytest.mark.parametrize("spec, refusal", [
    (lambda: build_specs([_ring_array()])[0], "KV 'kv' is a ring, which takes"),
    (lambda: CrossAttentionSpec(resource_key="cross", nodes={"dit"}, config=CrossAttentionConfig(
        kv_cache="ctx", query_kv_cache="kv", context_label="main")), "cross attention 'cross' takes"),
    (lambda: BlockCausalAttentionSpec(resource_key="flow", nodes={"dit"}, block_size=8, config=AttentionConfig(
        kv_cache="kv", backend=AttnBackend.FLASHINFER)), "block-causal attention 'flow' takes"),
], ids=["ring", "cross", "block_causal"])
def test_a_pool_that_takes_no_split_steps_stays_off_a_splitting_node(spec, refusal):
    shard = SeqShardSpec("shard", {"dit"}, ("main",), 0)
    assert spec().split_step_refusals("dit", shard, 1, 2, {}) == [
        f"{refusal} no split steps, so node 'dit' can't split them: run it on TP ranks alone"
    ], "the engine's build would refuse the node later, naming the manager rather than the pool"
