"""A node's sequence shard over several labels, and what a node that splits its steps refuses at load.

Cosmos3 splits its generation steps over SP ranks with both guidance branches packed into one plan, so its shard lists
``cond`` and ``uncond`` and brackets their combined plan. A shard that bracketed one branch would cut the packed tokens
by that branch's range, and a pool that can't run on a rank's share would attend one rank's entries with no error, so
each case is decided here before any step splits.
"""

from __future__ import annotations

import sys
import types

sys.path.insert(0, ".")

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_state_kept_reads import _CpuVarlenSdpa, _sdpa
from test_state_windows import _ring_array

from mstar.communication.tensors import LocalTransferEngine
from mstar.distributed.communication import CommGroup, JointGroups
from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    CrossAttentionConfig,
    CrossAttentionSpec,
)
from mstar.engine.resources.attn.dense import DenseAttentionManager
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.bounded.config import BoundedKVConfig, BoundedKVSpec, SinkWindow
from mstar.engine.resources.kv.config import KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.sequence_shard import SeqShardSpec
from mstar.engine.resources.spec import resolve_spec_dependencies
from mstar.engine.resources.step import ResourceStep, Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep, Attend, Bidirectional, KVEntry, Read, SeqShard, gather, scatter
from mstar.engine.state.adapters import StateSetup, _resolve, build_handle, build_specs
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
    (lambda: BoundedKVSpec(resource_key="bounded", nodes={"dit"}, config=BoundedKVConfig(
        num_layers=1, num_heads=1, head_dim=8, rows_per_request=1, max_source_len=0, retention=SinkWindow(0, 4),
    )), "bounded KV 'bounded' takes"),
], ids=["ring", "cross", "bounded"])
def test_a_pool_that_takes_no_split_steps_stays_off_a_splitting_node(spec, refusal):
    shard = SeqShardSpec("shard", {"dit"}, ("main",), 0)
    assert spec().split_step_refusals("dit", shard, 1, 2, {}) == [
        f"{refusal} no split steps, so node 'dit' can't split them: run it on TP ranks alone"
    ], "the engine's build would refuse the node later, naming the manager rather than the pool"


# Two SP ranks against one, on CPU over gloo

SP, Q_HEADS, KV_HEADS, DIM = 2, 4, 2, 8
CPU = torch.device("cpu")


def _tokens(n: int, heads: int, seed: int) -> torch.Tensor:
    return torch.randn(n, heads, DIM, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


def _rank_config() -> PagedKVConfig:
    """One rank's pool, which holds its share of the heads."""
    return PagedKVConfig(
        num_layers=1, num_qo_heads=Q_HEADS // SP, num_kv_heads=KV_HEADS // SP, head_dim=DIM, max_seq_len=64,
        max_num_pages=16, page_size=4,
    )


class _Rank:
    """This rank's pool, dense kernel and sequence shard, which splits the gen walk, and the handles a node binds over
    them."""

    def __init__(self, group: CommGroup):
        joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=group, instance=group)
        array = Array("kv", nodes={"dit"}, layers=frozenset({0}), entry=KVEntry(heads=KV_HEADS, head_dim=DIM),
                      max_len=64, reads={"attn": Read(mask=Bidirectional(), op=Attend(q_heads=Q_HEADS))})
        declared = SeqShard("shard", nodes={"dit"}, walks=frozenset({"gen"}))
        (spec,) = build_specs([declared])
        resolve_spec_dependencies([
            KVSpec(resource_key="kv", nodes={"dit"}, config=_rank_config()),
            AttentionSpec(
                resource_key="attn", nodes={"dit"}, config=AttentionConfig(kv_cache="kv", backend=AttnBackend.DENSE),
            ),
            spec,
        ])
        self.shard = spec.resource_class.build(spec, EngineResourceInfo(device=CPU, joint_comm_group=joint))
        self.pool = KVManager(
            cfg=_rank_config(), name="kv", joint_comm_group=None,
            transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")), device=CPU,
            dtype=torch.float64,
        )
        self.dense = DenseAttentionManager(
            kv_cache="kv", device=CPU, dtype=torch.float64, kv_config=_rank_config(), wrapper_class=_CpuVarlenSdpa,
        )
        self.dense._shard_key, self.dense._split_exchange = "shard", spec.exchange
        self.dense.bind_dependencies({"kv": self.pool, "shard": self.shard})
        resolved = _resolve([array], StateSetup())
        self.handle = build_handle(array, {"kv": self.pool, "attn": self.dense}, resolved)
        self.bracket = build_handle(declared, {"shard": self.shard}, resolved)
        self.pool.ingest_request("a")

    def step(self, walk: str, keep: bool, q, k, v) -> tuple[torch.Tensor, bool]:
        """One step on ``walk`` of the whole ``q``, ``k`` and ``v``, cut to this rank's share as a node's stack is: its
        output, whole again, and whether it split."""
        segments = (Segment("a", "main", q.shape[0]),)
        ctx = StepContext(request_ids=("a",), graph_walk=walk, slot=0, capture=False)
        kv_step = KVStep(segments=segments, commit=keep)
        ctx.plan_results["shard"] = self.shard.plan(ResourceStep(segments=segments), ctx)
        assert self.pool.admit(kv_step, ctx).ok
        ctx.plan_results["kv"] = self.pool.plan(kv_step, ctx)
        self.dense.plan(AttentionStep(causal=False), ctx)
        step = types.SimpleNamespace(
            state={"kv": ArrayStep(keep=keep)}, segments=segments, steps={}, combined_labels={},
            ctx=types.SimpleNamespace(replays=False, capture=False, graph_walk=walk),
        )
        for opened in (self.handle, self.bracket):
            opened.begin_step(step)
        split = self.shard.split_range("main") is not None
        q, k, v = scatter(self.bracket, q, k, v)
        self.handle.add(0, k=k, v=v)
        out = gather(self.bracket, self.handle.read(0, q))
        for opened in (self.handle, self.bracket):
            opened.check_step(step)
            opened.end_step(step)
        self.pool.commit(kv_step, ctx)
        return out, split

    def kept(self) -> torch.Tensor:
        """The k this rank's pages hold for request a, in order."""
        stream, view, page = self.pool._streams["a"]["main"], self.pool.layer_view(0), self.pool.config.tokens_per_page
        return torch.stack([view[stream.page_indices[i // page], 0, i % page] for i in range(stream.stored_len)])


def _on_sp_ranks(rank: int, store: str, body) -> None:
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=SP)
    torch.set_num_threads(1)
    group = CommGroup(rank, rank, list(range(SP)))
    group.device_group, group.initialized = dist.group.WORLD, True
    try:
        body(rank, _Rank(group))
    finally:
        group.device_group = None
        dist.destroy_process_group()


def _kept_add_then_read(rank: int, on: _Rank) -> None:
    q, k, v = _tokens(7, Q_HEADS, 0), _tokens(7, KV_HEADS, 1), _tokens(7, KV_HEADS, 2)
    out, split = on.step("gen", True, q, k, v)
    assert split, "the step should split 7 tokens over the two ranks"
    torch.testing.assert_close(out, _sdpa(q, k, v))
    q, later_k, later_v = _tokens(5, Q_HEADS, 3), _tokens(5, KV_HEADS, 4), _tokens(5, KV_HEADS, 5)
    out, _ = on.step("gen", False, q, later_k, later_v)
    torch.testing.assert_close(out, _sdpa(q, torch.cat([k, later_k]), torch.cat([v, later_v])),
                               msg="a later step would read pages the split step's kept add never reached")


def test_a_kept_add_through_a_split_step_matches_one_rank(tmp_path):
    """Cosmos3 commits a window's entries on the dense kernel, which takes k and v at the read, through a split step's
    exchange; each rank pages its heads of every token, and a later step attends them as one rank would."""
    mp.spawn(_on_sp_ranks, args=(str(tmp_path / "store"), _kept_add_then_read), nprocs=SP)


def _copied_prefix(rank: int, on: _Rank) -> None:
    q, k, v = _tokens(7, Q_HEADS, 0), _tokens(7, KV_HEADS, 1), _tokens(7, KV_HEADS, 2)
    _, split = on.step("prefill", True, q, k, v)
    assert not split, "a walk the shard doesn't list should run copied"
    heads = KV_HEADS // SP
    torch.testing.assert_close(on.kept(), k[:, rank * heads:(rank + 1) * heads],
                               msg="a rank's pages would hold heads its exchange gives another rank")


def test_a_copied_step_keeps_its_rank_s_heads(tmp_path):
    """Cosmos3's text prefix runs copied on every rank into the pool its split steps read, which holds each rank's
    head group."""
    mp.spawn(_on_sp_ranks, args=(str(tmp_path / "store"), _copied_prefix), nprocs=SP)
