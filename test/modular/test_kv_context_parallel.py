"""Context parallelism on a paged KV pool: each page's slots striped over the ranks that
would otherwise hold the same KV head, with logical page ids the same on every rank; and
attention over it, each rank's partial over its slots exchanged and merged."""

from __future__ import annotations

import itertools
import logging
import math
import os
import sys
import types
from dataclasses import dataclass, replace
from unittest import mock

sys.path.insert(0, ".")

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from flashinfer_reference import reference_attention, reference_flashinfer
from test_kv_prefix_correctness import NODE, PAGE_SIZE, ROOT
from test_kv_prefix_lease import _StubTransfer
from test_kv_prefix_tp import WALKS, _ctx, _follow, _ingest, _send, _step, _tokens
from test_startup_graph_validation import _qwen3_5

from mstar.distributed import communication
from mstar.distributed.communication import CommGroup, JointGroups, WorkerParallelGroups
from mstar.engine import engine as engine_mod
from mstar.engine.accelerator_graph_config import BatchedAcceleratorGraphConfig
from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner
from mstar.engine.engine import Engine
from mstar.engine.resources.attn import flashinfer as fi_mod
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.attn.config import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    BlockCausalAttentionSpec,
)
from mstar.engine.resources.attn.ragged import wrappers as ragged_mod
from mstar.engine.resources.attn.wrappers import PagedKernel
from mstar.engine.resources.base import EngineResourceInfo, Resource, build_resource
from mstar.engine.resources.context_parallel import (
    OneSlotPages,
    OwnTokensKernel,
    SPPlacement,
    TPContextParallelWrapper,
    ZigzagWrapper,
    merge,
    zigzag_capture,
    zigzag_rows,
)
from mstar.engine.resources.convenience import AttentionCallable
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import KVCache, _kv_scatter_nhd_eager
from mstar.engine.resources.kv.config import KVLayout, KVReqConfig, KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SINK_PAGE, KVPlanOutput, KVPlanOutputs, SequenceView, build_paged_indptrs
from mstar.engine.resources.kv.transfer import KVReadInfo, ShmKVTransferEngine
from mstar.engine.resources.position.config import LongRoPE, PositionConfig, PositionReqConfig, PositionStep
from mstar.engine.resources.position.manager import LongRopeManager, RopeManager
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.sampler.config import SamplingReqConfig
from mstar.engine.resources.sequence_shard import SeqShardSpec, SequenceShard
from mstar.engine.resources.spec import NodeResourceSpec, apply_yaml_overrides, resolve_spec_dependencies
from mstar.engine.resources.step import BucketKey, ResourceStep, Segment, SlotLease, StepContext
from mstar.model.base import declared_resources
from mstar.model.qwen3_5.config import GDN_STATE, KV_CACHE
from mstar.model.submodule_base import NodeInputs, engine_step, with_segment_steps
from mstar.testing_utils.engine import make_engine, make_mgmt
from mstar.testing_utils.model import make_model

P = 2
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _group(rank: int, size: int) -> CommGroup:
    return CommGroup(my_global_rank=rank, my_group_rank=rank, group_members=list(range(size)))


def _sp_groups(*nodes: str) -> WorkerParallelGroups:
    """Rank 0 of a worker holding each of ``nodes`` over ``P`` SP ranks."""
    groups = WorkerParallelGroups(num_workers=P, global_rank=0)
    for node in nodes:
        group = _group(0, P)
        groups.add(node, group, "sp")
        groups.node_to_instance_comm[node] = group
    return groups


def _zero_unplanned_rows_on_cpu(out: torch.Tensor, qo_indptr: torch.Tensor) -> torch.Tensor:
    """``zero_unplanned_rows``, on a device whose Triton has no driver."""
    out[int(qo_indptr[-1]):] = 0
    return out


def _placed(rank: int) -> KVCache:
    config = PagedKVConfig(
        num_layers=2, num_kv_heads=1, head_dim=2, max_seq_len=64, num_qo_heads=2, max_num_pages=4,
        page_size=8, context_parallel=True,
    )
    joint = JointGroups(tp_group=_group(rank, P), sp_group=CommGroup.trivial(), instance=_group(rank, P))
    return KVCache(config, CPU, torch.float32, config.place(joint, "kv"))


def _manager(rank: int, cfg: PagedKVConfig) -> KVManager:
    # what `KVManager.build` does first: TP placement shards heads over the TP ranks
    cfg.shard(P)
    kv = KVManager(
        cfg=cfg, name="kv",
        joint_comm_group=JointGroups(tp_group=_group(rank, P), sp_group=CommGroup.trivial(), instance=_group(rank, P)),
        transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT, WALKS)
    kv._cg_max_seq_len = 64
    return kv


def _decode(kv: KVManager, rid: str, slot: int, leased: bool = True) -> tuple[KVStep, StepContext]:
    """One decode step of ``rid``, planned and not committed: into a capture slot's buffers when
    ``leased``, else eager."""
    step = KVStep(segments=(Segment(rid, "main", 1),))
    lease = SlotLease(slot=slot, bucket=BucketKey(graph_walk="decode", bs=1, num_tokens=1)) if leased else None
    ctx = StepContext(request_ids=(rid,), graph_walk="decode", slot=slot, capture=False, slot_lease=lease)
    ctx.set_padded_rids((rid,))
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    return step, ctx


def _assert_split_writes(ranks: list[KVManager], step: KVStep) -> None:
    """The ranks hold the same pages, and each token ``step`` writes goes to exactly one of
    them, the one owning its logical slot, at its physical slot; the other sends it to the sink."""
    logical = []
    for seg in step.segments:
        streams = [kv._streams[seg.request_id][seg.label] for kv in ranks]
        assert all(s.page_indices == streams[0].page_indices for s in streams), seg.request_id
        start = streams[0].stored_len
        pages = streams[0].page_indices
        logical += [(pages[pos // PAGE_SIZE], pos % PAGE_SIZE) for pos in range(start, start + seg.span)]
    writes = [list(zip(*(t.tolist() for t in kv.write_slots("main")), strict=True)) for kv in ranks]
    assert all(len(w) == len(logical) for w in writes)
    for token, (page, offset) in enumerate(logical):
        owners = [(rank, w[token]) for rank, w in enumerate(writes) if w[token] != (SINK_PAGE, 0)]
        assert owners == [(offset % P, (page, offset // P))], f"token {token}, logical ({page}, {offset})"


def test_two_ranks_keep_logical_pages_alike_and_split_every_write():
    # one config for both, as one process emulating TP builds them: each rank's placement is its own
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096, num_qo_heads=2, max_num_pages=12,
        page_size=PAGE_SIZE, prefix_cache_tp=True, context_parallel=True,
    )
    ranks = [_manager(0, cfg), _manager(1, cfg)]
    leader, follower = ranks
    a = _tokens(3 * PAGE_SIZE + 5, seed=1)
    for kv in ranks:
        _ingest(kv, "a", a)
    steps = [_step(kv, "a", len(a)) for kv in ranks]
    _assert_split_writes(ranks, steps[0])
    for kv, step in zip(ranks, steps, strict=True):
        kv.commit(step, _ctx("a"))

    # a prefix hit rank 0 decides and rank 1 adopts
    b = a[:3 * PAGE_SIZE] + _tokens(7, seed=2)
    for kv in ranks:
        _ingest(kv, "b", b)
    assert _follow(follower, _send(leader, "b"))
    steps = [_step(kv, "b", len(b)) for kv in ranks]
    assert steps[0].segments[0].span == 7
    _assert_split_writes(ranks, steps[0])
    for kv, step in zip(ranks, steps, strict=True):
        kv.commit(step, _ctx("b"))
    assert leader._streams["b"]["main"].page_indices[:3] == leader._streams["a"]["main"].page_indices[:3]

    # decode
    for slot in (0, 1):
        planned = [_decode(kv, "b", slot) for kv in ranks]
        _assert_split_writes(ranks, planned[0][0])
        for kv, (step, ctx) in zip(ranks, planned, strict=True):
            kv.commit(step, ctx)
    planned = [_decode(kv, "b", 0, leased=False) for kv in ranks]
    _assert_split_writes(ranks, planned[0][0])
    for kv, (step, ctx) in zip(ranks, planned, strict=True):
        kv.commit(step, ctx)

    # a request larger than the pool is refused alike
    for kv in ranks:
        kv.ingest_request("big", KVReqConfig())
        assert not kv.admit(KVStep(segments=(Segment("big", "main", 12 * PAGE_SIZE),)), _ctx("big")).ok
        kv.remove_request("big")

    # abort both; a new request evicts what they left cached
    c = _tokens(9 * PAGE_SIZE + 3, seed=3)
    for kv in ranks:
        kv.remove_request("a")
        kv.remove_request("b")
        _ingest(kv, "c", c)
    steps = [_step(kv, "c", len(c)) for kv in ranks]
    _assert_split_writes(ranks, steps[0])


@pytest.mark.parametrize("rank", range(P))
def test_a_placed_cache_maps_logical_token_ranges_to_this_ranks_slots(rank, tmp_path):
    source, destination = (_placed(rank) for _ in range(2))
    assert source.page_slots == 8 // P and source.tensor.shape[3] == 8 // P
    source.tensor.copy_(torch.arange(1, source.tensor.numel() + 1, dtype=torch.float32).reshape(source.tensor.shape))
    # logical slots 3-5 of a page: rank 0 owns 4 (physical 2), rank 1 owns 3 and 5 (physical 1 and 2)
    first, end = {0: (2, 3), 1: (1, 3)}[rank]
    page = source.tensor[:, 2]
    assert torch.equal(source.chunk_view(1, 2, 3, 6), page[1, :, first:end])
    rows = source.layer_rows(0, 2, 2, 3, 6)
    assert [r.tolist() for r in rows] == [page[:, kv, first:end].tolist() for kv in (0, 1)]
    ptrs, nbytes = source.chunk_ptrs(1, 2, 3, 6)
    assert ptrs == [page[1, kv, first].data_ptr() for kv in (0, 1)] and nbytes == page[1, 0, first:end].nbytes
    # a range holding none of this rank's slots is no transfer at all
    other = (1 - rank, 2 - rank)
    assert source.layer_rows(0, 2, 2, *other) == [] and source.chunk_ptrs(1, 2, *other) == ([], 0)

    # an SHM read of a partial page lands in this rank's slots of it, and nowhere else
    producer = ShmKVTransferEngine(source, "producer", str(tmp_path))
    consumer = ShmKVTransferEngine(destination, "consumer", str(tmp_path))
    info = producer.get_kv_transfer_info(request_id="r", label="main", page_indices=[1, 2], seq_len=14)
    assert consumer.read_batched_async(info, [KVReadInfo(layer, 0, 2, 3, 6) for layer in range(2)]) is None
    assert torch.equal(destination.tensor[:, 0, :, first:end], page[:, :, first:end])
    assert destination.tensor.count_nonzero() == page[:, :, first:end].numel()


class _Dependent(Resource):
    @classmethod
    def build(cls, spec, info):
        resource = cls()
        resource.supports_context_parallel = spec.supports
        resource.info = info
        return resource


@dataclass
class _DependentSpec(NodeResourceSpec):
    supports: bool = False

    @property
    def resource_class(self):
        return _Dependent

    def depends_on(self):
        return {KV_CACHE}


class _Submodule:
    disable_autocast = True
    forward = forward_batched = None

    def requires_grad_(self, flag):
        return self

    def bind_node_resources(self, resources):
        self.resources = resources


@pytest.mark.parametrize("supports", [False, True])
def test_an_sp_placed_node_shards_heads_over_tp_and_refuses_a_dependent_without_support(
    supports, tmp_path, monkeypatch,
):
    groups = _sp_groups(NODE)
    # Qwen3.5's own declarations (dummy shapes) with its pool placed, as on SP ranks, and a
    # dependent without support
    specs = declared_resources(_qwen3_5(NODE, {}, tmp_path, monkeypatch), {"resources": {
        KV_CACHE: {"context_parallel": True, "max_num_pages": 8}, GDN_STATE: {"max_slots": 4},
    }})
    kv_spec, = (spec for spec in specs if spec.resource_key == KV_CACHE)
    heads = (kv_spec.config.num_kv_heads, kv_spec.config.num_qo_heads)
    specs.append(_DependentSpec(resource_key="dependent", nodes={NODE}, supports=supports))
    engine = Engine(graph_runtime=None)
    load = lambda: engine.load_model(  # noqa: E731
        {NODE: _Submodule()}, specs=specs, parallel_groups=groups, device=CPU, transfer_engine_info=None,
    )
    if not supports:
        with pytest.raises(ValueError, match=f"'dependent'.*does not support context parallelism.*'{KV_CACHE}'"):
            load()
        return
    load()
    # heads over TP alone (1 rank), positions over the 2 SP ranks
    assert engine._resources["dependent"].info.head_shards == 1 and engine._submodules[NODE].head_shards == 1
    assert (kv_spec.config.num_kv_heads, kv_spec.config.num_qo_heads) == heads
    placement = engine._resources[KV_CACHE].placement
    assert isinstance(placement, SPPlacement) and (placement.size, placement.index) == (P, 0)
    assert engine._resources[KV_CACHE].kv_cache.page_slots == kv_spec.config.tokens_per_page // P
    # built by hand, as a harness does, with a node's info told nothing about the placement: a
    # resource over the sharded pool that shards it another way is refused
    unplaced = EngineResourceInfo(device=CPU, joint_comm_group=groups.get_joint_group_for_node(NODE),
                                  nodes=frozenset({NODE}))
    with pytest.raises(ValueError, match="already sharded 1 ways.*2 ways"):
        AttentionManager.build(
            AttentionSpec(resource_key="attn2", nodes={NODE}, config=AttentionConfig(kv_cache=KV_CACHE)),
            replace(unplaced, dependencies={KV_CACHE: kv_spec}),
        )


@pytest.mark.parametrize("kv, attn_spec, manager", [
    (dict(head_dim=576, layout=KVLayout.MLA, kv_lora_rank=512, qk_rope_head_dim=64),
     lambda: AttentionSpec(resource_key="attn", nodes={NODE}, config=AttentionConfig(
         kv_cache="kv", backend=AttnBackend.FLASHINFER_MLA, sm_scale=0.1,
     )), "FlashInferMLAManager"),
    (dict(head_dim=64),
     lambda: BlockCausalAttentionSpec(resource_key="attn", nodes={NODE}, block_size=8, config=AttentionConfig(
         kv_cache="kv", backend=AttnBackend.FLASHINFER,
     )), "FlashInferBlockCausalManager"),
], ids=["mla", "block_causal"])
def test_attention_over_a_placed_pool_that_cant_plan_it_is_refused_at_load(kv, attn_spec, manager):
    groups = _sp_groups(NODE)
    specs = [
        KVSpec(resource_key="kv", nodes={NODE}, config=PagedKVConfig(
            num_layers=1, num_kv_heads=1, max_seq_len=256, num_qo_heads=4, max_num_pages=8, page_size=PAGE_SIZE,
            context_parallel=True, **kv,
        )),
        attn_spec(),
    ]
    with pytest.raises(ValueError, match=f"'attn' .{manager}. does not support context parallelism.*'kv'"):
        Engine(graph_runtime=None).load_model(
            {NODE: _Submodule()}, specs=specs, parallel_groups=groups, device=CPU, transfer_engine_info=None,
        )


@pytest.mark.parametrize("kind", ["recurrent_state", "block_causal"])
def test_a_node_that_splits_its_steps_refuses_a_resource_that_cant_take_them(kind, tmp_path, monkeypatch):
    """Qwen3.5's recurrent state holds no split strategy, nor does block-causal attention (its
    builder sets no sequence shard): a node splitting its steps over SP ranks can't own one."""
    from mstar.engine.state.adapters import StateSetup
    from mstar.model.base import enabled_resources

    groups = _sp_groups(NODE)
    if kind == "recurrent_state":
        # what a deployment without a `lora:` block builds
        specs = enabled_resources(_qwen3_5(NODE, {}, tmp_path, monkeypatch), {}, StateSetup())
        apply_yaml_overrides(specs, {"resources": {KV_CACHE: {"max_num_pages": 8}, GDN_STATE: {"max_slots": 4}}})
    else:
        specs = [
            KVSpec(resource_key="kv", nodes={NODE}, config=PagedKVConfig(
                num_layers=1, num_kv_heads=2, head_dim=64, max_seq_len=256, num_qo_heads=4, max_num_pages=8,
                page_size=PAGE_SIZE,
            )),
            BlockCausalAttentionSpec(resource_key="attn", nodes={NODE}, block_size=8, config=AttentionConfig(
                kv_cache="kv", backend=AttnBackend.FLASHINFER,
            )),
        ]
    specs.append(SeqShardSpec("shard", {NODE}, ("main",), 0))
    with pytest.raises(ValueError, match="does not support steps split over SP ranks"):
        Engine(graph_runtime=None).load_model(
            {NODE: _Submodule()}, specs=specs, parallel_groups=groups, device=CPU, transfer_engine_info=None,
        )


def test_a_read_no_exchange_serves_on_a_node_that_splits_its_steps_is_refused_at_load(monkeypatch):
    """SDPA over keys the sequence shard splits would attend one rank's share. The engine refuses it from the
    declarations, before any step splits (`split_read_problems`). The node splits by its other read, on the ragged
    resource (a node whose every read is on SDPA never splits)."""
    from test_state_cross_reads import _array, _on_sdpa

    from mstar.engine.state.adapters import build_specs

    # other heads, so the two reads don't share one resource
    video, audio = _array("video", "video_self", nodes=(NODE,)), _array("audio", nodes=(NODE,), heads=2)
    model = types.SimpleNamespace(checkpoint_path=lambda: None, declare_state=lambda: [video, audio])
    monkeypatch.setattr(engine_mod, "sequence_parallel_refusals", lambda *args: [])
    setup = _on_sdpa("video_self")
    engine = Engine(graph_runtime=None, state_setup=setup)
    specs = [*build_specs([video, audio], setup=setup), SeqShardSpec("shard", {NODE}, ("video", "audio"), 0)]
    with pytest.raises(ValueError, match=r"backend sdpa over labels \['video'\]"):
        engine.load_model(
            {NODE: _Submodule()}, specs=specs,
            parallel_groups=_sp_groups(NODE), device=CPU, transfer_engine_info=None, model=model,
        )


@dataclass
class _SharedSpec(_DependentSpec):
    def depends_on(self):
        return set()


def test_a_resource_over_nodes_that_split_heads_differently_is_refused(monkeypatch):
    """A resource is sized by one head-shard count, so it can't serve a node that places its
    positions over its SP ranks (heads over TP) beside one splitting its steps (heads over TP x SP)."""
    other = "other"
    groups = _sp_groups(NODE, other)
    monkeypatch.setattr(engine_mod, "sequence_parallel_refusals", lambda *args: [])
    monkeypatch.setattr(engine_mod, "position_pools", lambda specs, node: {"kv": node == NODE})
    with pytest.raises(ValueError, match="'shared' serves nodes that split heads different ways"):
        Engine(graph_runtime=None).load_model(
            {NODE: _Submodule(), other: _Submodule()}, specs=[_SharedSpec(resource_key="shared", nodes={NODE, other})],
            parallel_groups=groups, device=CPU, transfer_engine_info=None,
        )


# --- attention over a placed pool -------------------------------------------------------------

HEAD_DIM, Q_HEADS, CP_PAGE = 64, 4, 4
# (stored context, new tokens) per row. Prefill: a row with no context, one whose 1-token
# context only rank 0 holds, and contexts ending on a page whose one token rank 1 lacks.
PREFILL_ROWS = ((0, 5), (1, 3), (5, 2), (6, 1))
# decode, after the write: lengths 1 (rank 1 holds nothing), 5 (a last page rank 1 lacks), 7
DECODE_ROWS = ((0, 1), (4, 1), (6, 1))


@dataclass
class _Case:
    """One step's KV plan, the logical keys of its rows, and its queries."""
    plan: KVPlanOutputs
    keys: list[torch.Tensor]       # per row, every position's [2, 1, D] K and V
    q: torch.Tensor                # [N, Q_HEADS, D]
    rows: tuple[tuple[int, int], ...]

    @classmethod
    def build(cls, rows, device, dtype, padding: tuple[int, int] = (0, 0)) -> _Case:
        """``padding``: ``(rows, span)`` of a replay's padding rows after ``rows``, on SINK_PAGE as
        `KVManager` plans them; their tokens follow the step's in ``inputs``."""
        gen = torch.Generator().manual_seed(0)
        views, keys, page = [], [], 1
        for i, (context, new) in enumerate(rows):
            length = context + new
            pages = list(range(page, page + -(-length // CP_PAGE)))
            page += len(pages)
            views.append(SequenceView(f"r{i}", "main", pages, length, new))
            keys.append(torch.randn(length, 2, 1, HEAD_DIM, generator=gen).to(device, dtype))
        q = torch.randn(sum(new for _, new in rows), Q_HEADS, HEAD_DIM, generator=gen).to(device, dtype)
        span = padding[1]
        views += [SequenceView(f"pad{i}", "main", [SINK_PAGE] * -(-span // CP_PAGE), span, span)
                  for i in range(padding[0])]
        indptrs = build_paged_indptrs(views, CP_PAGE)
        return cls(KVPlanOutputs({"main": KVPlanOutput(indptrs, views)}), keys, q, rows)

    @property
    def views(self) -> list[SequenceView]:
        """The real rows' views, without the padding rows'."""
        return self.plan["main"].views[:len(self.rows)]

    def cache(self, placement, pages: int = 0, stored_only: bool = False) -> torch.Tensor:
        """This rank's layer of the pool, of at least ``pages`` pages: each owned slot at its
        physical place, NaN elsewhere (SINK_PAGE included; with ``stored_only``, the step's own slots too)."""
        slots = CP_PAGE // placement.size
        pages = max(pages, max(view.page_idxs[-1] for view in self.views) + 1)
        cache = torch.full((pages, 2, slots, 1, HEAD_DIM), math.nan, device=self.q.device, dtype=self.q.dtype)
        for view, kv in zip(self.views, self.keys, strict=True):
            for pos in range(view.length - view.to_compute if stored_only else view.length):
                offset = pos % CP_PAGE
                if offset % placement.size == placement.index:
                    page, slot = view.page_idxs[pos // CP_PAGE], offset // placement.size
                    cache[page, :, slot].copy_(kv[pos])
        return cache

    def new_keys(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The step's own tokens' K and V ``[N, 1, D]``, as the layer passes them to ``run``."""
        new = torch.cat([kv[context:] for kv, (context, _) in zip(self.keys, self.rows, strict=True)])
        return new[:, 0], new[:, 1]

    def inputs(self, placement, heads=slice(None), keys=True, pad=0, pages=0) -> list[torch.Tensor | None]:
        """What ``runner`` passes ``attn.run``: the query, the new keys (None without ``keys``) and
        this rank's cache (``cache``). ``pad`` NaN tokens follow the step's, as the padding rows of a
        captured bucket larger than the step hold whatever the model's layers left there."""
        q, k, v = self.q[:, heads].contiguous(), *self.new_keys()
        if not keys:
            k = v = None
        if pad:
            q, k, v = (torch.cat([t, t.new_full((pad, *t.shape[1:]), math.nan)]) for t in (q, k, v))
        return [q, k, v, self.cache(placement, pages)]

    def runner(self, attn, placement, heads=slice(None), keys=True, pad=0, inputs=None):
        """``attn.run`` for ``heads`` as a call whose ``inputs`` are built once (so it can be
        captured), by default ``self.inputs(placement, heads, keys, pad)``."""
        if inputs is None:
            inputs = self.inputs(placement, heads, keys, pad)
        q, k, v, cache = inputs
        return lambda: attn.run(q, "main", kv_cache_layer=cache, k=k, v=v)

    def want(self, causal: bool = True) -> torch.Tensor:
        outs = []
        qo = 0
        for kv, (_, new) in zip(self.keys, self.rows, strict=True):
            outs.append(reference_attention(self.q[qo:qo + new], kv[:, 0], kv[:, 1], causal)[0])
            qo += new
        return torch.cat(outs)


def _cp_attention(
    joint: JointGroups, device: torch.device, dtype: torch.dtype, shard: SequenceShard | None = None,
    exchange: str = "auto", kv_resource=None, **attention,
):
    """A ``FlashInferManager`` over a placed pool, and this rank's placement; under ``shard`` (key
    ``"shard"``) with its spec's ``exchange``, bound to the KV resource ``kv_resource(placement)``
    builds (by default one holding only the placement)."""
    kv = KVSpec(resource_key="kv", nodes={NODE}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=HEAD_DIM, max_seq_len=64, num_qo_heads=Q_HEADS,
        max_num_pages=16, page_size=CP_PAGE, context_parallel=True,
    ))
    dependencies = {"kv": kv}
    spec = AttentionSpec(resource_key="attn", nodes={NODE}, config=AttentionConfig(kv_cache="kv", **attention))
    if shard is not None:
        spec.shard_key = "shard"
        dependencies["shard"] = SeqShardSpec("shard", {NODE}, ("main",), 0, exchange=exchange)
    info = EngineResourceInfo(
        device=device, joint_comm_group=joint, kv_dtype=dtype, dependencies=dependencies,
        # SP placement replicates the node over the SP ranks, heads split over TP alone
        replicated_over_sp=joint.sp_group.world_size > 1,
    )
    attn = AttentionManager.build(spec, info)
    placement = kv.config.place(joint, "kv")
    # by default the KV resource's one field attention reads
    resource = types.SimpleNamespace(placement=placement, bind_sequence_shard=lambda shard: None) \
        if kv_resource is None else kv_resource(placement)
    attn.bind_dependencies({"kv": resource, "shard": shard})
    return attn, placement


def _plan(
    attn, case: _Case, slot: int = 0, bucket: BucketKey | None = None, capture=False, shard=None, **step,
) -> None:
    """Plans ``case``, its padding rows as a replay's (a capture's rows are all its own), and first
    on ``shard`` (key ``"shard"``) if given."""
    views = case.plan["main"].views
    real = views if capture else case.views
    ctx = StepContext(request_ids=tuple(v.request_id for v in real), graph_walk="w", slot=slot, capture=capture,
                      slot_lease=None if bucket is None else SlotLease(slot, bucket))
    if len(real) < len(views):
        ctx.set_padded_rids(tuple(v.request_id for v in views))
    ctx.plan_results["kv"] = case.plan
    if shard is not None:
        segments = tuple(Segment(v.request_id, "main", v.to_compute) for v in views)
        ctx.plan_results["shard"] = shard.plan(ResourceStep(segments=segments), ctx)
    attn.plan(AttentionStep(**step), ctx)


class _PlacedKV:
    """The KV resource an attention over a placed pool binds, and its ``AttentionCallable`` writes
    through: one layer holding ``case``'s stored context, the step's slots NaN until written.
    ``write_kv`` is ``KVManager``'s own, over a plan addressing each step token as a placed
    pool's does (``write_slots``) and the cache's own scatter."""

    write_kv = KVManager.write_kv

    def __init__(self, placement, case: _Case):
        self.placement, self._sequence_shard = placement, None
        self._default_label, self._default_layer_idx, self._layer_group = "main", 0, [0]
        self.cache = case.cache(placement, stored_only=True)
        logical = [(view.page_idxs[pos // CP_PAGE], pos % CP_PAGE) for view in case.plan["main"].views
                   for pos in range(view.length - view.to_compute, view.length)]
        pages, offsets = placement.write_slots(*(np.array(a) for a in zip(*logical, strict=True)))
        page, slot = (torch.as_tensor(a, dtype=torch.long, device=case.q.device) for a in (pages, offsets))
        self._current_plan_states = {("main", 0): types.SimpleNamespace(
            ready=None, token_to_page=page, token_to_cache=slot, total_tokens=len(logical),
        )}
        self.kv_cache = types.SimpleNamespace(
            write_tokens=lambda layer_idx, k, v, page_idx, cache_idx: _kv_scatter_nhd_eager(
                self.cache, k, v, page_idx, cache_idx,
            ),
        )

    def bind_sequence_shard(self, shard):
        self._sequence_shard = shard

    def set_default_label(self, label):
        self._default_label = label

    def set_default_layer_idx(self, layer_idx):
        assert layer_idx == 0

    def layer_view(self):
        return self.cache


def _over_ranks(rank, world, backend, store, over_sp, rows, captured=False, padding=(0, 0), causal=True, split=None):
    """One rank of a ``world``-rank placement running ``rows`` through ``FlashInferManager.run``
    against one rank attending every key; every rank issues the same collectives, same shapes.
    ``captured`` (NCCL only) does what production does for a graph bucket: captures ``run`` with
    its collectives on a step of exactly the bucket's tokens on every rank, then plans the step and
    replays it. ``padding``: ``(rows, span)`` of the replay's padding rows, which make its bucket
    larger than the step: on SINK_PAGE as `KVManager` plans them, and NaN in their queries, keys
    and the sink. ``causal`` is the step's. With ``split`` (the shard's ``exchange``, SP placement),
    the step runs through a sequence shard instead (``_split_step``), whose ring hops pair up rank
    by rank rather than match."""
    os.environ["MSTAR_WORKSPACE_BUFFER_MB"] = "64" if backend == "nccl" else "1"
    dist.init_process_group(backend, init_method=f"file://{store}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    cuda = backend == "nccl"
    device = torch.device("cuda", rank) if cuda else CPU
    if cuda:
        torch.cuda.set_device(device)
    else:
        sys.modules["flashinfer"] = reference_flashinfer()
    calls = []
    # `CommGroup.all_gather` calls the into-tensor gather it picked at import
    for owner, name in ((dist, "all_to_all_single"), (communication, "_all_gather_into_tensor")):
        def record(*args, _name=name, _call=getattr(owner, name), **kwargs):
            calls.append((_name, tuple(args[0].shape), tuple(args[1].shape), *args[2:]))
            return _call(*args, **kwargs)
        setattr(owner, name, record)
    group = CommGroup(rank, rank, list(range(world)))
    group.device_group, group.initialized = dist.group.WORLD, True
    graph = None
    try:
        joint = (JointGroups(tp_group=CommGroup.trivial(), sp_group=group, instance=group) if over_sp
                 else JointGroups(tp_group=group, sp_group=CommGroup.trivial(), instance=group))
        dtype = torch.bfloat16 if cuda else torch.float32
        if split is not None:
            _split_step(joint, device, dtype, split, causal)
            everyone = [None] * world
            dist.all_gather_object(everyone, calls)
            _assert_paired(everyone)
            return
        attn, placement = _cp_attention(joint, device, dtype)
        case = _Case.build(rows, device, dtype, padding)
        # the slot plans the other kind of step first
        _plan(attn, _Case.build(DECODE_ROWS if rows == PREFILL_ROWS else PREFILL_ROWS, device, dtype))
        # a captured prefill's bucket holds 8 tokens past the step's rows, its padding rows' included
        n = case.q.shape[0]
        tail = 8 if captured and rows == PREFILL_ROWS else 0
        pad = padding[0] * padding[1] + tail
        bucket = BucketKey("w", bs=len(rows) + padding[0], num_tokens=n + pad) if captured else None
        heads = slice(None) if over_sp else slice(rank * Q_HEADS // world, (rank + 1) * Q_HEADS // world)
        # the bucket's capture step: the step with the bucket's every row and token, the padding rows
        # on the sink (FlashInfer's run refuses a q of another length than its plan; a replay runs none)
        extended = (*rows[:-1], (rows[-1][0], rows[-1][1] + tail))
        full = _Case.build(extended, device, dtype, padding) if pad else case
        _plan(attn, full, bucket=bucket, capture=captured, causal=causal)
        # a one-token-a-row step attends the pages alone, so a fused KV-write kernel can serve it
        assert attn.reads_step_kv() == (rows == PREFILL_ROWS)
        if rows == PREFILL_ROWS:
            with pytest.raises(ValueError, match="got none"):
                full.runner(attn, placement, heads, keys=False)()
        tol = {"atol": 2e-2, "rtol": 2e-2} if cuda else {}
        inputs = full.inputs(placement, heads, pad=padding[0] * padding[1])
        run = full.runner(attn, placement, inputs=inputs)
        out = run()
        if captured:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = run()
            _plan(attn, case, bucket=bucket, causal=causal)
            for into, step in zip(inputs, case.inputs(placement, heads, pad=pad, pages=inputs[-1].shape[0]),
                                  strict=True):
                into.copy_(step)
            # what the checks read is the replay's
            out.fill_(math.nan)
            graph.replay()
            torch.cuda.synchronize()
        if rows == PREFILL_ROWS:
            # a prefill's padding merges to 0: its own-token part plans the real rows alone
            assert torch.isfinite(out).all() and not out[n:].any()
        else:
            # a padding row's output may be NaN (a padded decode's row attends the NaN sink); a real
            # row's may not
            assert torch.isfinite(out[:n]).all()
        torch.testing.assert_close(out[:n].float(), case.want(causal)[:, heads], **tol)
        if rows == PREFILL_ROWS and not captured:
            # a first prefill: no row holds stored context on any rank, so TP exchanges nothing and
            # SP gathers its partners' shares of the own tokens alone
            first = _Case.build(((0, 5), (0, 3)), device, dtype)
            _plan(attn, first)
            exchanged = len(calls)
            context = attn._current_plan_states["main"].context
            attend_slots, context.run = context.run, None
            torch.testing.assert_close(first.runner(attn, placement, heads)().float(), first.want()[:, heads], **tol)
            context.run = attend_slots
            assert len(calls) == exchanged + over_sp, "a step with no stored context ran the exchange"
            # one stored token, which one rank alone holds: TP's exchange would otherwise run on that
            # rank alone; SP gathers either way
            one = _Case.build(((1, 3),), device, dtype)
            _plan(attn, one)
            torch.testing.assert_close(one.runner(attn, placement, heads)().float(), one.want()[:, heads], **tol)
        everyone = [None] * world
        dist.all_gather_object(everyone, calls)
        assert calls and all(c == calls for c in everyone), everyone
    finally:
        if graph is not None:
            # a live graph holds its captured NCCL work, and destroy waits on it
            graph.reset()
        # clear our reference so destroy drops the group's last one and joins gloo's threads,
        # whatever else holds the group
        group.device_group = None
        dist.destroy_process_group()


def _split_step(joint: JointGroups, device: torch.device, dtype: torch.dtype, split: str, causal: bool) -> None:
    """``PREFILL_ROWS`` (odd, rows with and without stored context, one shorter than P, a riding
    decode row) through a sequence shard and ``AttentionCallable`` over a placed pool whose step
    slots hold NaN until the step writes them: split by ``split``, then copied on the same node
    (``min_tokens`` above the step), with the step's ``causal``. Each gathers to one rank's
    attention, and the sampler's rows are each row's last token of the whole step."""
    group = joint.sp_group
    shard = SequenceShard(("main",), 0, None, 0, group, stripe_pool="kv" if split == "ring" else None, device=device)
    case = _Case.build(PREFILL_ROWS, device, dtype)

    def placed_kv(placement):
        nonlocal kv
        kv = _PlacedKV(placement, case)
        return kv

    kv = None
    attn, _ = _cp_attention(joint, device, dtype, shard=shard, exchange=split, kv_resource=placed_kv)
    attend = AttentionCallable(kv, attn)
    attend.bind_step("main")
    attend.set_layer_idx(0)
    q, (k, v) = case.q, case.new_keys()
    n, tol = q.shape[0], {"atol": 2e-2, "rtol": 2e-2} if device.type == "cuda" else {}
    for split_it in (True, False):
        shard.min_tokens = 0 if split_it else n + 1
        kv.cache = case.cache(kv.placement, stored_only=True)
        _plan(attn, case, shard=shard, causal=causal)
        assert (shard.split_range("main") is not None) == split_it
        # a copied step over a pool stored by position attends as a node without a shard does, so a
        # layer's fused norm/RoPE/KV write stays on (`AttentionCallable.norm_rope_attend`)
        assert attn.step_exchanges() == split_it
        if split_it:
            # a K/V write without attention lands each token on the rank storing it (page 0, the sink, aside)
            attend.write(*shard.scatter("main", k, v))
            torch.testing.assert_close(kv.cache[1:], case.cache(kv.placement)[1:], equal_nan=True)
        # the ring's hops carry queries in their own dtype and partials in the kernel's
        queries = q.double() if split == "ring" and split_it else q
        out = shard.gather("main", attend(*shard.scatter("main", queries, k, v)))
        assert torch.isfinite(out).all(), "a step token's K/V wasn't written"
        torch.testing.assert_close(out.float(), case.want(causal), **tol)
        last = attn.select_last_hidden(torch.arange(n, device=device)[:, None])
        assert last.flatten().tolist() == (np.cumsum([new for _, new in PREFILL_ROWS]) - 1).tolist()


def _assert_paired(everyone: list) -> None:
    """Every rank ran the same collectives in the same order; each all-gather the same shapes, and
    each all-to-all's counts pair up: what rank a sends b is what b receives from a."""
    assert everyone[0] and all([c[0] for c in calls] == [c[0] for c in everyone[0]] for calls in everyone), everyone
    for i, call in enumerate(everyone[0]):
        if call[0] != "all_to_all_single":
            assert all(calls[i] == call for calls in everyone), everyone
            continue
        # (name, recv shape, send shape, recv counts, send counts)
        for a, sender in enumerate(everyone):
            assert [receiver[i][3][a] for receiver in everyone] == sender[i][4], everyone


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("split", ["all_gather", "ring"])
def test_sp_ranks_split_a_step_over_a_placed_pool_like_one_rank(tmp_path, split, world, causal):
    mp.spawn(_over_ranks, nprocs=world,
             args=(world, "gloo", str(tmp_path / "store"), True, PREFILL_ROWS, False, (0, 0), causal, split))


def test_a_ring_plans_each_partners_first_query_apart_where_it_sees_no_key_here(monkeypatch):
    """A fresh prompt's first tokens sit before some ranks' first keys: each such query plans as
    a one-query row over a sink slot, so no causal row holds more queries than keys (which
    FlashInfer refuses), and a partner holding none of a row plans none of it."""
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    monkeypatch.setitem(sys.modules, "flashinfer", reference_flashinfer())
    plans = []
    paged = sys.modules["flashinfer"].BatchPrefillWithPagedKVCacheWrapper.plan

    def planned(self, qo_indptr=None, *args, seq_lens=None, causal=False, **kwargs):
        plans.append((np.diff(qo_indptr.numpy()).tolist(), seq_lens.tolist(), causal))
        return paged(self, qo_indptr, *args, seq_lens=seq_lens, causal=causal, **kwargs)

    monkeypatch.setattr(sys.modules["flashinfer"].BatchPrefillWithPagedKVCacheWrapper, "plan", planned)
    # the ranks are planned in one process, with no group to connect
    monkeypatch.setattr(CommGroup, "connect_pairs", lambda self, device: None)
    world = 4
    case = _Case.build(((0, 6), (0, 1)), CPU, torch.float32)
    for rank in range(world):
        joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=_group(rank, world), instance=_group(rank, world))
        shard = SequenceShard(("main",), 0, None, 0, joint.sp_group, stripe_pool="kv")
        attn, _ = _cp_attention(
            joint, CPU, torch.float32, shard=shard, exchange="ring", kv_resource=lambda p: _PlacedKV(p, case),
        )
        plans.clear()
        _plan(attn, case, shard=shard)
        assert len(plans) == world and all(causal for _, _, causal in plans)
        for j, (queries, keys, _) in enumerate(plans):
            # partner j holds positions j and j + 4 of the first row, and position 0 of the second if j is 0
            assert sum(queries) == (2 if j < 2 else 1) + (j == 0), (rank, j, queries)
            assert all(q <= k for q, k in zip(queries, keys, strict=True)), (rank, j, queries, keys)
            # a first query this rank has no key at or before: its own row of one, over a sink slot
            assert (queries[0] == 1 and len(queries) > 1) == (j < rank and j < 2), (rank, j, queries)


@pytest.mark.parametrize("size", [2, 3, 4])
def test_zigzag_rows_split_a_steps_tokens_over_the_ranks_with_equal_causal_work(size):
    """SP partners attend a step's own tokens in zigzag chunks: together every token once, and on
    rows a multiple of 2P long, each the same number of query-key pairs."""
    for lens in ([2 * size, 6 * size, 4 * size], [6 * size + 1, 2 * size - 1, 3, 1]):
        qo = np.concatenate([[0], np.cumsum(lens)])
        shares = [zigzag_rows(qo, size, rank)[0] for rank in range(size)]
        assert sorted(np.concatenate(shares).tolist()) == list(range(qo[-1]))
        if all(n % (2 * size) == 0 for n in lens):
            # a query at position t of its row sees t + 1 keys
            work = {int((s - qo[np.searchsorted(qo, s, side="right") - 1] + 1).sum()) for s in shares}
            assert len(work) == 1, work


class _IdleContext(PagedKernel):
    """A context kernel for a test of the own-token part alone."""

    def plan(self, **kwargs):
        pass

    def run(self, q, kv_cache_layer, return_lse=False):
        raise AssertionError("the context isn't run here")


class _OneSlotPrefill(PagedKernel):
    """A paged prefill as ``OneSlotPages`` plans it, as torch math over its pages of one token."""

    def plan(self, qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, kv_lens, causal, dtype):
        self.qo, self.kv = qo_indptr.tolist(), paged_kv_indptr.tolist()
        self.indices, self.causal = paged_kv_indices.long(), causal

    def run(self, q, kv, return_lse):
        k, v = (pages.flatten(0, 1) for pages in kv)
        parts = [
            reference_attention(q[a:b], k[self.indices[c:d]], v[self.indices[c:d]], self.causal)
            for (a, b), (c, d) in zip(itertools.pairwise(self.qo), itertools.pairwise(self.kv), strict=True)
        ]
        return torch.cat([out for out, _ in parts]), torch.cat([lse for _, lse in parts])


def test_placed_attention_refuses_a_kernel_outside_its_contract(monkeypatch):
    """A backend's kernels are checked when they are built, not when a step first runs them."""
    monkeypatch.setitem(sys.modules, "flashinfer", reference_flashinfer())
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    attn, _ = _cp_attention(joint, CPU, torch.float32)
    kwargs = dict(attn._wrapper_kv_kwargs)
    good = attn.paged_kernel
    monkeypatch.setattr(attn, "paged_kernel", lambda *args: types.SimpleNamespace())
    with pytest.raises(ValueError, match="PagedKernel"):
        attn._context_parallel_wrapper("main", 0, False, kwargs)
    monkeypatch.setattr(attn, "paged_kernel", good)
    for wrong in (types.SimpleNamespace(), None):
        monkeypatch.setattr(attn, "own_tokens_kernel", lambda *args, wrong=wrong: wrong)
        with pytest.raises(ValueError, match="OwnTokensKernel"):
            attn._context_parallel_wrapper("main", 0, False, kwargs)
    # a kernel for whole rows alone, where SP's zigzag share would plan other rows on it
    joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=_group(0, P), instance=_group(0, P))
    attn, _ = _cp_attention(joint, CPU, torch.float32)
    whole_only = type("WholeOnly", (_GatheredKeys,), {"whole_rows_only": True})
    monkeypatch.setattr(attn, "own_tokens_kernel", lambda *args: whole_only())
    with pytest.raises(ValueError, match="whole rows"):
        attn._context_parallel_wrapper("main", 0, False, dict(attn._wrapper_kv_kwargs))
    # a kernel that says it is sized for other than the capture it was built for, or for none
    other = zigzag_capture(P, 1, 8)
    for built, says in ((16, None), (16, other), (None, other)):
        monkeypatch.setattr(attn, "own_tokens_kernel", lambda *args, says=says: _GatheredKeys(says))
        with pytest.raises(ValueError, match="says it is sized for"):
            attn._context_parallel_wrapper("main", 0, False, dict(attn._wrapper_kv_kwargs, batch_size=2), built)


class _GatheredKeys(OwnTokensKernel):
    """A kernel with no pages: each row's keys gathered by their token ids, one copy of them."""

    def __init__(self, captured=None):
        self.captured = captured

    def plan(self, rows, causal):
        self.rows, self.causal = rows, causal

    def run(self, q, k, v):
        qo, kv, tokens = self.rows
        parts = [
            reference_attention(q[a:b], k[tokens[c:d]], v[tokens[c:d]], self.causal)
            for (a, b), (c, d) in zip(itertools.pairwise(qo.tolist()), itertools.pairwise(kv.tolist()), strict=True)
        ]
        # a captured kernel's queries past its plan's: the wrapper drops them
        spare = q.shape[0] - int(qo[-1])
        outs = [out for out, _ in parts] + [q.new_zeros(spare, *q.shape[1:])]
        lses = [lse for _, lse in parts] + [q.new_full((spare, q.shape[1]), -math.inf)]
        return torch.cat(outs), torch.cat(lses)


@pytest.mark.parametrize("causal", [True, False], ids=["causal", "noncausal"])
@pytest.mark.parametrize("kernel", [
    lambda captured: OneSlotPages(_OneSlotPrefill(), torch.float32, captured), _GatheredKeys,
], ids=["one-slot-pages", "no-pages"])
def test_captured_zigzag_shares_replan_in_place_and_merge_to_attention_over_a_larger_bucket(causal, kernel):
    """SP partners' zigzag shares of a step's own tokens on wrappers captured for a bucket larger
    than the step, planned for one step and then another (with a zero-length padding row, as the
    graph runner pads): each plan writes the buffers a replay reads, the bucket's padding tokens
    merge to 0, and the real ones to their rows' attention, on a paged kernel or on one without
    pages."""
    rows, tokens = 3, 16
    captured = zigzag_capture(P, rows, tokens)
    wrappers = [ZigzagWrapper(SPPlacement(P, index, CommGroup.trivial()), kernel(captured), CPU) for index in range(P)]
    addresses = [(w._gather.data_ptr(), w._scatter.data_ptr()) for w in wrappers]
    gen = torch.Generator().manual_seed(0)
    for lens in ((5, 3, 1), (2, 7, 0)):
        qo = torch.tensor(np.concatenate([[0], np.cumsum(lens)]), dtype=torch.int32)
        q, k, v = (torch.randn(tokens, heads, HEAD_DIM, generator=gen) for heads in (Q_HEADS, 1, 1))
        shares = []
        for wrapper in wrappers:
            wrapper.plan(qo, causal=causal)
            shares.append(wrapper.run(q, k, v))
        assert [(w._gather.data_ptr(), w._scatter.data_ptr()) for w in wrappers] == addresses
        got = merge(torch.stack([out for out, _ in shares]), torch.stack([lse for _, lse in shares]))
        assert torch.isfinite(got).all() and not got[int(qo[-1]):].any()
        want = torch.cat([
            reference_attention(q[a:b], k[a:b], v[a:b], causal)[0] for a, b in itertools.pairwise(qo.tolist())
        ])
        torch.testing.assert_close(got[: int(qo[-1])], want)


class _OneRank:
    """A Qwen3.5 layer's KV handle as one rank attending every key: ``case``'s stored context, then
    the k and v the layer adds, which it writes into ``case`` as the step's own tokens."""

    def __init__(self, case: _Case):
        self.case = case

    def _add_read(self, layer, q, *, keep, k, v):
        assert keep is None, "this fake keeps every add"
        self.add(layer, k=k, v=v)
        return self.read(layer, q)

    def add(self, layer, *, k, v):
        new, at = torch.stack([k, v], 1), 0
        for kv, (context, n) in zip(self.case.keys, self.case.rows, strict=True):
            kv[context:] = new[at:at + n]
            at += n

    def read(self, layer, q):
        self.case.q = q.contiguous()
        return self.case.want()


def _layer_over_ranks(rank, world, store):
    """One rank of an SP placement running a Qwen3.5 attention layer through its KV handle on a
    prefill, against the same layer attending every key on one rank."""
    from mstar.engine.state.adapters import StateSetup, _resolve, build_handle
    from mstar.engine.state.analysis import position_key
    from mstar.engine.state.contract import ArrayStep
    from mstar.model.qwen3_5.components.language_model import Qwen3_5Attention, RopeCache
    from mstar.model.qwen3_5.config import ATTN

    os.environ["MSTAR_WORKSPACE_BUFFER_MB"] = "1"
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    sys.modules["flashinfer"] = reference_flashinfer()
    group = CommGroup(rank, rank, list(range(world)))
    group.device_group, group.initialized = dist.group.WORLD, True
    try:
        joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=group, instance=group)
        attn, placement = _cp_attention(joint, CPU, torch.float32)
        (array,) = [a for a in _qwen3_5(NODE, {}, None, None).declare_state() if a.name == KV_CACHE]
        (layer_id,) = array.layers
        torch.manual_seed(0)
        rope = RopeCache()
        layer = Qwen3_5Attention(
            rope_cache=rope, hidden_size=32, num_heads=Q_HEADS, num_kv_heads=1, head_dim=HEAD_DIM, qk_norm=True,
            attn_key=ATTN, kv_key=KV_CACHE, pos_key=None, layer=layer_id,
        )
        for param in layer.parameters():
            param.data.normal_(0, 0.1)
        case = _Case.build(PREFILL_ROWS, CPU, torch.float32)
        n = case.q.shape[0]
        hidden = torch.randn(n, 32)
        # a rotation of none: the handle, not the rope, is under test
        rope.set(torch.ones(n, 16), torch.zeros(n, 16))
        layer.bind_state({KV_CACHE: _OneRank(case)})
        want = layer(hidden)

        _plan(attn, case)
        cache = case.cache(placement)
        pool = types.SimpleNamespace(
            layer_view=lambda i: cache, write_kv=lambda *args, **kwargs: None, opens_label=lambda *args: True,
        )
        resources = {KV_CACHE: pool, ATTN: attn, position_key(array): None}
        handle = build_handle(array, resources, _resolve([array], StateSetup()))
        step = types.SimpleNamespace(
            state={KV_CACHE: ArrayStep()}, segments=tuple(
                Segment(view.request_id, "main", new) for view, (_, new) in zip(case.views, case.rows, strict=True)
            ),
            ctx=types.SimpleNamespace(replays=False, capture=False),
        )
        handle.begin_step(step)
        layer.bind_state({KV_CACHE: handle})
        torch.testing.assert_close(layer(hidden), want)
        handle.end_step(step)
    finally:
        group.device_group = None
        dist.destroy_process_group()


def test_a_qwen3_5_layer_attends_a_placed_pool_through_its_handle_like_one_rank(tmp_path):
    # a prefill attends its own tokens from the k and v the layer adds, beside the pool
    mp.spawn(_layer_over_ranks, args=(P, str(tmp_path / "store")), nprocs=P)


@pytest.mark.parametrize("attention, refused", [
    # as long as the pool's sequences, so it keeps no window of pages and gets past the window checks
    (dict(sliding_window=64), "sliding_window"),
    # on a host without FA3 dense degrades to FlashInfer, which would serve the pool
    (dict(backend=AttnBackend.DENSE), "backend 'dense'"),
    # its decode lse base isn't known
    (dict(flashinfer_backend="cudnn"), "flashinfer_backend 'cudnn'"),
])
def test_attention_a_placed_pool_cannot_serve_is_refused_at_build(attention, refused):
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    with pytest.raises(ValueError, match=f"'attn': {refused} is unsupported over KV pool 'kv'"):
        _cp_attention(joint, CPU, torch.float32, **attention)


def test_head_shards_that_dont_divide_the_kv_heads_are_refused():
    config = PagedKVConfig(num_layers=1, num_kv_heads=8, head_dim=8, max_seq_len=64, num_qo_heads=48, page_size=8)
    with pytest.raises(ValueError, match="8 KV and 48 query heads don't split 12 ways"):
        config.shard(12)


def test_a_pool_that_keeps_a_sliding_window_is_refused_a_placement():
    """Its pages hold another page size for the windowed layers, which the placed storage doesn't
    stripe; refused where the storage is placed, before it is allocated."""
    config = PagedKVConfig(
        num_layers=2, num_kv_heads=1, head_dim=8, max_seq_len=64, num_qo_heads=2, max_num_pages=8,
        context_parallel=True,
    )
    config.keep_window(16, (1,))
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    with pytest.raises(ValueError, match="context_parallel is unsupported over a pool that keeps a sliding window"):
        config.place(joint, "kv")


# a step that isn't causal (BAGEL's image generation, say) attends every token of its rows
STEPS = pytest.mark.parametrize("rows, causal", [
    (PREFILL_ROWS, True), (PREFILL_ROWS, False), (DECODE_ROWS, True),
], ids=["prefill", "prefill-noncausal", "decode"])


@STEPS
@pytest.mark.parametrize("over_sp", [False, True], ids=["tp", "sp"])
def test_two_ranks_attend_a_placed_pool_like_one_rank_with_the_same_collectives(tmp_path, over_sp, rows, causal):
    mp.spawn(_over_ranks, args=(P, "gloo", str(tmp_path / "store"), over_sp, rows, False, (0, 0), causal), nprocs=P)


def test_a_split_label_reached_outside_an_exchange_is_refused():
    """A layer that writes its K/V itself (``write_target``) or calls ``run`` directly would treat
    this rank's range as the whole step, or over a pool stored by head write its TP heads into this
    rank's slice of them; it raises on every rank at the first such step."""
    shard = SequenceShard(("main",), 0, None, 0, _group(0, P))
    kv = _manager(0, PagedKVConfig(
        num_layers=2, num_kv_heads=2, head_dim=2, max_seq_len=64, num_qo_heads=2, max_num_pages=4, page_size=8,
    ))
    kv.bind_sequence_shard(shard)
    kv.bind_sequence_shard(shard)
    # a pool keeps one shard for its writes to check, so a second node's shard would misjudge them
    with pytest.raises(ValueError, match="two sequence shards"):
        kv.bind_sequence_shard(SequenceShard(("main",), 0, None, 0, _group(0, P)))
    ctx = StepContext(request_ids=("r0",), graph_walk="w", slot=0, capture=False)
    shard.plan(ResourceStep(segments=(Segment("r0", "main", 6),)), ctx)
    k = torch.zeros(3, 1, 2)
    for write in (lambda: kv.write_kv(k, k, label="main"), lambda: kv.write_slots("main"),
                  lambda: kv.write_target(label="main")):
        with pytest.raises(ValueError, match="outside an exchange"):
            write()
    # through an exchange the guard lets it by (to the plan this test never made)
    shard.exchanging = True
    with pytest.raises(KeyError):
        kv.write_slots("main")
    # the pool stores this rank's head slice, so a copied step's direct write would land the TP
    # heads in it too: only the exchange's head slice may write
    shard.exchanging = False
    shard.commit(None, ctx)
    shard.plan(ResourceStep(segments=(Segment("r0", "main", 1),)), ctx)
    assert shard.split_range("main") is None
    with pytest.raises(ValueError, match="'main', whose KV this rank stores a head slice of, outside an exchange"):
        kv.write_target(label="main")

    # an attention's run over a placed pool, on either rank, under a lease whose replay pads 6
    # real tokens to its 16-token bucket, which the step splits 8 and 8
    lease = SlotLease(0, BucketKey("w", bs=1, num_tokens=16))
    for rank in range(P):
        joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=_group(rank, P), instance=_group(rank, P))
        shard = SequenceShard(("main",), 0, None, 0, joint.sp_group)
        attn, _ = _cp_attention(joint, CPU, torch.float32, shard=shard)
        ctx = StepContext(request_ids=("r0",), graph_walk="w", slot=0, capture=False, slot_lease=lease,
                          force_prefill=True)
        shard.plan(ResourceStep(segments=(Segment("r0", "main", 6),)), ctx)
        assert shard.split_range("main").counts == (8, 8)
        q = torch.zeros(8, Q_HEADS, HEAD_DIM)
        with pytest.raises(ValueError, match="attention run reached label 'main'.*outside an exchange"):
            attn.run(q, "main", kv_cache_layer=None, k=q[:, :1], v=q[:, :1])


def test_a_ring_deployment_loads_with_the_pool_it_stripes_by_planned_before_its_shard(monkeypatch):
    """The shard's exchange ring makes it depend on the pool whose stored positions it
    stripes by: the engine takes that edge to a pool stored by position, and plans the pool
    first whatever the keys' order."""
    # one rank of the two, with no group to connect
    monkeypatch.setattr(CommGroup, "connect_pairs", lambda self, device: None)
    groups = _sp_groups(NODE)
    specs = [
        KVSpec(resource_key="kv", nodes={NODE}, config=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=HEAD_DIM, max_seq_len=64, num_qo_heads=Q_HEADS, max_num_pages=8,
            page_size=CP_PAGE,
        )),
        AttentionSpec(resource_key="attn", nodes={NODE}, config=AttentionConfig(kv_cache="kv")),
        SeqShardSpec("a_shard", {NODE}, ("main",), 0),
    ]
    apply_yaml_overrides(specs, {"resources": {"kv": {"context_parallel": True}, "a_shard": {"exchange": "ring"}}})
    engine = Engine(graph_runtime=None)
    engine.load_model({NODE: _Submodule()}, specs=specs, parallel_groups=groups, device=CPU, transfer_engine_info=None)
    order = engine._runner.order
    assert order.index("kv") < order.index("a_shard") < order.index("attn"), order


@pytest.mark.parametrize("longrope", [False, True], ids=["rope", "longrope"])
def test_rope_in_a_layer_rotates_a_split_labels_range_at_its_own_positions(monkeypatch, longrope):
    """A layer that ropes through the position resource holds its range of a split label, and so
    do the ids it rotates at (a LongRoPE request's table ids, past its positions by the long
    half's offset); an explicit ``pos_ids`` of the whole label is refused (a LongRoPE request refuses
    any)."""
    shard = SequenceShard(("main",), 0, None, 0, _group(1, P))
    if longrope:
        table = LongRoPE((1.0,) * 4, (2.0,) * 4, original_max_position_embeddings=16, max_position_embeddings=512)
        pos = LongRopeManager(
            config=PositionConfig(kv_cache="kv", rotary_dim=8, longrope=table), device=CPU, max_seq_len=64, head_dim=8,
        )
        for i in range(len(PREFILL_ROWS)):
            pos.ingest_request(f"r{i}", PositionReqConfig(max_position=100))
    else:
        pos = RopeManager(config=PositionConfig(kv_cache="kv"), device=CPU, max_seq_len=64, head_dim=8)
    pos._shard_key = "shard"
    pos.bind_dependencies({"shard": shard})
    rows = [(f"r{i}", "main", new) for i, (_, new) in enumerate(PREFILL_ROWS)]
    ctx = StepContext(request_ids=tuple(rid for rid, _, _ in rows), graph_walk="w", slot=0, capture=False)
    ctx.plan_results["kv"] = KVPlanOutputs({"main": KVPlanOutput(None, rows=rows)})
    segments = tuple(Segment(*row) for row in rows)
    ctx.plan_results["shard"] = shard.plan(ResourceStep(segments=segments), ctx)
    pos.plan(PositionStep(segments=segments), ctx)
    rotated = []
    monkeypatch.setattr(torch.ops.mstar, "rope_apply_qk_inplace", lambda q, k, ids, *a, **kw: rotated.append(ids))
    # rank 1 of 2 holds the step's last 5 of 11 tokens
    q = torch.zeros(5, 2, 8, dtype=torch.bfloat16)
    pos.apply_qk(q, q.clone(), label="main")
    assert rotated[0].tolist() == [i + 64 * longrope for i in (1, 2, 0, 1, 0)]
    assert pos.pos_ids("main").shape[0] == 11
    with pytest.raises(ValueError, match="explicit pos_ids" if longrope else "this rank's range's positions"):
        pos.apply_qk(q, q.clone(), label="main", pos_ids=torch.arange(11))


# --- Cosmos3's DiT, its steps split over its SP ranks ------------------------------------------


def _cosmos3(reasoner: bool = False):
    """A toy Cosmos3 (Edge's backbone family, paged attention alone), and its DiT submodule over a
    seeded tower."""
    from mstar.model.cosmos3.components.transformer import Cosmos3OmniTransformer
    from mstar.model.cosmos3.cosmos3_model import Cosmos3Model
    from mstar.model.cosmos3.submodules import Cosmos3DiTSubmodule
    from mstar.model.cosmos3.tests.test_edge import _init_all, _tiny_edge_config

    cfg = _tiny_edge_config(enable_reasoner=reasoner, attention_backend="flashinfer", compile_denoise=False)
    model = Cosmos3Model(model_path_hf="unused", skip_weight_loading=True)
    model.config = cfg
    tower = Cosmos3OmniTransformer(cfg).eval()
    _init_all(tower)
    sub = Cosmos3DiTSubmodule(tower, cfg, scheduler=None)
    sub.get_device = lambda: CPU
    return model, sub


def _cosmos3_resources(model, joint: JointGroups) -> dict:
    """The model's own declaration, built and bound as the engine does, small."""
    from mstar.model.cosmos3.submodules import KV_CACHE as C3_KV

    specs = model.get_node_resources()
    apply_yaml_overrides(specs, {"resources": {C3_KV: {"max_num_pages": 16, "page_size": 16, "max_seq_len": 256}}})
    by_key = resolve_spec_dependencies(specs)
    res = {
        key: build_resource(spec, EngineResourceInfo(
            device=CPU, joint_comm_group=joint, kv_dtype=torch.float32, nodes=frozenset(spec.nodes),
            dependencies={dep: by_key[dep] for dep in spec.depends_on()},
        ))
        for key, spec in by_key.items()
    }
    for key, spec in by_key.items():
        res[key].bind_dependencies({dep: res[dep] for dep in spec.depends_on()})
    return res


def _run_step(runner, step, walk, rids=("r0",)):
    step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False))
    assert runner.admit(step).ok
    runner.plan(step)


def _gen_input(static):
    """A two-branch eager denoise step's inputs, as ``prepare_inputs`` makes them."""
    from mstar.model.cosmos3.submodules import GenStepInfo
    from mstar.model.submodule_base import ARNodeInputs

    return ARNodeInputs(
        input_seq_len=static["num_vision_tokens"],
        resource_step_info=GenStepInfo(cfg=True, cfg_active=True, capture_key=None),
    )


def _cosmos3_steps(sub, resources, statics, latents, timesteps) -> tuple[list[torch.Tensor], StepRunner]:
    """One request through the DiT: its two-branch text prefill (copied: the prefill walk splits
    nothing), then a denoise step three ways: both branches as one combined label (the compiled
    step's pack, and the cross-request step's eager bracket), and each branch on its own label (the
    sequential path). Returns every velocity, and the runner, the request still on it."""
    from mstar.model.cosmos3 import constants as C
    from mstar.model.cosmos3.submodules import ATTN, CFG_BATCHED_LABEL, COND_LABEL, UNCOND_LABEL

    tower = sub.transformer
    sub.bind_node_resources(resources)
    runner = StepRunner(resources)
    runner.ingest_request("r0", {"kv": KVReqConfig(needed_labels=[COND_LABEL, UNCOND_LABEL])})
    cond, uncond = statics
    prefill = sub._get_prefill_node_inputs(cond, uncond)
    step = engine_step(sub, C.PREFILL_WALK, ["r0"], [prefill])
    _run_step(runner, step, C.PREFILL_WALK)
    labels = (COND_LABEL, UNCOND_LABEL)
    tower.prefill_und(
        torch.cat([prefill.tensor_inputs[f"input_ids_{label}"] for label in labels]),
        torch.cat([prefill.tensor_inputs[f"text_mrope_ids_{label}"] for label in labels], dim=1),
        CFG_BATCHED_LABEL,
    )
    runner.commit(step)

    gen = _gen_input(cond)
    layout = (cond["vision_token_shapes"], cond["vision_noisy_frame_indexes"], cond["mse_gen_indexes"])
    attn = resources[ATTN]
    out = []
    for batched_cfg in (True, False):
        sub.batched_cfg = batched_cfg
        for run in range(2 if batched_cfg else 1):
            step = engine_step(sub, C.IMAGE_GEN_WALK, ["r0"], [gen])
            _run_step(runner, step, C.IMAGE_GEN_WALK)
            if not batched_cfg:
                out += [tower.denoise_step(latents, timesteps, s["rotary"], *layout, label, attn)
                        for s, label in zip(statics, labels, strict=True)]
            elif run == 0:
                out += tower.denoise_step_batched_cfg(
                    latents, timesteps, cond["rotary"], uncond["rotary"], *layout, CFG_BATCHED_LABEL, attn,
                )
            else:
                row = tower.denoise_step_row(cond, latents, timesteps, cond["rotary"], uncond["rotary"])
                out += tower.denoise_step_batched([row], CFG_BATCHED_LABEL, attn)[0]
            runner.commit(step)
    return out, runner


def _cosmos3_over_ranks(rank, world, store):
    """Two SP ranks against one rank over a stand-in attention (SDPA over stored K/V): the copied
    prefill writes each rank's heads (a head slice, the Edge K re-written through the callable),
    the denoise steps split their tokens and attend by Ulysses. After them, outside a step, the
    bracket is the identity, as the startup warmup compile (after the captures) runs it. Then a
    compiled denoise step splits and copies at two sizes without recompiling a size it saw."""
    from torch._dynamo.utils import counters

    from mstar.model.cosmos3 import constants as C
    from mstar.model.cosmos3.submodules import ATTN, CFG_BATCHED_LABEL, SEQ_SHARD, _WarmupAttention
    from mstar.model.cosmos3.tests.test_engine_cache import _SdpaResources

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    sys.modules["flashinfer"] = reference_flashinfer()
    manager_mod.KVTransferManager = _StubTransfer
    group = CommGroup(rank, rank, list(range(world)))
    group.device_group, group.initialized = dist.group.WORLD, True
    try:
        model, sub = _cosmos3()
        tower = sub.transformer

        def inputs(height, width):
            statics = [sub._build_static(list(range(n)), height, width, 1, 24.0, False, CPU) for n in (5, 3)]
            g = torch.Generator().manual_seed(height * width)
            latents = torch.randn(sub._latent_shape(height, width, 1), generator=g)
            return statics, latents, torch.full((statics[0]["num_noisy_vision_tokens"],), 500.0)

        statics, latents, timesteps = inputs(32, 64)
        with torch.no_grad():
            want, _ = _cosmos3_steps(sub, _SdpaResources().as_dict(), statics, latents, timesteps)
            res = _cosmos3_resources(model, JointGroups(tp_group=CommGroup.trivial(), sp_group=group, instance=group))
            got, runner = _cosmos3_steps(sub, res, statics, latents, timesteps)
            row = tower.denoise_step_row(statics[0], latents, timesteps, statics[0]["rotary"], statics[0]["rotary"])
            warm = tower.denoise_step_batched([row, row], CFG_BATCHED_LABEL, _WarmupAttention())
            assert [v.shape for branches in warm for v in branches] == [latents.shape] * 4
        assert len(got) == len(want) == 6
        for g, w in zip(got, want, strict=True):
            torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-4)

        torch._dynamo.reset()
        counters.clear()
        cfg_step = torch.compile(tower.denoise_step_batched_cfg, backend="eager", dynamic=False)
        graphs = []
        sub.batched_cfg = True
        with torch.no_grad():
            for (height, width), min_tokens in [((32, 64), 0), ((64, 64), 0), ((32, 64), 1 << 20)]:
                (cond, uncond), latents, timesteps = inputs(height, width)
                res[SEQ_SHARD].min_tokens = min_tokens
                for _ in range(2):
                    step = engine_step(sub, C.IMAGE_GEN_WALK, ["r0"], [_gen_input(cond)])
                    _run_step(runner, step, C.IMAGE_GEN_WALK)
                    assert (res[SEQ_SHARD].split_range(CFG_BATCHED_LABEL) is None) == bool(min_tokens)
                    cfg_step(
                        latents, timesteps, cond["rotary"], uncond["rotary"], cond["vision_token_shapes"],
                        cond["vision_noisy_frame_indexes"], cond["mse_gen_indexes"], CFG_BATCHED_LABEL, res[ATTN],
                    )
                    runner.commit(step)
                    graphs.append(counters["stats"]["unique_graphs"])
        # a second step of a size and split it compiled for compiles nothing
        assert graphs[0] > 0 and graphs[0::2] == graphs[1::2], graphs
    finally:
        group.device_group = None
        dist.destroy_process_group()


def test_two_sp_ranks_run_cosmos3s_dit_like_one_rank(tmp_path):
    mp.spawn(_cosmos3_over_ranks, args=(P, str(tmp_path / "store")), nprocs=P)


def test_cosmos3s_reasoner_shares_the_dits_sequence_shard_and_plans_it(monkeypatch):
    """Edge's reasoner attends the DiT's pool through the same attention, so the sequence shard
    spans both nodes (one over the DiT alone is refused), and the reasoner's steps plan it (the
    engine adds its step of the node's segments): one runs through the runner on one rank, where
    the attention refuses a step without its plan."""
    from dataclasses import replace as replaced

    from mstar.model.cosmos3 import constants as C
    from mstar.model.cosmos3.cosmos3_model import DIT_NODE, REASONER_NODE
    from mstar.model.cosmos3.submodules import REASONER_LABEL, SAMPLER, SEQ_SHARD, Cosmos3ReasonerSubmodule
    from mstar.model.submodule_base import ARNodeInputs

    monkeypatch.setitem(sys.modules, "flashinfer", reference_flashinfer())
    model, sub = _cosmos3(reasoner=True)
    specs = model.get_node_resources()
    shard, = (spec for spec in specs if spec.resource_key == SEQ_SHARD)
    assert shard.nodes == {DIT_NODE, REASONER_NODE}
    with pytest.raises(ValueError, match=r"spans nodes \['reasoner'\] without the sequence shard"):
        resolve_spec_dependencies([replaced(spec, nodes={DIT_NODE}) if spec is shard else spec for spec in specs])

    trivial = CommGroup.trivial()
    res = _cosmos3_resources(model, JointGroups(tp_group=trivial, sp_group=trivial, instance=trivial))
    runner = StepRunner(res)
    runner.ingest_request("t", {"kv": KVReqConfig(needed_labels=[REASONER_LABEL]), SAMPLER: SamplingReqConfig()})
    reasoner = Cosmos3ReasonerSubmodule(sub.transformer, model.config)
    reasoner.bind_node_resources(res)
    step = engine_step(reasoner, C.REASONER_PREFILL_WALK, ["t"], [ARNodeInputs(input_ids=torch.arange(6),
                                                                                input_seq_len=6)])
    _run_step(runner, step, C.REASONER_PREFILL_WALK, rids=("t",))
    assert not runner.resources[SEQ_SHARD].can_split


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("captured, padding", [
    (False, None), (True, None),
    # a bucket of more rows than the step: padding rows of 2 tokens (a decode's of 1), and of none
    (True, 2), (True, 0),
], ids=["eager", "captured", "captured-padding-rows", "captured-span-0-rows"])
@STEPS
@pytest.mark.parametrize("over_sp", [False, True], ids=["tp", "sp"])
def test_two_gpus_attend_a_placed_pool_like_one(tmp_path, over_sp, rows, causal, captured, padding):
    pytest.importorskip("flashinfer")
    if padding is None:
        padding = (0, 0)
    elif rows == DECODE_ROWS:
        if padding == 0:
            pytest.skip("a decode bucket's padding rows hold a token each")
        # each attends the first slot of SINK_PAGE, which one rank holds and the other masks
        padding = (2, 1)
    else:
        padding = (2, padding)
    mp.spawn(_over_ranks, args=(P, "nccl", str(tmp_path / "store"), over_sp, rows, captured, padding, causal), nprocs=P)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("split", ["all_gather", "ring"])
def test_two_gpus_split_a_step_over_a_placed_pool_like_one(tmp_path, split):
    pytest.importorskip("flashinfer")
    mp.spawn(
        _over_ranks, args=(P, "nccl", str(tmp_path / "store"), True, PREFILL_ROWS, False, (0, 0), True, split),
        nprocs=P,
    )


# a replay's padding rows after the step's, as `KVManager` plans them: none (the bucket's tokens past
# the step alone), two of 2 tokens, and two of none
@pytest.mark.parametrize("padding", [(0, 0), (2, 2), (2, 0)], ids=["tail", "rows", "span-0-rows"])
def test_a_captured_buckets_padding_merges_to_0_and_reaches_no_real_row(monkeypatch, padding):
    """The own-token part of a bucket larger than its step: the kernel writes only the planned rows,
    and the padding's context partials are masked (out 0, lse -inf). So only the part's own out 0
    and lse -inf past its plan keep the padding finite (``_over_ranks`` checks it on GPU). The
    padding's keys are whatever the model's layers left in those rows, NaN included, and the kernel
    reads keys past each row's end (FA3), so the part plans the real rows alone and zeroes every key
    past them, the padding rows' included."""
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    monkeypatch.setitem(sys.modules, "flashinfer", reference_flashinfer())
    # the captured context wrapper stages its plan through a CUDA event; this needs only the own part
    context = _IdleContext()
    monkeypatch.setattr(ragged_mod, "zero_unplanned_rows", _zero_unplanned_rows_on_cpu)
    monkeypatch.setattr(fi_mod, "FlashInferPrefillWrapper", lambda **kwargs: context)
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    attn, placement = _cp_attention(joint, CPU, torch.float32)
    case = _Case.build(PREFILL_ROWS, CPU, torch.float32, padding=padding)
    n, pad = case.q.shape[0], padding[0] * padding[1] + 8
    _plan(attn, case, bucket=BucketKey("w", bs=len(PREFILL_ROWS) + padding[0], num_tokens=n + pad))
    # this rank's query heads, which its own tokens attend for alone
    q, k, v, _ = case.inputs(placement, heads=slice(0, Q_HEADS // P), pad=pad)
    out, lse = attn._current_plan_states["main"].new.run(q, k, v)
    merged = merge(torch.stack([torch.zeros_like(out), out]), torch.stack([torch.full_like(lse, -math.inf), lse]))
    assert torch.isfinite(merged).all() and not merged[n:].any()
    qo = np.cumsum([0, *(new for _, new in PREFILL_ROWS)]).tolist()
    want = torch.cat([reference_attention(q[a:b], k[a:b], v[a:b])[0] for a, b in itertools.pairwise(qo)])
    torch.testing.assert_close(merged[:n], want)


def test_a_captured_step_exchanges_with_no_stored_context(monkeypatch):
    """A graph replays the collectives of its capture whatever a later step stores, so a captured
    step of no stored context, as a capture step usually is, can't skip them."""
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    monkeypatch.setitem(sys.modules, "flashinfer", reference_flashinfer())
    # the captured context wrapper stages its plan through a CUDA event; this needs only its partial

    class Context(_IdleContext):
        def run(self, q, cache, return_lse):
            return torch.zeros_like(q), torch.full(q.shape[:2], -math.inf)

    monkeypatch.setattr(ragged_mod, "zero_unplanned_rows", _zero_unplanned_rows_on_cpu)
    monkeypatch.setattr(fi_mod, "FlashInferPrefillWrapper", lambda **kwargs: Context())
    exchanged = []

    def exchange(self, out, lse, valid):
        exchanged.append(out)
        return out[None], lse[None]

    monkeypatch.setattr(TPContextParallelWrapper, "_exchange_queries", lambda self, q: q)
    monkeypatch.setattr(TPContextParallelWrapper, "_exchange_partials", exchange)
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    attn, placement = _cp_attention(joint, CPU, torch.float32)
    case = _Case.build(((0, 5), (0, 3)), CPU, torch.float32)
    _plan(attn, case, bucket=BucketKey("w", bs=2, num_tokens=8))
    heads = slice(0, Q_HEADS // P)
    torch.testing.assert_close(case.runner(attn, placement, heads)(), case.want()[:, heads])
    assert len(exchanged) == 1


@pytest.mark.parametrize("batch_sizes, multiplier", [([1, 2], 1), ([1], 2)], ids=["padded", "one_row"])
def test_a_capture_combining_labels_over_a_placed_pool_is_refused_at_load_if_it_can_pad(batch_sizes, multiplier):
    """A combined plan is label-major (cond's rows, then uncond's), so a replay's padding rows sit
    between real ones, which attention over a placed pool can't plan (Chatterbox T3 with CFG). The
    capture's own declaration decides, as Cosmos3's prefill combines labels with no token multiplier;
    a capture of one row never pads."""
    config = BatchedAcceleratorGraphConfig(
        capture_graph_walk="decode", single_request_inputs=NodeInputs(tensor_inputs={}, input_seq_len=1),
        capture_batch_sizes=batch_sizes, total_tokens_multiplier=multiplier,
    )
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    kv = types.SimpleNamespace(can_capture=lambda: True, lean_capture=False, placement=PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, page_size=8, context_parallel=True,
    ).place(joint, "kv"))
    step = types.SimpleNamespace(steps={"kv": KVStep(combined_labels={("cond", "uncond"): "cfg"})}, state={})
    runner = AcceleratorGraphRunner(
        submodule_name="t3", resources={"kv": kv}, step_runner=mock.Mock(), device=CPU, autocast_dtype=None,
        submodule=types.SimpleNamespace(get_accelerator_graph_configs=lambda *args: [config],
                                        declare_step=lambda **kwargs: step, compiled_entries_per_shape=1),
        joint_comm_group=joint, num_slots=1, head_shards=1,
    )
    if len(batch_sizes) == 1:
        assert runner.prepare_for_capture()
        return
    with pytest.raises(ValueError, match=r"t3: the capture of 'decode' at batch size 2 combines labels.*\['kv'\]"):
        runner.prepare_for_capture()


def test_a_dropped_preplan_leaves_its_own_token_wrappers_nothing_to_skip(monkeypatch):
    """A ragged own-token wrapper skips a plan whose layout it already holds. A dropped pre-plan's
    staging went on the plan stream, which the step planned instead never waits on, so that step
    plans its own tokens again."""
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    fi = reference_flashinfer()
    plans = []
    ragged_plan = fi.BatchPrefillWithRaggedKVCacheWrapper.plan

    def counted(self, *args, **kwargs):
        plans.append(args[0].tolist())
        return ragged_plan(self, *args, **kwargs)

    monkeypatch.setattr(fi.BatchPrefillWithRaggedKVCacheWrapper, "plan", counted)
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    # the captured context wrapper stages its plan through a CUDA event; this needs only the own part
    context = _IdleContext()
    monkeypatch.setattr(fi_mod, "FlashInferPrefillWrapper", lambda **kwargs: context)
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    attn, _ = _cp_attention(joint, CPU, torch.float32)
    case = _Case.build(PREFILL_ROWS, CPU, torch.float32)
    lease = SlotLease(1, BucketKey(graph_walk="w", bs=len(PREFILL_ROWS), num_tokens=16))
    ctx = StepContext(request_ids=tuple(v.request_id for v in case.plan["main"].views), graph_walk="w",
                      slot=1, capture=False, slot_lease=lease, is_preplan=True)
    ctx.plan_results["kv"] = case.plan
    attn.plan(AttentionStep(), ctx)
    attn.clear_preplan()
    planned = len(plans)
    _plan(attn, case, slot=1, bucket=lease.bucket)
    assert len(plans) == planned + 1, "the step skipped the own-token plan the dropped pre-plan staged"


def test_a_placed_pool_plans_on_fa2_for_auto_from_sm100(monkeypatch):
    """From SM100 FlashInfer's auto can send short rows to cuDNN, whose lse the merge isn't known
    to read right."""
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    monkeypatch.setattr(fi_mod, "current_platform", types.SimpleNamespace(cuda_capability=lambda device=None: (10, 0)))
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    attn, _ = _cp_attention(joint, CPU, torch.float32)
    assert attn._wrapper_kv_kwargs["backend"] == attn._new_kwargs["backend"] == "fa2"


@pytest.mark.parametrize(("over_sp", "capability", "backend", "outcome"), [
    (True, (9, 0), "fa2", "warned"),
    (True, (9, 0), "auto", "quiet"),  # auto prefills on fa3 on SM90
    (True, (10, 0), "fa2", "warned"),
    (True, (10, 0), "auto", "warned"),  # auto runs fa2 from SM100
    (True, (8, 0), "auto", "warned"),  # and below SM90, where FlashInfer has no fa3
    (True, (8, 0), "fa2", "warned"),
    (True, None, "fa2", "warned"),  # off NVIDIA too
    (False, (9, 0), "fa2", "quiet"),
    (False, (10, 0), "fa2", "quiet"),
])
def test_fa2_over_an_sp_placed_pool(monkeypatch, caplog, over_sp, capability, backend, outcome):
    """One SP-placed server hung in an fa2 prefill, cause unknown: a warning wherever an SP-placed
    pool prefills on fa2, and a TP-placed pool doesn't care."""
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    platform = types.SimpleNamespace(cuda_capability=lambda device=None: capability)
    monkeypatch.setattr(fi_mod, "current_platform", platform)
    if over_sp:
        joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=_group(0, P), instance=_group(0, P))
    else:
        joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    with caplog.at_level(logging.WARNING, logger=fi_mod.logger.name):
        # nothing here touches the device but the check of its type, so no GPU is needed
        _cp_attention(joint, torch.device("cuda", 0), torch.float32, flashinfer_backend=backend)
    assert any("fa2" in r.getMessage() for r in caplog.records) == (outcome == "warned")


@pytest.mark.parametrize("over_sp", [False, True], ids=["tp", "sp"])
def test_a_placed_pools_own_tokens_attend_at_the_contexts_sm_scale(over_sp, monkeypatch):
    """The two parts merge by log-sum-exp, so a declared ``sm_scale`` must reach both: TP's whole
    rows on the ragged prefill, SP's zigzag share on one-slot pages."""
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    monkeypatch.setitem(sys.modules, "flashinfer", reference_flashinfer())
    if over_sp:
        joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=_group(0, P), instance=_group(0, P))
    else:
        joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    attn, _ = _cp_attention(joint, CPU, torch.float32, sm_scale=0.5)
    _plan(attn, _Case.build(PREFILL_ROWS, CPU, torch.float32))
    state = attn._current_plan_states["main"]
    own = state.new.kernel.kernel if over_sp else state.new._ragged
    assert own.sm_scale == state.context.sm_scale == 0.5


def test_tp_placed_decode_runs_on_tensor_cores_whatever_decode_tensor_cores_says(monkeypatch):
    """A rank attends every partner's query heads over one KV head, a GQA group FlashInfer's
    CUDA-core decode has no kernel for unless it is one of a few sizes."""
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    monkeypatch.setitem(sys.modules, "flashinfer", reference_flashinfer())
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    attn, _ = _cp_attention(joint, CPU, torch.float32, decode_tensor_cores=False)
    _plan(attn, _Case.build(DECODE_ROWS, CPU, torch.float32))
    assert attn._current_plan_states["main"].context.attn_wrapper.use_tensor_cores


# --- the Qwen3-Omni Thinker, its steps split over SP ranks ---------------------------------------

# where the Thinker's KV pool stores its tokens over the SP ranks, and the split steps' exchange
THINKER_STORAGE = pytest.mark.parametrize("placed, exchange", [
    pytest.param(False, "ulysses", id="by_head-ulysses"),
    pytest.param(True, "all_gather", id="by_position-all_gather"),
    pytest.param(True, "ring", id="by_position-ring"),
])


def _thinker_over_ranks(rank, world, store, placed, exchange):
    """One SP rank of a small Qwen3-Omni Thinker (its real layers: a dense one, MoE ones masking the
    step's last rows as padding, deepstack, the Talker's layer-N states) over the model's own Thinker
    declarations, built and bound as the engine builds them. Three steps (a first prefill, one over
    stored context with a fresh row and a riding decode row, then a decode) run split where they can,
    eager and compiled, and give what they give copied; a split prefill samples each row's last token.
    The fused norm/RoPE/KV write is the copied decode's over KV stored by position alone."""
    from mstar.engine.resources.sequence_shard import sequence_parallel_refusals
    from mstar.engine.resources.step import SubmoduleStep
    from mstar.engine.state import ArrayStep
    from mstar.engine.state.adapters import node_handles, node_resources, state_steps
    from mstar.model.components import MoeParallelConfig
    from mstar.model.qwen3_omni.components.attention import qk_norm_rope_table_attend
    from mstar.model.qwen3_omni.components.thinker import Qwen3OmniThinkerModel
    from mstar.model.qwen3_omni.config import (
        THINKER_ATTN,
        THINKER_KV,
        THINKER_POS,
        THINKER_SAMPLER,
        THINKER_SHARD,
        Qwen3OmniModelConfig,
    )
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel
    from mstar.model.qwen3_omni.submodules import ThinkerSubmodule

    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    sys.modules["flashinfer"] = reference_flashinfer()
    manager_mod.KVTransferManager = _StubTransfer
    group = CommGroup(rank, rank, list(range(world)))
    group.device_group, group.initialized = dist.group.WORLD, True
    try:
        cfg = Qwen3OmniModelConfig()
        tc = cfg.thinker_text
        tc.vocab_size, tc.hidden_size, tc.num_hidden_layers = 8, 2 * HEAD_DIM, 3
        tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim = 4, 2, HEAD_DIM
        tc.num_experts, tc.num_experts_per_tok, tc.moe_intermediate_size = 4, 2, 16
        tc.intermediate_size, tc.mlp_only_layers = 32, [0]
        cfg.talker.accept_hidden_layer = 1
        thinker = Qwen3OmniThinkerModel(cfg, moe_parallel=MoeParallelConfig(mask_padding=True))
        gen = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for p in thinker.parameters():
                # norms 1, weights at unit gain
                gain = p.shape[-1] ** -0.5
                p.copy_(torch.ones_like(p) if p.dim() == 1 else torch.randn(p.shape, generator=gen) * gain)

        omni = object.__new__(Qwen3OmniModel)
        omni.config = cfg
        keys = (THINKER_KV, THINKER_ATTN, THINKER_POS, THINKER_SHARD)
        specs = [spec for spec in node_resources(omni) if spec.resource_key in keys]
        overrides = {
            THINKER_KV: dict(max_num_pages=32, page_size=CP_PAGE, max_seq_len=64, context_parallel=placed),
            **({THINKER_SHARD: dict(exchange=exchange)} if exchange else {}),
        }
        apply_yaml_overrides(specs, {"resources": overrides})
        # the engine's order: derive, then the SP refusals
        by_key = resolve_spec_dependencies(specs)
        assert sequence_parallel_refusals(specs, "Thinker", 1, world) == []
        joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=group, instance=group)

        # (rid, tokens) per step: two first prefills, then their next tokens beside a fresh prompt,
        # one row a decode; 13 tokens, the last 2 the MoE's padding; then a decode, never split
        steps = [(("r0", 5), ("r1", 6)), (("r0", 2), ("r1", 1), ("r2", 10)), (("r0", 1), ("r1", 1), ("r2", 1))]
        inputs = []
        for rows in steps:
            n = sum(span for _, span in rows)
            hidden, cos, sin = (torch.randn(n, d, generator=gen) for d in (tc.hidden_size, HEAD_DIM, HEAD_DIM))
            # vision features on every third row
            deepstack = [torch.randn(n, tc.hidden_size, generator=gen) * (torch.arange(n)[:, None] % 3 == 0)
                         for _ in range(2)]
            inputs.append(dict(input_embeds=hidden, cos_sin_3d=(cos, sin), deepstack_visual_embeds=deepstack,
                               num_valid_tokens=torch.tensor([n - 2])))

        def run(split: bool, compiled: bool) -> list[torch.Tensor]:
            res = {
                key: build_resource(spec, EngineResourceInfo(
                    device=CPU, joint_comm_group=joint, kv_dtype=torch.float32, nodes=frozenset(spec.nodes),
                    dependencies={dep: by_key[dep] for dep in spec.depends_on()},
                    replicated_over_sp=placed,
                ))
                for key, spec in by_key.items()
            }
            for key, spec in by_key.items():
                res[key].bind_dependencies({dep: res[dep] for dep in spec.depends_on()})
            # the Thinker declares a rebuild policy, which the engine refuses beside a resource that
            # can't drop a request's state: its shard and attention must load on every storage
            node = make_mgmt(
                resources=res, submodule=types.SimpleNamespace(split_batches_by_capture_key=lambda walk: False),
            )
            make_engine(_submodules={"Thinker": node})._check_rebuild_policies(
                make_model(rebuild_policies=omni.rebuild_policies, combined_walk_of=dict)
            )
            handles = node_handles(omni, "Thinker", {**res, THINKER_SAMPLER: None})
            for module in thinker.modules():
                if hasattr(module, "bind_state"):
                    module.bind_state(handles)
            # the submodule hands the model its node's shard
            ThinkerSubmodule(thinker, cfg).bind_node_resources(res)
            assert thinker.sequence_shard is res[THINKER_SHARD]
            res[THINKER_SHARD].min_tokens = 0 if split else 1 << 20
            runner = StepRunner(res)
            for rid in ("r0", "r1", "r2"):
                runner.ingest_request(rid, {THINKER_KV: KVReqConfig(needed_labels=["main"])})
            torch._dynamo.reset()
            forward = torch.compile(thinker, backend="eager") if compiled else thinker
            out = []
            for rows, step_inputs in zip(steps, inputs, strict=True):
                decode = all(span == 1 for _, span in rows)
                step = state_steps(with_segment_steps(SubmoduleStep(
                    segments=[Segment(rid, "main", span) for rid, span in rows], state={THINKER_KV: ArrayStep()},
                ), [THINKER_SHARD]), handles)
                step.set_ctx(StepContext(request_ids=tuple(rid for rid, _ in rows), graph_walk="prefill_text",
                                         slot=0, capture=False))
                assert runner.admit(step).ok
                runner.plan(step)
                splits = split and not decode
                assert (res[THINKER_SHARD].split_range("main") is not None) == splits
                # a copied step over KV stored by head still slices its heads
                assert res[THINKER_ATTN].step_exchanges() == (splits or not placed)
                step.begin_calls()
                # on CUDA the state op fuses the write (CPU always takes its reference): never on a step that
                # exchanges, and on the copied decode over KV stored by position, which reads only the pages
                kv = handles[THINKER_KV]
                kv.fused_ops = frozenset({qk_norm_rope_table_attend.op.name})
                if decode and placed and not splits:
                    assert kv.fused_ops
                else:
                    assert not kv.fused_ops
                kv.fused_ops = frozenset()
                with torch.no_grad():
                    hidden, layer_0, layer_n = forward(**step_inputs)
                step.end_calls()
                # what the Thinker samples a prefill from: each row's last token of the whole step
                ends = torch.tensor([span for _, span in rows]).cumsum(0) - 1
                torch.testing.assert_close(res[THINKER_ATTN].select_last_hidden(hidden), hidden[ends])
                runner.commit(step)
                out += [hidden, layer_0, layer_n]
            return out

        copied = run(split=False, compiled=False)
        for compiled in (False, True):
            for got, want in zip(run(split=True, compiled=compiled), copied, strict=True):
                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    finally:
        group.device_group = None
        dist.destroy_process_group()


@THINKER_STORAGE
def test_two_ranks_split_the_thinkers_steps_like_they_run_copied(tmp_path, placed, exchange):
    mp.spawn(_thinker_over_ranks, args=(P, str(tmp_path / "store"), placed, exchange), nprocs=P)
