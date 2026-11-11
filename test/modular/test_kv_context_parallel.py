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
from mstar.engine.resources.base import EngineResourceInfo, Resource
from mstar.engine.resources.context_parallel import ContextParallelWrapper, merge
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import KVCache
from mstar.engine.resources.kv.config import KVLayout, KVReqConfig, KVSpec, KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SINK_PAGE, KVPlanOutput, KVPlanOutputs, SequenceView, build_paged_indptrs
from mstar.engine.resources.kv.transfer import KVReadInfo, ShmKVTransferEngine
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext
from mstar.model.base import declared_resources
from mstar.model.qwen3_5.config import GDN_STATE, KV_CACHE
from mstar.model.submodule_base import NodeInputs

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
    assert (placement.size, placement.index, placement.over_sp) == (P, 0, True)
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


@dataclass
class _SharedSpec(_DependentSpec):
    def depends_on(self):
        return set()


def test_a_resource_over_nodes_that_split_heads_differently_is_refused(monkeypatch):
    """A resource is sized by one head-shard count, so it can't serve a node that places its
    positions over its SP ranks (heads over TP) beside one running its own SP (heads over TP x SP)."""
    other = "other"
    groups = _sp_groups(NODE, other)
    monkeypatch.setattr(engine_mod, "positions_over_sp", lambda specs, node, sp, enabled: node == NODE)
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


def _reference(q, k, v, causal=True):
    """Attention of ``q`` ``[T, Hq, D]`` over ``k``/``v`` ``[L, Hkv, D]``, the queries the last ``T``
    positions when causal; fp32 ``(out, lse)``, lse natural-log ``[T, Hq]``."""
    group = q.shape[1] // k.shape[1]
    k, v = (t.float().repeat_interleave(group, dim=1) for t in (k, v))
    scores = torch.einsum("thd,lhd->htl", q.float(), k) * q.shape[-1] ** -0.5
    if causal:
        t, length = q.shape[0], k.shape[0]
        seen = torch.arange(length)[None, :] <= torch.arange(t)[:, None] + (length - t)
        scores = scores.masked_fill(~seen.to(scores.device), -math.inf)
    return torch.einsum("htl,lhd->thd", torch.softmax(scores, dim=-1), v), torch.logsumexp(scores, -1).T


def _reference_flashinfer() -> types.ModuleType:
    """FlashInfer's paged prefill/decode and ragged wrappers as dense torch math on CPU, with
    their lse bases. A paged plan refuses a row FlashInfer's split-KV planner would miscount: no
    pages, or a last page holding no slot (see ``ContextParallelWrapper.local_indptrs``). The ragged one
    reads keys past each row's end as FA3 does."""
    fi = types.ModuleType("flashinfer")

    class Paged:
        base = "log2"

        def __init__(self, *args, **kwargs):
            pass

        def plan(self, qo_indptr=None, paged_kv_indptr=None, paged_kv_indices=None,
                 paged_kv_last_page_len=None, *, page_size, seq_lens, causal=False, indptr=None,
                 indices=None, last_page_len=None, **kwargs):
            kvp = (paged_kv_indptr if indptr is None else indptr).tolist()
            self.indices = (paged_kv_indices if indices is None else indices).tolist()
            last = (paged_kv_last_page_len if last_page_len is None else last_page_len).tolist()
            self.rows = [(kvp[b], kvp[b + 1], int(seq_lens[b])) for b in range(len(last))]
            for (start, end, length), tail in zip(self.rows, last, strict=True):
                assert end > start and 1 <= tail <= page_size and length == (end - start - 1) * page_size + tail
            self.qo = qo_indptr.tolist() if qo_indptr is not None else list(range(len(last) + 1))
            self.causal = causal

        def run(self, q, kv, return_lse=False, lse_base=None):
            outs, lses = [], []
            for b, (start, end, length) in enumerate(self.rows):
                pages = kv[self.indices[start:end]]
                k, v = (pages[:, i].flatten(0, 1)[:length] for i in (0, 1))
                out, lse = _reference(q[self.qo[b]:self.qo[b + 1]], k, v, self.causal)
                outs.append(out)
                lses.append(lse)
            out, lse = torch.cat(outs).to(q.dtype), torch.cat(lses)
            if (lse_base or self.base) == "log2":
                lse = lse / math.log(2)
            return (out, lse) if return_lse else out

    class Decode(Paged):
        def __init__(self, *args, use_tensor_cores=False, **kwargs):
            self.use_tensor_cores = use_tensor_cores

        def plan(self, **kwargs):
            super().plan(**kwargs)
            self.causal = False

    class Ragged:
        def __init__(self, *args, qo_indptr_buf=None, kv_indptr_buf=None, **kwargs):
            self.bufs = (qo_indptr_buf, kv_indptr_buf)

        def plan(self, qo_indptr, kv_indptr, *args, causal=False, **kwargs):
            self.qo, self.causal = qo_indptr.tolist(), causal
            # graph mode: the layouts into the caller's buffers
            for buf, indptr in zip(self.bufs, (qo_indptr, kv_indptr), strict=True):
                if buf is not None:
                    buf.copy_(indptr)

        def run(self, q, k, v, out=None, lse=None, return_lse=False, lse_base="log2"):
            parts = []
            for a, b in itertools.pairwise(self.qo):
                part = _reference(q[a:b], k[a:b], v[a:b], self.causal)
                # FA3 reads the keys past a row's end, the next rows' and past the plan, to its last
                # 128-key tile and masks them additively: nothing from a finite key, NaN in the row
                # from a NaN one
                over = slice(b, min(k.shape[0], a + -(-(b - a) // 128) * 128))
                poison = (k[over].float().sum() + v[over].float().sum()) * 0
                parts.append((part[0] + poison, part[1] + poison))
            o, got = torch.cat([p[0] for p in parts]).to(q.dtype), torch.cat([p[1] for p in parts])
            got = got if lse_base == "ln" else got / math.log(2)
            # the planned rows alone, into the caller's buffers or q-sized ones of its own (FlashInfer's
            # torch.empty; NaN here)
            if out is None:
                out = torch.full((q.shape[0], *o.shape[1:]), math.nan, dtype=o.dtype, device=q.device)
            if lse is None:
                lse = torch.full((q.shape[0], *got.shape[1:]), math.nan, dtype=got.dtype, device=q.device)
            out[:len(o)], lse[:len(got)] = o, got
            return (out, lse) if return_lse else out

    fi.BatchPrefillWithPagedKVCacheWrapper = Paged
    fi.BatchDecodeWithPagedKVCacheWrapper = Decode
    fi.BatchPrefillWithRaggedKVCacheWrapper = Ragged
    return fi


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

    def cache(self, placement, pages: int = 0) -> torch.Tensor:
        """This rank's layer of the pool, of at least ``pages`` pages: each owned slot at its
        physical place, NaN elsewhere (SINK_PAGE included)."""
        slots = CP_PAGE // placement.size
        pages = max(pages, max(view.page_idxs[-1] for view in self.views) + 1)
        cache = torch.full((pages, 2, slots, 1, HEAD_DIM), math.nan, device=self.q.device, dtype=self.q.dtype)
        for view, kv in zip(self.views, self.keys, strict=True):
            for pos in range(view.length):
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

    def want(self) -> torch.Tensor:
        outs = []
        qo = 0
        for kv, (_, new) in zip(self.keys, self.rows, strict=True):
            outs.append(_reference(self.q[qo:qo + new], kv[:, 0], kv[:, 1])[0])
            qo += new
        return torch.cat(outs)


def _cp_attention(
    joint: JointGroups, device: torch.device, dtype: torch.dtype, **attention,
):
    """A ``FlashInferManager`` over a placed pool, and this rank's placement."""
    kv = KVSpec(resource_key="kv", nodes={NODE}, config=PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=HEAD_DIM, max_seq_len=64, num_qo_heads=Q_HEADS,
        max_num_pages=16, page_size=CP_PAGE, context_parallel=True,
    ))
    info = EngineResourceInfo(
        device=device, joint_comm_group=joint, kv_dtype=dtype, dependencies={"kv": kv},
        # SP placement replicates the node over the SP ranks, heads split over TP alone
        replicated_over_sp=joint.sp_group.world_size > 1,
    )
    spec = AttentionSpec(resource_key="attn", nodes={NODE}, config=AttentionConfig(kv_cache="kv", **attention))
    attn = AttentionManager.build(spec, info)
    # the KV resource's one field attention reads
    placement = kv.config.place(joint, "kv")
    attn.bind_dependencies({"kv": types.SimpleNamespace(placement=placement)})
    return attn, placement


def _plan(attn, case: _Case, slot: int = 0, bucket: BucketKey | None = None, capture=False, **step) -> None:
    """Plans ``case``, its padding rows as a replay's (a capture's rows are all its own)."""
    views = case.plan["main"].views
    real = views if capture else case.views
    ctx = StepContext(request_ids=tuple(v.request_id for v in real), graph_walk="w", slot=slot, capture=capture,
                      slot_lease=None if bucket is None else SlotLease(slot, bucket))
    if len(real) < len(views):
        ctx.set_padded_rids(tuple(v.request_id for v in views))
    ctx.plan_results["kv"] = case.plan
    attn.plan(AttentionStep(**step), ctx)


def _over_ranks(rank, world, backend, store, over_sp, rows, captured=False, padding=(0, 0)):
    """One rank of a ``world``-rank placement running ``rows`` through ``FlashInferManager.run``
    against one rank attending every key; every rank issues the same collectives, same shapes.
    ``captured`` (NCCL only) does what production does for a graph bucket: captures ``run`` with
    its collectives on a step of exactly the bucket's tokens on every rank, then plans the step and
    replays it. ``padding``: ``(rows, span)`` of the replay's padding rows, which make its bucket
    larger than the step: on SINK_PAGE as `KVManager` plans them, and NaN in their queries, keys
    and the sink."""
    os.environ["MSTAR_WORKSPACE_BUFFER_MB"] = "64" if backend == "nccl" else "1"
    dist.init_process_group(backend, init_method=f"file://{store}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    cuda = backend == "nccl"
    device = torch.device("cuda", rank) if cuda else CPU
    if cuda:
        torch.cuda.set_device(device)
    else:
        sys.modules["flashinfer"] = _reference_flashinfer()
    calls = []
    # `CommGroup.all_gather` calls the into-tensor gather it picked at import
    for owner, name in ((dist, "all_to_all_single"), (communication, "_all_gather_into_tensor")):
        def record(*args, _name=name, _call=getattr(owner, name), **kwargs):
            calls.append((_name, tuple(args[0].shape), tuple(args[1].shape), *(str(a) for a in args[2:])))
            return _call(*args, **kwargs)
        setattr(owner, name, record)
    group = CommGroup(rank, rank, list(range(world)))
    group.device_group, group.initialized = dist.group.WORLD, True
    graph = None
    try:
        joint = (JointGroups(tp_group=CommGroup.trivial(), sp_group=group, instance=group) if over_sp
                 else JointGroups(tp_group=group, sp_group=CommGroup.trivial(), instance=group))
        dtype = torch.bfloat16 if cuda else torch.float32
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
        _plan(attn, full, bucket=bucket, capture=captured)
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
            _plan(attn, case, bucket=bucket)
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
        torch.testing.assert_close(out[:n].float(), case.want()[:, heads], **tol)
        if rows == PREFILL_ROWS and not captured:
            # a first prefill: no row holds stored context on any rank, so nothing is exchanged
            first = _Case.build(((0, 5), (0, 3)), device, dtype)
            _plan(attn, first)
            exchanged = len(calls)
            torch.testing.assert_close(first.runner(attn, placement, heads)().float(), first.want()[:, heads], **tol)
            assert len(calls) == exchanged, "a step with no stored context ran the exchange"
        everyone = [None] * world
        dist.all_gather_object(everyone, calls)
        assert calls and all(c == calls for c in everyone), everyone
    finally:
        if graph is not None:
            # a live graph holds its captured NCCL work, and destroy waits on it
            graph.reset()
        # defensive: clear our reference so destroy drops the group's last one and joins gloo's
        # threads, whatever else holds the group
        group.device_group = None
        dist.destroy_process_group()


class _OneRank:
    """A Qwen3.5 layer's KV handle as one rank attending every key: ``case``'s stored context, then
    the k and v the layer adds, which it writes into ``case`` as the step's own tokens."""

    def __init__(self, case: _Case):
        self.case = case

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
    from mstar.engine.state.adapters import build_handle
    from mstar.engine.state.analysis import position_key
    from mstar.engine.state.contract import ArrayStep
    from mstar.model.qwen3_5.components.language_model import Qwen3_5Attention, RopeCache
    from mstar.model.qwen3_5.config import ATTN

    os.environ["MSTAR_WORKSPACE_BUFFER_MB"] = "1"
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=world)
    torch.set_num_threads(1)
    sys.modules["flashinfer"] = _reference_flashinfer()
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
        cache, writes = case.cache(placement), []
        pool = types.SimpleNamespace(
            layer_view=lambda i: cache, write_kv=lambda k, v, **kwargs: writes.append((k, v)),
            opens_label=lambda *args: True,
        )
        handle = build_handle(array, {KV_CACHE: pool, ATTN: attn, position_key(array): None})
        step = types.SimpleNamespace(
            state={KV_CACHE: ArrayStep()}, segments=tuple(
                Segment(view.request_id, "main", new) for view, (_, new) in zip(case.views, case.rows, strict=True)
            ),
            ctx=types.SimpleNamespace(replays=False, capture=False),
        )
        handle.begin_step(step)
        layer.bind_state({KV_CACHE: handle})
        torch.testing.assert_close(layer(hidden), want)
        handle.check_step(step)
        handle.end_step(step)
        # the step's k and v, written once
        (written,) = writes
        torch.testing.assert_close(written, case.new_keys())
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


@pytest.mark.parametrize("rows", [PREFILL_ROWS, DECODE_ROWS], ids=["prefill", "decode"])
@pytest.mark.parametrize("over_sp", [False, True], ids=["tp", "sp"])
def test_two_ranks_attend_a_placed_pool_like_one_rank_with_the_same_collectives(tmp_path, over_sp, rows):
    mp.spawn(_over_ranks, args=(P, "gloo", str(tmp_path / "store"), over_sp, rows), nprocs=P)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("captured, padding", [
    (False, None), (True, None),
    # a bucket of more rows than the step: padding rows of 2 tokens (a decode's of 1), and of none
    (True, 2), (True, 0),
], ids=["eager", "captured", "captured-padding-rows", "captured-span-0-rows"])
@pytest.mark.parametrize("rows", [PREFILL_ROWS, DECODE_ROWS], ids=["prefill", "decode"])
@pytest.mark.parametrize("over_sp", [False, True], ids=["tp", "sp"])
def test_two_gpus_attend_a_placed_pool_like_one(tmp_path, over_sp, rows, captured, padding):
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
    mp.spawn(_over_ranks, args=(P, "nccl", str(tmp_path / "store"), over_sp, rows, captured, padding), nprocs=P)


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
    monkeypatch.setitem(sys.modules, "flashinfer", _reference_flashinfer())
    # the captured context wrapper stages its plan through a CUDA event; this needs only the own part
    context = types.SimpleNamespace(plan=lambda **kwargs: None)
    monkeypatch.setattr(ragged_mod, "zero_unplanned_rows", _zero_unplanned_rows_on_cpu)
    monkeypatch.setattr(fi_mod, "FlashInferPrefillWrapper", lambda **kwargs: context)
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    attn, placement = _cp_attention(joint, CPU, torch.float32)
    case = _Case.build(PREFILL_ROWS, CPU, torch.float32, padding=padding)
    n, pad = case.q.shape[0], padding[0] * padding[1] + 8
    _plan(attn, case, bucket=BucketKey("w", bs=len(PREFILL_ROWS) + padding[0], num_tokens=n + pad))
    # this rank's query heads, which its own tokens attend for alone
    q, k, v, _ = case.inputs(placement, heads=slice(0, Q_HEADS // P), pad=pad)
    out, lse = attn._current_plan_states["main"].new.run(q, k, v, return_lse=True)
    merged = merge(torch.stack([torch.zeros_like(out), out]), torch.stack([torch.full_like(lse, -math.inf), lse]))
    assert torch.isfinite(merged).all() and not merged[n:].any()
    qo = np.cumsum([0, *(new for _, new in PREFILL_ROWS)]).tolist()
    want = torch.cat([_reference(q[a:b], k[a:b], v[a:b])[0] for a, b in itertools.pairwise(qo)])
    torch.testing.assert_close(merged[:n], want)


def test_a_captured_step_exchanges_with_no_stored_context(monkeypatch):
    """A graph replays the collectives of its capture whatever a later step stores, so a captured
    step of no stored context, as a capture step usually is, can't skip them."""
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    monkeypatch.setitem(sys.modules, "flashinfer", _reference_flashinfer())
    # the captured context wrapper stages its plan through a CUDA event; this needs only its partial
    context = types.SimpleNamespace(
        plan=lambda **kwargs: None,
        run=lambda q, cache, return_lse: (torch.zeros_like(q), torch.full(q.shape[:2], -math.inf)),
    )
    monkeypatch.setattr(ragged_mod, "zero_unplanned_rows", _zero_unplanned_rows_on_cpu)
    monkeypatch.setattr(fi_mod, "FlashInferPrefillWrapper", lambda **kwargs: context)
    exchanged = []

    def exchange(self, out, lse, valid):
        exchanged.append(out)
        return out[None], lse[None]

    monkeypatch.setattr(ContextParallelWrapper, "_exchange_partials", exchange)
    joint = JointGroups(tp_group=CommGroup.trivial(), sp_group=_group(0, P), instance=_group(0, P))
    attn, placement = _cp_attention(joint, CPU, torch.float32)
    case = _Case.build(((0, 5), (0, 3)), CPU, torch.float32)
    _plan(attn, case, bucket=BucketKey("w", bs=2, num_tokens=8))
    torch.testing.assert_close(case.runner(attn, placement)(), case.want())
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
    fi = _reference_flashinfer()
    plans = []
    ragged_plan = fi.BatchPrefillWithRaggedKVCacheWrapper.plan

    def counted(self, *args, **kwargs):
        plans.append(args[0].tolist())
        return ragged_plan(self, *args, **kwargs)

    monkeypatch.setattr(fi.BatchPrefillWithRaggedKVCacheWrapper, "plan", counted)
    monkeypatch.setitem(sys.modules, "flashinfer", fi)
    # the captured context wrapper stages its plan through a CUDA event; this needs only the own part
    context = types.SimpleNamespace(plan=lambda **kwargs: None)
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


def test_a_placed_pools_own_tokens_attend_at_the_contexts_sm_scale(monkeypatch):
    """The two parts merge by log-sum-exp, so a declared ``sm_scale`` must reach both."""
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    monkeypatch.setitem(sys.modules, "flashinfer", _reference_flashinfer())
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    attn, _ = _cp_attention(joint, CPU, torch.float32, sm_scale=0.5)
    _plan(attn, _Case.build(PREFILL_ROWS, CPU, torch.float32))
    state = attn._current_plan_states["main"]
    assert state.new.sm_scale == state.context.sm_scale == 0.5


def test_tp_placed_decode_runs_on_tensor_cores_whatever_decode_tensor_cores_says(monkeypatch):
    """A rank attends every partner's query heads over one KV head, a GQA group FlashInfer's
    CUDA-core decode has no kernel for unless it is one of a few sizes."""
    monkeypatch.setenv("MSTAR_WORKSPACE_BUFFER_MB", "1")
    monkeypatch.setitem(sys.modules, "flashinfer", _reference_flashinfer())
    joint = JointGroups(tp_group=_group(0, P), sp_group=CommGroup.trivial(), instance=_group(0, P))
    attn, _ = _cp_attention(joint, CPU, torch.float32, decode_tensor_cores=False)
    _plan(attn, _Case.build(DECODE_ROWS, CPU, torch.float32))
    assert attn._current_plan_states["main"].context.attn_wrapper.use_tensor_cores
