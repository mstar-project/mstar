"""Self KV and attention on persistent rows: a captured decode graph builds
its write addressing and FlashInfer's paged arrays from the rows, and every
replay writes each row's token where its stream says and attends exactly its
stream's tokens, across joins, leaves, removals, page crossings and slots.

The fills themselves are checked on CPU against host-built arrays; the graph
test needs CUDA and FlashInfer.
"""
import random

import numpy as np
import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig
from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.engine.resources.kv.rows import KVDeviceRows, decode_addressing, decode_indptrs
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext

PAGE, MAX_PAGES = 4, 5


def _rows(lens, pages):
    table = torch.full((len(lens), MAX_PAGES), SINK_PAGE, dtype=torch.int32)
    for i, held in enumerate(pages):
        table[i, :len(held)] = torch.tensor(held, dtype=torch.int32)
    return KVDeviceRows(torch.tensor(lens, dtype=torch.int32), table, PAGE)


# lengths before the step's token; the last two rows are padding
_LENS = [0, 3, 4, 13, 0, 0]
_PAGES = [[7], [2], [5, 9], [1, 3, 6, 8], [], []]


def test_each_row_writes_its_token_after_its_stored_length():
    rows = _rows(_LENS, _PAGES)
    page, offset = torch.empty(6, dtype=torch.long), torch.empty(6, dtype=torch.long)
    decode_addressing(rows, 6, page, offset)
    assert page.tolist() == [7, 2, 9, 8, SINK_PAGE, SINK_PAGE]
    assert offset.tolist() == [0, 3, 0, 1, 0, 0]


def test_the_paged_arrays_cover_each_row_and_its_token():
    rows = _rows(_LENS, _PAGES)
    indptr = torch.full((7,), -1, dtype=torch.int32)
    indices = torch.full((40,), -1, dtype=torch.int32)
    last = torch.full((6,), -1, dtype=torch.int32)
    decode_indptrs(rows, 6, indptr, indices, last)
    assert indptr.tolist() == [0, 1, 2, 4, 8, 9, 10]
    assert indices[:10].tolist() == [7, 2, 5, 9, 1, 3, 6, 8, SINK_PAGE, SINK_PAGE]
    assert last.tolist() == [1, 4, 1, 2, 1, 1]
    # what a host plan of the same rows hands FlashInfer
    lens = np.array(_LENS) + 1
    assert (((indptr[1:] - indptr[:-1]).numpy() - 1) * PAGE + last.numpy()).tolist() == lens.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fills' Triton kernels")
@pytest.mark.parametrize(("n", "max_pages", "page"), [(1, 5, 4), (6, 5, 4), (37, 70, 16), (64, 3, 128)])
def test_the_device_fills_match_the_torch_ones(n, max_pages, page):
    pytest.importorskip("triton")
    gen = torch.Generator().manual_seed(n)
    lens = torch.randint(0, (max_pages - 1) * page, (n + 3,), generator=gen, dtype=torch.int32)
    lens[n:] = 0
    table = torch.randint(1, 1000, (n + 3, max_pages), generator=gen, dtype=torch.int32)
    rows = KVDeviceRows(lens, table, page)
    cuda = KVDeviceRows(lens.cuda(), table.cuda(), page)
    out = {}
    for dev, r in (("cpu", rows), ("cuda", cuda)):
        page_at, offset = (torch.full((n,), -1, dtype=torch.long, device=dev) for _ in range(2))
        decode_addressing(r, n, page_at, offset)
        indptr = torch.full((n + 1,), -1, dtype=torch.int32, device=dev)
        indices = torch.full((n * max_pages,), -1, dtype=torch.int32, device=dev)
        last = torch.full((n,), -1, dtype=torch.int32, device=dev)
        decode_indptrs(r, n, indptr, indices, last)
        out[dev] = [t.cpu() for t in (page_at, offset, indptr, last)] + [indices.cpu()[:int(indptr[-1])]]
    for want, got in zip(out["cpu"], out["cuda"], strict=True):
        assert torch.equal(want, got)


def test_a_capture_records_only_its_own_leases_latest_fills():
    """The runner plans a spec once per warm-up forward, and other specs in
    between, before it captures: only the last plan's fills, and only the
    captured lease's, belong in its graph."""
    from mstar.engine.rows import RowFills

    ran = []
    fills = RowFills()
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=4, num_tokens=4))
    other = SlotLease(slot=1, bucket=lease.bucket)
    for plan in range(3):
        fills.arm(lease, "main", lambda plan=plan: ran.append(("main", plan)))
        fills.arm(other, "main", lambda: ran.append("other"))
    fills.arm(lease, "cross", lambda: ran.append("cross"))
    fills.before_forward(lease)
    assert ran == [("main", 2), "cross"]
    fills.before_forward(lease)  # consumed by its capture
    assert ran == [("main", 2), "cross"]
    fills.before_forward(other)
    assert ran[2:] == ["other"]


# -- a captured graph over real KV and attention -------------------------------

flashinfer = pytest.importorskip("flashinfer") if torch.cuda.is_available() else None
WALK, KEY, LABEL, BUCKET, HANDLES = "decode", "kv", "main", 4, 12
HEADS, HEAD_DIM, GRAPH_PAGE = 2, 64, 16


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        pass

    def owns_transfer_info(self, transfer_info, **kwargs):
        return False

    def cleanup(self):
        pass

    def remove_request(self, request_id):
        pass


class _Harness:
    """What the engine does around a persistent walk's captured steps, by hand."""

    def __init__(self, tensor_cores: bool, max_live: int | None = None):
        from mstar.engine.resources.attn.flashinfer import FlashInferManager
        from mstar.engine.rows import NodeRows, RowWalk

        dev = torch.device("cuda")
        cfg = PagedKVConfig(
            num_layers=1, num_kv_heads=HEADS, head_dim=HEAD_DIM, max_seq_len=96,
            max_num_pages=96, page_size=GRAPH_PAGE,
        )
        self.kv = manager_mod.KVManager(
            cfg=cfg, name=KEY, joint_comm_group=None, transfer_engine_info=None,
            device=dev, dtype=torch.bfloat16,
        )
        self.kv._cg_max_seq_len = BUCKET
        # every page holds something to attend; a step's own token overwrites its slot
        self.kv.kv_cache.tensor.normal_()
        self.attn = FlashInferManager(
            kv_cache=KEY, device=dev, dtype=torch.bfloat16, kv_config=cfg,
            decode_tensor_cores=tensor_cores,
        )
        self.node_rows = NodeRows.build(
            "toy", {WALK: RowWalk(labels={KEY: (LABEL,)})}, lambda walk: BUCKET,
            {KEY: self.kv, "attn": self.attn}, dev, max_live=max_live,
        )
        self.space = self.node_rows.space(WALK)
        self.q = torch.zeros(BUCKET, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
        self.k = torch.zeros_like(self.q)
        self.v = torch.zeros_like(self.q)
        # slot -> (graph, its output)
        self.graphs = {slot: self._capture(slot) for slot in (0, 1)}

    def _forward(self):
        self.kv.write_kv(self.k, self.v, layer_idx=0, label=LABEL)
        return self.attn.run(self.q, LABEL, self.kv.layer_view(0))

    def _lease(self, slot):
        return SlotLease(slot=slot, bucket=BucketKey(graph_walk=WALK, bs=BUCKET, num_tokens=BUCKET))

    def _plan(self, ctx, rids):
        from mstar.engine.resources.attn.config import AttentionStep

        step = KVStep(segments=tuple(Segment(rid, LABEL, 1) for rid in rids))
        assert self.kv.admit(step, ctx).ok
        ctx.plan_results = {KEY: self.kv.plan(step, ctx)}
        self.attn.plan(AttentionStep(causal=True), ctx)
        return step

    def _capture(self, slot):
        pads = tuple(-1 - i for i in range(BUCKET))
        ctx = StepContext(request_ids=(), graph_walk=WALK, slot=slot, capture=True, slot_lease=self._lease(slot))
        ctx.set_padded_rids(pads)
        # as the runner does: a plan per warm-up forward, then the capture's
        self._plan(ctx, pads)
        self._plan(ctx, pads)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._forward()  # warm-up, as the runner's: no fills
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.node_rows.before_forward(WALK, self._lease(slot))
            out = self._forward()
            self.space.after_forward()
        return graph, out

    def prefill(self, rid, n):
        self.kv.ingest_request(rid, KVReqConfig())
        step = KVStep(segments=(Segment(rid, LABEL, n),))
        ctx = StepContext(request_ids=(rid,), graph_walk="prefill", slot=0, capture=False)
        assert self.kv.admit(step, ctx).ok
        self.kv.plan(step, ctx)
        self.kv.commit(step, ctx)

    def step(self, rids, slot):
        """One replayed step over ``rids``; returns them in row order, with
        each one's attention output and the token it wrote."""
        ctx = StepContext(
            request_ids=tuple(rids), graph_walk=WALK, slot=slot, capture=False, slot_lease=self._lease(slot),
        )
        ctx.rows = self.node_rows.begin(WALK, rids)
        order = list(ctx.rows.order)
        ctx.request_ids = tuple(order)
        padded = tuple(order) + tuple(-1 - i for i in range(BUCKET - len(order)))
        ctx.set_padded_rids(padded)
        step = self._plan(ctx, padded)
        ctx.rows.stage()
        self.node_rows.commit(WALK, ctx.rows)
        ctx.rows.apply()
        self.node_rows.gather(WALK, BUCKET)
        for t in (self.q, self.k, self.v):
            t.normal_()
        graph, out = self.graphs[slot]
        graph.replay()
        self.kv.commit(step, ctx)
        torch.cuda.synchronize()
        n = len(order)
        return order, out[:n].float().clone(), self.k[:n].clone(), self.v[:n].clone(), self.q[:n].float().clone()

    def stream_kv(self, rid):
        stream = self.kv._streams[rid][LABEL]
        layer = self.kv.kv_cache.tensor[0]  # [pages, 2, page, heads, dim]
        pages = layer[torch.tensor(stream.page_indices.tolist(), device=layer.device)]
        kv = pages.permute(1, 0, 2, 3, 4).reshape(2, -1, HEADS, HEAD_DIM)[:, :stream.stored_len]
        return kv[0], kv[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="captures a CUDA graph")
@pytest.mark.parametrize("tensor_cores", [False, True])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("max_live", [None, 8])
def test_replays_write_and_attend_each_rows_own_stream(monkeypatch, tensor_cores, seed, max_live):
    """Rows a step doesn't run stay resident while there is room (a step's
    rows without ``max_live``), and the steps gather the rows they run."""
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    rng = random.Random(seed)
    h = _Harness(tensor_cores, max_live)
    live: set[int] = set()
    free = list(range(HANDLES))
    slot, idle = 0, 0
    for _ in range(60):
        if live and rng.random() < 0.08:
            rid = rng.choice(sorted(live))
            h.node_rows.forget(rid)
            h.kv.remove_request(rid)
            live.discard(rid)
            free.append(rid)
        while free and len(live) < 6 and rng.random() < 0.3:
            rid = free.pop(0)
            h.prefill(rid, rng.randrange(1, 40))
            live.add(rid)
        if not live:
            continue
        rids = rng.sample(sorted(live), rng.randint(1, min(BUCKET, len(live))))
        order, out, k, v, q = h.step(rids, slot)
        idle += h.space.batch.n_held > len(order)
        assert h.space.batch.n_held <= (max_live or BUCKET)
        slot ^= 1
        for i, rid in enumerate(order):
            keys, values = h.stream_kv(rid)
            # the step's token landed after the stream's earlier ones
            assert torch.equal(keys[-1], k[i]) and torch.equal(values[-1], v[i]), rid
            scores = torch.einsum("hd,thd->ht", q[i], keys.float()) / HEAD_DIM ** 0.5
            want = torch.einsum("ht,thd->hd", scores.softmax(-1), values.float())
            torch.testing.assert_close(out[i], want, atol=2e-2, rtol=2e-2)
    h.space.check_mirrors()
    assert idle, "some steps ran while other rows stayed resident"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="captures a CUDA graph")
def test_a_step_that_writes_nothing_after_a_replay_zeroes_the_addressing(monkeypatch):
    """The graph writes the slot's addressing, so a captured step on that
    slot that writes nothing can't assume the buffer is still zero."""
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    h = _Harness(tensor_cores=False)
    h.prefill(0, 20)

    def read():
        lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="read", bs=BUCKET, num_tokens=BUCKET))
        padded = (0, *(-1 - i for i in range(BUCKET - 1)))
        ctx = StepContext(request_ids=(0,), graph_walk="read", slot=0, capture=False, slot_lease=lease)
        ctx.set_padded_rids(padded)
        step = KVStep(segments=tuple(Segment(rid, LABEL, 0) for rid in padded))
        assert h.kv.admit(step, ctx).ok
        h.kv.plan(step, ctx)
        h.kv.commit(step, ctx)
        torch.cuda.synchronize()
        return h.kv._current_plan_states[LABEL].staging

    assert not read().any()
    h.step([0], 0)  # the replay writes row 0's page and offset
    assert not read().any()
