"""Resident planning: a captured decode step reads its rows off device tables and
updates them itself, and what its graph computes matches what a host plan would
have staged.

The resources here are the real ones on the CPU. ``gather`` and ``commit_in_graph``
run the ops a capture records, so each step below is what a replay does; a second
set of resources plans the same requests on the host as the reference.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.resources import resident as resident_mod  # noqa: E402
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec  # noqa: E402
from mstar.engine.resources.base import CGSlotKey, Resource  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import KVLayout, KVReqConfig, KVSpec, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.kv.plan import SINK_PAGE  # noqa: E402
from mstar.engine.resources.position.config import PositionConfig, PositionStep, PosScheme  # noqa: E402
from mstar.engine.resources.position.manager import RopeManager  # noqa: E402
from mstar.engine.resources.recurrent.config import (  # noqa: E402
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import SINK_SLOT, RecurrentStatePool  # noqa: E402
from mstar.engine.resources.resident import TRASH_SLOT, ResidentPlanner, build_planner  # noqa: E402
from mstar.engine.resources.spec import resolve_spec_dependencies  # noqa: E402
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext, SubmoduleStep  # noqa: E402
from mstar.utils import knobs  # noqa: E402

PAGE = 4
WALK = "decode"
CPU = torch.device("cpu")


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def remove_request(self, request_id):
        del request_id

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _resources(cpu_offload_pages: int = 0, max_num_pages: int = 64):
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=512,
            max_num_pages=max_num_pages, page_size=PAGE, cpu_offload_pages=cpu_offload_pages,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )
    kv._cg_max_seq_len = 64
    pool = RecurrentStatePool(device=CPU, config=RecurrentStateConfig(
        num_layers=1, max_slots=9, blocks={"state": RecurrentBlockConfig(shape=(2,), dtype=torch.float32)},
    ))
    rope = RopeManager(PositionConfig(kv_cache="kv"), CPU, max_seq_len=512, head_dim=8)
    # page-locked memory needs a GPU, and nothing here copies out of it asynchronously
    rope._pinned_pos_ids_buffer = lambda key, n: torch.empty(n, dtype=torch.long)
    return {"kv": kv, "pool": pool, "rope": rope}


def _ingest(res: dict[str, Resource], rid: int) -> None:
    res["kv"].ingest_request(rid, KVReqConfig())
    res["pool"].ingest_request(rid, None)
    res["rope"].ingest_request(rid, None)


def _host_step(res, rids, spans, walk="prefill", advance=None):
    """A host-planned step, eager, as a prefill or mixed step runs."""
    segments = tuple(Segment(r, "main", n) for r, n in zip(rids, spans, strict=True))
    ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)
    kv, pool = KVStep(segments=segments), RecurrentStep(segments=segments)
    rope = PositionStep(segments=segments, advance=advance)
    assert res["kv"].admit(kv, ctx).ok
    assert res["pool"].admit(pool, ctx).ok
    ctx.plan_results["kv"] = res["kv"].plan(kv, ctx)
    res["pool"].plan(pool, ctx)
    res["rope"].plan(rope, ctx)
    res["kv"].commit(kv, ctx)
    res["pool"].commit(pool, ctx)
    res["rope"].commit(rope, ctx)
    return SubmoduleStep(steps={"kv": kv}, segments=list(segments)), ctx


def _captured_step(res, planner: ResidentPlanner | None, rids, slot, bs, preplan=False, run=True):
    """One decode step on a bucket of ``bs`` rows. With a planner, the replay of a
    resident capture; without one, the host plan of the same step."""
    padded = tuple(rids) + tuple(-(i + 1) for i in range(bs - len(rids)))
    bucket = BucketKey(graph_walk=WALK, bs=bs, num_tokens=bs)
    lease = SlotLease(slot=slot, bucket=bucket)
    ctx = StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=slot, capture=False, slot_lease=lease)
    ctx.set_padded_rids(padded)
    segments = tuple(Segment(r, "main", 1) for r in padded)
    kv, pool, rope = KVStep(segments=segments), RecurrentStep(segments=segments), PositionStep(segments=segments)
    step = SubmoduleStep(steps={"kv": kv}, segments=list(segments))
    assert res["kv"].admit(kv, ctx).ok
    assert res["pool"].admit(pool, ctx).ok
    # a capture's first plan builds the static buffers on the host
    first = StepContext(request_ids=tuple(padded), graph_walk=WALK, slot=slot, capture=True, slot_lease=lease)
    if (bucket, slot) not in res.setdefault("captured", set()):
        res["captured"].add((bucket, slot))
        first.plan_results["kv"] = res["kv"].plan(kv, first)
        res["pool"].plan(pool, first)
        res["rope"].plan(rope, first)
    captured = None
    if planner is not None:
        captured = planner.capture_step(bucket, slot, step)
        ctx.resident = planner
    if planner is None or preplan:
        ctx.is_preplan = preplan
        if planner is not None:
            planner.plan(step, ctx)
        ctx.plan_results["kv"] = res["kv"].plan(kv, ctx)
        res["pool"].plan(pool, ctx)
        res["rope"].plan(rope, ctx)
        ctx.is_preplan = False
    out = SimpleNamespace(step=step, ctx=ctx, kv=kv, pool=pool, rope=rope, captured=captured)
    if run:
        _replay(res, planner, out)
    return out


def _promote(res, planner, s):
    ctx = s.ctx
    planner.plan(s.step, ctx)
    ctx.plan_results["kv"] = res["kv"].plan(s.kv, ctx)
    res["pool"].plan(s.pool, ctx)
    res["rope"].plan(s.rope, ctx)


def _replay(res, planner, s, promote=True):
    """What the GPU thread does from the promote on."""
    ctx, bucket, lease = s.ctx, s.ctx.slot_lease.bucket, s.ctx.slot_lease
    if planner is not None:
        if promote:
            _promote(res, planner, s)
        planner.flush(ctx.resident_plan)
        planner.gather(s.captured)
    state = res["kv"]._static_plan_state(lease.slot, ("main", 0))
    ck = CGSlotKey(bucket=bucket, slot=lease.slot, label="main")
    s.pages = state.token_to_page[: bucket.bs].tolist()
    s.offsets = state.token_to_cache[: bucket.bs].tolist()
    s.positions = res["rope"]._static_pos_ids[ck][: bucket.bs].tolist()
    s.pool_slots = res["pool"]._cg_addressing[ck].slot_indices[: bucket.bs].tolist()
    if planner is not None:
        planner.commit_in_graph(s.captured)
    res["kv"].commit(s.kv, ctx)
    res["pool"].commit(s.pool, ctx)
    res["rope"].commit(s.rope, ctx)
    if planner is not None:
        planner.commit(s.step, ctx)
    return s


def _setup(prompts: dict[int, int], cpu_offload_pages: int = 0, max_num_pages: int = 64):
    """The same requests prefilled on resident resources and on a host-planned reference."""
    live, ref = _resources(cpu_offload_pages, max_num_pages), _resources(cpu_offload_pages, max_num_pages)
    planner = ResidentPlanner("n", {WALK}, {k: live[k] for k in ("kv", "pool", "rope")}, CPU, max_slots=8)
    for res in (live, ref):
        for rid, n in prompts.items():
            _ingest(res, rid)
            _host_step(res, [rid], [n])
    return live, ref, planner


def _same(a, b, real: int):
    assert a.pages[:real] == b.pages[:real]
    assert a.offsets[:real] == b.offsets[:real]
    assert a.positions[:real] == b.positions[:real]
    assert a.pool_slots[:real] == b.pool_slots[:real]


def _row(planner: ResidentPlanner, live, rid: int):
    tables = live["kv"]._resident
    slot = planner.slot_of(rid)
    n = int(tables.kv_len.view[slot, 0])
    cols = n // PAGE + (1 if n % PAGE else 0)
    return n, tables.block_table.view[slot, :cols].tolist()


def _assert_row(planner, live, rid):
    stream = live["kv"]._streams[rid]["main"]
    n, pages = _row(planner, live, rid)
    assert n == stream.stored_len
    assert pages == stream.page_indices[: -(-n // PAGE)]


# ---- the rows ----

def test_a_new_row_is_written_whole_by_its_first_step():
    live, ref, planner = _setup({1: 9})
    assert planner.slot_of(1) is None
    a = _captured_step(live, planner, [1], slot=0, bs=1)
    b = _captured_step(ref, None, [1], slot=0, bs=1)
    _same(a, b, 1)
    _assert_row(planner, live, 1)
    assert planner.slot_of(1) != TRASH_SLOT


def test_steps_with_crossings_match_the_host_plan_and_the_host_lengths():
    live, ref, planner = _setup({1: 3, 2: 5, 3: 8})
    rids = [1, 2, 3]
    for t in range(40):
        a = _captured_step(live, planner, rids, slot=t % 2, bs=4)
        b = _captured_step(ref, None, rids, slot=t % 2, bs=4)
        _same(a, b, 3)
    tables = live["kv"]._resident
    for rid in rids:
        _assert_row(planner, live, rid)
        slot = planner.slot_of(rid)
        assert tables.host_len[slot] == live["kv"]._streams[rid]["main"].stored_len
        assert tables.host_cols[slot] == live["kv"]._streams[rid]["main"].stored_len // PAGE + (
            1 if live["kv"]._streams[rid]["main"].stored_len % PAGE else 0
        )


def test_a_steady_batch_keeps_its_row():
    live, ref, planner = _setup({1: 1, 2: 1})
    del ref
    _captured_step(live, planner, [1, 2], slot=0, bs=2)
    buffers = planner.buffers(BucketKey(WALK, 2, 2), 0)
    before = buffers.slots.clone()
    last = planner._last_slots
    _captured_step(live, planner, [1, 2], slot=0, bs=2)
    assert planner._last_slots is last
    assert torch.equal(buffers.slots, before)


def test_padding_rows_read_the_trash_row_and_leave_it_alone():
    live, ref, planner = _setup({1: 6})
    for t in range(9):
        a = _captured_step(live, planner, [1], slot=t % 2, bs=4)
        b = _captured_step(ref, None, [1], slot=t % 2, bs=4)
        _same(a, b, 1)
        assert a.pages[1:] == [SINK_PAGE] * 3
        assert a.offsets[1:] == [0] * 3
        assert a.pool_slots[1:] == [SINK_SLOT] * 3
        assert a.positions[1:] == [0] * 3
    tables = live["kv"]._resident
    assert int(tables.kv_len.view[TRASH_SLOT, 0]) == 0
    assert tables.block_table.view[TRASH_SLOT].eq(SINK_PAGE).all()
    assert int(live["pool"]._resident_table.view[TRASH_SLOT, 0]) == SINK_SLOT


def test_a_mixed_step_leaves_the_next_decode_step_a_ready_row():
    live, ref, planner = _setup({1: 3})
    for t in range(3):
        _captured_step(live, planner, [1], slot=t % 2, bs=1)
        _captured_step(ref, None, [1], slot=t % 2, bs=1)
    # a request's prompt beside the decode row, on the host plan; the decode row crosses here
    for res in (live, ref):
        _ingest(res, 2)
        step, ctx = _host_step(res, [1, 2], [1, 7], walk="mixed")
        if res is live:
            planner.commit(step, ctx)
    # and again, so the decode row is two columns behind
    for _ in range(5):
        for res in (live, ref):
            step, ctx = _host_step(res, [1, 2], [1, 1], walk="mixed")
            if res is live:
                planner.commit(step, ctx)
    a = _captured_step(live, planner, [1, 2], slot=0, bs=2)
    b = _captured_step(ref, None, [1, 2], slot=0, bs=2)
    _same(a, b, 2)
    _assert_row(planner, live, 1)
    _assert_row(planner, live, 2)


def test_a_resident_step_leaves_the_addressing_memo_to_host_steps():
    live, ref, planner = _setup({1: 3, 2: 5})
    del ref
    pool = live["pool"]
    for slot in (0, 1):
        _captured_step(live, planner, [1, 2], slot=slot, bs=2)
    _host_step(live, [1, 2], [1, 1], walk="mixed")
    memo, version = pool._addressing_memo["main"], pool._version
    for t in range(6):
        s = _captured_step(live, planner, [1, 2], slot=t % 2, bs=2)
        assert s.pool_slots == [pool._slots[r]["main"].index for r in (1, 2)]
    assert pool._addressing_memo["main"] is memo
    assert pool._version == version
    _host_step(live, [1, 2], [1, 1], walk="mixed")
    addressing = pool._current["main"]
    assert addressing.host_slot_indices is memo[1]
    assert addressing.host_slot_indices == [pool._slots[r]["main"].index for r in (1, 2)]


def test_a_resident_step_keeps_the_cached_kv_plan_attention_reads():
    live, ref, planner = _setup({1: 3, 2: 5, 3: 8})
    for res in (live, ref):
        res["kv"]._plan_cache_mode = 1
    rids = [1, 2, 3]
    pages = set()
    for t in range(40):
        a = _captured_step(live, planner, rids, slot=t % 2, bs=4)
        b = _captured_step(ref, None, rids, slot=t % 2, bs=4)
        _same(a, b, 3)
        x, y = (s.ctx.plan_results["kv"]["main"].cpu_indptrs for s in (a, b))
        for p, q in zip(x[:5], y[:5], strict=True):
            assert torch.equal(torch.as_tensor(p), torch.as_tensor(q))
        pages.add(a.pages[0])
    assert len(pages) > 1
    assert live["kv"].plan_cache_stats["hit"] > 20


def test_a_stage_is_timed_under_the_step_plan_phase(monkeypatch):
    live, ref, planner = _setup({1: 3})
    del ref
    seen = []
    laps = resident_mod.PhaseLaps
    monkeypatch.setattr(resident_mod, "PhaseLaps", lambda prefix: seen.append(prefix) or laps(prefix))
    s = _captured_step(live, planner, [1], slot=0, bs=1, run=False)
    s.ctx.plan_phase = "res.plan_fresh"
    _replay(live, planner, s)
    assert seen == ["res.plan_fresh.resident."]


def test_an_abandoned_plan_that_rewrote_a_row_behind_by_columns_writes_it_again():
    live, ref, planner = _setup({1: 3})
    a = _captured_step(live, planner, [1], slot=0, bs=1)
    b = _captured_step(ref, None, [1], slot=0, bs=1)
    _same(a, b, 1)
    # two page boundaries go by on the host
    for res in (live, ref):
        for _ in range(6):
            step, ctx = _host_step(res, [1], [1], walk="mixed")
            if res is live:
                planner.commit(step, ctx)
    assert 1 in planner._moved
    staged = _captured_step(live, planner, [1], slot=1, bs=1, preplan=True, run=False)
    assert 1 in staged.ctx.resident_plan.taken
    for key in ("kv", "pool", "rope"):
        live[key].clear_preplan()
    c = _captured_step(live, planner, [1], slot=0, bs=1)
    d = _captured_step(ref, None, [1], slot=0, bs=1)
    _same(c, d, 1)
    _assert_row(planner, live, 1)


def test_a_position_offset_carries_through_a_row():
    """A vision prompt advances positions by less than its tokens."""
    live, ref, planner = _setup({})
    for res in (live, ref):
        _ingest(res, 1)
        _host_step(res, [1], [10], walk="prefill_vision", advance=(6,))
    a = _captured_step(live, planner, [1], slot=0, bs=1)
    b = _captured_step(ref, None, [1], slot=0, bs=1)
    _same(a, b, 1)
    assert a.positions[0] == 6


def test_a_chunked_prefill_row_is_whole_at_its_first_decode_step():
    live, ref, planner = _setup({})
    for res in (live, ref):
        _ingest(res, 1)
        for chunk in (8, 8, 3):
            step, ctx = _host_step(res, [1], [chunk], walk="prefill")
            if res is live:
                planner.commit(step, ctx)
    a = _captured_step(live, planner, [1], slot=0, bs=1)
    b = _captured_step(ref, None, [1], slot=0, bs=1)
    _same(a, b, 1)
    _assert_row(planner, live, 1)


def test_more_chunks_after_decode_rewrite_the_row():
    live, ref, planner = _setup({1: 2})
    _captured_step(live, planner, [1], slot=0, bs=1)
    _captured_step(ref, None, [1], slot=0, bs=1)
    for res in (live, ref):
        step, ctx = _host_step(res, [1], [9], walk="prefill")
        if res is live:
            planner.commit(step, ctx)
    assert 1 in planner._stale
    a = _captured_step(live, planner, [1], slot=1, bs=1)
    b = _captured_step(ref, None, [1], slot=1, bs=1)
    _same(a, b, 1)
    _assert_row(planner, live, 1)
    assert not planner._stale


def test_a_fork_marks_the_row_stale():
    live, ref, planner = _setup({1: 9})
    del ref
    _captured_step(live, planner, [1], slot=0, bs=1)
    kv = live["kv"]
    kv._touch_plan_cache(1)
    assert 1 in planner._stale


def test_a_row_whose_pages_moved_is_rewritten():
    """What a reload leaves: the stream on other pages. Its hook is `_touch_plan_cache(rid)`;
    `test_real_offload_and_reload_mark_the_row` runs the real one on a GPU."""
    live, ref, planner = _setup({1: 9})
    del ref
    _captured_step(live, planner, [1], slot=0, bs=1)
    kv = live["kv"]
    stream = kv._streams[1]["main"]
    moved = kv._arena.acquire(len(stream.page_indices))
    kv._arena.release(stream.page_indices)
    stream.page_indices = moved
    kv._touch_plan_cache(1)
    assert 1 in planner._stale
    s = _captured_step(live, planner, [1], slot=1, bs=1)
    assert s.pages[0] == moved[-1]
    _assert_row(planner, live, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the host page pool is page-locked")
def test_real_offload_and_reload_mark_the_row():
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, max_num_pages=16, page_size=PAGE,
            cpu_offload_pages=16,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cuda"), dtype=torch.float32,
    )
    planner = ResidentPlanner("n", {WALK}, {"kv": kv}, torch.device("cuda"), max_slots=2)
    kv.ingest_request(1, KVReqConfig())
    segments = (Segment(1, "main", 9),)
    ctx = StepContext(request_ids=(1,), graph_walk="prefill", slot=0, capture=False)
    step = KVStep(segments=segments)
    assert kv.admit(step, ctx).ok
    ctx.plan_results["kv"] = kv.plan(step, ctx)
    kv.commit(step, ctx)
    planner._slot_of[1] = 1
    assert kv.offload(1) > 0
    assert 1 in planner._stale
    planner._stale.clear()
    torch.cuda.synchronize()
    assert kv.reload(1)
    assert 1 in planner._stale


def test_an_abandoned_preplan_writes_nothing_and_gives_its_rows_back():
    live, ref, planner = _setup({1: 3})
    a = _captured_step(live, planner, [1], slot=0, bs=1)
    b = _captured_step(ref, None, [1], slot=0, bs=1)
    _same(a, b, 1)
    # a host step dirties the row, a pre-plan takes it, and the step never runs
    for res in (live, ref):
        step, ctx = _host_step(res, [1], [6], walk="prefill")
        if res is live:
            planner.commit(step, ctx)
    table_before = planner.store.clone()
    staged = _captured_step(live, planner, [1], slot=1, bs=1, preplan=True, run=False)
    assert staged.ctx.resident_plan.taken == {1}
    assert torch.equal(planner.store, table_before)
    live["kv"].clear_preplan()
    live["pool"].clear_preplan()
    live["rope"].clear_preplan()
    # the next plan is another step: the row comes back, and is written whole then
    c = _captured_step(live, planner, [1], slot=0, bs=1)
    d = _captured_step(ref, None, [1], slot=0, bs=1)
    _same(c, d, 1)
    _assert_row(planner, live, 1)


def test_a_new_row_abandoned_before_its_first_step_is_still_written():
    live, ref, planner = _setup({1: 6})
    staged = _captured_step(live, planner, [1], slot=0, bs=1, preplan=True, run=False)
    assert planner.slot_of(1) is not None
    for key in ("kv", "pool", "rope"):
        live[key].clear_preplan()
    del staged
    c = _captured_step(live, planner, [1], slot=1, bs=1)
    d = _captured_step(ref, None, [1], slot=1, bs=1)
    _same(c, d, 1)
    _assert_row(planner, live, 1)


def test_a_promoted_preplan_reuses_its_stage():
    live, ref, planner = _setup({1: 3})
    del ref
    _captured_step(live, planner, [1], slot=0, bs=1)
    s = _captured_step(live, planner, [1], slot=1, bs=1, preplan=True, run=False)
    plan = s.ctx.resident_plan
    assert planner._staged is plan
    _replay(live, planner, s)
    assert planner._staged is None
    assert s.ctx.resident_plan is plan
    _assert_row(planner, live, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernels run on CUDA")
def test_the_captured_kernels_match_the_torch_ops():
    from mstar.engine.resources.kv.resident import KVResidentTables
    from mstar.engine.resources.resident import advance_column, gather_column, scatter_cells

    def run(device):
        store = torch.arange(100, 100 + 7 * 20, device=device, dtype=torch.long)
        slots = torch.tensor([3, 1, 0, 0], device=device)
        cells = torch.tensor([[7, 0, 15, 0], [55, 0, 66, 0]], device=device)
        scatter_cells(store, cells)
        out = torch.zeros(4, dtype=torch.long, device=device)
        small = torch.zeros(4, dtype=torch.int32, device=device)
        gather_column(store, slots, 7, 1, out)
        gather_column(store, slots, 7, 2, small, plus=out)
        advance_column(store, slots, 7, 1)
        tables = SimpleNamespace(
            planner=SimpleNamespace(store=store), page_size=4,
            block_table=SimpleNamespace(stride=7, col=3), kv_len=SimpleNamespace(stride=7, col=1),
        )
        store[slots * 7 + 1] = torch.tensor([9, 5, 0, 0], device=device)
        pages = torch.zeros(4, dtype=torch.long, device=device)
        offsets = torch.zeros(4, dtype=torch.long, device=device)
        step = SimpleNamespace(
            slots=slots, bucket=SimpleNamespace(bs=4), buffers=SimpleNamespace(
                cells=torch.zeros((2, 8), dtype=torch.long, device=device), kv_len=out,
            ),
        )
        KVResidentTables.gather(tables, step, pages, offsets)
        return [t.tolist() for t in (store, out, small, pages, offsets)]

    assert run(torch.device("cuda")) == run(CPU)


# ---- slots ----

def test_a_freed_slot_waits_for_the_last_step_that_read_it():
    live, ref, planner = _setup({1: 3})
    del ref
    # step N reads the slot and is still running: planned, not committed
    s = _captured_step(live, planner, [1], slot=0, bs=1, preplan=True, run=False)
    _promote(live, planner, s)
    slot = s.ctx.resident_plan.slots[0]
    planner.release(1)
    planner._free.clear()
    assert (int(slot), s.ctx.resident_plan.seq) in planner._cooling
    with pytest.raises(RuntimeError, match="slots are held"):
        with planner._lock:
            planner._take_slot()
    # commit lets the slot go; its late advance landed on the old owner's row
    _replay(live, planner, s, promote=False)
    with planner._lock:
        assert planner._take_slot() == slot


def test_a_late_commit_cannot_touch_the_next_owner():
    live, ref, planner = _setup({1: 3, 2: 5})
    del ref
    s = _captured_step(live, planner, [1], slot=0, bs=1, preplan=True, run=False)
    old = int(s.ctx.resident_plan.slots[0])
    planner.release(1)
    for key in ("kv", "pool", "rope"):
        live[key].clear_preplan()
    a = _captured_step(live, planner, [2], slot=1, bs=1, preplan=True, run=False)
    assert int(a.ctx.resident_plan.slots[0]) != old


def test_release_purges_the_handle():
    live, ref, planner = _setup({1: 3})
    del ref
    _captured_step(live, planner, [1], slot=0, bs=1)
    planner.mark_stale(1)
    planner.release(1)
    assert planner.slot_of(1) is None
    assert not planner._stale
    assert planner._last_rids is None


def test_a_stage_over_a_released_row_does_not_promote():
    live, ref, planner = _setup({1: 3})
    del ref
    s = _captured_step(live, planner, [1], slot=0, bs=1, preplan=True, run=False)
    old = s.ctx.resident_plan
    planner.release(1)
    assert planner._staged is old
    for key in ("kv", "pool", "rope"):
        live[key].clear_preplan()
    _replay(live, planner, s)
    assert s.ctx.resident_plan is not old
    assert planner.slot_of(1) is not None


def test_a_recycled_handle_gets_a_fresh_row():
    live, ref, planner = _setup({1: 9})
    del ref
    _captured_step(live, planner, [1], slot=0, bs=1)
    kv = live["kv"]
    kv.remove_request(1)
    live["pool"].remove_request(1)
    live["rope"].remove_request(1)
    planner.release(1)
    _ingest(live, 1)
    _host_step(live, [1], [2])
    _captured_step(live, planner, [1], slot=1, bs=1)
    _assert_row(planner, live, 1)


def test_running_out_of_slots_says_how_to_fix_it():
    live, ref, planner = _setup({})
    del ref
    planner._free.clear()
    _ingest(live, 1)
    _host_step(live, [1], [2])
    with pytest.raises(RuntimeError, match="resident_slots"):
        _captured_step(live, planner, [1], slot=0, bs=1)


# ---- refusals ----

class _Spare(Resource):
    @classmethod
    def build(cls, spec, info):
        return cls()


def _build(resources, nodes=None, walks=(WALK,)):
    return build_planner("n", walks, resources, nodes or {}, CPU, 8)


def test_every_resident_resource_passes():
    res = _resources()
    planner, why = _build({k: res[k] for k in ("kv", "pool", "rope")})
    assert planner is not None and why == ""


def test_no_walks_no_planner():
    res = _resources()
    planner, why = _build({"kv": res["kv"]}, walks=())
    assert planner is None and "no resident walk" in why


def test_a_resource_without_a_table_refuses():
    planner, why = _build({"other": _Spare()})
    assert planner is None and "_Spare keeps no resident table" in why


def test_a_pool_shared_by_nodes_refuses():
    res = _resources()
    planner, why = _build({"kv": res["kv"]}, nodes={"kv": ("a", "b")})
    assert planner is None and "shared by nodes" in why


def test_a_pool_read_across_nodes_refuses():
    res = _resources()
    res["kv"]._needs_remote_transfer = True
    planner, why = _build({"kv": res["kv"]})
    assert planner is None and "across nodes" in why


def test_the_mla_layout_refuses():
    """GLM-5.2's latent pages have no K/V axis to write through."""
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=512, max_num_pages=64, page_size=PAGE,
            layout=KVLayout.MLA, kv_lora_rank=4, qk_rope_head_dim=4,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )
    planner, why = _build({"kv": kv})
    assert planner is None and "MLA" in why


def test_a_sliding_window_pool_refuses():
    cfg = PagedKVConfig(num_layers=3, num_kv_heads=1, head_dim=8, max_seq_len=512, max_num_pages=64, page_size=8)
    resolve_spec_dependencies([
        KVSpec(resource_key="kv", nodes={"n"}, config=cfg),
        AttentionSpec(resource_key="local", nodes={"n"}, config=AttentionConfig("kv", sliding_window=6, layers=(1, 2))),
        AttentionSpec(resource_key="full", nodes={"n"}, config=AttentionConfig("kv")),
    ])
    kv = KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )
    planner, why = _build({"kv": kv})
    assert planner is None and "sliding-window" in why


def test_block_positions_refuse():
    rope = RopeManager(PositionConfig(kv_cache="kv", scheme=PosScheme.BLOCK), CPU, max_seq_len=512, head_dim=8)
    assert "doesn't advance" in rope.resident_refusal()


def test_a_pool_without_a_sink_refuses():
    pool = RecurrentStatePool(device=CPU, config=RecurrentStateConfig(
        num_layers=1, max_slots=4, blocks={"state": RecurrentBlockConfig(shape=(2,), dtype=torch.float32)},
        disable_sink_slot=True,
    ))
    assert "sink" in pool.resident_refusal()


@pytest.mark.parametrize("module,name", [
    ("mstar.engine.resources.attn.cross", "CrossAttentionManager"),
    ("mstar.engine.resources.kv.bounded.manager", "BoundedKVManager"),
    ("mstar.engine.resources.attn.dense", "DenseAttentionManager"),
    ("mstar.engine.resources.attn.flashinfer_mla", "FlashInferMLAManager"),
    ("mstar.engine.resources.linear_attn.mamba2", "Mamba2Manager"),
    ("mstar.engine.resources.kv.ring.manager", "RingKVManager"),
])
def test_resources_with_no_table_keep_the_base_refusal(module, name):
    """These stay on the host plan: none overrides the base's refusal."""
    import importlib

    cls = getattr(importlib.import_module(module), name)
    assert cls.resident_refusal is Resource.resident_refusal


def test_flashinfer_refuses_a_sliding_window():
    from mstar.engine.resources.attn.flashinfer import FlashInferManager

    manager = FlashInferManager.__new__(FlashInferManager)
    manager._window_left = 5
    assert "sliding window" in manager.resident_refusal()
    manager._window_left = -1
    assert manager.resident_refusal() is None


@pytest.mark.parametrize("field,value", [
    ("pre_forks", (("a", "b"),)), ("post_forks", (("a", "b"),)), ("sink", frozenset({(1, "main")})),
])
def test_a_resident_walk_can_not_declare_forks_or_a_sink(field, value):
    _, _, planner = _setup({})
    segments = [Segment(1, "main", 1)]
    kv = KVStep(segments=tuple(segments), **{field: value})
    with pytest.raises(ValueError, match=field):
        planner.capture_step(BucketKey(WALK, 1, 1), 0, SubmoduleStep(steps={"kv": kv}, segments=segments))


def test_a_resident_walk_is_one_token_a_row_in_one_stream():
    _, _, planner = _setup({})
    two = [Segment(1, "main", 2)]
    with pytest.raises(ValueError, match="one token per row"):
        planner.capture_step(BucketKey(WALK, 1, 2), 0, SubmoduleStep(steps={}, segments=two))
    labels = [Segment(1, "a", 1), Segment(1, "b", 1)]
    with pytest.raises(ValueError, match="one token per row"):
        planner.capture_step(BucketKey(WALK, 2, 2), 0, SubmoduleStep(steps={}, segments=labels))


# ---- the engine ----

def _engine(walks, resources, device=CPU):
    from mstar.engine.engine import Engine
    from mstar.engine.resources.runner import StepRunner

    engine = Engine.__new__(Engine)
    engine._device = device
    engine._runner = StepRunner(resources)
    engine._resource_nodes = {key: frozenset({"n"}) for key in resources}
    submodule = SimpleNamespace(resident_walks=lambda: frozenset(walks))
    engine._submodules = {"n": SimpleNamespace(submodule=submodule, resources=resources, resident=None)}
    return engine, engine._submodules["n"]


def test_the_engine_builds_a_planner_for_a_capable_node(monkeypatch):
    monkeypatch.setattr(knobs, "resident_slots", lambda: 8)
    res = _resources()
    engine, mgmt = _engine({WALK}, {k: res[k] for k in ("kv", "pool", "rope")})
    engine._build_resident_planner("n")
    assert isinstance(mgmt.resident, ResidentPlanner)
    assert mgmt.resident.covers(BucketKey(WALK, 2, 2))
    assert not mgmt.resident.covers(BucketKey("prefill_text", 2, 64))


def test_the_knob_turns_it_off(monkeypatch):
    monkeypatch.setenv("MSTAR_RESIDENT_PLANNING", "off")
    res = _resources()
    engine, mgmt = _engine({WALK}, {"kv": res["kv"]})
    engine._build_resident_planner("n")
    assert mgmt.resident is None
    assert res["kv"]._resident is None


def test_a_refused_node_stays_on_the_host_and_says_why(monkeypatch, caplog):
    monkeypatch.setattr(knobs, "resident_slots", lambda: 8)
    res = _resources()
    engine, mgmt = _engine({WALK}, {"kv": res["kv"], "other": _Spare()})
    with caplog.at_level("WARNING"):
        engine._build_resident_planner("n")
    assert mgmt.resident is None
    assert "_Spare keeps no resident table" in caplog.text
    assert res["kv"]._resident is None


def test_a_node_with_no_resident_walk_is_left_alone(caplog):
    res = _resources()
    engine, mgmt = _engine((), {"kv": res["kv"]})
    with caplog.at_level("WARNING"):
        engine._build_resident_planner("n")
    assert mgmt.resident is None
    assert not caplog.text


def test_an_eager_step_marks_its_rows():
    live, ref, planner = _setup({1: 3})
    del ref
    _captured_step(live, planner, [1], slot=0, bs=1)
    step, ctx = _host_step(live, [1], [1], walk="decode")
    planner.commit(step, ctx)
    assert 1 in planner._moved
    assert _captured_step(live, planner, [1], slot=1, bs=1).ctx.resident_plan.moved == {1}
    _assert_row(planner, live, 1)


def test_qwen35_decode_needs_the_rope_inside_the_graph(monkeypatch):
    from mstar.model.qwen3_5 import submodules

    monkeypatch.setattr(submodules, "ingraph_decode_rope", lambda: False)
    llm = next(
        cls for cls in vars(submodules).values()
        if isinstance(cls, type) and cls.__module__ == submodules.__name__ and "resident_walks" in vars(cls)
    )
    assert llm.resident_walks(None) == frozenset()
    monkeypatch.setattr(submodules, "ingraph_decode_rope", lambda: True)
    assert llm.resident_walks(None) == {"decode"}


# ---- the knob ----

def test_the_knob_takes_auto_and_off():
    assert knobs.check({"resident_planning": "auto"}) == {"resident_planning": "auto"}
    assert knobs.check({"resident_planning": False}) == {"resident_planning": "off"}
    with pytest.raises(ValueError, match="auto or off"):
        knobs.check({"resident_planning": "maybe"})
    with pytest.raises(ValueError, match="auto or off"):
        knobs.check({"resident_planning": True})


def test_the_knob_defaults_to_auto(monkeypatch):
    monkeypatch.delenv("MSTAR_RESIDENT_PLANNING", raising=False)
    knobs.configure(None)
    assert knobs.resident_planning() == "auto"
    monkeypatch.setenv("MSTAR_RESIDENT_PLANNING", "off")
    assert knobs.resident_planning() == "off"
