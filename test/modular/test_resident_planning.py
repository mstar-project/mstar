"""Resident planning: a captured decode step reads its rows off device tables and
updates them itself, and what its graph computes matches what a host plan would
have staged.

The resources here are the real ones on the CPU. ``gather`` and ``commit_in_graph``
run the ops a capture records, so each step below is what a replay does; a second
set of resources plans the same requests on the host as the reference.
"""
from __future__ import annotations

import random
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner  # noqa: E402
from mstar.engine.resources import resident as resident_mod  # noqa: E402
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec  # noqa: E402
from mstar.engine.resources.base import CGSlotKey, MoveKind, Residency, Resource  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.config import KVLayout, KVReqConfig, KVSpec, KVStep, PagedKVConfig  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.engine.resources.kv.plan import SINK_PAGE  # noqa: E402
from mstar.engine.resources.position.config import (  # noqa: E402
    LongRoPE,
    PositionConfig,
    PositionReqConfig,
    PositionStep,
    PosScheme,
)
from mstar.engine.resources.position.manager import LongRopeManager, RopeManager  # noqa: E402
from mstar.engine.resources.recurrent.config import (  # noqa: E402
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import SINK_SLOT, RecurrentStatePool  # noqa: E402
from mstar.engine.resources.resident import TRASH_SLOT, ResidentPlanner, build_planner  # noqa: E402
from mstar.engine.resources.spec import resolve_spec_dependencies  # noqa: E402
from mstar.engine.resources.step import (  # noqa: E402
    AllocationFailed,
    BucketKey,
    Segment,
    SlotLease,
    StepContext,
    SubmoduleStep,
)
from mstar.model.submodule_base import BatchedModelOutput
from mstar.testing_utils.resource import StubResource
from mstar.testing_utils.worker import make_worker
from mstar.utils import knobs  # noqa: E402
from mstar.utils.ipc_format import OffloadDelta  # noqa: E402
from mstar.worker.rebuild import Rebuilds  # noqa: E402
from mstar.worker.scheduled_batch import ScheduledBatch  # noqa: E402
from mstar.worker.step_repeats import RepeatConfig, StepRepeats  # noqa: E402

PAGE = 4
WALK = "decode"
CPU = torch.device("cpu")
MAX_SEQ_LEN = 512
# a request whose prompt reaches past 16 positions rotates with the long half
LONGROPE = LongRoPE((1.0,) * 4, (2.0,) * 4, original_max_position_embeddings=16, max_position_embeddings=1024)

# Every test here also runs on LongRoPE (`_all_on_longrope`), where each ingest of a
# handle flips its table, so a recycled or displaced row changes halves.
_ON_LONGROPE = {"on": False}
_INGESTS: dict[tuple[int, int], int] = {}


@pytest.fixture(autouse=True, params=[False, True], ids=["rope", "longrope"])
def _all_on_longrope(request):
    _ON_LONGROPE["on"] = request.param
    _INGESTS.clear()
    yield
    _ON_LONGROPE["on"] = False


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


def _resources(
    cpu_offload_pages: int = 0, max_num_pages: int = 64, pool_slots: int = 9, longrope: bool | None = None,
):
    if longrope is None:
        longrope = _ON_LONGROPE["on"]
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=MAX_SEQ_LEN,
            max_num_pages=max_num_pages, page_size=PAGE, cpu_offload_pages=cpu_offload_pages,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )
    kv._cg_max_seq_len = 64
    pool = RecurrentStatePool(device=CPU, config=RecurrentStateConfig(
        num_layers=1, max_slots=pool_slots, blocks={"state": RecurrentBlockConfig(shape=(2,), dtype=torch.float32)},
    ))
    if longrope:
        rope = LongRopeManager(
            PositionConfig(kv_cache="kv", longrope=LONGROPE), CPU, max_seq_len=MAX_SEQ_LEN, head_dim=8,
        )
    else:
        rope = RopeManager(PositionConfig(kv_cache="kv"), CPU, max_seq_len=MAX_SEQ_LEN, head_dim=8)
    # page-locked memory needs a GPU, and nothing here copies out of it asynchronously
    rope._pinned_pos_ids_buffer = lambda key, n: torch.empty(n, dtype=torch.long)
    return {"kv": kv, "pool": pool, "rope": rope}


def _ingest(res: dict[str, Resource], rid: int, max_position: int | None = None) -> None:
    if max_position is None and isinstance(res["rope"], LongRopeManager):
        # a handle's tables alternate from one ingest to the next, from its id on
        n = _INGESTS.get((id(res["rope"]), rid), 0)
        _INGESTS[(id(res["rope"]), rid)] = n + 1
        max_position = 20 if (rid + n) % 2 else 3
    res["kv"].ingest_request(rid, KVReqConfig())
    res["pool"].ingest_request(rid, None)
    res["rope"].ingest_request(rid, None if max_position is None else PositionReqConfig(max_position=max_position))


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


def _captured_step(res, planner: ResidentPlanner | None, rids, slot, bs, preplan=False, run=True, repeats=1):
    """One decode step on a bucket of ``bs`` rows. With a planner, the replay of a
    resident capture, ``repeats`` times; without one, the host plan of the same step."""
    padded = tuple(rids) + tuple(-(i + 1) for i in range(bs - len(rids)))
    bucket = BucketKey(graph_walk=WALK, bs=bs, num_tokens=bs)
    lease = SlotLease(slot=slot, bucket=bucket)
    ctx = StepContext(
        request_ids=tuple(rids), graph_walk=WALK, slot=slot, capture=False, slot_lease=lease, repeats=repeats,
    )
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
    state = res["kv"]._static_plan_state(lease.slot, ("main", 0), bucket.num_tokens)
    ck = CGSlotKey(bucket=bucket, slot=lease.slot, label="main")
    s.replays = []
    rope = res["rope"]
    for _ in range(ctx.repeats):
        if planner is not None:
            planner.gather(s.captured)
        s.pages = state.token_to_page[: bucket.bs].tolist()
        s.offsets = state.token_to_cache[: bucket.bs].tolist()
        if isinstance(rope, LongRopeManager):
            # its positions and table ids, the two rows of one buffer
            both = rope._static_pos_ids[CGSlotKey(bucket=bucket, slot=lease.slot, label="main#longrope")]
            s.positions, s.table_ids = both.view(2, bucket.num_tokens)[:, : bucket.bs].tolist()
        else:
            s.positions = rope._static_pos_ids[ck][: bucket.bs].tolist()
        s.pool_slots = res["pool"]._cg_addressing[ck].slot_indices[: bucket.bs].tolist()
        s.replays.append(SimpleNamespace(
            pages=s.pages, offsets=s.offsets, positions=s.positions, pool_slots=s.pool_slots,
            table_ids=getattr(s, "table_ids", None),
        ))
        if planner is not None:
            planner.commit_in_graph(s.captured)
    res["kv"].commit(s.kv, ctx)
    res["pool"].commit(s.pool, ctx)
    res["rope"].commit(s.rope, ctx)
    if planner is not None:
        planner.commit(s.step, ctx)
    return s


def _setup(
    prompts: dict[int, int], cpu_offload_pages: int = 0, max_num_pages: int = 64, pool_slots: int = 9,
    longrope: bool | None = None,
):
    """The same requests prefilled on resident resources and on a host-planned
    reference. With ``longrope``, each request's prompt length picks its table;
    left None, the module's LongRoPE pass alternates each handle's table."""
    live = _resources(cpu_offload_pages, max_num_pages, pool_slots, longrope)
    ref = _resources(cpu_offload_pages, max_num_pages, pool_slots, longrope)
    planner = ResidentPlanner("n", {WALK}, {k: live[k] for k in ("kv", "pool", "rope")}, CPU, max_bs=4)
    for res in (live, ref):
        for rid, n in prompts.items():
            _ingest(res, rid, n if longrope else None)
            _host_step(res, [rid], [n])
    return live, ref, planner


def _same(a, b, real: int):
    assert a.pages[:real] == b.pages[:real]
    assert a.offsets[:real] == b.offsets[:real]
    assert a.positions[:real] == b.positions[:real]
    assert a.pool_slots[:real] == b.pool_slots[:real]
    if hasattr(b, "table_ids"):
        assert a.table_ids[:real] == b.table_ids[:real]


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


@pytest.mark.parametrize("repeats", [2, 3, PAGE], ids=lambda n: f"{n}_replays")
def test_a_repeated_step_addresses_each_replay_as_host_plans_would(repeats):
    """Every replay of a step that repeats writes and attends where a host plan
    of that iteration would, page crossings included, with no host work between
    replays; the host's mirrors then stand where the stream does."""
    # 17 is past LongRoPE's original length, so request 2 starts on the long table
    live, ref, planner = _setup({1: 3, 2: 17, 3: 8}, longrope=True)
    rids = [1, 2, 3]
    for t in range(6):
        a = _captured_step(live, planner, rids, slot=t % 2, bs=4, repeats=repeats)
        for i, replay in enumerate(a.replays):
            b = _captured_step(ref, None, rids, slot=(t + i) % 2, bs=4)
            assert (replay.offsets[:3], replay.positions[:3]) == (b.offsets[:3], b.positions[:3])
            if replay.table_ids is not None:  # each replay's table row advances with its position
                assert replay.table_ids[:3] == b.table_ids[:3]
            # a run reserves ahead, so its page numbers aren't the host's: each is the row's own
            for row, rid in enumerate(rids):
                page = live["kv"]._streams[rid]["main"].page_indices[replay.positions[row] // PAGE]
                assert replay.pages[row] == page
    tables = live["kv"]._resident
    for rid in rids:
        _assert_row(planner, live, rid)
        assert tables.host_len[planner.slot_of(rid)] == live["kv"]._streams[rid]["main"].stored_len


def test_a_repeat_feeds_each_replay_the_last_one_s_token_and_keeps_a_column_per_replay():
    """A replay overwrites its stop buffers and reads its static input, so a step
    that repeats hands each replay the token before it and keeps every one, on
    the host while the next step on its slot (here its only one) replays."""
    tokens = torch.tensor([5, 7])
    ids = torch.zeros(2, 1, dtype=torch.long)

    class Graph:
        def replay(self):
            tokens.copy_(ids.reshape(-1) + 1)

    outputs = BatchedModelOutput(check_stop_buffers={"new_token": tokens, "text_inputs": tokens})
    slot = SimpleNamespace(run=None, graph=Graph(), static_inputs={"input_ids": ids}, static_outputs=outputs)
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner.slot_for = lambda lease: slot
    runner._device, runner._repeat_carries = torch.device("cpu"), {}
    runner._buckets = {"bucket": SimpleNamespace(config=SimpleNamespace(capture_graph_walk=WALK), slots=[slot])}
    runner.accept_repeats(WALK, {"new_token": "input_ids"}, 3)
    lease = SimpleNamespace(bucket=SimpleNamespace(graph_walk=WALK))
    wide = runner.repeat(lease, outputs, 3).check_stop_buffers
    assert wide["new_token"].tolist() == [[5, 6, 7], [7, 8, 9]]
    assert wide["text_inputs"] is wide["new_token"]
    tokens.copy_(torch.tensor([1, 2]))
    assert runner.repeat(lease, outputs, 3).check_stop_buffers["new_token"].tolist() == [[1, 2, 3], [2, 3, 4]]
    assert wide["new_token"].tolist() == [[5, 6, 7], [7, 8, 9]], "the next step landed its own columns"

    # a walk that never accepted repeats can't run several: every resource would commit them
    slot.run, runner._repeat_carries = None, {}
    assert runner.repeat(lease, outputs, 1) is outputs
    with pytest.raises(ValueError, match="never accepted repeats"):
        runner.repeat(lease, outputs, 3)


@pytest.mark.parametrize(("in_chain", "fresh", "same_node", "room", "want"), [
    (1, 0, True, 100, 8),
    (0, 0, True, 100, 1),  # other ready work hasn't had its look in
    (1, 1, True, 100, 1),  # a fresh row needs a plan of its own
    (1, 0, False, 100, 1),
    (1, 0, True, 3, 3),  # a row's cap ends the run, never passes it
], ids=["steady", "chain_start", "fresh_row", "other_node", "near_cap"])
def test_a_step_repeats_only_when_its_rows_all_continue(in_chain, fresh, same_node, room, want):
    rows = dict.fromkeys(range(2 + fresh), 0)
    speculation = SimpleNamespace(
        scheduled_batch=SimpleNamespace(
            node_name="n", graph_walk=WALK, request_to_worker_graph=rows, chunk_ranges={},
        ),
        continuing_rids={0, 1}, is_new_iter=True, is_same_node=same_node,
    )
    repeats = StepRepeats({"n": RepeatConfig(max=8)}, {("n", WALK): None})
    repeats._walks.add(("n", WALK))
    assert repeats.ceiling(speculation, in_chain, lambda: room) == want


@pytest.mark.parametrize("repeats", [1, 3], ids=lambda n: f"{n}_replays")
def test_a_row_a_host_step_moved_does_not_rewind_on_each_replay(repeats):
    """A one-token host step leaves a row's length to be refreshed. A run after
    it refreshes it once, not at the head of every replay; a one-iteration step
    writes it as a cell on the graph's scatter, with nothing to flush."""
    live, ref, planner = _setup({1: 5})
    _captured_step(live, planner, [1], slot=0, bs=1)
    _captured_step(ref, None, [1], slot=0, bs=1)
    step, ctx = _host_step(live, [1], [1], walk=WALK)
    planner.commit(step, ctx)
    _host_step(ref, [1], [1], walk=WALK)
    a = _captured_step(live, planner, [1], slot=1, bs=1, repeats=repeats)
    for i, replay in enumerate(a.replays):
        b = _captured_step(ref, None, [1], slot=(1 + i) % 2, bs=1)
        assert (replay.offsets[:1], replay.positions[:1]) == (b.offsets[:1], b.positions[:1])
    if repeats == 1:
        assert not a.ctx.resident_plan.writes.rows, "a one-iteration step flushed a run the scatter writes as a cell"


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


def test_longrope_rows_read_their_own_half_of_the_table():
    """Short and long requests side by side, each row's table id on its own half:
    through page crossings, a row joining, and rows trading places, so a slot's
    neighbour reads the other half."""
    prompts = {1: 3, 2: 17, 3: 8, 4: 20}
    long = {rid for rid, n in prompts.items() if LONGROPE.is_long(n)}
    assert long == {2, 4}
    live, ref, planner = _setup({rid: prompts[rid] for rid in (1, 2, 3)}, longrope=True)
    batches = [[1, 2, 3]] * 6 + [[2, 1, 4, 3]] * 6 + [[4, 3, 2, 1]] * 6
    for t, rids in enumerate(batches):
        if t == 6:
            for res in (live, ref):
                _ingest(res, 4, prompts[4])
                _host_step(res, [4], [prompts[4]])
        a = _captured_step(live, planner, rids, slot=t % 2, bs=4)
        b = _captured_step(ref, None, rids, slot=t % 2, bs=4)
        _same(a, b, len(rids))
        half = [MAX_SEQ_LEN if rid in long else 0 for rid in rids]
        assert a.table_ids[: len(rids)] == [p + h for p, h in zip(a.positions[: len(rids)], half, strict=True)]
        # a padding row reads the trash row, in range of the table
        assert all(0 <= i < 2 * MAX_SEQ_LEN for i in a.table_ids)


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
    """A stream on other pages, as a fork or reset can leave it. Its hook is `_touch_plan_cache(rid)`."""
    live, ref, planner = _setup({1: 9})
    del ref
    _captured_step(live, planner, [1], slot=0, bs=1)
    kv = live["kv"]
    stream = kv._streams[1]["main"]
    moved = kv._arena.acquire(len(stream.page_indices))
    kv._arena.release(stream.page_indices)
    stream.page_indices.assign(moved)
    kv._touch_plan_cache(1)
    assert 1 in planner._stale
    s = _captured_step(live, planner, [1], slot=1, bs=1)
    assert s.pages[0] == moved[-1]
    _assert_row(planner, live, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the host page pool is page-locked")
def test_a_real_offload_gives_the_slot_up_and_its_reload_is_written_whole():
    device = torch.device("cuda")
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=64, max_num_pages=16, page_size=PAGE,
            cpu_offload_pages=16,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=device, dtype=torch.float32,
    )
    kv._cg_max_seq_len = 64
    planner = ResidentPlanner("n", {WALK}, {"kv": kv}, device, max_bs=2)
    engine = _offload_engine({"kv": kv}, planner)
    kv.ingest_request(1, KVReqConfig())
    segments = (Segment(1, "main", 9),)
    ctx = StepContext(request_ids=(1,), graph_walk="prefill", slot=0, capture=False)
    step = KVStep(segments=segments)
    assert kv.admit(step, ctx).ok
    ctx.plan_results["kv"] = kv.plan(step, ctx)
    kv.commit(step, ctx)
    with planner._lock:
        planner._slot_of[1] = planner._free.pop()
    assert engine.offload_request("n", 1) > 0
    assert planner.slot_of(1) is None
    assert not planner._stale
    torch.cuda.synchronize()
    assert engine.reload_request("n", 1)
    assert not planner._stale
    assert planner.slot_of(1) is None


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


def test_a_refused_admit_drops_the_stage_naming_its_pages():
    """The unwind gives back the page a pre-plan staged a crossing into. With no
    commit in between, a retry over the same rows reserves another page, and it
    must not promote the stage's cell naming the old one."""
    live, ref, planner = _setup({1: 3, 2: 4})
    for res, p in ((live, planner), (ref, None)):
        _captured_step(res, p, [1], slot=0, bs=1)
        # row 1 holds 4 tokens: the next one opens a page, which the pre-plan stages
        s = _captured_step(res, p, [1], slot=1, bs=1, preplan=True, run=False)
        # refused, as `StepRunner.admit` unwinds it
        res["pool"].rollback_admit(s.pool, s.ctx)
        res["kv"].rollback_admit(s.kv, s.ctx)
        for key in ("kv", "pool", "rope"):
            res[key].clear_preplan()
        # a teardown frees pages before the retry, so its admit draws another
        for key in ("kv", "pool", "rope"):
            res[key].remove_request(2)
    a = _captured_step(live, planner, [1], slot=1, bs=1)
    b = _captured_step(ref, None, [1], slot=1, bs=1)
    _same(a, b, 1)
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


def test_an_abandoned_stage_of_a_steady_batch_is_staged_again(monkeypatch):
    """Its slot row went out on the plan stream, and the step replacing it waits on no
    event for that, so an equal row is no reason to skip the copy on its own stream."""
    live, ref, planner = _setup({1: 3, 2: 5})
    del ref
    _captured_step(live, planner, [1, 2], slot=0, bs=2)
    _captured_step(live, planner, [1, 2], slot=0, bs=2, preplan=True, run=False)
    for key in ("kv", "pool", "rope"):
        live[key].clear_preplan()
    buffers = planner.buffers(BucketKey(WALK, 2, 2), 0)
    copies = []
    real = planner._stager.copy_
    monkeypatch.setattr(planner._stager, "copy_", lambda dst, *a, **k: (copies.append(dst), real(dst, *a, **k))[1])
    s = _captured_step(live, planner, [1, 2], slot=0, bs=2)
    assert any(dst is buffers.slots for dst in copies)
    _assert_step_read_its_streams(live, s, [1, 2])
    # and a steady step after it still copies nothing
    copies.clear()
    _captured_step(live, planner, [1, 2], slot=0, bs=2)
    assert not any(dst is buffers.slots for dst in copies)


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
            planner=SimpleNamespace(store=store), page_size=4, key="kv",
            block_table=SimpleNamespace(stride=7, col=3), kv_len=SimpleNamespace(stride=7, col=1),
        )
        store[slots * 7 + 1] = torch.tensor([9, 5, 0, 0], device=device)
        pages = torch.zeros(4, dtype=torch.long, device=device)
        offsets = torch.zeros(4, dtype=torch.long, device=device)
        step = SimpleNamespace(
            slots=slots, bucket=SimpleNamespace(bs=4), buffers=SimpleNamespace(
                cells=torch.zeros((2, 8), dtype=torch.long, device=device), kv_len={"kv": out},
            ), streams={"kv": resident_mod.ResidentStream("main", reads=False)}, block_tables={},
        )
        KVResidentTables.gather(tables, step, pages, offsets)
        return [t.tolist() for t in (store, out, small, pages, offsets)]

    assert run(torch.device("cuda")) == run(CPU)



def test_the_host_lengths_of_a_graph_indexed_step_match_the_host_plan():
    """What the KV plan hands an attention that reads its pages off the tables:
    the same indptr, lengths and last page lengths as the full host plan of the
    step, padding rows included."""
    live, ref, planner = _setup({1: 3, 2: 5, 3: 8})
    rids = [1, 2, 3]
    for t in range(12):
        a = _captured_step(live, planner, rids, slot=t % 2, bs=4, run=False)
        _promote(live, planner, a)
        got = live["kv"]._resident.decode_indptrs(a.ctx.resident_plan, 4)
        b = _captured_step(ref, None, rids, slot=t % 2, bs=4, run=False)
        want = b.ctx.plan_results["kv"]["main"].cpu_indptrs
        for field in ("qo_indptr", "paged_kv_indptr", "paged_kv_last_page_len", "kv_lens"):
            assert getattr(got, field).tolist() == getattr(want, field).tolist(), (t, field)
        _replay(live, planner, a, promote=False)
        _replay(ref, None, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel runs on CUDA")
@pytest.mark.parametrize("appends", [True, False])
def test_the_graph_built_page_arrays_match_the_host_lengths_and_the_block_table(appends):
    from mstar.engine.resources.attn.wrappers import FlashInferDecodeWrapper
    from mstar.engine.resources.kv.resident import KVResidentTables

    cuda = torch.device("cuda")
    stride, block_col, page = 40, 2, 4
    store = torch.zeros(6 * stride, dtype=torch.long, device=cuda)
    lens = {1: 9, 3: 0, 4: 4, 5: 130}
    for slot, n in lens.items():
        store[slot * stride + 1] = n
        cols = n // page + 1
        store[slot * stride + block_col: slot * stride + block_col + cols] = torch.arange(
            100 * slot, 100 * slot + cols,
        )
    order = [5, 1, 3, 4]
    bs = 6  # two padding rows, at the trash row
    slots = torch.tensor(order + [TRASH_SLOT] * (bs - len(order)), device=cuda)
    kv_len = store[slots * stride + 1].clone()
    capacity = 64
    attn = SimpleNamespace(
        _paged_kv_indptr_buf=torch.full((bs + 1,), -1, dtype=torch.int32, device=cuda),
        _paged_kv_indices_buf=torch.full((capacity,), -1, dtype=torch.int32, device=cuda),
        _paged_kv_last_page_len_buf=torch.full((bs,), -1, dtype=torch.int32, device=cuda),
    )
    FlashInferDecodeWrapper.gather_resident(
        SimpleNamespace(attn_wrapper=attn, page_size=page), store, slots, kv_len, stride, block_col, appends=appends,
    )

    host_len = torch.zeros(6, dtype=torch.long).numpy()
    for slot, n in lens.items():
        host_len[slot] = n
    tables = SimpleNamespace(host_len=host_len, page_size=page, key="kv", _qo_indptr={})
    streams = {"kv": SimpleNamespace(reads=not appends)}
    plan = SimpleNamespace(slots=torch.tensor(order).numpy(), writes=SimpleNamespace(streams=streams), repeats=1)
    want = KVResidentTables.decode_indptrs(tables, plan, bs)
    assert attn._paged_kv_indptr_buf.tolist() == want.paged_kv_indptr.tolist()
    assert attn._paged_kv_last_page_len_buf.tolist() == want.paged_kv_last_page_len.tolist()
    read = {s: max(lens[s] + appends, 1) for s in order}
    pages = [list(range(100 * s, 100 * s + (read[s] - 1) // page + 1)) for s in order]
    pages += [[SINK_PAGE]] * (bs - len(order))
    end = int(want.paged_kv_indptr[-1])
    assert attn._paged_kv_indices_buf[:end].tolist() == [p for row in pages for p in row]

@pytest.mark.parametrize("reads", [False, True], ids=["appends", "reads"])
def test_a_step_that_repeats_plans_its_schedule_for_its_last_replay(reads):
    """The graph builds each replay's page arrays, but the host plans the
    schedule once: for the longest rows, which a shorter replay's lengths fit."""
    from mstar.engine.resources.kv.resident import KVResidentTables

    tables = SimpleNamespace(host_len=torch.tensor([0, 15, 3]).numpy(), page_size=16, key="kv", _qo_indptr={})
    streams = {"kv": SimpleNamespace(reads=reads)}
    plan = SimpleNamespace(slots=torch.tensor([1, 2]).numpy(), writes=SimpleNamespace(streams=streams), repeats=4)

    got = KVResidentTables.decode_indptrs(tables, plan, 3)

    # the padding row reads its one sink token
    assert got.kv_lens.tolist() == ([15, 3, 1] if reads else [19, 7, 1])
    assert got.paged_kv_indptr.tolist() == ([0, 1, 2, 3] if reads else [0, 2, 3, 4])


# ---- slots ----

def test_a_freed_slot_waits_for_the_last_step_that_read_it():
    live, ref, planner = _setup({1: 3})
    del ref
    # step N reads the slot and is still running: planned, not committed
    s = _captured_step(live, planner, [1], slot=0, bs=1, preplan=True, run=False)
    _promote(live, planner, s)
    slot = int(s.ctx.resident_plan.slots[0])
    seq = s.ctx.resident_plan.seq
    planner.release(1)
    assert (slot, seq) in planner._cooling
    with planner._lock:
        assert slot not in planner._take_slots(7, seq + 1)
    # commit lets the slot go; its late advance landed on the old owner's row
    _replay(live, planner, s, promote=False)
    with planner._lock:
        assert planner._take_slots(1, seq + 2) == [slot]


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


def _slot_ledger(planner: ResidentPlanner) -> None:
    """Every request slot is exactly one of free, cooling or held."""
    held = list(planner._slot_of.values())
    cooling = [slot for slot, _ in planner._cooling]
    assert sorted(held + cooling + planner._free) == list(range(1, planner.max_slots))


def _assert_step_read_its_streams(live, s, rids) -> None:
    """What the replay addressed against the host, which has committed the step since."""
    for i, rid in enumerate(rids):
        stream = live["kv"]._streams[rid]["main"]
        wrote = stream.stored_len - 1
        assert s.pages[i] == stream.page_indices[wrote // PAGE]
        assert s.offsets[i] == wrote % PAGE
        assert s.positions[i] == live["rope"].position(rid, "main") - 1
        assert s.pool_slots[i] == live["pool"]._slots[rid]["main"].index
        rope = live["rope"]
        if isinstance(rope, LongRopeManager):
            assert s.table_ids[i] == s.positions[i] + (MAX_SEQ_LEN if rid in rope._long_rids else 0)


def _assert_held_rows(planner, live) -> None:
    """A row marked neither stale nor moved holds what its stream says."""
    for rid in planner._slot_of:
        if rid not in planner._stale and rid not in planner._moved:
            _assert_row(planner, live, rid)


def _recycle(live, planner, rid: int, tokens: int) -> None:
    """The request ends, and its handle comes back as another's."""
    for key in ("kv", "pool", "rope"):
        live[key].remove_request(rid)
    planner.release(rid)
    _ingest(live, rid)
    step, ctx = _host_step(live, [rid], [tokens])
    planner.commit(step, ctx)


@pytest.mark.parametrize("seed", range(4))
def test_more_live_requests_than_slots_run_off_the_cache(seed):
    """B=4, so 8 slots and the trash row, under 20 live requests taking turns, with
    pre-plans, abandoned ones, host steps and teardowns mixed in. Nothing raises,
    a displaced row is written whole when it next runs, the victims are the least
    recently run, and every row a step reads is its stream."""
    rng = random.Random(seed)
    prompts = {rid: 1 + rid % 7 for rid in range(1, 21)}
    live, ref, planner = _setup(prompts, max_num_pages=1024, pool_slots=32)
    del ref
    assert planner.max_slots == 9
    order, at, displaced, returned = list(prompts), 0, set(), 0
    for t in range(300):
        rows = [order[(at + i) % len(order)] for i in range(rng.randint(1, 4))]
        at += len(rows)
        roll = rng.random()
        if roll < 0.1:
            recycled = rng.choice(order)
            _recycle(live, planner, recycled, rng.randint(1, 9))
            displaced.discard(recycled)
        elif roll < 0.2:
            step, ctx = _host_step(live, rows, [1] * len(rows), walk="mixed")
            planner.commit(step, ctx)
        elif roll < 0.3:
            _captured_step(live, planner, rows, slot=t % 2, bs=4, preplan=True, run=False)
            for key in ("kv", "pool", "rope"):
                live[key].clear_preplan()
        before = {rid: int(planner._last_read[slot]) for rid, slot in planner._slot_of.items()}
        s = _captured_step(live, planner, rows, slot=t % 2, bs=4, preplan=rng.random() < 0.5)
        _assert_step_read_its_streams(live, s, rows)
        for rid in rows:
            _assert_row(planner, live, rid)
            assert planner.slot_of(rid) != TRASH_SLOT
            if rid in displaced:
                assert rid in s.ctx.resident_plan.taken
                returned += 1
                displaced.discard(rid)
        gone = before.keys() - planner._slot_of.keys()
        displaced |= gone
        kept = before.keys() - gone - set(rows)
        if gone and kept:
            assert max(before[rid] for rid in gone) <= min(before[rid] for rid in kept)
        _slot_ledger(planner)
        _assert_held_rows(planner, live)
        live["kv"].assert_pages_conserved()
    assert returned > 100


def test_a_displaced_row_is_written_whole_with_its_lengths():
    live, ref, planner = _setup({rid: 5 + rid for rid in range(1, 12)}, max_num_pages=256, pool_slots=16)
    del ref
    # eight rows fill the slots, the oldest step's first
    _captured_step(live, planner, [1, 2, 3, 4], slot=0, bs=4)
    _captured_step(live, planner, [5, 6, 7, 8], slot=1, bs=4)
    assert not planner._free
    victim_slot = planner.slot_of(1)
    s = _captured_step(live, planner, [9], slot=0, bs=4)
    assert planner.slot_of(1) is None
    assert planner.slot_of(9) == victim_slot
    assert s.ctx.resident_plan.taken == {9}
    tables = live["kv"]._resident
    assert tables.host_len[victim_slot] == live["kv"]._streams[9]["main"].stored_len
    _assert_row(planner, live, 9)
    # the next victims are the rest of the oldest step, in the order they took slots
    _captured_step(live, planner, [10], slot=1, bs=4)
    assert planner.slot_of(2) is None and planner.slot_of(3) is not None
    # and the first comes back through the new-request path
    s = _captured_step(live, planner, [1], slot=0, bs=4)
    assert s.ctx.resident_plan.taken == {1}
    _assert_step_read_its_streams(live, s, [1])
    _assert_row(planner, live, 1)
    assert tables.host_len[planner.slot_of(1)] == live["kv"]._streams[1]["main"].stored_len


def test_a_step_does_not_displace_its_own_rows_or_the_step_behind_it():
    live, ref, planner = _setup({rid: 3 for rid in range(1, 16)}, max_num_pages=256, pool_slots=20)
    del ref
    _captured_step(live, planner, [1, 2, 3, 4], slot=0, bs=4)
    _captured_step(live, planner, [5, 6, 7, 8], slot=1, bs=4)
    # planned behind a step that hasn't committed: its rows are out of reach
    behind = _captured_step(live, planner, [5, 6, 7, 8], slot=1, bs=4, preplan=True, run=False)
    _promote(live, planner, behind)
    s = _captured_step(live, planner, [9, 10, 11, 12], slot=0, bs=4, run=False)
    planner.plan(s.step, s.ctx)
    assert set(planner._slot_of) == {5, 6, 7, 8, 9, 10, 11, 12}
    # the late commit advances the rows it read, none of them taken
    _replay(live, planner, behind, promote=False)
    _assert_row(planner, live, 5)


def test_a_step_keeps_its_own_oldest_row_and_a_victim_leaves_no_marks():
    live, ref, planner = _setup({rid: 3 for rid in range(1, 12)}, max_num_pages=256, pool_slots=16)
    del ref
    _captured_step(live, planner, [1, 2, 3, 4], slot=0, bs=4)
    _captured_step(live, planner, [5, 6, 7, 8], slot=1, bs=4)
    planner.mark_stale(2)
    planner._moved.add(3)
    # 1 is the oldest and in the step, so the victims come after it
    s = _captured_step(live, planner, [1, 9, 10], slot=0, bs=4)
    assert planner.slot_of(1) is not None
    assert planner.slot_of(2) is None and planner.slot_of(3) is None and planner.slot_of(4) is not None
    assert not planner._stale and not planner._moved
    _slot_ledger(planner)
    _assert_step_read_its_streams(live, s, [1, 9, 10])


def test_plans_that_never_ran_hold_no_slots():
    """Refused retries and abandoned pre-plans, one after another with no commit
    between: however many there are, only the newest can look uncommitted."""
    live, ref, planner = _setup({rid: 3 for rid in range(1, 26)}, max_num_pages=256, pool_slots=32)
    del ref
    rid = 1
    for t in range(6):
        rows = list(range(rid, rid + 4))
        rid += 4
        s = _captured_step(live, planner, rows, slot=t % 2, bs=4, preplan=True, run=False)
        assert set(s.ctx.resident_plan.rids) == set(rows)
        for key in ("kv", "pool", "rope"):
            live[key].clear_preplan()
        _slot_ledger(planner)


def test_no_slot_is_an_invariant_violation_not_a_knob():
    live, ref, planner = _setup({1: 3})
    del ref
    # every slot read by the step planned just now, which no step can do
    with planner._lock:
        planner._slot_of.update({100 + i: planner._free.pop() for i in range(8)})
        planner._last_read[1:] = 5
        planner._plan_seq = 5
        with pytest.raises(RuntimeError, match="accounting is wrong"):
            planner._take_slots(1, 6)


def test_a_rank_whose_release_lands_on_the_other_side_of_a_commit_still_runs():
    """Ranks replay teardowns between the same steps, but a release can land
    before or after a commit on the GPU thread, so the same slot is cooling on one
    rank and free on the other. Neither may raise, and each rank's rows are its own."""
    ranks = []
    for release_before_commit in (False, True):
        live, ref, planner = _setup({rid: 2 + rid for rid in range(1, 21)}, max_num_pages=1024, pool_slots=32)
        del ref
        ranks.append((live, planner, release_before_commit))
    next_rid = 21
    for t in range(60):
        rows = [1 + (4 * t + i) % 20 for i in range(4)]
        gone = rows[0]
        for live, planner, early in ranks:
            s = _captured_step(live, planner, rows, slot=t % 2, bs=4, preplan=True, run=False)
            _promote(live, planner, s)
            if early:
                planner.release(gone)
            _replay(live, planner, s, promote=False)
            _assert_step_read_its_streams(live, s, rows)
            if not early:
                planner.release(gone)
            for key in ("kv", "pool", "rope"):
                live[key].remove_request(gone)
            _ingest(live, gone)
            step, ctx = _host_step(live, [gone], [next_rid % 5 + 1])
            planner.commit(step, ctx)
            _slot_ledger(planner)
        next_rid += 1


def test_ranks_with_the_same_history_displace_the_same_requests():
    runs = []
    for shuffle in (0, 1):
        live, ref, planner = _setup({rid: 3 for rid in range(1, 21)}, max_num_pages=1024, pool_slots=32)
        del ref
        # handles and slot numbers are per rank, so a tie must not depend on either
        if shuffle:
            planner._free.reverse()
        seen = []
        for t in range(40):
            rows = [1 + (3 * t + i) % 20 for i in range(4)]
            _captured_step(live, planner, rows, slot=t % 2, bs=4)
            seen.append(frozenset(planner._slot_of))
        runs.append(seen)
    assert runs[0] == runs[1]


def test_the_store_is_sized_by_the_widest_capture():
    res = _resources()
    planner = ResidentPlanner("n", {WALK}, {k: res[k] for k in ("kv", "pool", "rope")}, CPU, max_bs=8)
    assert planner.max_slots == 17
    assert len(planner._free) == 16
    assert TRASH_SLOT not in planner._free


def test_a_flush_of_the_largest_step_stages_without_growing(monkeypatch):
    res = _resources(max_num_pages=64)
    planner = ResidentPlanner("n", {WALK}, {k: res[k] for k in ("kv", "pool", "rope")}, CPU, max_bs=4)
    stride = planner._tables[0].stride
    assert planner._stager._seg_numel >= 4 * stride + 4 * 4
    monkeypatch.setattr(planner._stager, "_grow", lambda numel: pytest.fail(f"grew to {numel}"))
    rows = [(slot, 0, list(range(stride))) for slot in range(1, 5)]
    plan = SimpleNamespace(writes=SimpleNamespace(rows=rows))
    planner.flush(plan)
    assert planner.store.view(9, stride)[1:5].tolist() == [list(range(stride))] * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the stager pins and copies on a GPU")
def test_a_flush_of_the_largest_step_stages_on_the_device_without_growing(monkeypatch):
    device = torch.device("cuda")
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=512, max_num_pages=64, page_size=PAGE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=device, dtype=torch.float32,
    )
    planner = ResidentPlanner("n", {WALK}, {"kv": kv}, device, max_bs=4)
    stride = planner._tables[0].stride
    monkeypatch.setattr(planner._stager, "_grow", lambda numel: pytest.fail(f"grew to {numel}"))
    rows = [(slot, 0, list(range(100, 100 + stride))) for slot in range(1, 5)]
    planner.flush(SimpleNamespace(writes=SimpleNamespace(rows=rows)))
    planner._stager.copy_(planner.buffers(BucketKey(WALK, 4, 4), 0).cells, [0] * 16)
    torch.cuda.synchronize()
    assert planner.store.view(9, stride)[1:5].tolist() == [list(range(100, 100 + stride))] * 4


# ---- the block table's width ----

def _kv(max_seq_len: int, max_num_pages: int) -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=max_seq_len,
            max_num_pages=max_num_pages, page_size=PAGE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )


def test_a_row_is_as_wide_as_the_pool_when_the_context_is_wider():
    kv = _kv(max_seq_len=1 << 20, max_num_pages=64)
    ResidentPlanner("n", {WALK}, {"kv": kv}, CPU, max_bs=1)
    assert kv._resident.block_table.width == kv.config.request_pages == 63


def test_a_row_is_as_wide_as_the_context_when_the_pool_is_wider():
    kv = _kv(max_seq_len=4 * 10 + 1, max_num_pages=64)
    ResidentPlanner("n", {WALK}, {"kv": kv}, CPU, max_bs=1)
    assert kv._resident.block_table.width == 11


def test_a_stream_wider_than_a_row_raises():
    kv = _kv(max_seq_len=8, max_num_pages=64)
    planner = ResidentPlanner("n", {WALK}, {"kv": kv}, CPU, max_bs=1)
    tables = kv._resident
    assert tables.block_table.width == 2
    writes = resident_mod.ResidentWrites({"kv": resident_mod.ResidentStream("main", reads=False)})
    with pytest.raises(ValueError, match="fit a block table row"):
        tables.row(9, [1, 2, 3], 1, [0] * planner.narrow_cols, writes)
    plan = SimpleNamespace(
        repeats=1, slots=torch.tensor([1]).numpy(), rids=(7,), taken=set(), writes=writes, crossed={}, moved=set(),
    )
    tables.host_len[1], tables.host_cols[1] = 8, 2
    with pytest.raises(ValueError, match="fit a block table row"):
        tables.stage_step(plan, lambda rid: 8, lambda rid: [1, 2, 3], planner.rewrite)
    assert not plan.writes.cells


# ---- what a raise in the plan leaves ----

def _preplan_worker(engine):
    return make_worker(worker_id="w", engine_manager=SimpleNamespace(get_engine=lambda node: engine))


def _preplan_engine(res, planner):
    from mstar.engine.resources.runner import StepRunner
    from mstar.testing_utils.engine import make_engine, make_mgmt

    resources = {k: res[k] for k in ("kv", "pool", "rope")}
    runner = SimpleNamespace(
        plan_stream=lambda: None, pad_inputs=lambda lease, inputs, uniform=None: inputs,
        step_ids=lambda lease, rids: list(rids), release=lambda lease, rows: None,
    )

    def declare(graph_walk, request_ids, inputs, **kwargs):
        del graph_walk, inputs, kwargs
        segments = [Segment(rid, "main", 1) for rid in request_ids]
        keys = {"kv": KVStep(), "pool": RecurrentStep(), "rope": PositionStep()}
        return SubmoduleStep(segments=segments, steps=keys)

    engine = make_engine(_runner=StepRunner(resources), _submodules={"n": make_mgmt(
        SimpleNamespace(declare_step=declare, step_is_input_free=lambda walk: False),
        accelerator_graph_runner=runner,
        resident=planner,
    )})
    engine.can_pre_plan = lambda node: True
    engine.reserve_replay_slot = lambda batch: batch.step_context.slot_lease
    return engine


class _RecordedEvent:
    def __init__(self, device=None):
        self.recorded = False

    def record(self, stream=None):
        self.recorded = True


def _preplan_batch(planner, rids, bs):
    bucket = BucketKey(graph_walk=WALK, bs=bs, num_tokens=bs)
    lease = SlotLease(slot=0, bucket=bucket)
    ctx = StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False, slot_lease=lease)
    ctx.resident = planner
    return SimpleNamespace(
        node_name="n", inputs=[SimpleNamespace() for _ in rids], request_ids=list(rids), per_request_info={},
        step_context=ctx, preplan_event=None, preplanned_rids=None, step=None, uniform_inputs=None,
    )


@pytest.mark.parametrize("raiser", ["resident", "runner"])
def test_a_raise_in_the_preplan_gives_its_reservation_back(monkeypatch, raiser):
    """`pre_admit` reserved a crossing page and something raises before the stage
    is done. The plan thread's wrapper unwinds the pre-plan as it does one thrown
    away: the page goes back, so rank 0 holds no page a TP follower that never
    planned the step lacks, the in-flight mark goes, and exec's admit takes the page
    again. The same for a raise from the planner and from the runner's own pre-plan."""
    res = _resources()
    planner = ResidentPlanner("n", {WALK}, {k: res[k] for k in ("kv", "pool", "rope")}, CPU, max_bs=4)
    _ingest(res, 1)
    step, ctx = _host_step(res, [1], [3])
    planner.commit(step, ctx)
    s = _captured_step(res, planner, [1], slot=0, bs=1)
    _assert_step_read_its_streams(res, s, [1])
    kv, stream = res["kv"], res["kv"]._streams[1]["main"]
    assert stream.stored_len == 4 and len(stream.page_indices) == 1

    # a CPU torch.Event can't be made, and the pre-plan records one even when it raises
    monkeypatch.setattr(torch, "Event", _RecordedEvent)
    engine = _preplan_engine(res, planner)
    batch = _preplan_batch(planner, [1], 1)
    target = planner if raiser == "resident" else engine._runner
    name = "plan" if raiser == "resident" else "pre_plan"
    real = getattr(target, name)

    def boom(*args, **kwargs):
        raise RuntimeError("raised mid pre-plan")

    monkeypatch.setattr(target, name, boom)
    assert _preplan_worker(engine).executor._preplan_spec(None, SimpleNamespace(node_batch=batch)) is False
    monkeypatch.setattr(target, name, real)

    assert batch.preplan_event is None and batch.preplanned_rids is None and batch.step is None
    assert engine._pre_plan_done["n"].recorded
    assert engine._runner._pre_admitted is None and not engine._runner.staged
    assert not stream.step_in_flight
    assert len(stream.page_indices) == 1
    free = kv._arena.num_free
    kv.assert_pages_conserved()

    # exec re-drives it and takes the crossing page itself
    s = _captured_step(res, planner, [1], slot=1, bs=1)
    assert len(stream.page_indices) == 2 and kv._arena.num_free == free - 1
    assert s.pages[0] == stream.page_indices[1]
    _assert_step_read_its_streams(res, s, [1])
    _assert_row(planner, res, 1)
    kv.assert_pages_conserved()


def test_a_plan_that_raises_leaves_its_rows_to_the_next_step(monkeypatch):
    live, ref, planner = _setup({1: 3, 2: 5})
    del ref
    kv = live["kv"]
    real = kv.resident_row

    def boom(rid, slot, row, writes):
        if rid == 2:
            raise ValueError("a row that can't be written")
        real(rid, slot, row, writes)

    monkeypatch.setattr(kv, "resident_row", boom)
    s = _captured_step(live, planner, [1, 2], slot=0, bs=2, run=False)
    with pytest.raises(ValueError, match="can't be written"):
        planner.plan(s.step, s.ctx)
    assert planner._stale == {1, 2}
    monkeypatch.setattr(kv, "resident_row", real)
    _replay(live, planner, s)
    _assert_row(planner, live, 1)
    _assert_row(planner, live, 2)


# ---- what an offload does to a slot ----

class _HostPagesKV:
    """The offload hooks of a KV manager over a real one's streams and arena, for a
    host with no page-locked memory: a stream's pages go back to the pool and the
    reload takes others, which is all a row can see of the real move."""

    def __init__(self, kv: KVManager):
        self.kv = kv
        self.away: dict[int, int] = {}
        kv.__class__ = type("_Offloading", (KVManager,), {
            "offload": lambda _, rid: self.offload(rid),
            "reload": lambda _, rid: self.reload(rid),
            "is_offloaded": lambda _, rid: rid in self.away,
            "can_reload": lambda _, rid: True,
            "supports_eviction": property(lambda _: True),
            "reclaimable": lambda _, rid: len(kv._streams[rid]["main"].page_indices) if rid not in self.away else 0,
        })

    def offload(self, rid: int) -> int:
        stream = self.kv._streams[rid]["main"]
        if stream.step_in_flight:
            return 0
        pages = stream.page_indices.tolist()
        self.kv._touch_plan_cache(rid, row=False)
        stream.page_indices.clear()
        self.kv._arena.release(pages)
        self.away[rid] = len(pages)
        return len(pages)

    def reload(self, rid: int) -> bool:
        self.kv._touch_plan_cache(rid, row=False)
        stream = self.kv._streams[rid]["main"]
        stream.page_indices.assign(self.kv._arena.acquire(self.away.pop(rid)))
        return True


def _offload_engine(resources, planner):
    from mstar.testing_utils.engine import make_engine, make_mgmt

    engine = make_engine(
        _resources=resources,
        _resource_nodes={key: frozenset({"n"}) for key in resources},
        _submodules={"n": make_mgmt(resources=resources, resident=planner)},
        _graph_runtime=SimpleNamespace(get_rid_string=str, get_rid_handle=int),
        # these requests were taken in by their resources alone, so no admission queue holds them
        _runner=SimpleNamespace(readmit=lambda rid, node: None),
    )
    return engine


def _refused_worker(engine, line):
    """The real handler and the real victim search over a real engine, with
    ``line`` the requests in the order a step reaches them."""
    pushed = []
    worker = make_worker(
        graph_runtime=SimpleNamespace(push_back_node=lambda node, rids, graphs: pushed.append(rids)),
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        scheduler=SimpleNamespace(
            could_refill=lambda *a: False,
            last_in_line=lambda node, walk, rids: max(rids, key=line.index),
        ),
    )
    for rid in line:
        worker.lifecycle.add(rid, str(rid))
    worker.pushed = pushed
    return worker


def _refused(batch, outcome_reason, failed_resource):
    return SimpleNamespace(
        node_name=batch.node_name, admit_error=outcome_reason, failed_resource=failed_resource,
        request_ids=list(batch.request_to_worker_graph), cached_rids=set(), no_forward_rids=set(),
    )


def test_a_refused_admit_offloads_a_victim_and_its_reload_is_written_whole():
    """Through the worker's refusal path: the KV pool refuses, `_handle_admit_failure`
    offloads the row furthest from a step, its slot goes with it, the retry runs, and the
    victim's reload is written whole at its own lengths."""
    from mstar.engine.resources.runner import StepRunner

    # seven pages to hand out, all held: each of 1 and 2 has room for one more token
    live = _resources(max_num_pages=8)
    owned = {k: live[k] for k in ("kv", "pool", "rope")}
    planner = ResidentPlanner("n", {WALK}, owned, CPU, max_bs=4)
    for rid, tokens in ((1, 7), (2, 7), (3, 11)):
        _ingest(live, rid)
        step, ctx = _host_step(live, [rid], [tokens])
        planner.commit(step, ctx)
    kv = live["kv"]
    _HostPagesKV(kv)
    _captured_step(live, planner, [1, 2, 3], slot=0, bs=3)
    assert planner.slot_of(3) is not None and kv._arena.num_free == 0
    engine = _offload_engine(owned, planner)

    # both cross a page now
    segments = tuple(Segment(rid, "main", 1) for rid in (1, 2))
    step = SubmoduleStep(
        segments=list(segments), steps={"kv": KVStep(), "pool": RecurrentStep(), "rope": PositionStep()},
    )
    step.set_ctx(StepContext(request_ids=(1, 2), graph_walk=WALK, slot=0, capture=False))
    outcome = StepRunner(owned).admit(step)
    assert not outcome.ok and isinstance(outcome.reason, AllocationFailed) and outcome.failed_resource == "kv"

    worker = _refused_worker(engine, line=[1, 2, 3])
    batch = ScheduledBatch(node_name="n", graph_walk=WALK, request_to_worker_graph={1: "g", 2: "g"})
    worker.pipeline._handle_admit_failure(batch, _refused(batch, outcome.reason, outcome.failed_resource))
    assert worker.pushed == [[1, 2]] and list(worker.admission.held_until) == [3]
    assert engine.residency("n", 3) is Residency.ON_HOST
    assert planner.slot_of(3) is None and 3 not in planner._stale

    # the retry runs on the victim's pages
    s = _captured_step(live, planner, [1, 2], slot=0, bs=2)
    _assert_step_read_its_streams(live, s, [1, 2])
    _assert_row(planner, live, 1)

    for key in ("kv", "pool", "rope"):
        live[key].remove_request(2)
    planner.release(2)
    assert engine.reload_request("n", 3)
    assert planner.slot_of(3) is None
    s = _captured_step(live, planner, [3], slot=0, bs=1)
    assert s.ctx.resident_plan.taken == {3}
    stream = kv._streams[3]["main"]
    assert stream.stored_len == 13
    assert kv._resident.host_len[planner.slot_of(3)] == stream.stored_len
    _assert_step_read_its_streams(live, s, [3])
    _assert_row(planner, live, 3)


def test_a_recurrent_pool_refusal_holds_the_request_since_the_pool_cannot_evict():
    """The pool refuses with `AllocationFailed`, which is backpressure: it has no
    eviction, so nothing is offloaded and the row it named is held for a smaller retry."""
    live = _resources(pool_slots=2)
    planner = ResidentPlanner("n", {WALK}, {k: live[k] for k in ("kv", "pool", "rope")}, CPU, max_bs=4)
    for rid in (1, 2):
        _ingest(live, rid)
    step, ctx = _host_step(live, [1], [3])
    planner.commit(step, ctx)
    _HostPagesKV(live["kv"])
    pool_step = RecurrentStep(segments=(Segment(2, "main", 1),))
    pool_ctx = StepContext(request_ids=(2,), graph_walk="prefill", slot=0, capture=False)
    live["kv"].admit(KVStep(segments=(Segment(2, "main", 1),)), pool_ctx)
    outcome = live["pool"].admit(pool_step, pool_ctx)
    assert not outcome.ok and isinstance(outcome.reason, AllocationFailed)
    assert outcome.reason.request_id == 2

    engine = _offload_engine({k: live[k] for k in ("kv", "pool", "rope")}, planner)
    worker = _refused_worker(engine, line=[1, 2])
    batch = ScheduledBatch(node_name="n", graph_walk="prefill", request_to_worker_graph={2: "g"})
    worker.pipeline._handle_admit_failure(batch, _refused(batch, outcome.reason, "pool"))
    assert worker.pushed == [[2]]
    assert list(worker.admission.held_until) == [2]
    assert engine.residency("n", 1) is Residency.RESIDENT


def test_an_offload_another_node_causes_gives_the_slot_up():
    """Node ``m`` shares ``n``'s KV pool and plans on the host. Its refused admit
    picks ``n``'s request as the victim (eviction searches every request the pool
    holds) and offloads it through ``m``: the slot must go on ``n``'s planner,
    or its row would name pages the pool has freed, and the reload is written
    whole."""
    from mstar.engine.resources.runner import StepRunner
    from mstar.testing_utils.engine import make_engine, make_mgmt

    live = _resources(max_num_pages=8)
    owned = {k: live[k] for k in ("kv", "pool", "rope")}
    planner = ResidentPlanner("n", {WALK}, owned, CPU, max_bs=4)
    for rid, tokens in ((1, 7), (2, 7), (3, 11)):
        _ingest(live, rid)
        step, ctx = _host_step(live, [rid], [tokens])
        planner.commit(step, ctx)
    kv = live["kv"]
    _HostPagesKV(kv)
    _captured_step(live, planner, [1, 2, 3], slot=0, bs=3)
    assert planner.slot_of(3) is not None and kv._arena.num_free == 0
    engine = make_engine(
        _resources=owned,
        _submodules={
            "n": make_mgmt(resources=owned, resident=planner),
            "m": make_mgmt(resources={"kv": kv}, resident=None),
        },
        _graph_runtime=SimpleNamespace(get_rid_string=str, get_rid_handle=int),
        _resource_planners=dict.fromkeys(owned, planner),
        # these requests were taken in by their resources alone, so no admission queue holds them
        _runner=SimpleNamespace(readmit=lambda rid, node: None),
    )

    # a step of m's over n's two short rows, crossing a page: the pool refuses
    step = SubmoduleStep(segments=[Segment(rid, "main", 1) for rid in (1, 2)], steps={"kv": KVStep()})
    step.set_ctx(StepContext(request_ids=(1, 2), graph_walk="other", slot=0, capture=False))
    outcome = StepRunner({"kv": kv}).admit(step)
    assert not outcome.ok and outcome.failed_resource == "kv"

    worker = _refused_worker(engine, line=[1, 2, 3])
    batch = ScheduledBatch(node_name="m", graph_walk="other", request_to_worker_graph={1: "g", 2: "g"})
    worker.pipeline._handle_admit_failure(batch, _refused(batch, outcome.reason, outcome.failed_resource))
    assert engine.residency("m", 3) is Residency.ON_HOST
    assert planner.slot_of(3) is None, "the offload left n's row naming freed pages"

    for key in ("kv", "pool", "rope"):
        live[key].remove_request(2)
    planner.release(2)
    assert engine.reload_request("n", 3)
    s = _captured_step(live, planner, [3], slot=0, bs=1)
    assert s.ctx.resident_plan.taken == {3}
    assert kv._resident.host_len[planner.slot_of(3)] == kv._streams[3]["main"].stored_len
    _assert_step_read_its_streams(live, s, [3])
    _assert_row(planner, live, 3)


def test_a_host_step_of_another_node_over_a_held_row_marks_it():
    """A node sharing the resident node's pool, without a planner, runs a host
    step: over a row the planner holds it marks the row as the node's own host
    step would; over other rows nothing."""
    live, ref, planner = _setup({1: 9, 2: 9})
    del ref
    _captured_step(live, planner, [1, 2], slot=0, bs=2)
    commits = planner._commits
    _ingest(live, 3)
    other, ctx = _host_step(live, [3], [5])
    planner.note_foreign_step(other, ctx)
    assert planner._commits == commits and not planner._moved and not planner._stale
    step, ctx = _host_step(live, [1, 2], [1, 4])
    planner.note_foreign_step(step, ctx)
    assert planner._commits == commits
    assert planner._moved == {1} and planner._stale == {2}


def test_a_follower_replaying_an_offload_gives_the_slot_up_too():
    live, ref, planner = _setup({1: 9})
    del ref
    _captured_step(live, planner, [1], slot=0, bs=1)
    _HostPagesKV(live["kv"])
    engine = _offload_engine({k: live[k] for k in ("kv", "pool", "rope")}, planner)
    engine._tp_follower_nodes = {"n"}
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "1", node="n")
    assert engine.apply_resident_delta(delta, Rebuilds({}, {}))
    assert planner.slot_of(1) is None


# ---- two KV pools on a node ----

def _pool(name: str, max_seq_len: int = 512, max_num_pages: int = 64) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=max_seq_len,
            max_num_pages=max_num_pages, page_size=PAGE,
        ),
        name=name, joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )
    kv._cg_max_seq_len = 64
    return kv


def _decoder(positions: str = "kv"):
    """A decoder's resources: the self-attention pool ``kv``, a second pool
    ``ctx`` holding each request's context, and positions over ``positions``."""
    rope = RopeManager(PositionConfig(kv_cache=positions), CPU, max_seq_len=512, head_dim=8)
    rope._pinned_pos_ids_buffer = lambda key, n: torch.empty(n, dtype=torch.long)
    return {"kv": _pool("kv"), "ctx": _pool("ctx", max_seq_len=64), "rope": rope}


def _decoder_setup(prompts: dict[int, tuple[int, int]], positions: str = "kv"):
    """Each request's prompt and context prefilled on resident resources and a
    host-planned reference. ``prompts`` is rid -> (prompt tokens, context tokens)."""
    live, ref = _decoder(positions), _decoder(positions)
    planner = ResidentPlanner("n", {WALK}, {k: live[k] for k in ("kv", "ctx", "rope")}, CPU, max_bs=4)
    for res in (live, ref):
        for rid, (prompt, context) in prompts.items():
            res["kv"].ingest_request(rid, KVReqConfig())
            res["ctx"].ingest_request(rid, KVReqConfig())
            res["rope"].ingest_request(rid, None)
            _decoder_host_step(res, [rid], [prompt], [context])
    return live, ref, planner


def _decoder_host_step(res, rids, spans, context_spans, walk="prefill"):
    ctx = StepContext(request_ids=tuple(rids), graph_walk=walk, slot=0, capture=False)
    segments = tuple(Segment(r, "main", n) for r, n in zip(rids, spans, strict=True))
    context = tuple(Segment(r, "main", n) for r, n in zip(rids, context_spans, strict=True))
    kv, cx, rope = KVStep(segments=segments), KVStep(segments=context), PositionStep(segments=segments)
    assert res["kv"].admit(kv, ctx).ok
    assert res["ctx"].admit(cx, ctx).ok
    ctx.plan_results["kv"] = res["kv"].plan(kv, ctx)
    ctx.plan_results["ctx"] = res["ctx"].plan(cx, ctx)
    res["rope"].plan(rope, ctx)
    for key, resource_step in (("kv", kv), ("ctx", cx), ("rope", rope)):
        res[key].commit(resource_step, ctx)
    return SubmoduleStep(steps={"kv": kv, "ctx": cx, "rope": rope}, segments=list(segments)), ctx


def _decoder_step(res, planner, rids, slot, bs, context_span=0):
    """One captured decode step: ``kv`` appends a token a row, and ``ctx`` is read
    at span 0, or appended to as well at span 1. With a planner, the replay of a
    resident capture; without one, the host plan of the same step."""
    padded = tuple(rids) + tuple(-(i + 1) for i in range(bs - len(rids)))
    bucket = BucketKey(graph_walk=WALK, bs=bs, num_tokens=bs)
    lease = SlotLease(slot=slot, bucket=bucket)
    ctx = StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=slot, capture=False, slot_lease=lease)
    ctx.set_padded_rids(padded)
    segments = tuple(Segment(r, "main", 1) for r in padded)
    context = tuple(Segment(r, "main", context_span) for r in padded)
    kv, cx, rope = KVStep(segments=segments), KVStep(segments=context), PositionStep(segments=segments)
    step = SubmoduleStep(steps={"kv": kv, "ctx": cx, "rope": rope}, segments=list(segments))
    assert res["kv"].admit(kv, ctx).ok
    assert res["ctx"].admit(cx, ctx).ok
    if (bucket, slot) not in res.setdefault("captured", set()):
        res["captured"].add((bucket, slot))
        first = StepContext(request_ids=padded, graph_walk=WALK, slot=slot, capture=True, slot_lease=lease)
        first.plan_results["kv"] = res["kv"].plan(kv, first)
        first.plan_results["ctx"] = res["ctx"].plan(cx, first)
        res["rope"].plan(rope, first)
    captured = None
    if planner is not None:
        captured = planner.capture_step(bucket, slot, step)
        ctx.resident = planner
        planner.plan(step, ctx)
    ctx.plan_results["kv"] = res["kv"].plan(kv, ctx)
    ctx.plan_results["ctx"] = res["ctx"].plan(cx, ctx)
    res["rope"].plan(rope, ctx)
    if planner is not None:
        planner.flush(ctx.resident_plan)
        planner.gather(captured)
    out = SimpleNamespace(plan=ctx.resident_plan, captured=captured, lens={})
    ck = CGSlotKey(bucket=bucket, slot=slot, label="main")
    out.positions = res["rope"]._static_pos_ids[ck][:bs].tolist()
    for key in ("kv", "ctx") if context_span else ("kv",):
        state = res[key]._static_plan_state(slot, ("main", 0), bucket.num_tokens)
        setattr(out, f"{key}_pages", state.token_to_page[:bs].tolist())
        setattr(out, f"{key}_offsets", state.token_to_cache[:bs].tolist())
    if planner is not None:
        out.lens = {key: lens[:bs].tolist() for key, lens in captured.buffers.kv_len.items()}
        planner.commit_in_graph(captured)
    for key, resource_step in (("kv", kv), ("ctx", cx), ("rope", rope)):
        res[key].commit(resource_step, ctx)
    if planner is not None:
        planner.commit(step, ctx)
    return out


def _assert_pool_row(planner, live, key: str, rid: int) -> None:
    """``key``'s cells of ``rid``'s row hold its stream: the stored length, and
    every page holding a token."""
    tables, stream = live[key]._resident, live[key]._streams[rid]["main"]
    slot = planner.slot_of(rid)
    n = int(tables.kv_len.view[slot, 0])
    assert n == stream.stored_len
    assert tables.block_table.view[slot, : -(-n // PAGE)].tolist() == stream.page_indices[: -(-n // PAGE)]


def _assert_trash_row(live) -> None:
    for key in ("kv", "ctx"):
        tables = live[key]._resident
        assert int(tables.kv_len.view[TRASH_SLOT, 0]) == 0
        assert tables.block_table.view[TRASH_SLOT].eq(SINK_PAGE).all()


def test_two_pools_lay_out_a_span_each_after_the_narrow_columns():
    live, _, planner = _decoder_setup({})
    kv, ctx = live["kv"]._resident, live["ctx"]._resident
    assert planner.narrow_cols == 3
    assert kv.block_table.col == 3 and kv.block_table.width == 63
    assert ctx.block_table.col == 3 + 63 and ctx.block_table.width == 16
    assert planner._stride == 3 + 63 + 16
    assert {kv.kv_len.col, ctx.kv_len.col, live["rope"]._resident_delta.col} == {0, 1, 2}
    for table in (kv.block_table, ctx.block_table, kv.kv_len, ctx.kv_len):
        assert table.view.data_ptr() == planner.store.data_ptr() + 8 * table.col


def test_a_self_attention_pool_and_a_context_pool_run_through_captured_steps():
    """``kv`` appends a token a step, ``ctx`` is read: every step addresses what
    the host plan does and gathers both pools' lengths, a padding row reads the
    trash row, and after every step both rows hold their streams."""
    prompts = {1: (3, 9), 2: (6, 13), 3: (8, 1)}
    live, ref, planner = _decoder_setup(prompts)
    rids, contexts = [1, 2, 3], [9, 13, 1]
    crossed = 0
    for t in range(14):
        before = [live["kv"]._streams[rid]["main"].stored_len for rid in rids]
        a = _decoder_step(live, planner, rids, slot=t % 2, bs=4)
        b = _decoder_step(ref, None, rids, slot=t % 2, bs=4)
        assert a.kv_pages[:3] == b.kv_pages[:3] and a.kv_offsets[:3] == b.kv_offsets[:3]
        assert a.positions[:3] == b.positions[:3]
        assert a.lens == {"kv": before + [0], "ctx": contexts + [0]}
        # the padding row
        assert (a.kv_pages[3], a.kv_offsets[3], a.positions[3]) == (SINK_PAGE, 0, 0)
        assert set(a.plan.crossed) <= {"kv"}
        crossed += "kv" in a.plan.crossed
        for rid in rids:
            _assert_pool_row(planner, live, "kv", rid)
            _assert_pool_row(planner, live, "ctx", rid)
        _assert_trash_row(live)
    ctx = live["ctx"]._resident
    assert [int(ctx.host_len[planner.slot_of(rid)]) for rid in rids] == contexts
    assert crossed >= 3


def test_a_crossing_on_the_appending_pool_writes_its_table_alone():
    live, ref, planner = _decoder_setup({1: (2, 9)})
    del ref
    for t in range(2):
        _decoder_step(live, planner, [1], slot=t % 2, bs=1)
    kv, ctx = live["kv"]._resident, live["ctx"]._resident
    slot = planner.slot_of(1)
    held = planner.store.view(planner.max_slots, planner._stride)[slot].clone()
    # stored 4 of a page of 4: this step's token goes to the row's second page
    s = _decoder_step(live, planner, [1], slot=0, bs=1)
    assert list(s.plan.crossed) == ["kv"]
    assert s.plan.writes.cells == [kv.block_table.cell(slot, 1)]
    row = planner.store.view(planner.max_slots, planner._stride)[slot]
    changed = (row != held).nonzero().flatten().tolist()
    assert changed == [kv.kv_len.col, kv.block_table.col + 1]
    assert row[ctx.block_table.col: ctx.block_table.col + 3].tolist() == live["ctx"]._streams[1]["main"].page_indices
    _assert_pool_row(planner, live, "kv", 1)
    _assert_pool_row(planner, live, "ctx", 1)


def test_a_stale_row_is_rewritten_whole_with_both_spans():
    live, ref, planner = _decoder_setup({1: (3, 9), 2: (6, 13)})
    del ref
    for t in range(3):
        _decoder_step(live, planner, [1, 2], slot=t % 2, bs=2)
    slot = planner.slot_of(2)
    kv, ctx = live["kv"]._resident, live["ctx"]._resident
    # whatever the device row held, a stale row comes back from the streams
    planner.store.view(planner.max_slots, planner._stride)[slot].fill_(77)
    planner.mark_stale(2)
    s = _decoder_step(live, planner, [1, 2], slot=1, bs=2)
    runs = [(col, len(cells)) for row_slot, col, cells in s.plan.writes.rows if row_slot == slot]
    assert runs == [(0, 3 + 9 // PAGE + 1), (ctx.block_table.col, 4)]
    assert kv.block_table.col == 3
    _assert_pool_row(planner, live, "kv", 2)
    _assert_pool_row(planner, live, "ctx", 2)
    assert int(live["rope"]._resident_delta.view[slot, 0]) == 0
    _assert_pool_row(planner, live, "kv", 1)


def test_a_context_moved_by_a_host_step_is_rewritten_and_an_unmoved_one_is_not():
    """A host step that ran a row one token leaves it to the next resident step
    to refresh; the context pool compares its own stream, which nothing but a
    host step moves, and rewrites the row only if it changed."""
    live, ref, planner = _decoder_setup({1: (3, 9), 2: (6, 13)})
    del ref
    for t in range(2):
        _decoder_step(live, planner, [1, 2], slot=t % 2, bs=2)
    # one token on each self stream; request 2's context grows by a page's worth, 1's doesn't
    step, ctx = _decoder_host_step(live, [1, 2], [1, 1], [0, 5], walk="mixed")
    planner.commit(step, ctx)
    assert planner._moved == {1, 2}
    s = _decoder_step(live, planner, [1, 2], slot=0, bs=2)
    assert s.plan.taken == {2}
    assert s.lens["ctx"] == [9, 18]
    for rid in (1, 2):
        _assert_pool_row(planner, live, "kv", rid)
        _assert_pool_row(planner, live, "ctx", rid)


def test_a_context_the_step_only_reads_never_advances():
    live, ref, planner = _decoder_setup({1: (3, 7)})
    del ref
    ctx = live["ctx"]._resident
    for t in range(6):
        s = _decoder_step(live, planner, [1], slot=t % 2, bs=2)
    slot = planner.slot_of(1)
    assert int(ctx.kv_len.view[slot, 0]) == int(ctx.host_len[slot]) == 7
    assert int(ctx.host_cols[slot]) == 2
    assert s.captured.streams["ctx"].reads and not s.captured.streams["kv"].reads
    assert "ctx" in s.captured.block_tables and "kv" in s.captured.block_tables


@pytest.mark.parametrize("positions", ["kv", "ctx"])
def test_positions_follow_the_pool_their_config_names(positions):
    """Two pools that both append, at different lengths: the position offset is
    taken against, and gathered with, the length of the pool positions name."""
    live, ref, planner = _decoder_setup({1: (3, 11), 2: (9, 2)}, positions=positions)
    for t in range(8):
        a = _decoder_step(live, planner, [1, 2], slot=t % 2, bs=4, context_span=1)
        b = _decoder_step(ref, None, [1, 2], slot=t % 2, bs=4, context_span=1)
        assert a.positions[:2] == b.positions[:2]
        for key in ("kv", "ctx"):
            assert getattr(a, f"{key}_pages")[:2] == getattr(b, f"{key}_pages")[:2]
            assert getattr(a, f"{key}_offsets")[:2] == getattr(b, f"{key}_offsets")[:2]
    assert planner.buffers(BucketKey(WALK, 4, 4), 0).cells.shape == (2, 2 * 4 * 2)


def test_positions_over_a_pool_the_step_only_reads_refuse_at_capture():
    live, ref, planner = _decoder_setup({1: (3, 7)}, positions="ctx")
    del ref
    with pytest.raises(ValueError, match="advance with the stream"):
        _decoder_step(live, planner, [1], slot=0, bs=1)


def test_capture_takes_a_label_for_each_resource():
    live, _, planner = _decoder_setup({})
    del live
    rows = [Segment(1, "self", 1)]
    step = SubmoduleStep(
        steps={"kv": KVStep(), "ctx": KVStep(segments=(Segment(1, "context", 0),)), "rope": PositionStep()},
        segments=rows,
    )
    captured = planner.capture_step(BucketKey(WALK, 1, 1), 0, step)
    streams = {key: (s.label, s.reads) for key, s in captured.streams.items()}
    assert streams == {"kv": ("self", False), "ctx": ("context", True), "rope": ("self", False)}
    assert planner._streams[BucketKey(WALK, 1, 1)] is captured.streams


def test_capture_refuses_two_labels_on_one_resource():
    live, _, planner = _decoder_setup({})
    del live
    rows = [Segment(1, "main", 1)]
    two = (Segment(1, "main", 1), Segment(1, "aux", 1))
    step = SubmoduleStep(steps={"kv": KVStep(segments=two), "ctx": KVStep(), "rope": PositionStep()}, segments=rows)
    with pytest.raises(ValueError, match="resource 'kv' declares 2 segments over labels"):
        planner.capture_step(BucketKey(WALK, 1, 1), 0, step)
    mixed = (Segment(1, "main", 1), Segment(2, "main", 0))
    step = SubmoduleStep(steps={"kv": KVStep(), "ctx": KVStep(segments=mixed), "rope": PositionStep()}, segments=rows)
    with pytest.raises(ValueError, match="resource 'ctx'"):
        planner.capture_step(BucketKey(WALK, 2, 2), 0, step)


class _Fmha:
    """A captured fmha decode wrapper's gather, recorded."""

    plans_in_graph = True
    supports_gather_resident = False
    supports_repeats = True

    def __init__(self):
        self.calls = []

    def gather_resident(self, store, slots, kv_len, stride, block_col):
        self.calls.append((kv_len, block_col))


def _fmha_manager(pool: str, wrapper):
    from mstar.engine.resources.attn.flashinfer import FlashInferManager

    manager = FlashInferManager.__new__(FlashInferManager)
    manager._kv_cache_name = pool
    manager._cg_plan_states = {CGSlotKey(bucket=BucketKey(WALK, 2, 2), slot=0, label="main"): wrapper}
    return manager


@pytest.mark.parametrize("pool", ["kv", "ctx"])
def test_the_fmha_gather_reads_its_own_pools_table_and_lengths(monkeypatch, pool):
    from mstar.engine.resources.attn import flashinfer as flashinfer_mod

    monkeypatch.setattr(flashinfer_mod, "FmhaDecodeWrapper", _Fmha)
    live, _, planner = _decoder_setup({})
    wrapper = _Fmha()
    manager = _fmha_manager(pool, wrapper)
    segments = [Segment(1, "main", 1), Segment(2, "main", 1)]
    # both pools append here, so either can be read off the tables
    step = SubmoduleStep(steps={"kv": KVStep(), "ctx": KVStep(), "rope": PositionStep()}, segments=segments)
    captured = planner.capture_step(BucketKey(WALK, 2, 2), 0, step)
    for key in ("kv", "ctx"):
        live[key].resident_gather(captured)
    manager.resident_gather(captured)
    assert wrapper.calls == [(captured.buffers.kv_len[pool], live[pool]._resident.block_table.col)]
    assert captured.reads_tables and not captured.host_planned


def test_an_fmha_wrapper_over_a_pool_the_step_only_reads_keeps_its_host_plan(monkeypatch):
    from mstar.engine.resources.attn import flashinfer as flashinfer_mod

    monkeypatch.setattr(flashinfer_mod, "FmhaDecodeWrapper", _Fmha)
    _, _, planner = _decoder_setup({})
    wrapper = _Fmha()
    manager = _fmha_manager("ctx", wrapper)
    segments = [Segment(1, "main", 1), Segment(2, "main", 1)]
    context = (Segment(1, "main", 0), Segment(2, "main", 0))
    steps = {"kv": KVStep(), "ctx": KVStep(segments=context), "rope": PositionStep()}
    captured = planner.capture_step(BucketKey(WALK, 2, 2), 0, SubmoduleStep(steps=steps, segments=segments))
    manager.resident_gather(captured)
    assert not wrapper.calls and captured.host_planned


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel runs on CUDA")
def test_the_fmha_gather_kernel_reads_the_second_pools_table():
    from mstar.engine.resources.attn.wrappers import FmhaDecodeWrapper

    device = torch.device("cuda")
    store = torch.zeros(5 * 20, dtype=torch.long, device=device)
    rows = store.view(5, 20)
    # the first pool's table at columns 4-9 and the second's at 10-19, one row of each per slot
    rows[:, 4:10] = torch.arange(5 * 6, device=device).view(5, 6) + 1000
    rows[:, 10:20] = torch.arange(5 * 10, device=device).view(5, 10) + 2000
    wrapper = FmhaDecodeWrapper.__new__(FmhaDecodeWrapper)
    wrapper.batch_size, wrapper._page_size = 3, 4
    wrapper._table = torch.zeros((3, 10), dtype=torch.int32, device=device)
    wrapper._lens = torch.zeros(3, dtype=torch.int32, device=device)
    slots = torch.tensor([2, 4, 0], device=device)
    kv_len = torch.tensor([5, 12, 0], device=device)
    wrapper.gather_resident(store, slots, kv_len, 20, 10)
    torch.cuda.synchronize()
    assert wrapper._lens.tolist() == [6, 13, 1]
    assert wrapper._table[0, :2].tolist() == rows[2, 10:12].tolist()
    assert wrapper._table[1, :4].tolist() == rows[4, 10:14].tolist()
    assert wrapper._table[2, :1].tolist() == rows[0, 10:11].tolist()


class _Attention(StubResource):
    def __init__(self, reads_tables: bool):
        self._reads = reads_tables

    def resident_refusal(self):
        return None

    def resident_gather(self, step):
        if self._reads:
            step.reads_tables = True
        else:
            step.host_planned = True


@pytest.mark.parametrize("attentions,free", [
    ((True,), True), ((True, True), True), ((True, False), False), ((False, True), False), ((), False),
])
def test_a_bucket_is_host_plan_free_only_when_every_attention_reads_the_tables(attentions, free):
    resources = {f"attn{i}": _Attention(reads) for i, reads in enumerate(attentions)}
    planner = ResidentPlanner("n", {WALK}, resources, CPU, max_bs=2)
    step = SubmoduleStep(steps={}, segments=[Segment(1, "main", 1)])
    planner.gather(planner.capture_step(BucketKey(WALK, 1, 1), 0, step))
    assert (BucketKey(WALK, 1, 1) in planner.host_plan_free) is free


def test_cross_attention_over_its_own_context_pool_runs_on_the_planner():
    from mstar.engine.resources.attn.cross import FlashInferCrossManager

    manager = FlashInferCrossManager.__new__(FlashInferCrossManager)
    manager._kv_cache_name, manager._query_kv_cache_name = "ctx", "kv"
    assert manager.resident_refusal() is None
    manager._query_kv_cache_name = "ctx"
    assert "second stream" in manager.resident_refusal()
    manager._query_kv_cache_name = "kv"
    manager._cg_plan_states = {}
    streams = {"kv": SimpleNamespace(label="main")}
    step = SimpleNamespace(
        reads_tables=False, host_planned=False, bucket="b", slot=0, streams=streams, lengths_only=set(),
    )
    FlashInferCrossManager.resident_gather(manager, step)
    assert step.host_planned and not step.reads_tables and not step.lengths_only


# ---- refusals ----

class _Spare(StubResource):
    pass


def _build(resources, nodes=None, walks=(WALK,), max_bs=4):
    return build_planner("n", walks, resources, nodes or {}, CPU, max_bs)


def test_a_node_with_no_table_owner_still_builds():
    planner = ResidentPlanner("n", {WALK}, {}, CPU, max_bs=2)
    assert planner.max_slots == 5


def test_every_resident_resource_passes():
    res = _resources()
    planner, why = _build({k: res[k] for k in ("kv", "pool", "rope")})
    assert planner is not None and why == ""


def test_no_walks_no_planner():
    res = _resources()
    planner, why = _build({"kv": res["kv"]}, walks=())
    assert planner is None and "no resident walk" in why


def test_no_capture_of_the_walks_no_planner():
    res = _resources()
    planner, why = _build({"kv": res["kv"]}, max_bs=0)
    assert planner is None and "nothing captures" in why
    assert res["kv"]._resident is None


def test_a_resource_without_a_table_refuses():
    planner, why = _build({"other": _Spare()})
    assert planner is None and "_Spare keeps no resident table" in why


def test_a_pool_shared_with_a_host_planned_node_builds():
    res = _resources()
    planner, why = _build({"kv": res["kv"]}, nodes={"kv": ("n", "m")})
    assert planner is not None and why == ""


def test_a_pool_shared_with_another_resident_node_refuses():
    # a resource binds one planner's tables
    res = _resources()
    planner, why = build_planner("n", (WALK,), {"kv": res["kv"]}, {"kv": ("n", "m")}, CPU, 4, resident_nodes={"m"})
    assert planner is None and "shared with resident node(s) ['m']" in why


def test_a_pool_shared_with_a_lockstep_node_refuses():
    # offloads are journalled per node, so a follower's replay order across nodes is undefined
    res = _resources()
    planner, why = build_planner("n", (WALK,), {"kv": res["kv"]}, {"kv": ("n", "m")}, CPU, 4, sharded_nodes={"m"})
    assert planner is None and "in lockstep" in why
    # unshared, a lockstep node keeps its planner
    planner, why = build_planner("n", (WALK,), {"kv": res["kv"]}, {"kv": ("n",)}, CPU, 4, sharded_nodes={"n"})
    assert planner is not None


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
    ("pre_forks", (("a", "b"),)), ("post_forks", (("a", "b"),)), ("sink", frozenset({(1, "main")})), ("commit", False),
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

def _capture_runner(max_bs=4):
    bound = {}
    return SimpleNamespace(
        max_capture_bs=lambda walks: max_bs, bind_resident=lambda planner: bound.update(p=planner), bound=bound,
    )


def _engine(walks, resources, device=CPU):
    from mstar.engine.resources.runner import StepRunner
    from mstar.testing_utils.engine import make_engine, make_mgmt

    submodule = SimpleNamespace(resident_walks=lambda: frozenset(walks))
    engine = make_engine(
        _device=device,
        _runner=StepRunner(resources),
        _resource_nodes={key: frozenset({"n"}) for key in resources},
        _submodules={"n": make_mgmt(submodule, resources=resources)},
    )
    return engine, engine._submodules["n"]


def test_the_engine_builds_a_planner_for_a_capable_node():
    res = _resources()
    engine, mgmt = _engine({WALK}, {k: res[k] for k in ("kv", "pool", "rope")})
    runner = _capture_runner()
    engine._build_resident_planner("n", runner)
    assert isinstance(mgmt.resident, ResidentPlanner)
    assert runner.bound["p"] is mgmt.resident
    assert mgmt.resident.max_slots == 9
    assert mgmt.resident.covers(BucketKey(WALK, 2, 2))
    assert not mgmt.resident.covers(BucketKey("prefill_text", 2, 64))


def test_the_knob_turns_it_off(monkeypatch):
    monkeypatch.setenv("MSTAR_RESIDENT_PLANNING", "off")
    res = _resources()
    engine, mgmt = _engine({WALK}, {"kv": res["kv"]})
    engine._build_resident_planner("n", _capture_runner())
    assert mgmt.resident is None
    assert res["kv"]._resident is None


def test_a_refused_node_stays_on_the_host_and_says_why(caplog):
    res = _resources()
    engine, mgmt = _engine({WALK}, {"kv": res["kv"], "other": _Spare()})
    with caplog.at_level("WARNING"):
        engine._build_resident_planner("n", _capture_runner())
    assert mgmt.resident is None
    assert "_Spare keeps no resident table" in caplog.text
    assert res["kv"]._resident is None


def test_a_node_with_no_resident_walk_is_left_alone(caplog):
    res = _resources()
    engine, mgmt = _engine((), {"kv": res["kv"]})
    with caplog.at_level("WARNING"):
        engine._build_resident_planner("n", _capture_runner())
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


def _qwen35_llm():
    from mstar.model.qwen3_5 import submodules

    return submodules, next(
        cls for cls in vars(submodules).values()
        if isinstance(cls, type) and cls.__module__ == submodules.__name__ and "resident_walks" in vars(cls)
    )


@pytest.mark.parametrize("knob", [False, True])
def test_qwen35_decode_builds_its_rope_in_the_graph_on_every_config(monkeypatch, knob):
    submodules, llm = _qwen35_llm()
    monkeypatch.setattr(submodules, "ingraph_decode_rope", lambda: knob)
    assert llm.resident_walks(llm) == {"decode"}
    assert llm._rope_in_graph(SimpleNamespace(captured=True), "decode")
    assert not llm._rope_in_graph(SimpleNamespace(captured=False), "decode")


@pytest.mark.parametrize("rows", ["loopback", "own_ids"])
def test_qwen35_captured_decode_preprocess_reads_nothing_the_tables_fill(rows):
    """The graph fills positions and write locations, after `preprocess`."""
    _, llm = _qwen35_llm()
    token = torch.tensor([5, 6])
    inputs = [
        SimpleNamespace(input_embeds=None, input_ids=None if rows == "loopback" else token[:1], custom_pos_ids=None,
                        input_seq_len=0 if rows == "loopback" else 1)
        for _ in range(2)
    ]
    node = llm(None, SimpleNamespace(stop_token_ids=(1,)))
    node._position_ids_3d = lambda inputs: pytest.fail("read positions")
    node.sampler = SimpleNamespace(
        _open=True, resource=SimpleNamespace(loopback_tokens=lambda rids: token), rows=(1, 2),
    )
    engine_inputs = SimpleNamespace(captured=True, request_ids=[1, 2])
    out = node.preprocess("decode", engine_inputs, inputs)
    assert set(out) == {"input_ids"}


def test_a_capture_runner_answers_for_the_walks_it_captures():
    from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner as Runner

    runner = Runner.__new__(Runner)
    runner._capture_configs = [
        SimpleNamespace(capture_graph_walk="decode", capture_batch_sizes=[1, 8, 128]),
        SimpleNamespace(capture_graph_walk="prefill", capture_batch_sizes=[1, 512]),
        SimpleNamespace(capture_graph_walk="other", capture_batch_sizes=None),
    ]
    assert runner.max_capture_bs({"decode"}) == 128
    assert runner.max_capture_bs({"other"}) == max(Runner.CAPTURE_BATCH_SIZES)
    assert runner.max_capture_bs({"absent"}) == 0


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


def test_the_slot_count_is_no_knob():
    with pytest.raises(ValueError):
        knobs.check({"resident_slots": 8})


def test_cross_attention_packs_a_resident_steps_queries_off_its_rows():
    """A resident step's query pool may hand back lengths only, with no views."""
    from mstar.engine.resources.attn.cross import FlashInferCrossManager

    cross = FlashInferCrossManager.__new__(FlashInferCrossManager)
    cross._kv_cache_name, cross._query_kv_cache_name = "ctx", "kv"
    cross._resident_qo_indptr = {}
    plan = SimpleNamespace(writes=SimpleNamespace(streams={"kv": SimpleNamespace(label="main")}))
    ctx = SimpleNamespace(resident=object(), resident_plan=plan, padded_request_ids=[3, 5, 9])
    packings = cross._query_packings(SimpleNamespace(), ctx)
    assert list(packings) == ["main"]
    assert packings["main"].request_ids == [3, 5, 9]
    assert packings["main"].qo_indptr.tolist() == [0, 1, 2, 3]


def test_cross_attention_schedules_a_context_once_per_lengths():
    """A context's schedule reads only its lengths, so a steady step plans nothing; a full
    plan on the wrapper in between starts over."""
    from mstar.engine.resources.attn.cross import FlashInferCrossManager

    cross = FlashInferCrossManager.__new__(FlashInferCrossManager)
    cross._kv_cache_name, cross._query_kv_cache_name, cross._context_label = "ctx", "kv", "context"
    cross._resident_qo_indptr, cross._scheduled_lens, cross._last_plans = {}, {}, {}
    cross._current_plan_states, cross._preplan_states, cross._preplanned = {}, {}, False
    cross.reset_default_cursors = lambda: None
    planned = []
    wrapper = SimpleNamespace(plan_schedule=lambda indptr, lens: planned.append(lens.tolist()))
    cross._wrapper_for = lambda *args, **kwargs: ("key", wrapper)
    lease = SimpleNamespace(bucket="bucket", slot=0)
    resident = SimpleNamespace(lengths_only={("bucket", "ctx")})
    streams = {"kv": SimpleNamespace(label="main")}

    def plan(lens):
        out = SimpleNamespace(cpu_indptrs=SimpleNamespace(
            paged_kv_indptr=torch.arange(len(lens) + 1, dtype=torch.int32),
            kv_lens=torch.tensor(lens, dtype=torch.int32),
        ))
        cross.plan(SimpleNamespace(segments=None, context_only=False), SimpleNamespace(
            slot_lease=lease, is_preplan=False, plan_phase="res.plan", resident=resident,
            resident_plan=SimpleNamespace(writes=SimpleNamespace(streams=streams)),
            padded_request_ids=list(range(len(lens))), plan_results={"ctx": {"context": out}}, slot=0,
            force_prefill=False,
        ))

    plan([1500, 1500, 1])
    plan([1500, 1500, 1])
    assert planned == [[1500, 1500, 1]]
    plan([1500, 1, 1])
    assert planned == [[1500, 1500, 1], [1500, 1, 1]]


def test_a_cross_attention_without_a_query_pool_plans_on_the_host():
    """A cross attention whose query side keeps no KV cache has no stream on the
    resident step to read: it stays host planned rather than looking one up."""
    from mstar.engine.resources.attn.cross import FlashInferCrossManager

    manager = FlashInferCrossManager.__new__(FlashInferCrossManager)
    manager._query_kv_cache_name, manager._kv_cache_name, manager._cg_plan_states = None, "ctx", {}
    step = SimpleNamespace(streams={}, host_planned=False, bucket=None, slot=0, lengths_only=set())
    manager.resident_gather(step)
    assert step.host_planned and not step.lengths_only


def test_a_walk_repeats_only_if_every_captured_wrapper_says_it_sees_each_replays_lengths():
    from mstar.engine.resources.attn.wrappers import FlashInferPrefillWrapper

    fmha = _fmha_manager("kv", _Fmha())
    fmha._repeat_walks = {}
    ctx = SimpleNamespace(graph_walk=WALK)
    assert fmha.max_repeats(None, ctx) > 1
    # a prefill wrapper's host plan fixes the lengths its replays see
    prefill = _fmha_manager("kv", FlashInferPrefillWrapper.__new__(FlashInferPrefillWrapper))
    prefill._repeat_walks = {}
    assert prefill.max_repeats(None, ctx) == 1
