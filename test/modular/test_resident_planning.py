"""Resident planning: a captured decode step reads its rows off device tables and
updates them itself, and what its graph computes matches what a host plan would
have staged.

The resources here are the real ones on the CPU. ``gather`` and ``commit_in_graph``
run the ops a capture records, so each step below is what a replay does; a second
set of resources plans the same requests on the host as the reference.
"""
from __future__ import annotations

import ctypes
import random
import sys
from functools import partial
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, ".")

from mstar.engine.accelerator_graph_runner import AcceleratorGraphRunner  # noqa: E402
from mstar.engine.device_run import RunState, TokenStream  # noqa: E402
from mstar.engine.resources import resident as resident_mod  # noqa: E402
from mstar.engine.resources import runner as runner_mod  # noqa: E402
from mstar.engine.resources.attn.config import AttentionConfig, AttentionSpec  # noqa: E402
from mstar.engine.resources.attn.flashinfer import FlashInferManager  # noqa: E402
from mstar.engine.resources.attn.wrappers import FlashInferDecodeWrapper  # noqa: E402
from mstar.engine.resources.base import CGSlotKey, Resource  # noqa: E402
from mstar.engine.resources.iteration_tables import IterationTables  # noqa: E402
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
from mstar.engine.resources.resident import NOT_ASKED, TRASH_SLOT, ResidentPlanner, build_planner  # noqa: E402
from mstar.engine.resources.resident_hooks import ANY_REPEATS  # noqa: E402
from mstar.engine.resources.runner import StepRunner  # noqa: E402
from mstar.engine.resources.spec import resolve_spec_dependencies  # noqa: E402
from mstar.engine.resources.step import (  # noqa: E402
    AllocationFailed,
    BucketKey,
    Segment,
    SlotLease,
    StepContext,
    SubmoduleStep,
)
from mstar.graph.base import EdgeKind, LoopEdges  # noqa: E402
from mstar.model.submodule_base import IterationInput  # noqa: E402
from mstar.utils import knobs, profiler  # noqa: E402
from mstar.utils.ipc_format import OffloadDelta  # noqa: E402
from mstar.worker.micro_scheduler import ScheduledBatch  # noqa: E402
from mstar.worker.step_repeats import RepeatConfig, StepRepeats, parse_step_repeats  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402

PAGE = 4
WALK = "decode"
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _resident_planning_on(monkeypatch):
    """Resident planning is opt-in; these tests are of a deployment that asks."""
    monkeypatch.setenv("MSTAR_RESIDENT_PLANNING", "on")


def _repeats(limit: int, nodes="n") -> StepRepeats:
    """Repeats up to ``limit`` asked of each of ``nodes``' WALK, a token loop."""
    edges = LoopEdges({"t": EdgeKind.LOOP_BACK}, {("t", "n"): EdgeKind.LOOP_BACK})
    return StepRepeats({n: RepeatConfig(max=limit) for n in nodes}, {(n, WALK): edges for n in nodes})


def _walk(*keys: str, reads: tuple[str, ...] = ()) -> dict:
    """WALK's streams as its template declares them: each resource on label
    "main", appending a token a row unless it only ``reads``."""
    return {WALK: {key: resident_mod.ResidentStream("main", key in reads) for key in keys}}


def _planner(res, max_bs: int, keys=("kv", "pool", "rope")) -> ResidentPlanner:
    return ResidentPlanner("n", _walk(*keys), {k: res[k] for k in keys}, CPU, max_bs=max_bs)


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


def _resources(cpu_offload_pages: int = 0, max_num_pages: int = 64, pool_slots: int = 9):
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=512,
            max_num_pages=max_num_pages, page_size=PAGE, cpu_offload_pages=cpu_offload_pages,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )
    kv._cg_max_seq_len = 64
    pool = RecurrentStatePool(device=CPU, config=RecurrentStateConfig(
        num_layers=1, max_slots=pool_slots, blocks={"state": RecurrentBlockConfig(shape=(2,), dtype=torch.float32)},
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


def _replay(res, planner, s, promote=True, head=None):
    """What the GPU thread does from the promote on. ``head``: the capture's
    ops before each replay's gather, given the replay's index."""
    ctx, bucket, lease = s.ctx, s.ctx.slot_lease.bucket, s.ctx.slot_lease
    if planner is not None:
        if promote:
            _promote(res, planner, s)
        planner.flush(ctx.resident_plan)
    state = res["kv"]._static_plan_state(lease.slot, ("main", 0), bucket.num_tokens)
    ck = CGSlotKey(bucket=bucket, slot=lease.slot, label="main")
    s.replays = []
    for replay in range(ctx.repeats):
        if head is not None:
            head(replay)
        if planner is not None:
            planner.gather(s.captured)
        s.pages = state.token_to_page[: bucket.bs].tolist()
        s.offsets = state.token_to_cache[: bucket.bs].tolist()
        s.positions = res["rope"]._static_pos_ids[ck][: bucket.bs].tolist()
        s.pool_slots = res["pool"]._cg_addressing[ck].slot_indices[: bucket.bs].tolist()
        s.replays.append(SimpleNamespace(
            pages=s.pages, offsets=s.offsets, positions=s.positions, pool_slots=s.pool_slots,
            has_state=res["pool"]._cg_addressing[ck].has_state[: bucket.bs].tolist(),
        ))
        if planner is not None:
            planner.commit_in_graph(s.captured)
    res["kv"].commit(s.kv, ctx)
    res["pool"].commit(s.pool, ctx)
    res["rope"].commit(s.rope, ctx)
    if planner is not None:
        planner.commit(s.step, ctx)
    return s


def _setup(prompts: dict[int, int], cpu_offload_pages: int = 0, max_num_pages: int = 64, pool_slots: int = 9):
    """The same requests prefilled on resident resources and on a host-planned reference."""
    live = _resources(cpu_offload_pages, max_num_pages, pool_slots)
    ref = _resources(cpu_offload_pages, max_num_pages, pool_slots)
    planner = _planner(live, max_bs=4)
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


@pytest.mark.parametrize("repeats", [2, 3, PAGE], ids=lambda n: f"{n}_replays")
def test_a_repeated_step_addresses_each_replay_as_host_plans_would(repeats):
    """Every replay of a step that repeats writes and attends where a host plan
    of that iteration would, page crossings included, with no host work between
    replays; the host's mirrors then stand where the stream does."""
    live, ref, planner = _setup({1: 3, 2: 5, 3: 8})
    rids = [1, 2, 3]
    for t in range(6):
        a = _captured_step(live, planner, rids, slot=t % 2, bs=4, repeats=repeats)
        for i, replay in enumerate(a.replays):
            b = _captured_step(ref, None, rids, slot=(t + i) % 2, bs=4)
            assert (replay.offsets[:3], replay.positions[:3]) == (b.offsets[:3], b.positions[:3])
            # a run reserves ahead, so its page numbers aren't the host's: each is the row's own
            for row, rid in enumerate(rids):
                page = live["kv"]._streams[rid]["main"].page_indices[replay.positions[row] // PAGE]
                assert replay.pages[row] == page
    tables = live["kv"]._resident
    for rid in rids:
        _assert_row(planner, live, rid)
        assert tables.host_len[planner.slot_of(rid)] == live["kv"]._streams[rid]["main"].stored_len


def test_a_resident_step_resumes_the_state_its_last_step_left():
    """``RecurrentStatePool.gather`` reads ``start`` for a row with no saved
    state, so a resident row must read as saved from its second iteration on,
    replays included (a transducer's state, restarted each step, repeats its
    first tokens); a fresh row and a padding row read ``start``."""
    live, _, planner = _setup({1: 3})
    _ingest(live, 2)
    a = _captured_step(live, planner, [1, 2], slot=0, bs=4, repeats=2)
    assert [r.has_state for r in a.replays] == [[True, False, False, False], [True, True, False, False]]
    b = _captured_step(live, planner, [1, 2], slot=1, bs=4)
    assert b.replays[0].has_state == [True, True, False, False]


def test_a_stopped_row_leaves_its_kv_and_recurrent_state_where_its_stop_left_them():
    """``device_stop: drop_rows``: from the replay after its stop column a row
    reads and writes the trash slot, so its stream length, page and recurrent
    slot stay where the stop left them, while the others advance; the next
    run reads the rows the host staged."""
    live, _, planner = _setup({1: 3, 2: 5})
    s = _captured_step(live, planner, [1, 2], slot=0, bs=2, run=False, repeats=3)
    bucket = s.ctx.slot_lease.bucket
    state = RunState(2, "t", {9}, CPU, planner.drop_rows(bucket, 0))
    state.stage(2, [False, False], 3)
    tokens = [[9, 1], [1, 1], [1, 1]]  # row 0 stops in the first column

    def head(replay):
        if replay:  # the stop kernel's mark
            state.marks |= (torch.tensor(tokens[replay - 1]).unsqueeze(-1) == state.stop_ids).any(-1)
        state.head()

    _replay(live, planner, s, head=head)
    assert [replay.pool_slots[0] == s.replays[0].pool_slots[0] for replay in s.replays] == [True, False, False]
    assert s.replays[0].pool_slots[1] == s.replays[2].pool_slots[1], "the other row keeps its slot"
    lengths = live["kv"]._resident.kv_len
    row, other = planner.slot_of(1), planner.slot_of(2)

    def stored(slot):
        return planner.store[slot * lengths.stride + lengths.col].item()

    assert stored(row) == 3 + 1, "the row stopped in the first replay"
    assert stored(other) == 5 + 3
    # the next run's first replay reads every row again
    state.stage(2, [False, False], 3)
    state.head()
    assert planner.buffers(bucket, 0).slots.tolist() == [row, other]


def test_a_run_that_ends_short_gives_back_what_its_commit_counted():
    """A yield ends a run where it stands: the host counted the whole run at
    commit, so what it didn't run goes back, and the row runs on from there."""
    live, _, planner = _setup({1: 3})
    s = _captured_step(live, planner, [1], slot=0, bs=1, repeats=3)
    stream = live["kv"]._streams[1]["main"]
    assert stream.stored_len == 6
    for key, step in (("kv", s.kv), ("pool", s.pool), ("rope", s.rope)):
        live[key].rewind_run(step, s.ctx, 2, {1})
    assert stream.stored_len == 4
    assert live["kv"]._resident.host_len[planner.slot_of(1)] == 4
    assert live["pool"]._slots[1]["main"].tokens == 4


def test_skipped_page_crossing_rewrites_cells_before_retry():
    live, _, planner = _setup({1: 2})
    _captured_step(live, planner, [1], slot=0, bs=1)
    # Commit a run crossing into the second page without replaying its cells.
    skipped = _captured_step(live, planner, [1], slot=0, bs=1, repeats=3, run=False)
    _promote(live, planner, skipped)
    planner.flush(skipped.ctx.resident_plan)
    for key, step in (("kv", skipped.kv), ("pool", skipped.pool), ("rope", skipped.rope)):
        live[key].commit(step, skipped.ctx)
    planner.commit(skipped.step, skipped.ctx)
    for key, step in (("kv", skipped.kv), ("pool", skipped.pool), ("rope", skipped.rope)):
        live[key].rewind_run(step, skipped.ctx, 3, {1})
    retry = _captured_step(live, planner, [1], slot=0, bs=1, repeats=3)
    expected = live["kv"]._streams[1]["main"].page_indices
    assert [r.pages[0] for r in retry.replays] == [expected[0], expected[1], expected[1]]


@pytest.mark.parametrize(("world_size", "policies", "why"), [
    (2, {}, "deployment: it runs on more than one rank"),
    (1, {}, "graph: it has no declared stop policy"),
])
def test_a_walk_says_why_it_doesnt_skip_a_stopped_replay(world_size, policies, why):
    from mstar.engine.engine import Engine

    engine = Engine.__new__(Engine)
    mgmt = SimpleNamespace(
        joint_comm_group=SimpleNamespace(world_size=world_size), device_stops={}, accelerator_graph_runner=None,
        submodule=SimpleNamespace(get_stop_policies=lambda walk: policies),
    )
    engine._submodules = {"n": mgmt}
    engine._ask_run("n", None, WALK, SimpleNamespace(), ())
    assert engine.device_stop_refusal("n", WALK) == why


@pytest.mark.parametrize("captured", [True, False])
def test_a_step_with_no_capture_for_its_batch_runs_one_iteration(captured):
    """Only a replay repeats: a batch that leases no graph (a size or shape
    the node didn't capture) runs its eager forward once."""
    from mstar.engine.engine import Engine

    def lease_slot(graph_walk, bs, slot, num_tokens=None, cg_key_info=None):
        return SlotLease(slot, BucketKey(graph_walk, bs, bs)) if captured else None

    engine = object.__new__(Engine)
    engine._submodules = {"n": SimpleNamespace(
        submodule=SimpleNamespace(cg_key_info=lambda walk, info, **kw: None), lora_pools=(), resident=None,
        accelerator_graph_runner=SimpleNamespace(lease_slot=lease_slot, is_packed=lambda lease: False),
    )}
    batch = SimpleNamespace(
        node_name="n", slot=0, request_ids=["a", "b"], inputs=None, per_request_info_wrapped={},
        per_request_input_metadata={},
        step_context=StepContext(request_ids=["a", "b"], graph_walk=WALK, slot=0, capture=False, repeats=4),
        lease_slot=lambda lease: None,
    )
    engine.reserve_replay_slot(batch)
    assert batch.step_context.repeats == (4 if captured else 1)


def _raw_backend(slot):
    """A backend whose raw launch replays ``slot``'s graph and whose 2D copy
    moves bytes on the host, through the C entry points the Rust loop calls;
    and the calls it took, in order."""
    from mstar.engine.accelerator_graph_backend import RawLaunch

    calls = []

    @ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    def launch(graph_exec, stream):
        assert (graph_exec, stream) == (1, 2)
        calls.append("launch")
        slot.graph.replay()
        return 0

    @ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
    )
    def copy(dst, dst_pitch, src, src_pitch, width, height, kind, stream):
        assert kind == 4 and stream == 2
        calls.append("copy")
        for row in range(height):
            ctypes.memmove(dst + row * dst_pitch, src + row * src_pitch, width)
        return 0

    addresses = [ctypes.cast(fn, ctypes.c_void_p).value for fn in (launch, copy)]
    backend = SimpleNamespace(raw_launch=lambda graph: RawLaunch(*addresses, 1, 2), keep=(launch, copy))
    return backend, calls


@pytest.mark.parametrize("raw", [False, True], ids=["host_loop", "rust_loop"])
def test_a_repeat_carries_every_loop_back_signal_and_keeps_a_column_per_replay(raw):
    """A replay overwrites its outputs and reads its static inputs, so a step
    that repeats hands each replay what the one before left of every
    loop-back signal (a token, a float embedding under another name, a long
    counter) and keeps a column of each per-iteration output, whichever loop
    launches it."""
    from mstar.model.submodule_base import CarryPair

    tokens, ids = torch.tensor([5, 7]), torch.zeros(2, 1, dtype=torch.long)
    embeds, embeds_in = torch.tensor([[0.5], [1.5]]), torch.zeros(2, 1)
    step, step_in = torch.tensor([10, 20]), torch.zeros(2, dtype=torch.long)
    frames = torch.tensor([[5.0, -5.0], [7.0, -7.0]])

    class Graph:
        def replay(self):
            tokens.copy_(ids.reshape(-1) + 1)
            embeds.copy_(embeds_in * 2)
            step.copy_(step_in + 1)
            frames.copy_(tokens[:, None] * torch.tensor([1.0, -1.0]))

    slot = SimpleNamespace(
        graph=Graph(), raw_launch=raw, static_inputs={"input_ids": ids, "input_embeds": embeds_in, "step": step_in},
        static_outputs=SimpleNamespace(
            check_stop_buffers={"new_token": tokens, "text_inputs": tokens},
            row_outputs={"frame": frames, "embeds": embeds, "step": step}, output_spans=None, packed_outputs={},
        ),
    )
    carries = {
        "text_inputs": CarryPair("new_token", "input_ids"), "embeds": CarryPair("embeds", "input_embeds"),
        "step": CarryPair("step", "step"),
    }
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner.slot_for = lambda lease: slot
    runner._graph_backend, calls = _raw_backend(slot)
    landing = TokenStream(2, 3, torch.device("cpu"))
    wide, rows, _ = runner.repeat(None, carries, 3, (), ["frame"], stream=(landing, "new_token"))
    if raw:
        # each replay's columns out, then the next one's carries in and its launch
        assert calls == ["copy"] * 4 + (["copy"] * 3 + ["launch"] + ["copy"] * 4) * 2
    # a replay's tokens land behind it, and ``landed`` behind them
    assert landing.tokens.tolist() == wide["new_token"].tolist() and landing.landed.item() == 3
    assert wide["new_token"].tolist() == [[5, 6, 7], [7, 8, 9]]
    assert wide["text_inputs"] is wide["new_token"]
    # the last replay's outputs, each from the replay before's
    assert embeds.reshape(-1).tolist() == [2.0, 6.0] and step.tolist() == [12, 22]
    # a per-iteration output keeps each replay's rows: [rows, replays, ...]
    assert rows["frame"].tolist() == [[[5, -5], [6, -6], [7, -7]], [[7, -7], [8, -8], [9, -9]]]


def test_a_run_a_copy_of_which_casts_stays_on_the_hosts_loop():
    from mstar.engine.accelerator_graph_runner import _launch_words

    assert _launch_words([None, (torch.zeros(2), torch.zeros(2))]) is not None
    assert _launch_words([None, (torch.zeros(2), torch.zeros(2, dtype=torch.long))]) is None


def test_a_capture_that_draws_from_a_torch_generator_keeps_the_hosts_loop():
    """A raw launch doesn't advance torch's generators, so its draws would repeat."""
    slot = SimpleNamespace(graph=object(), raw_launch=False)
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._buckets = {0: SimpleNamespace(config=SimpleNamespace(capture_graph_walk=WALK), slots=[slot])}
    runner._graph_backend = SimpleNamespace(launch_refusal=lambda: None)
    runner._generator_walks = {WALK}
    assert "torch generator" in runner.launch_refusal(WALK) and slot.raw_launch is False
    runner._generator_walks = set()
    assert runner.launch_refusal(WALK) is None and slot.raw_launch is True
    runner._graph_backend = SimpleNamespace(launch_refusal=lambda: "no runtime")
    assert runner.launch_refusal(WALK) == "backend: no runtime"


def test_a_decode_wrappers_lengths_move_one_a_replay_as_a_column():
    wrapper = FlashInferDecodeWrapper.__new__(FlashInferDecodeWrapper)
    wrapper._paged_kv_last_page_len_buf = torch.tensor([3, 5, 9], dtype=torch.int32)
    wrapper._replay_rows = 2
    column, target = wrapper.replay_column(3)
    assert column.tolist() == [[3, 4, 5], [5, 6, 7]] and column.dtype == torch.int32
    assert target.data_ptr() == wrapper._paged_kv_last_page_len_buf.data_ptr() and len(target) == 2


@pytest.mark.parametrize(("preprocess", "refused"), [
    (torch.cat, False),
    (lambda rows: torch.cat(rows) * 2, True),
    # a zero condition added to a template of zeros would pass without a probe value
    (lambda rows: torch.cat(rows) + torch.zeros(1), False),
    (lambda rows: torch.cat(rows) + 1.0, True),
], ids=["identity", "scaled", "zero_add", "condition_add"])
def test_a_preprocess_that_changes_a_loop_back_signal_keeps_its_walk_from_repeating(preprocess, refused):
    """A run skips ``preprocess`` between replays, so at load the engine feeds
    each loop-back signal a probe value through the capture template's
    ``preprocess`` and refuses a walk that changes it on the way."""
    from mstar.engine.accelerator_graph_runner import CaptureTemplate
    from mstar.model.submodule_base import CarryPair, NodeInputs

    rows = [NodeInputs(tensor_inputs={"embeds": torch.zeros(1, 4)}) for _ in range(2)]
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._widest = lambda walk: SimpleNamespace(bucket=SimpleNamespace(graph_walk=walk))
    runner._template = lambda spec: CaptureTemplate([0, 1], rows, None, None)
    runner._dummy_rows = SimpleNamespace(reset=lambda rids, free: None)
    runner._submodule = SimpleNamespace(preprocess=lambda graph_walk, engine_inputs, inputs: {
        "input_embeds": preprocess([row.tensor_inputs["embeds"] for row in inputs]),
    })
    refusal = runner.prepare_template(WALK, {"embeds": CarryPair("embeds", "input_embeds")})
    assert (refusal is not None) is refused
    assert refusal is None or "changes loop-back 'embeds'" in refusal


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
    repeats = _repeats(8)
    repeats.settle(lambda *key: None, lambda *key: None)
    assert repeats.ceiling(speculation, in_chain, lambda: room)[0] == want


@pytest.mark.parametrize("limit", [1, 4])
def test_the_startup_line_names_the_owner_that_refuses_each_capability(caplog, limit):
    """Each owner answers for itself; the worker prints one line per Loop walk,
    at every limit, naming the first that refused each capability."""
    from mstar.engine.engine import Engine
    from mstar.engine.resources.resident_hooks import ResidentHooks

    willing = SimpleNamespace(repeat_refusal=lambda walk, host_step: None)
    engine = Engine.__new__(Engine)
    engine._submodules = {node: SimpleNamespace(
        resident=SimpleNamespace(walks={WALK} - refused.keys()), resident_refusals=refused,
        joint_comm_group=SimpleNamespace(world_size=1),
        submodule=SimpleNamespace(repeat_carry=lambda walk, signals: {}),
        resources=resources, carry_refusals={}, carries={}, accelerator_graph_runner=None, template_steps={},
        row_signals={},
    ) for node, resources, refused in (
        ("a", {"kv": willing}, {}), ("b", {"kv": willing, "pool": ResidentHooks()}, {}),
        ("c", {"kv": willing}, {WALK: "resource 'kv': why"}),
    )}
    with caplog.at_level("INFO", logger="mstar.worker.step_repeats"):
        _repeats(limit, "abc").settle(
            engine.resident_refusal, engine.repeat_refusal,
            {"rows": lambda node, walk: NOT_ASKED if limit == 1 else engine.rows_refusal(node, walk)},
        )
    lines = {line.split(":", 1)[0]: line for line in caplog.messages}
    want = (
        "resident on | repeats on, up to 4 | rows off: runner: the node captures no graphs" if limit > 1
        else "resident on | repeats off: not asked for | rows off: not asked for"
    )
    assert lines[f"a/{WALK}"].endswith(want)
    if limit > 1:
        assert "repeats off: resource 'pool': ResidentHooks keeps no per-replay support |" in lines[f"b/{WALK}"]
    assert lines[f"c/{WALK}"].startswith(f"c/{WALK}: resident off: resource 'kv': why | repeats off")


def _token_loop(*extra_outputs, streamed=()):
    """Loop "loop" over node "n", whose token "t" loops back and goes to the
    client, beside a constant input "cond" and a Loop output "latent"."""
    from mstar.graph.base import GraphEdge, GraphNode, Loop
    from mstar.graph.special_destinations import EMIT_TO_CLIENT

    node = GraphNode(name="n", input_names={"t", "cond", *streamed}, outputs=[
        GraphEdge("n", "t"), GraphEdge(EMIT_TO_CLIENT, "t"), GraphEdge("vae", "latent"), *extra_outputs,
    ])
    node._register_streaming(set(streamed))
    return Loop(section=node, max_iters=8, outputs=[GraphEdge("vae", "latent")], name="loop")


def test_each_edge_of_a_loop_has_its_own_kind():
    """The kind is the edge's, not the signal's: "t" loops back and goes to the
    client at once."""
    from mstar.graph.base import loop_edges
    from mstar.graph.special_destinations import EMIT_TO_CLIENT

    edges = loop_edges(_token_loop(), "n")
    assert edges.inputs == {"t": EdgeKind.LOOP_BACK, "cond": EdgeKind.CONSTANT}
    assert edges.outputs == {
        ("t", "n"): EdgeKind.LOOP_BACK, ("t", EMIT_TO_CLIENT): EdgeKind.PER_ITERATION,
        ("latent", "vae"): EdgeKind.TERMINAL,
    }
    assert loop_edges(_token_loop(), "absent") is None


@pytest.mark.parametrize("accumulated", [True, False])
def test_a_loop_back_its_node_never_reads_is_an_item_only_of_what_the_loop_accumulates(accumulated):
    """It routes nothing to the next iteration, so a run carries none of it:
    an accumulated one keeps a column a replay, else it is no edge at all."""
    from mstar.graph.base import GraphEdge, GraphNode, Loop, loop_edges
    from mstar.graph.special_destinations import EMIT_TO_CLIENT
    from mstar.worker.step_repeats import loop_back, run_columns

    node = GraphNode(name="n", input_names={"h"}, outputs=[GraphEdge("n", "h"), GraphEdge("n", "p")])
    loop = Loop(section=node, max_iters=8, outputs=[], name="loop", accumulated_outputs=(
        [GraphEdge(EMIT_TO_CLIENT, "p")] if accumulated else []
    ))
    edges = loop_edges(loop, "n")
    assert loop_back(edges) == {"h"}
    assert run_columns(edges, inline=()) == ({"p"} if accumulated else set())
    assert (("p", "n") in edges.outputs) == accumulated


def _frames(streaming: bool):
    from mstar.graph.base import GraphEdge

    return GraphEdge("codec", "frame", is_streaming=streaming)


@pytest.mark.parametrize(("loop", "why"), [
    (_token_loop(), None),
    (_token_loop(_frames(streaming=True)), None),
    (_token_loop(_frames(streaming=False)), "graph runtime: 'frame' -> 'codec' is read once an iteration"),
    (_token_loop(streamed=("chunk",)), "node: its streamed input 'chunk' isn't an iteration input"),
], ids=["served", "streamed_out", "next_node", "streamed_in"])
def test_a_walk_repeats_only_if_a_run_serves_each_of_its_edges(loop, why):
    from mstar.graph.base import loop_edges
    from mstar.worker.step_repeats import edge_refusal

    refusal = edge_refusal(loop_edges(loop, "n"))
    assert refusal is None if why is None else refusal.startswith(why)


def test_an_input_another_node_of_the_loop_makes_each_iteration_is_not_constant():
    """BAGEL's split CFG: combine_cfg reads each branch's velocity once an
    iteration. A run of combine alone would replay on the first velocity while
    its count skipped the branch's iterations, so it doesn't repeat alone."""
    from mstar.graph.base import GraphEdge, GraphNode, Loop, Parallel, Sequential, loop_edges
    from mstar.worker.step_repeats import edge_refusal

    branch = GraphNode(name="b", input_names=["x"], outputs=[GraphEdge("c", "v")], enable_async_scheduling=False)
    combine = GraphNode(
        name="c", input_names=["v", "x"], outputs=[GraphEdge("b", "x"), GraphEdge("c", "x")],
        enable_async_scheduling=False,
    )
    edges = loop_edges(Loop(section=Sequential([Parallel([branch]), combine]), max_iters=8, outputs=[], name="l"), "c")
    assert edges.inputs == {"v": EdgeKind.PER_ITERATION, "x": EdgeKind.LOOP_BACK}
    assert edges.made_by == {"v": "b", "x": "c"} and edges.streamed == frozenset()
    assert edge_refusal(edges) == "graph runtime: its input 'v' is made by 'b' once an iteration"


@pytest.mark.parametrize(("parallel", "why"), [(False, None), (True, "TP: each rank's window")])
def test_under_tp_a_walk_reading_a_streamed_iteration_input_does_not_repeat(parallel, why):
    """Each rank's window is what reached it, so a follower would read other items."""
    from mstar.graph.base import loop_edges
    from mstar.worker.step_repeats import edge_refusal

    refusal = edge_refusal(loop_edges(_token_loop(streamed=("chunk",)), "n"), {"chunk"}, parallel)
    assert refusal is None if why is None else refusal.startswith(why)


def test_a_run_keeps_a_column_of_each_per_iteration_output_the_token_column_doesnt_carry():
    from mstar.graph.base import loop_edges
    from mstar.worker.step_repeats import run_columns

    edges = loop_edges(_token_loop(_frames(streaming=True)), "n")
    assert run_columns(edges, inline={"t"}) == {"frame"}
    assert run_columns(edges, inline=()) == {"frame", "t"}


class _LengthsFixedDecode(FlashInferDecodeWrapper):
    lengths_move = False  # fa3 or cuDNN


class _TensorCoreDecode(FlashInferDecodeWrapper):
    lengths_move = True


def _sampler(**attrs):
    from mstar.engine.resources.sampler.resource import SamplerResource

    sampler = SamplerResource.__new__(SamplerResource)
    for name, value in {
        "_history": None, **attrs,
    }.items():
        setattr(sampler, name, value)
    return sampler


def _attention(*wrappers):
    manager = FlashInferManager.__new__(FlashInferManager)
    bucket = BucketKey(graph_walk=WALK, bs=1, num_tokens=1)
    manager._cg_plan_states = {
        CGSlotKey(bucket=bucket, slot=slot, label="main"): cls.__new__(cls) for slot, cls in enumerate(wrappers)
    }
    return manager


@pytest.mark.parametrize(("resource", "refuses"), [
    (_sampler, False),
    # the repetition window moves in the graph each replay, like the offset
    (lambda: _sampler(_history=object()), False),
    (lambda: _attention(_TensorCoreDecode), False),
    (lambda: _attention(_LengthsFixedDecode, _LengthsFixedDecode), True),
], ids=["sampler", "history", "tensor_core", "lengths_fixed"])
def test_a_resource_refuses_repeats_at_load_for_what_it_knows_at_load(resource, refuses):
    assert (resource().repeat_refusal(WALK) is not None) is refuses


@pytest.mark.parametrize(("groups", "why"), [
    (1, None),  # either layout: its writes go through the same two tables
    (2, "a sliding-window group frees its front pages at commit"),
], ids=["one_group", "sliding_window"])
def test_a_kv_pool_says_what_keeps_it_off_the_tables(groups, why):
    kv = KVManager.__new__(KVManager)
    kv._groups = [object()] * groups
    kv._needs_remote_transfer = False
    refusal = kv.resident_refusal("kv", {"kv": object()})
    assert refusal is None if why is None else why in refusal


def test_mla_attention_is_resident_on_its_host_plan_and_says_why_it_doesnt_repeat():
    from mstar.engine.resources.attn.flashinfer_mla import FlashInferMLAManager

    mla = FlashInferMLAManager.__new__(FlashInferMLAManager)
    assert mla.resident_refusal("attn", {}) is None
    step = SimpleNamespace(host_planned=False)
    mla.resident_gather(step)
    assert step.host_planned
    assert "takes each row's length from its plan" in mla.repeat_refusal(WALK)
    assert mla.repeat_refusal(WALK, host_step=object()) is None


@pytest.mark.parametrize(("walk", "tensor_cores", "note"), [
    (WALK, True, "tensor-core decode for it"),
    ("other", False, None),  # a walk that doesn't ask keeps the config's choice
])
def test_a_walk_that_asks_for_repeats_gets_tensor_core_decode(monkeypatch, walk, tensor_cores, note):
    from mstar.engine.resources.attn import flashinfer

    built = {}
    monkeypatch.setattr(flashinfer, "FlashInferDecodeWrapper", lambda **kw: built.update(kw) or object())
    manager = FlashInferManager.__new__(FlashInferManager)
    manager._cg_plan_states, manager._splits, manager._decode_fmha = {}, {}, False
    manager._decode_tensor_cores, manager._repeat_walks = False, set()
    manager._wrapper_kv_kwargs, manager._kv_config, manager._group = {}, None, None
    manager._workspaces = SimpleNamespace(get=lambda label, slot: "workspace")
    monkeypatch.setattr(flashinfer, "captured_page_capacity", lambda *args: 8)
    manager.ask_repeats(WALK)
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk=walk, bs=4, num_tokens=4))
    manager._cg_wrapper(lease, "main", 4, False)
    assert built["use_tensor_cores"] is tensor_cores and manager.repeat_note(walk) == note


@pytest.mark.parametrize(("block", "why"), [
    *(({"n": {"max": limit}}, "integer of at least 1") for limit in (2.5, True, "4", 0)),
    ({"n": {"max": 2, "maxx": 3}}, "not settings"),
    ({"m": {"max": 2}}, "not nodes of the model"),
    ({"n": 4}, "must be a mapping"),
    ({"n": {"max": 2, "device_stop": "maybe"}}, "must be one of"),
    ({"n": {"device_stop": "skip"}}, "need max above 1"),
    ({"n": {"max": 2, "refill_at": 1.5}}, "must be in"),
    ({"n": {"max": 2, "device_stop": "skip", "refill_at": "0.5"}}, "must be in"),
    ({"n": {"max": 2, "refill_at": 0.5}}, "needs device_stop"),
    ({"n": {"max": 2, "device_launch": "on"}}, "not settings"),
    ({"n": {"max": 2, "ready_waits": "yes"}}, "must be a boolean"),
    ({"n": {"ready_waits": True}}, "need max above 1"),
    ({"n": {"auto": True}}, "need max above 1"),
    ({"n": {"max": 2, "eos_odds": 1.5}}, "must be in"),
    ({"n": {"max": 2, "stream_queue": {"depth": 0}}}, "must be {depth"),
])
def test_a_step_repeats_block_that_asks_for_nothing_real_is_refused(block, why):
    with pytest.raises(ValueError, match=why):
        parse_step_repeats(block, {"n"})


def test_auto_raises_the_run_at_once_and_lowers_it_after_eight_quiet_steps():
    """A gap is host time the run didn't hide: the next run is long enough to
    fill it, under ``max``. Quiet steps lower it one at a time, to see whether
    less still hides the host."""
    from mstar.worker.step_repeats import AutoRepeats

    auto = AutoRepeats(limit=8)
    ms = 1e-3
    # 1 ms of GPU an iteration, a 2.5 ms gap: the host's period is 3.5 iterations
    assert auto.observe(2.5 * ms, 1 * ms, 1) == 4
    # a longer gap than the max fills: capped
    assert auto.observe(40 * ms, 4 * ms, 4) == 8
    for _ in range(7):
        assert auto.observe(0.0, 8 * ms, 8) == 8
    assert auto.observe(0.0, 8 * ms, 8) == 7
    # a launch's own gap is no host time
    assert all(auto.observe(5e-6, 7 * ms, 7) == 7 for _ in range(7))
    assert auto.observe(5e-6, 7 * ms, 7) == 6


def test_a_run_whose_rows_near_a_stop_shortens_the_next_one():
    """Past ``eos_odds`` the next run takes the iterations such a stop takes on
    average; below it, the full length again."""
    repeats = StepRepeats({"n": RepeatConfig(max=8, eos_odds=0.2)}, {})
    key = ("n", WALK)
    for odds, cap in [(0.05, None), (0.3, 3), (0.6, 1), (0.1, None)]:
        repeats.observe_odds(key, odds)
        assert repeats._odds_caps.get(key) == cap


@pytest.mark.parametrize("capacity", [2, 4])
@pytest.mark.parametrize("preplan", [False, True])
def test_a_tp_count_is_checked_before_any_resource_reserves(capacity, preplan):
    """The last step of a chain needs no successor message to detect a
    follower's smaller capacity; a larger capacity never expands its count."""
    reserved = []

    class Limited(Resource):
        supports_preplan = True

        @classmethod
        def build(cls, spec, info):
            return cls()

        def max_repeats(self, step, ctx):
            return capacity

        def admit(self, step, ctx):
            reserved.append(ctx.repeats)
            return super().admit(step, ctx)

    runner = StepRunner({"limited": Limited()})
    ctx = StepContext(request_ids=[1], graph_walk=WALK, slot=0, capture=False, repeats=3)
    ctx.tp_repeats = 3
    step = SubmoduleStep(steps={"limited": SimpleNamespace(segments=())})
    step.set_ctx(ctx)
    admit = runner.pre_admit if preplan else runner.admit
    if capacity < 3:
        with pytest.raises(SystemExit, match="leader.*3"):
            admit(step)
        assert reserved == []
    else:
        assert admit(step).ok
        assert reserved == [3]
        assert ctx.repeats == 3


@pytest.mark.parametrize("fallback", ["lane", "uncaptured", "bucket"])
@pytest.mark.parametrize("authoritative", [False, True])
def test_an_eager_tp_fallback_cannot_change_a_broadcast_count(fallback, authoritative):
    from mstar.engine.engine import Engine, ExecutingBatch

    ctx = StepContext(request_ids=[1], graph_walk=WALK, slot=0, capture=False, repeats=3)
    ctx.tp_repeats = 3 if authoritative else None
    batch = ExecutingBatch(node_name="n", per_request_info={1: SimpleNamespace()}, step_context=ctx)
    engine = Engine.__new__(Engine)
    engine._walk_lanes = {("n", WALK): 1} if fallback == "lane" else {}
    engine._beside_chain = lambda node, walk: True
    engine._submodules = {"n": SimpleNamespace(
        num_slots=2, lease_slot=lambda: 0, lora_pools=(), resident=None,
        submodule=SimpleNamespace(cg_key_info=lambda *a, **kw: None),
        accelerator_graph_runner=None if fallback == "uncaptured" else SimpleNamespace(lease_slot=lambda **kw: None),
    )}
    if authoritative:
        with pytest.raises(SystemExit, match="leader.*3"):
            engine.reserve_replay_slot(batch)
    else:
        assert engine.reserve_replay_slot(batch) is None
        assert ctx.repeats == 1


def test_explicit_device_stop_requires_a_loop_in_the_whole_model():
    from mstar.graph.base import GraphNode, Loop

    node = GraphNode(name="n", input_names=[], outputs=[])
    config = {"n": {"max": 4, "device_stop": "skip"}}
    with pytest.raises(ValueError, match="device_stop.*Loop"):
        parse_step_repeats(config, {"n"}, walks={"prefill": node})
    # A prefill-only worker still accepts a node whose loop runs elsewhere.
    walks = {"prefill": node, "decode": Loop(section=node, max_iters=8, outputs=[])}
    assert parse_step_repeats(config, {"n"}, walks=walks)["n"].device_stop == "skip"


def test_each_node_asks_for_its_own_repeats():
    configs = parse_step_repeats({"thinker": {"max": 4}, "talker": {"max": 1}}, {"thinker", "talker", "codec"})
    assert configs == {"thinker": RepeatConfig(max=4), "talker": RepeatConfig()}


def test_a_step_with_no_plan_of_its_own_leaves_the_last_steps_wrappers_alone():
    """A bucket whose kernels read the tables plans nothing, so the wrappers the
    manager still holds are an earlier step's, on another slot or eager."""
    bucket = BucketKey(graph_walk=WALK, bs=1, num_tokens=1)
    manager = FlashInferManager.__new__(FlashInferManager)
    manager._current_plan_states = {"main": FlashInferDecodeWrapper.__new__(FlashInferDecodeWrapper)}
    assert manager.replay_columns(None, SimpleNamespace(
        slot_lease=SlotLease(slot=0, bucket=bucket), resident=SimpleNamespace(host_plan_free={bucket}),
    ), 4) == []


def test_cached_pages_are_room_for_a_step_that_repeats():
    """A long-running server's free pages sink to none while the prefix cache
    holds them, and those go on demand."""
    kv = KVManager.__new__(KVManager)
    kv._arena = SimpleNamespace(num_free=1)
    kv._groups = [SimpleNamespace(page_size=PAGE)]
    kv._index = SimpleNamespace(evictable=lambda: {5, 6})
    kv.config = SimpleNamespace(max_seq_len=10 * PAGE, request_pages=100)
    kv._streams = {}
    step = SimpleNamespace(segments=())
    assert kv.max_repeats(step, SimpleNamespace(request_ids=(1, 2, 3), repeats=2, resident=object())) == PAGE
    assert kv.max_repeats(step, SimpleNamespace(request_ids=(1, 2, 3, 4), repeats=2, resident=object())) == 1


def test_under_tp_a_steps_repeats_never_narrow_on_a_ranks_own_free_pages():
    """Each rank of a lockstep instance frees pages when its own teardown runs,
    so its free count is its own: narrowing on it would settle the ranks at
    different counts of the same collectives (review-r2 F3)."""
    kv = KVManager.__new__(KVManager)
    kv._arena = SimpleNamespace(num_free=1)
    kv._groups = [SimpleNamespace(page_size=PAGE)]
    kv._index = None
    kv.config = SimpleNamespace(max_seq_len=10 * PAGE, request_pages=100)
    kv._streams = {}
    kv._world_size = 2
    step = SimpleNamespace(segments=())
    assert kv.max_repeats(step, SimpleNamespace(request_ids=(1, 2, 3, 4), repeats=2, resident=object())) == PAGE


def test_a_steps_log_probs_narrow_it_only_by_its_own_rows():
    """Which requests ask log-probs, pool-wide, depends on when each rank
    ingested them; the step's own rows are the same on every rank (review-r2 F3)."""
    from mstar.engine.resources.sampler.resource import SamplerResource

    sampler = SamplerResource.__new__(SamplerResource)
    sampler._logprob_rids = {9}
    sampler._kept_rids_of = lambda step, ctx: None
    assert sampler.max_repeats(object(), SimpleNamespace(request_ids=(1, 2))) > 1
    assert sampler.max_repeats(object(), SimpleNamespace(request_ids=(1, 9))) == 1


def test_a_run_past_a_rows_cap_stays_inside_the_rows_table():
    """``past_cap`` lets a run outlast a row's count, and a row at the end of
    its context has nowhere to put those tokens."""
    kv = KVManager.__new__(KVManager)
    kv._arena = SimpleNamespace(num_free=8)
    kv._groups = [SimpleNamespace(page_size=PAGE)]
    kv._index = None
    kv.config = SimpleNamespace(max_seq_len=10 * PAGE, request_pages=100)
    kv._streams = {1: {"main": SimpleNamespace(stored_len=10 * PAGE - 5)}}
    step = SimpleNamespace(segments=[SimpleNamespace(request_id=1, label="main", span=1)])
    assert kv.max_repeats(step, SimpleNamespace(request_ids=(1,), repeats=3, resident=object())) == 2
    kv._streams[1]["main"].stored_len = 10 * PAGE
    assert kv.max_repeats(step, SimpleNamespace(request_ids=(1,), repeats=3, resident=object())) == 1
    # nor past max_seq_len inside its last page: a learned position table ends there (review-r2 F14)
    kv.config.max_seq_len = 10 * PAGE - 3
    kv._streams[1]["main"].stored_len = 10 * PAGE - 8
    assert kv.max_repeats(step, SimpleNamespace(request_ids=(1,), repeats=3, resident=object())) == 2
    # a full stream the step only reads goes nowhere: Whisper's cross KV holds max_seq_len frames
    kv._streams[1]["main"].stored_len = kv.config.max_seq_len
    step.segments[0].span = 0
    assert kv.max_repeats(step, SimpleNamespace(request_ids=(1,), repeats=3, resident=object())) > 3


def test_a_narrowed_run_says_which_resource_did_it(monkeypatch):
    """The walk's line reads "repeats on" from load, so without this a resource that
    holds every step to one iteration looks like a run (review-r6 F6). Each step
    counts; the log has one line for it, and one more once it comes down to 1."""
    answers = {"cross_kv": 1, "sampler": ANY_REPEATS}
    runner = StepRunner.__new__(StepRunner)
    runner._resources = {
        key: SimpleNamespace(max_repeats=lambda step, ctx, key=key: answers[key]) for key in answers
    }
    runner._keys_cache = {frozenset(answers): list(answers)}
    runner._node_order = {"decoder": list(answers)}
    runner._narrowed = set()
    logged = []
    monkeypatch.setattr(runner_mod.logger, "info", lambda *args: logged.append(args))
    monkeypatch.setattr(profiler, "PHASE_PERIOD", 1)
    samples = profiler.phase_buffer()
    samples.clear()

    def settle(asked):
        ctx = StepContext(request_ids=[], graph_walk=WALK, slot=0, capture=False, repeats=asked)
        runner._settle_repeats(SimpleNamespace(ctx=ctx, keys=answers.keys, get=lambda key: None))
        return ctx.repeats

    answers["cross_kv"] = 5
    assert [settle(8), settle(8)] == [5, 5]
    assert len(logged) == 1 and "cross_kv" in logged[0] and ["decoder"] in logged[0] and WALK in logged[0]
    answers["cross_kv"] = 1
    assert [settle(8), settle(8)] == [1, 1]
    assert len(logged) == 2
    assert samples[f"repeats.narrowed.{WALK}.cross_kv#"] == [5, 5, 1, 1]
    # each resource is named with what it allowed, not the step's settled count (review-r7 G6)
    answers["cross_kv"], answers["sampler"] = 5, 1
    assert settle(8) == 1
    assert samples[f"repeats.narrowed.{WALK}.cross_kv#"][-1] == 5
    assert samples[f"repeats.narrowed.{WALK}.sampler#"] == [1]
    assert logged[-1][-2:] == (1, 8) and "sampler" in logged[-1]
    assert settle(1) == 1 and len(samples[f"repeats.narrowed.{WALK}.cross_kv#"]) == 5, "nothing was asked"
    samples.clear()


def test_a_step_its_resources_held_to_one_iteration_moves_its_rows_by_one():
    """The worker asks for a ceiling, and the step's resources settle what it
    runs. Once it has admitted, its rows' Loops move by what it runs."""
    rows = {0: 0, 1: 0}
    batch = ScheduledBatch(node_name="n", graph_walk=WALK, request_to_worker_graph=dict(rows))
    ctx = SimpleNamespace(repeats=1)
    worker = Worker.__new__(Worker)
    worker._step_repeats = _repeats(8)
    worker._step_repeats.settle(lambda *key: None, lambda *key: None)
    worker._innermost_loop_by_node = {(WALK, "n"): "loop"}
    worker._graph_runtime = SimpleNamespace(loop_room=lambda *args: 100)
    speculation = SimpleNamespace(
        scheduled_batch=batch, node_batch=SimpleNamespace(step_context=ctx), loop_name="loop",
        continuing_rids={0, 1}, is_new_iter=True, is_same_node=True, consumed_streaming_edges={},
    )
    in_flight = SimpleNamespace(batch=SimpleNamespace(repeats=1))
    worker._stamp_repeats(speculation, in_flight, 1)
    assert ctx.repeats == 8
    ctx.repeats = 1  # what its resources allowed, at admit
    pending = SimpleNamespace(batch=batch, node_batch=SimpleNamespace(
        step_context=ctx, launch_started_event=SimpleNamespace(is_set=lambda: True), stream_windows={},
    ))
    worker._await_admit_settled(pending)
    assert batch.loop_strides(list(rows)) in (None, [1, 1])


@pytest.mark.parametrize("chained", [False, True])
def test_a_step_of_a_node_that_never_chains_repeats_where_the_scheduler_picks_it(chained):
    """A node that opts out of async scheduling has no speculated next step, so
    the step the scheduler picks is where it repeats: as far as its Loop's room
    with nothing of it in flight. A chained node's picked step stays one
    iteration (its chain's next step repeats)."""
    rows = {0: 0, 1: 0}
    edges = LoopEdges({"t": EdgeKind.LOOP_BACK}, {("t", "n"): EdgeKind.LOOP_BACK}, chained=chained)
    worker = Worker.__new__(Worker)
    worker._step_repeats = StepRepeats({"n": RepeatConfig(max=8)}, {("n", WALK): edges})
    worker._step_repeats.settle(lambda *key: None, lambda *key: None)
    worker._innermost_loop_by_node = {(WALK, "n"): "loop"}
    asked = []
    worker._graph_runtime = SimpleNamespace(loop_room=lambda *args: asked.append(args) or 3)
    batch = ScheduledBatch(node_name="n", graph_walk=WALK, request_to_worker_graph=dict(rows))
    assert worker._unchained_repeats(batch) == (1 if chained else 3)
    assert asked == ([] if chained else [("n", WALK, "loop", [0, 1], 0, False)])


def test_a_step_cut_to_one_before_admit_says_why(monkeypatch):
    """Admit can't narrow an ask of 1, so the walk's "repeats on" line hid every cut
    the worker made itself: pi0.5's Loop with no room was one. Counted per step,
    logged once per reason."""
    from mstar.worker import step_repeats as step_repeats_mod

    edges = LoopEdges({"t": EdgeKind.LOOP_BACK}, {("t", "n"): EdgeKind.LOOP_BACK}, chained=False)
    worker = Worker.__new__(Worker)
    worker._step_repeats = StepRepeats({"n": RepeatConfig(max=8)}, {("n", WALK): edges})
    worker._step_repeats.settle(lambda *key: None, lambda *key: None)
    worker._innermost_loop_by_node = {(WALK, "n"): "loop"}
    room = [0]
    worker._graph_runtime = SimpleNamespace(loop_room=lambda *args: room[0])
    logged = []
    monkeypatch.setattr(step_repeats_mod.logger, "info", lambda *args: logged.append(args))
    monkeypatch.setattr(profiler, "PHASE_PERIOD", 1)
    samples = profiler.phase_buffer()
    samples.clear()
    batch = ScheduledBatch(node_name="n", graph_walk=WALK, request_to_worker_graph={0: 0})

    assert [worker._unchained_repeats(batch) for _ in range(2)] == [1, 1]
    room[0] = 5
    assert worker._unchained_repeats(batch) == 5
    worker._innermost_loop_by_node = {}
    assert worker._unchained_repeats(batch) == 1
    assert samples[f"repeats.asked.{WALK}#"] == [1, 1, 5, 1]
    assert [args[-1] for args in logged] == ["its Loop has room for 0", "the runtime holds no Loop of it here"]
    samples.clear()


def test_an_unnamed_loop_is_asked_about_by_the_name_the_runtime_holds():
    """A Loop without a name draws a fresh one each time the model builds its walks,
    and the runtime holds the worker graphs' build. pi0.5's action loop is one: its
    unchained steps asked about a name the runtime didn't know, got no room, and ran
    one iteration under a "repeats on" line."""
    from mstar.graph.base import GraphEdge, GraphNode, Loop
    from mstar.model.base import WorkerGraph
    from mstar.worker.step_repeats import walk_edges

    def build():
        node = GraphNode(
            name="n", input_names=["t"], outputs=[GraphEdge(next_node="n", name="t")], enable_async_scheduling=False,
        )
        return {WALK: Loop(section=node, max_iters=10, outputs=[])}

    held = WorkerGraph(section=build()[WALK], graph_walks={WALK})
    worker = Worker.__new__(Worker)
    worker._innermost_loop_by_node = Worker._build_innermost_loop_cache(
        (walk, wg.section) for wg in [held] for walk in wg.graph_walks
    )
    worker._step_repeats = StepRepeats({"n": RepeatConfig(max=8)}, walk_edges(build(), [("n", WALK)]))
    worker._step_repeats.settle(lambda *key: None, lambda *key: None)
    names = {held.section.name: 10}
    worker._graph_runtime = SimpleNamespace(loop_room=lambda node, walk, loop, *rest: names.get(loop, 0))
    batch = ScheduledBatch(node_name="n", graph_walk=WALK, request_to_worker_graph={0: 0})
    assert worker._unchained_repeats(batch) == 8


@pytest.mark.parametrize("state, streamed, why", [
    (dict(parallel_nodes={"n"}), frozenset(), "TP group"),
    ({}, frozenset({"chunk"}), "stream window"),
    (dict(_ring_signals={"n": {"codes"}}), frozenset(), "ring"),
    (dict(_signal_consumer={"t": "codec"}, _walk_drivers={"codec": object()}), frozenset(), "consumer"),
])
def test_a_node_that_never_chains_says_why_it_cant_repeat(state, streamed, why):
    """Its ask has only the Loop's room to go on, so a walk that needs a chain's
    other caps says so on its line rather than overrunning them, or reading
    "on" while it runs one iteration (review-r2 F16)."""
    worker = Worker.__new__(Worker)
    worker.parallel_nodes, worker._ring_signals, worker._signal_consumer, worker._walk_drivers = set(), {}, {}, {}
    worker.__dict__.update(state)
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: SimpleNamespace(
        streamed_iteration_inputs=lambda node: streamed,
    ))
    worker._graph_runtime = SimpleNamespace(get_output_signals=lambda node, walk: ["t"])
    assert why in worker._unchained_refusal("n", WALK)
    worker.__dict__.update(
        parallel_nodes=set(), _ring_signals={}, _signal_consumer={}, _walk_drivers={},
    )
    streamed = frozenset()
    assert worker._unchained_refusal("n", WALK) is None


def test_a_row_declaring_another_stream_than_its_walk_fails_its_step_not_the_worker():
    """A resident step, and each replay of one, advances every row on the walk's
    streams; a row whose declared label differs would be read and written on
    the wrong stream."""
    live, _, planner = _setup({1: 3, 2: 3})
    _captured_step(live, planner, [1], slot=0, bs=2)
    padded = (1, 2)
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk=WALK, bs=2, num_tokens=2))
    ctx = StepContext(request_ids=padded, graph_walk=WALK, slot=0, capture=False, slot_lease=lease, repeats=4)
    step = SubmoduleStep(steps={}, segments=[Segment(1, "main", 1), Segment(2, "aux", 1)])
    with pytest.raises(ValueError, match="row 2 declares 'aux'"):
        planner.plan(step, ctx)


def test_a_row_a_host_step_moved_does_not_rewind_on_each_replay():
    """A one-token host step leaves a row's length to be refreshed. A run after
    it refreshes it once, not at the head of every replay."""
    live, ref, planner = _setup({1: 5})
    _captured_step(live, planner, [1], slot=0, bs=1)
    _captured_step(ref, None, [1], slot=0, bs=1)
    step, ctx = _host_step(live, [1], [1], walk=WALK)
    planner.commit(step, ctx)
    _host_step(ref, [1], [1], walk=WALK)
    a = _captured_step(live, planner, [1], slot=1, bs=1, repeats=3)
    for i, replay in enumerate(a.replays):
        b = _captured_step(ref, None, [1], slot=(1 + i) % 2, bs=1)
        assert (replay.offsets[:1], replay.positions[:1]) == (b.offsets[:1], b.positions[:1])


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
    planner = ResidentPlanner("n", _walk("kv"), {"kv": kv}, device, max_bs=2)
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
    planner = _planner(res, max_bs=8)
    assert planner.max_slots == 17
    assert len(planner._free) == 16
    assert TRASH_SLOT not in planner._free


def test_a_flush_of_the_largest_step_stages_without_growing(monkeypatch):
    res = _resources(max_num_pages=64)
    planner = _planner(res, max_bs=4)
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
    planner = ResidentPlanner("n", _walk("kv"), {"kv": kv}, device, max_bs=4)
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
    ResidentPlanner("n", _walk("kv"), {"kv": kv}, CPU, max_bs=1)
    assert kv._resident.block_table.width == kv.config.request_pages == 63


def test_a_row_is_as_wide_as_the_context_when_the_pool_is_wider():
    kv = _kv(max_seq_len=4 * 10 + 1, max_num_pages=64)
    ResidentPlanner("n", _walk("kv"), {"kv": kv}, CPU, max_bs=1)
    assert kv._resident.block_table.width == 11


def test_a_stream_wider_than_a_row_raises():
    kv = _kv(max_seq_len=8, max_num_pages=64)
    planner = ResidentPlanner("n", _walk("kv"), {"kv": kv}, CPU, max_bs=1)
    tables = kv._resident
    assert tables.block_table.width == 2
    writes = resident_mod.ResidentWrites({"kv": resident_mod.ResidentStream("main", reads=False)})
    with pytest.raises(ValueError, match="fit a block table row"):
        tables.row(9, [1, 2, 3], 1, [0] * planner.narrow_cols, writes)
    plan = SimpleNamespace(
        slots=torch.tensor([1]).numpy(), rids=(7,), taken=set(), writes=writes, crossed={}, moved=set(),
    )
    tables.host_len[1], tables.host_cols[1] = 8, 2
    with pytest.raises(ValueError, match="fit a block table row"):
        tables.stage_step(plan, lambda rid: 8, lambda rid: [1, 2, 3], planner.rewrite)
    assert not plan.writes.cells


# ---- what a raise in the plan leaves ----

def _preplan_worker(engine):
    worker = Worker.__new__(Worker)
    worker.worker_id = "w"
    worker.enable_nvtx = False
    worker._phase_period = 0
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    return worker


def _preplan_engine(res, planner):
    from mstar.engine.engine import Engine
    from mstar.engine.resources.runner import StepRunner

    resources = {k: res[k] for k in ("kv", "pool", "rope")}
    engine = Engine.__new__(Engine)
    engine._enable_nvtx = False
    engine._device = CPU
    engine._pre_plan_done, engine._plan_peers = {}, {"n": ("n",)}
    engine._runner = StepRunner(resources)
    runner = SimpleNamespace(
        plan_stream=lambda: None, pad_inputs=lambda lease, inputs, uniform=None: inputs,
        step_ids=lambda lease, rids: list(rids), release=lambda lease, rows: None,
    )

    def declare(graph_walk, request_ids, inputs, **kwargs):
        del graph_walk, inputs, kwargs
        segments = [Segment(rid, "main", 1) for rid in request_ids]
        keys = {"kv": KVStep(), "pool": RecurrentStep(), "rope": PositionStep()}
        return SubmoduleStep(segments=segments, steps=keys)

    engine._submodules = {"n": SimpleNamespace(
        submodule=SimpleNamespace(declare_step=declare, step_is_input_free=lambda walk: False),
        accelerator_graph_runner=runner, piecewise_runners={}, resident=planner,
    )}
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
def test_a_raise_in_the_preplan_leaves_the_reservation_for_exec(monkeypatch, raiser):
    """`pre_admit` reserved a crossing page and something raises before the stage
    is done. The plan thread's wrapper resets the pre-plan: the page stays with the
    stream, the in-flight mark goes, and exec's admit finds the page and takes no
    other. The same for a raise from the planner and from the runner's own pre-plan."""
    res = _resources()
    planner = _planner(res, max_bs=4)
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
    assert _preplan_worker(engine)._preplan_spec(None, SimpleNamespace(node_batch=batch)) is False
    monkeypatch.setattr(target, name, real)

    assert batch.preplan_event is None and batch.preplanned_rids is None and batch.step is None
    assert engine._pre_plan_done["n"].recorded
    assert engine._runner._pre_admitted is None and not engine._runner.staged
    assert not stream.step_in_flight
    assert len(stream.page_indices) == 2
    pages, free = list(stream.page_indices), kv._arena.num_free
    kv.assert_pages_conserved()

    # exec re-drives it: the page it finds is the one pre_admit took
    s = _captured_step(res, planner, [1], slot=1, bs=1)
    assert stream.page_indices == pages and kv._arena.num_free == free
    assert s.pages[0] == pages[1]
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
    from mstar.engine.engine import Engine

    engine = Engine.__new__(Engine)
    engine._tp_follower_nodes = set()
    engine._resident_delta = {}
    engine._resources = resources
    engine._submodules = {"n": SimpleNamespace(resources=resources, resident=planner)}
    engine._graph_runtime = SimpleNamespace(get_rid_string=str, get_rid_handle=int)
    return engine


def _refused_worker(engine, line):
    """The real handler and the real victim search over a real engine."""
    worker = Worker.__new__(Worker)
    worker.parallel_nodes = set()
    worker.parallel_leader_nodes = set()
    worker._hold_logged = {}
    worker.pushed = []
    worker.held = []
    worker._apply_removes_whose_step_landed = lambda: None
    worker._apply_pending_removes_safe_to_drop = lambda rids: None
    worker._graph_runtime = SimpleNamespace(push_back_node=lambda node, rids, graphs: worker.pushed.append(rids))
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    worker.eviction_policy = SimpleNamespace(value="last_in_line")
    worker.scheduler = SimpleNamespace(
        arrival=list(line), hold_requests=worker.held.extend, could_refill=lambda *a: False,
        last_in_line=lambda node, walk, rids: max(rids, key=line.index),
    )
    return worker


def test_a_refused_admit_offloads_a_victim_and_its_reload_is_written_whole():
    """Through the worker's refusal path: the KV pool refuses, `_handle_allocation_failure`
    offloads the row furthest from a step, its slot goes with it, the retry runs, and the
    victim's reload is written whole at its own lengths."""
    from mstar.engine.resources.runner import StepRunner

    # seven pages to hand out, all held: each of 1 and 2 has room for one more token
    live = _resources(max_num_pages=8)
    owned = {k: live[k] for k in ("kv", "pool", "rope")}
    planner = ResidentPlanner("n", _walk(*owned), owned, CPU, max_bs=4)
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
    batch = SimpleNamespace(node_name="n", graph_walk=WALK, request_to_worker_graph={1: "g", 2: "g"})
    worker._handle_allocation_failure(batch, SimpleNamespace(
        node_name="n", admit_error=outcome.reason, failed_resource=outcome.failed_resource,
    ))
    assert worker.pushed == [[1, 2]] and worker.held == [3]
    assert engine.is_offloaded("n", 3)
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
    planner = _planner(live, max_bs=4)
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
    batch = SimpleNamespace(node_name="n", graph_walk="prefill", request_to_worker_graph={2: "g"})
    worker._handle_allocation_failure(batch, SimpleNamespace(
        node_name="n", admit_error=outcome.reason, failed_resource="pool",
    ))
    assert worker.pushed == [[2]]
    assert worker.held == [2]
    assert not engine.is_offloaded("n", 1)


def test_a_follower_replaying_an_offload_gives_the_slot_up_too():
    live, ref, planner = _setup({1: 9})
    del ref
    _captured_step(live, planner, [1], slot=0, bs=1)
    _HostPagesKV(live["kv"])
    engine = _offload_engine({k: live[k] for k in ("kv", "pool", "rope")}, planner)
    engine._tp_follower_nodes = {"n"}
    delta = OffloadDelta.new()
    delta.add_offloaded("1")
    assert engine.apply_resident_delta("n", delta)
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


def _decoder_setup(prompts: dict[int, tuple[int, int]], positions: str = "kv", context_span: int = 0):
    """Each request's prompt and context prefilled on resident resources and a
    host-planned reference. ``prompts`` is rid -> (prompt tokens, context tokens);
    decode steps append ``context_span`` tokens a row to the context."""
    live, ref = _decoder(positions), _decoder(positions)
    walk = _walk("kv", "ctx", "rope", reads=() if context_span else ("ctx",))
    planner = ResidentPlanner("n", walk, {k: live[k] for k in ("kv", "ctx", "rope")}, CPU, max_bs=4)
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
    live, ref, planner = _decoder_setup({1: (3, 11), 2: (9, 2)}, positions=positions, context_span=1)
    for t in range(8):
        a = _decoder_step(live, planner, [1, 2], slot=t % 2, bs=4, context_span=1)
        b = _decoder_step(ref, None, [1, 2], slot=t % 2, bs=4, context_span=1)
        assert a.positions[:2] == b.positions[:2]
        for key in ("kv", "ctx"):
            assert getattr(a, f"{key}_pages")[:2] == getattr(b, f"{key}_pages")[:2]
            assert getattr(a, f"{key}_offsets")[:2] == getattr(b, f"{key}_offsets")[:2]
    assert planner.buffers(BucketKey(WALK, 4, 4), 0).cells.shape == (2, 2 * 4 * 2)


def test_positions_over_a_pool_the_step_only_reads_are_refused_at_load():
    res = _decoder("ctx")
    steps = {WALK: SubmoduleStep(
        steps={"ctx": KVStep(segments=(Segment(1, "main", 0),))}, segments=[Segment(1, "main", 1)],
    )}
    planner, refused = build_planner("n", steps, res, {}, CPU, lambda walks: 4)
    assert planner is None and "advance with the stream" in refused[WALK]


def test_a_walk_takes_a_label_for_each_resource():
    rows = [Segment(1, "self", 1)]
    step = SubmoduleStep(
        steps={"kv": KVStep(), "ctx": KVStep(segments=(Segment(1, "context", 0),)), "rope": PositionStep()},
        segments=rows,
    )
    streams = {key: tuple(s) for key, s in resident_mod.declared_streams(("kv", "ctx", "rope"), step).items()}
    assert streams == {"kv": ("self", False), "ctx": ("context", True), "rope": ("self", False)}


def test_a_walk_with_two_labels_on_one_resource_is_refused():
    rows = [Segment(1, "main", 1)]
    two = (Segment(1, "main", 1), Segment(1, "aux", 1))
    step = SubmoduleStep(steps={"kv": KVStep(segments=two), "ctx": KVStep(), "rope": PositionStep()}, segments=rows)
    assert "resource 'kv' declares 2 segments over labels" in resident_mod.step_refusal(("kv", "ctx", "rope"), step)
    # a resource with no segments has no stream: the line names the one that refuses
    bagel = SubmoduleStep(steps={"kv": step.steps["kv"], "sampler": SimpleNamespace(segments=())})
    assert "resource 'kv' declares" in resident_mod.step_refusal(("sampler", "kv"), bagel)
    mixed = (Segment(1, "main", 1), Segment(2, "main", 0))
    step = SubmoduleStep(steps={"kv": KVStep(), "ctx": KVStep(segments=mixed), "rope": PositionStep()}, segments=rows)
    assert "resource 'ctx'" in resident_mod.step_refusal(("kv", "ctx", "rope"), step)


def test_a_bucket_that_declares_other_streams_than_its_walk_stops_the_worker():
    _, _, planner = _decoder_setup({})
    rows = [Segment(1, "main", 1)]
    step = SubmoduleStep(steps={"kv": KVStep(), "ctx": KVStep(), "rope": PositionStep()}, segments=rows)
    with pytest.raises(SystemExit, match="aren't its walk's"):
        planner.capture_step(BucketKey(WALK, 1, 1), 0, step)


class _Fmha:
    """A captured fmha decode wrapper's gather, recorded."""

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
    live, _, planner = _decoder_setup({}, context_span=1)
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


class _Attention(Resource):
    @classmethod
    def build(cls, spec, info):
        return cls()

    def __init__(self, reads_tables: bool):
        self._reads = reads_tables

    def resident_refusal(self, key, streams):
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
    planner = ResidentPlanner("n", _walk(*resources), resources, CPU, max_bs=2)
    step = SubmoduleStep(steps={}, segments=[Segment(1, "main", 1)])
    planner.gather(planner.capture_step(BucketKey(WALK, 1, 1), 0, step))
    assert (BucketKey(WALK, 1, 1) in planner.host_plan_free) is free


def test_cross_attention_over_its_own_context_pool_runs_on_the_planner():
    from mstar.engine.resources.attn.cross import FlashInferCrossManager

    manager = FlashInferCrossManager.__new__(FlashInferCrossManager)
    manager._kv_cache_name, manager._query_kv_cache_name = "ctx", "kv"
    assert manager.resident_refusal("x", {}) is None
    manager._query_kv_cache_name = "ctx"
    assert "second stream" in manager.resident_refusal("x", {})
    step = SimpleNamespace(reads_tables=False, host_planned=False)
    FlashInferCrossManager.resident_gather(manager, step)
    assert step.host_planned and not step.reads_tables


# ---- refusals ----

class _Spare(Resource):
    @classmethod
    def build(cls, spec, info):
        return cls()


_ONE_TOKEN = SubmoduleStep(steps={}, segments=[Segment(1, "main", 1)])


def _build(resources, nodes=None, step=_ONE_TOKEN):
    """The planner for WALK on its declared ``step``, and why not."""
    planner, refused = build_planner("n", {WALK: step}, resources, nodes or {}, CPU, lambda walks: 4)
    return planner, refused.get(WALK, "")


def test_every_resident_resource_passes():
    res = _resources()
    planner, why = _build({k: res[k] for k in ("kv", "pool", "rope")})
    assert planner is not None and why == ""


@pytest.mark.parametrize(("resources", "step", "why"), [
    ({}, _ONE_TOKEN, "no resource of it keeps a table"),
    ({"kv": None}, None, "declares no step"),
], ids=["no_table_owner", "no_step"])
def test_a_looped_node_with_no_step_or_no_table_owner_is_refused_at_load(resources, step, why):
    if "kv" in resources:
        resources = {"kv": _resources()["kv"]}
    planner, refused = _build(resources, step=step)
    assert planner is None and why in refused


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


def test_the_mla_layout_takes_the_tables():
    """GLM-5.2's latent pages take ``write_kv``, through the two buffers the tables fill."""
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=512, max_num_pages=64, page_size=PAGE,
            layout=KVLayout.MLA, kv_lora_rank=4, qk_rope_head_dim=4,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None, device=CPU, dtype=torch.float32,
    )
    planner, why = _build({"kv": kv})
    assert planner is not None and why == ""


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
    assert "doesn't advance" in rope.resident_refusal("rope", {})


def test_a_pool_without_a_sink_refuses():
    pool = RecurrentStatePool(device=CPU, config=RecurrentStateConfig(
        num_layers=1, max_slots=4, blocks={"state": RecurrentBlockConfig(shape=(2,), dtype=torch.float32)},
        disable_sink_slot=True,
    ))
    assert "sink" in pool.resident_refusal("pool", {})


@pytest.mark.parametrize("module,name", [
    ("mstar.engine.resources.attn.cross", "CrossAttentionManager"),
    ("mstar.engine.resources.attn.dense", "DenseAttentionManager"),
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
    assert "sliding window" in manager.resident_refusal("attn", {})
    manager._window_left = -1
    assert manager.resident_refusal("attn", {}) is None


@pytest.mark.parametrize("field,value", [
    ("pre_forks", (("a", "b"),)), ("post_forks", (("a", "b"),)), ("sink", frozenset({(1, "main")})), ("commit", False),
])
def test_a_walk_declaring_forks_or_a_sink_is_refused_at_load(field, value):
    res = _resources()
    segments = [Segment(1, "main", 1)]
    kv = KVStep(segments=tuple(segments), **{field: value})
    planner, why = _build({"kv": res["kv"]}, step=SubmoduleStep(steps={"kv": kv}, segments=segments))
    assert planner is None and field in why


@pytest.mark.parametrize("segments", [
    [Segment(1, "main", 2)], [Segment(1, "a", 1), Segment(1, "b", 1)],
], ids=["two_tokens", "two_labels"])
def test_a_walk_that_isnt_one_token_a_row_in_one_stream_is_refused_at_load(segments):
    res = _resources()
    planner, why = _build({"kv": res["kv"]}, step=SubmoduleStep(steps={}, segments=segments))
    assert planner is None and "not one token a row" in why


# ---- the engine ----

def _capture_runner(max_bs=4, refusal=None, step=_ONE_TOKEN):
    bound = {}
    return SimpleNamespace(
        resident_refusal=lambda walk: refusal, template_step=lambda walk: step,
        keyed_template_steps=lambda walk: {None: step}, prepare_template=lambda walk: None,
        max_capture_bs=lambda walks: max_bs, bind_resident=lambda planner: bound.update(p=planner), bound=bound,
    )


def _engine(resources, device=CPU):
    from mstar.engine.engine import Engine
    from mstar.engine.resources.runner import StepRunner

    engine = Engine.__new__(Engine)
    engine._device = device
    engine._runner = StepRunner(resources)
    engine._resource_nodes = {key: frozenset({"n"}) for key in resources}
    engine._submodules = {"n": SimpleNamespace(
        submodule=SimpleNamespace(), resources=resources, resident=None, resident_refusals={},
    )}
    return engine, engine._submodules["n"]


def test_the_engine_puts_a_loop_walk_on_resident_tables_with_nothing_declared():
    res = _resources()
    engine, mgmt = _engine({k: res[k] for k in ("kv", "pool", "rope")})
    runner = _capture_runner()
    engine._build_resident_planner("n", runner, {WALK})
    assert isinstance(mgmt.resident, ResidentPlanner)
    assert runner.bound["p"] is mgmt.resident
    assert mgmt.resident.max_slots == 9
    assert engine.resident_refusal("n", WALK) is None
    assert mgmt.resident.covers(BucketKey(WALK, 2, 2))
    assert not mgmt.resident.covers(BucketKey("prefill_text", 2, 64))


@pytest.mark.parametrize(("knob", "runner", "resources", "why"), [
    ("off", _capture_runner(), ("kv",), "not asked for"),
    ("on", _capture_runner(refusal="it has no batched capture"), ("kv",), "runner: it has no batched capture"),
    ("on", _capture_runner(), ("kv", "other"), "_Spare keeps no resident table"),
], ids=["knob", "capture", "resource"])
def test_a_refused_walk_stays_on_the_host_and_says_why(monkeypatch, knob, runner, resources, why):
    monkeypatch.setenv("MSTAR_RESIDENT_PLANNING", knob)
    res = {**_resources(), "other": _Spare()}
    engine, mgmt = _engine({k: res[k] for k in resources})
    engine._build_resident_planner("n", runner, {WALK})
    assert mgmt.resident is None and res["kv"]._resident is None
    assert why in engine.resident_refusal("n", WALK)


def _probe_runner(preprocess, freed: list):
    """A runner whose one batched capture of WALK preprocesses with ``preprocess``."""
    from mstar.engine.accelerator_graph_config import AcceleratorGraphConfigType
    from mstar.engine.resources import CGSlotSpec

    config = SimpleNamespace(
        capture_graph_walk=WALK, replay_graph_walks=[WALK], capture_batch_sizes=[2],
        get_config_type=lambda: AcceleratorGraphConfigType.BASIC_BATCHED, get_node_inputs=lambda bs, n: [None] * bs,
    )
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._capture_configs = [config]
    runner._prepared_slot_specs = [CGSlotSpec(bucket=BucketKey(WALK, 2, 2), slot=0, config=config, config_idx=0)]
    runner._graph_backend = SimpleNamespace(is_available=lambda: True)
    runner._resources, runner._resident = {}, None
    runner._dummy_rows = SimpleNamespace(
        ensure=lambda key, bs: [-1, -2], reset=lambda rids, free=False: freed.append((rids, free)),
    )
    runner._step_runner = SimpleNamespace(admit=lambda step: SimpleNamespace(ok=True), plan=lambda step: None)
    runner._submodule = SimpleNamespace(
        declare_step=lambda **kwargs: SubmoduleStep(steps={}, segments=[Segment(r, "main", 1) for r in (-1, -2)]),
        preprocess=lambda graph_walk, engine_inputs, inputs: preprocess(),
    )
    return runner


@pytest.mark.parametrize("reads", [False, True])
def test_a_walk_whose_captured_preprocess_reads_what_the_graph_fills_is_refused_naming_it(reads):
    """The promise a model can't state in code is checked on the template its
    capture runs, at load, whose padding rows the probe leaves at rest."""
    res = _resources()
    freed = []
    runner = _probe_runner(lambda: res["rope"].pos_ids("main") if reads else None, freed)
    engine, mgmt = _engine({k: res[k] for k in ("kv", "pool", "rope")})
    engine._build_resident_planner("n", runner, {WALK})
    assert freed == [([-1, -2], True)]
    if reads:
        assert mgmt.resident is None and "rope.pos_ids" in engine.resident_refusal("n", WALK)
    else:
        assert engine.resident_refusal("n", WALK) is None
    res["rope"].pos_ids("main")  # outside a resident preprocess, a read is the forward's


@pytest.mark.parametrize("reader", ["preprocess", "forward"])
def test_a_resident_preprocess_that_reads_what_the_graph_fills_stops_the_worker(reader):
    """A branch the load-time template didn't take: the forward reads the same
    buffer legitimately, after the graph has filled it."""
    from mstar.engine.engine import Engine

    rope = _resources()["rope"]

    def read(where):
        if where == reader:
            rope.pos_ids("main")
        return {}

    engine = Engine.__new__(Engine)
    engine._enable_nvtx = engine._enable_profile = False
    engine._forward = lambda *args: read("forward")
    submodule = SimpleNamespace(preprocess=lambda walk, engine_inputs, inputs: read("preprocess"))
    mgmt = SimpleNamespace(
        accelerator_graph_runner=None, submodule=submodule, piecewise_runners={}, fp32_precision=None, resources={},
        loopback_rows=None,
    )
    ctx = StepContext(request_ids=(1,), graph_walk=WALK, slot=0, capture=False)
    ctx.resident = object()
    batch = SimpleNamespace(node_name="n", per_request_input_metadata={}, launch_started_event=None)
    drive = partial(engine._drive_step, batch, mgmt, [1], [], {}, ctx, None, True, None, False)
    if reader == "preprocess":
        with pytest.raises(ValueError, match="rope.pos_ids"):
            drive()
    else:
        assert drive() == ({}, None)


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

    return submodules, submodules.LLMSubmodule


def test_qwen35s_resident_set_is_derived_as_its_decode_walk():
    """Nothing declared: the LLM node's Loop walks are decode and the chunked
    text prefill, and only decode has a batched capture."""
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    model = object.__new__(Qwen3_5DenseModel)
    model.config = SimpleNamespace(max_position_embeddings=4096)
    walks = model.get_graph_walk_graphs().items()
    looped = {walk for walk, node in Worker._build_innermost_loop_cache(walks) if node == "LLM"}
    _, llm = _qwen35_llm()
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner._capture_configs = object.__new__(llm).get_accelerator_graph_configs(CPU)
    runner._graph_backend = SimpleNamespace(is_available=lambda: True)
    runner._prepared_slot_specs = [SimpleNamespace(bucket=BucketKey(walk, 1, 1)) for walk in looped]
    assert {walk for walk in looped if runner.resident_refusal(walk) is None} == {"decode"}


@pytest.mark.parametrize("knob", [False, True])
def test_qwen35_decode_builds_its_rope_in_the_graph_on_every_config(monkeypatch, knob):
    submodules, llm = _qwen35_llm()
    monkeypatch.setattr(submodules, "ingraph_decode_rope", lambda: knob)
    assert llm._rope_in_graph(SimpleNamespace(captured=True), "decode")
    assert not llm._rope_in_graph(SimpleNamespace(captured=False), "decode")


@pytest.mark.parametrize("rows", ["loopback", "own_ids"])
def test_qwen35_captured_decode_preprocess_reads_nothing_the_tables_fill(rows):
    """The graph fills positions and write locations, after `preprocess`."""
    submodules, llm = _qwen35_llm()
    token = torch.tensor([5, 6])
    inputs = [
        SimpleNamespace(input_embeds=None, input_ids=None if rows == "loopback" else token[:1], custom_pos_ids=None,
                        input_seq_len=0 if rows == "loopback" else 1)
        for _ in range(2)
    ]
    stub = SimpleNamespace(
        _rope_in_graph=llm._rope_in_graph, _position_ids_3d=lambda inputs: pytest.fail("read positions"),
        loopback_tokens=lambda engine_inputs: token,
    )
    engine_inputs = SimpleNamespace(captured=True, request_ids=[1, 2])
    out = llm.preprocess(stub, "decode", engine_inputs, inputs)
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

def test_the_knob_takes_on_off_and_auto():
    assert knobs.check({"resident_planning": "on"}) == {"resident_planning": "on"}
    # yaml reads a bare on/off as a bool
    assert knobs.check({"resident_planning": True}) == {"resident_planning": "on"}
    assert knobs.check({"resident_planning": False}) == {"resident_planning": "off"}
    assert knobs.check({"resident_planning": "auto"}) == {"resident_planning": "auto"}
    with pytest.raises(ValueError, match="on, off or auto"):
        knobs.check({"resident_planning": "maybe"})


@pytest.mark.parametrize(("setting", "by_default", "asked"), [
    (None, True, True), (None, False, False), ("on", False, True), ("off", True, False),
])
def test_a_node_resident_by_default_is_resident_with_no_knob(monkeypatch, setting, by_default, asked):
    """#1192 ran Qwen3.5's decode resident by default: a user who sets nothing
    keeps it (``resident_by_default``), and the knob still turns it on or off."""
    from mstar.engine.engine import Engine
    from mstar.model.qwen3_5.submodules import LLMSubmodule

    assert LLMSubmodule.resident_by_default
    monkeypatch.delenv("MSTAR_RESIDENT_PLANNING", raising=False)
    knobs.configure(None if setting is None else {"resident_planning": setting})
    engine = Engine.__new__(Engine)
    mgmt = SimpleNamespace(submodule=SimpleNamespace(resident_by_default=by_default), resident_refusals={})
    engine._submodules = {"n": mgmt}
    runner = SimpleNamespace(resident_refusal=lambda walk: "probed")
    try:
        engine._build_resident_planner("n", runner, {"decode"})
    finally:
        knobs.configure(None)
    assert mgmt.resident_refusals["decode"] == ("runner: probed" if asked else NOT_ASKED)


@pytest.mark.parametrize(("setting", "rows"), [(None, {"keeps"}), ("on", {"keeps", "other"}), ("off", set())])
def test_a_node_that_keeps_resident_planning_keeps_its_rows_under_auto_as_under_on(monkeypatch, setting, rows):
    """The default (``auto``) asks for a ``resident_by_default`` node what ``on``
    asks, so a config needn't set the knob (review-r2 F30): rows included."""
    from mstar.worker.engine_manager import _row_nodes

    monkeypatch.delenv("MSTAR_RESIDENT_PLANNING", raising=False)
    knobs.configure(None if setting is None else {"resident_planning": setting})
    loop = SimpleNamespace(get_nodes=lambda: {"keeps": None, "other": None})
    model = SimpleNamespace(get_graph_walk_graphs=lambda: {"w": SimpleNamespace(get_loops=lambda: {"l": loop})})
    submodules = {
        "keeps": SimpleNamespace(resident_by_default=True), "other": SimpleNamespace(resident_by_default=False),
    }
    try:
        assert _row_nodes(model, {}, submodules) == rows
    finally:
        knobs.configure(None)


def test_the_slot_count_is_no_knob():
    with pytest.raises(ValueError):
        knobs.check({"resident_slots": 8})


# ---- host-planned steps that keep nothing ----

def test_a_host_planned_step_that_keeps_nothing_repeats_on_one_iterations_lengths():
    """A step that commits nothing (an action loop's denoise over a fixed
    prefix) rewrites the same scratch tokens each replay: a run of it reserves
    and plans one replay's pages, as it does at one iteration."""
    res = _resources()
    for rid in (1, 2):
        _ingest(res, rid)
    _host_step(res, [1, 2], [5, 6])

    def plan(repeats):
        segments = (Segment(1, "main", 3), Segment(2, "main", 3))
        ctx = StepContext(request_ids=(1, 2), graph_walk="action", slot=0, capture=False, repeats=repeats)
        kv = KVStep(segments=segments, commit=False)
        assert res["kv"].max_repeats(kv, ctx) > 4
        assert res["kv"].admit(kv, ctx).ok
        out = res["kv"].plan(kv, ctx)["main"].cpu_indptrs
        res["kv"].commit(kv, ctx)
        return [t.tolist() for t in out[:4]]

    assert plan(4) == plan(1)


def test_a_host_planned_walk_repeats_only_if_its_step_keeps_nothing():
    """Each resource answers for its part of a host-planned walk's step: a KV
    step that commits, or positions that advance, refuse with their line."""
    from mstar.engine.engine import Engine

    res = _resources()
    segments = (Segment(1, "main", 3),)

    def refusal(commit, advance, declares=True):
        engine = Engine.__new__(Engine)
        step = SubmoduleStep(steps={
            "kv": KVStep(segments=segments, commit=commit), "rope": PositionStep(segments=segments, advance=advance),
        }, segments=list(segments)) if declares else None
        engine._submodules = {"n": SimpleNamespace(
            resident=None, resident_refusals={WALK: "not asked for"}, template_steps={WALK: step},
            joint_comm_group=SimpleNamespace(world_size=1), submodule=SimpleNamespace(repeat_carry=lambda w, s: {}),
            resources={"kv": res["kv"], "rope": res["rope"]}, carry_refusals={}, carries={},
            accelerator_graph_runner=None,
        )}
        return engine.repeat_refusal("n", WALK)

    assert refusal(commit=False, advance=(0,)) is None
    assert refusal(commit=True, advance=(0,)).startswith("resource 'kv': a host-planned step that commits")
    assert refusal(commit=False, advance=None).startswith("resource 'rope': a host-planned step that advances")
    # a step that touches no resource (a DiT's) has nothing to plan, so nothing refuses
    assert refusal(commit=True, advance=None, declares=False) is None


# ---- iteration inputs and the watched loop counter ----

class _Conditioned(torch.nn.Module):
    """A node whose ``cond`` input is row i's ``trailing[k]`` at iteration k."""

    ITERATION_INPUTS = {"cond": IterationInput("trailing", shape=(2,), pad="pad", capture_fill=-1.0)}

    def __init__(self):
        super().__init__()
        self.request_states = {}


def test_a_mixed_step_stages_each_rows_iteration_of_its_own_walk():
    """A decode row at k=3 reads trailing[3]; a prefill row of the same step,
    outside the Loop, reads trailing[0]; a padding row takes its fill."""
    from mstar.engine.engine import Engine, ForwardPassInfoWrapper
    from mstar.model.submodule_base import InputMetadata, PerRequestState

    node = _Conditioned()
    for rid in (1, 2):
        node.request_states[rid] = PerRequestState()
        node.request_states[rid].add("trailing", torch.arange(10.0).reshape(5, 2) + 100 * rid)
    engine = Engine.__new__(Engine)
    from mstar.graph.loop_indices import LoopLayout
    engine._device, engine._loops = CPU, {("n", "decode"): LoopLayout("loop")}
    ctx = StepContext(request_ids=(1, 2), graph_walk="mixed", slot=0, capture=False, repeats=1)
    ctx.request_walks = {1: "decode", 2: "prefill"}
    info = {
        rid: ForwardPassInfoWrapper(SimpleNamespace(), InputMetadata(dynamic_loop_iter_counts={"loop": k}))
        for rid, k in ((1, 3), (2, 0))
    }
    node.iteration_tables = IterationTables(CPU)
    cond = engine._iteration_columns("n", node, ctx, [1, 2, -1], info)["cond"]
    assert cond[:, 0].tolist() == [[106.0, 107.0], [200.0, 201.0], [-1.0, -1.0]]


def test_a_run_stages_each_replays_iteration_input_before_it():
    cond_in, seen = torch.zeros(2), []

    class Graph:
        def replay(self):
            seen.append(cond_in.tolist())

    slot = SimpleNamespace(
        graph=Graph(), raw_launch=False, static_inputs={"cond": cond_in},
        static_outputs=SimpleNamespace(check_stop_buffers={"t": torch.zeros(2)}, row_outputs={}, output_spans=None),
    )
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner.slot_for = lambda lease: slot
    columns = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cond_in.copy_(columns[:, 0])  # the first replay's, staged with the step
    seen.append(cond_in.tolist())
    runner.repeat(None, {}, 3, iteration={"cond": columns})
    assert seen == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


@pytest.mark.parametrize("repeats", [1, 2])
def test_a_repeating_step_that_reads_the_loop_counter_stops_the_worker(repeats):
    """A run's later replays don't run prepare_inputs or preprocess again, so a
    value either derives from the counter would hold for all of them."""
    from mstar.engine.engine import ForwardPassInfoWrapper, _refuse_counter_read
    from mstar.engine.resources.resident import LOOP_COUNTER, watch_table_reads
    from mstar.model.submodule_base import InputMetadata

    info = ForwardPassInfoWrapper(SimpleNamespace(), InputMetadata(dynamic_loop_iter_counts={"loop": 4}))
    assert info.dynamic_loop_iter_counts["loop"] == 4  # unwatched: free
    with watch_table_reads() as reads:
        assert info.dynamic_loop_iter_counts.get("loop") == 4
    assert reads == [LOOP_COUNTER]
    ctx = SimpleNamespace(repeats=repeats, graph_walk=WALK)
    if repeats == 1:
        _refuse_counter_read("n", ctx, reads, "preprocess")
    else:
        with pytest.raises(ValueError, match="preprocess read the loop counter"):
            _refuse_counter_read("n", ctx, reads, "preprocess")


def test_a_captured_preprocess_that_reads_the_loop_counter_keeps_its_walk_from_repeating():
    """Probed too where the capture also serves another walk (V-JEPA2's
    rollouts), which keeps the walk off resident tables but not from repeating."""
    from mstar.engine.accelerator_graph_runner import CaptureTemplate
    from mstar.engine.engine import Engine

    def preprocess(graph_walk, engine_inputs, inputs):
        info = next(iter(engine_inputs.per_request_info.values()))
        return {"x": torch.tensor([info.dynamic_loop_iter_counts.get("loop", 0)])}

    rows = [SimpleNamespace(tensor_inputs={})]
    template_info = {-1: SimpleNamespace(dynamic_loop_iter_counts={})}
    runner = AcceleratorGraphRunner.__new__(AcceleratorGraphRunner)
    runner.batched_refusal = lambda walk: None
    runner.resident_refusal = lambda walk: "its capture also replays other walks"
    runner.template_step = lambda walk: None
    runner._widest = lambda walk: SimpleNamespace(bucket=SimpleNamespace(graph_walk=walk))
    runner._template = lambda spec: CaptureTemplate(
        [-1], rows, SimpleNamespace(per_request_info=template_info), None,
    )
    runner._dummy_rows = SimpleNamespace(reset=lambda rids, free: None)
    runner._submodule = SimpleNamespace(preprocess=preprocess, ITERATION_INPUTS={})
    engine = Engine.__new__(Engine)
    mgmt = SimpleNamespace(
        submodule=SimpleNamespace(repeat_carry=lambda walk, signals: {}), template_steps={}, carry_refusals={},
    )
    engine._submodules = {"n": mgmt}
    engine._probe_carries("n", runner, WALK, ())
    assert mgmt.carry_refusals[WALK] == (
        "node: its captured preprocess reads the loop counter, which a run's later replays don't renew"
    )


def test_a_step_that_reads_a_stream_window_repeats_as_far_as_every_row_has_items():
    """The stream answers how far a step may run: the fewest items any row's
    window holds; a row's window is cut an item a replay."""
    from mstar.engine.engine import _window_tables
    from mstar.streaming.stream_buffer import StreamChunkInfo, StreamingEdge

    rows = {0: 0, 1: 0}
    batch = ScheduledBatch(node_name="n", graph_walk=WALK, request_to_worker_graph=dict(rows))
    ctx = SimpleNamespace(repeats=1)

    def window(items):
        return [StreamingEdge("chunk", "n", list(range(items)), StreamChunkInfo(0, 0, items, False))]

    node_batch = SimpleNamespace(step_context=ctx)
    worker = Worker.__new__(Worker)
    worker._step_repeats = _repeats(8)
    worker._step_repeats.settle(lambda *key: None, lambda *key: None)
    worker._innermost_loop_by_node = {(WALK, "n"): "loop"}
    worker._graph_runtime = SimpleNamespace(loop_room=lambda *args: 100)
    speculation = SimpleNamespace(
        scheduled_batch=batch, node_batch=node_batch, loop_name="loop", continuing_rids={0, 1},
        is_new_iter=True, is_same_node=True, consumed_streaming_edges={0: window(3), 1: window(5)},
    )
    worker._stamp_repeats(speculation, SimpleNamespace(batch=SimpleNamespace(repeats=1)), 1)
    assert ctx.repeats == 3 and set(node_batch.stream_windows) == {0, 1}
    # a row with one item has one iteration of it
    speculation.consumed_streaming_edges = {0: window(1), 1: window(5)}
    worker._stamp_repeats(speculation, SimpleNamespace(batch=SimpleNamespace(repeats=1)), 1)
    assert ctx.repeats == 1

    spec = IterationInput("chunk", shape=(2,), streamed=True)
    items = torch.arange(6.0).reshape(3, 2)
    metadata = SimpleNamespace(stream_chunks={"chunk": StreamChunkInfo(0, 0, 3, False)})
    assert torch.equal(_window_tables([spec], {"chunk": [items]}, metadata)["chunk"], items)


def test_the_omni_talker_reads_the_thinkers_items_then_tts_eos_once_then_tts_pad():
    """Whatever the runs, a Talker row's text inputs are the host path's: each
    Thinker item once, then tts_eos for one iteration, then tts_pad."""
    from mstar.engine.engine import _window_tables
    from mstar.model.qwen3_omni.submodules import TalkerSubmodule
    from mstar.model.submodule_base import iteration_columns
    from mstar.streaming.stream_buffer import StreamChunkInfo

    sub = TalkerSubmodule.__new__(TalkerSubmodule)
    torch.nn.Module.__init__(sub)
    sub.model = SimpleNamespace(text_projection=SimpleNamespace(linear_fc1=SimpleNamespace(weight=torch.zeros(1))))
    sub.config = SimpleNamespace(thinker_hidden_size=2)
    sub.request_states, sub._iteration_inputs = {}, {}
    sub._tts_eos_raw, sub._tts_pad_raw = torch.full((1, 2), -1.0), torch.zeros(1, 2)
    sub.get_device = lambda: CPU
    (spec,) = sub.ITERATION_INPUTS.values()
    item = iter(range(1, 100))
    seen: dict[int, list[float]] = {0: [], 1: []}

    def step(windows: dict[int, int], repeats: int) -> None:
        rows = []
        for rid, items in windows.items():
            values = torch.tensor([[float(next(item))] * 2 for _ in range(items)])
            inputs = {"talker_input_embeds": [torch.zeros(1, 2)], "thinker_states": [values] if items else []}
            info = SimpleNamespace(rid_handle=rid)
            sub.prepare_inputs("talker_decode", info, inputs)
            chunks = SimpleNamespace(stream_chunks={"thinker_states": StreamChunkInfo(0, 0, items, False)})
            rows.append((rid, 0, _window_tables([spec], inputs, chunks)))
        column = iteration_columns(sub, rows, repeats, CPU)["text_raw"]
        for (rid, _, _), values in zip(rows, column, strict=True):
            seen[rid].extend(values[:, 0].tolist())

    step({0: 1, 1: 1}, 1)
    step({0: 2, 1: 0}, 2)  # row 1's stream has ended; the run is as long as row 0's window
    step({0: 0, 1: 0}, 3)  # and now row 0's
    step({0: 0, 1: 0}, 2)
    assert seen[0] == [1, 3, 4, -1, 0, 0, 0, 0]
    assert seen[1] == [2, -1, 0, 0, 0, 0, 0, 0]


def test_a_run_stops_short_of_the_iteration_whose_items_run_another_consumer_walk():
    """A pass stamps one consumer walk on what it streams, so a run's items
    must all run the walk its first does: a hook that moves on at k=3 holds a
    run from k=1 to two iterations."""
    from mstar.engine.engine import InputMetadata
    from mstar.streaming.topology import Connection

    batch = ScheduledBatch(node_name="n", graph_walk=WALK, request_to_worker_graph={0: 0})
    hook = Connection.__new__(Connection)
    hook.consumer_walk = lambda ctx: "late" if ctx.loop_iters["loop"] >= 3 else "early"
    worker = Worker.__new__(Worker)
    worker._walk_drivers = {"talker": hook}
    worker._innermost_loop_by_node = {(WALK, "n"): "loop"}
    worker._signal_consumer = {"states": "talker"}
    worker._graph_runtime = SimpleNamespace(get_output_signals=lambda node, walk: ["states"])
    speculation = SimpleNamespace(
        scheduled_batch=batch, loop_name="loop", continuing_rids={0},
        node_batch=SimpleNamespace(
            per_request_input_metadata={0: InputMetadata(dynamic_loop_iter_counts={"loop": 1})},
            per_request_info={0: None},
        ),
    )
    assert worker._steady_consumer_walks(speculation, 4) == 2
    worker._walk_drivers = {}
    assert worker._steady_consumer_walks(speculation, 4) == 4



def test_a_token_step_node_carries_its_token_from_the_sample_to_its_input_ids():
    """A decoder that feeds its token back as ``text_inputs`` (canary, canary-
    qwen, qwen3-asr, whisper) carries it within a run as its preprocess stages
    it, from the sampled ``new_token`` into ``input_ids``."""
    from mstar.model.canary_qwen.submodules import CanaryQwenLLMSubmodule
    from mstar.model.submodule_base import CarryPair

    sub = CanaryQwenLLMSubmodule.__new__(CanaryQwenLLMSubmodule)
    torch.nn.Module.__init__(sub)
    assert sub.repeat_carry("decode", {"text_inputs"}) == {"text_inputs": CarryPair("new_token", "input_ids")}


def test_cross_attention_over_a_fixed_context_repeats():
    from mstar.engine.resources.attn.cross import FlashInferCrossManager

    cross = FlashInferCrossManager.__new__(FlashInferCrossManager)
    assert cross.repeat_refusal(WALK) is None


def _cut(table, pad, start, repeats):
    """The column a row's table gives from ``start``: past its end, the pad's rows,
    then the pad's last row over and over."""
    rows = torch.cat([table, pad]) if pad is not None else table
    at = [min(i, rows.shape[0] - 1) for i in range(start, start + repeats)]
    return rows[at]


@pytest.mark.parametrize("repeats", [1, 4])
def test_iteration_inputs_read_each_row_from_its_table_with_one_gather(repeats):
    """Inside the table, across into a state pad, past the pad, a constant pad,
    an empty table, an offset, a padding row; a pad-less read past the end
    raises; a reassigned source is read anew; a column keeps its table's dtype;
    and once the tables are made a step's device work doesn't grow with its
    rows (review-r2 F6)."""
    from torch.utils._python_dispatch import TorchDispatchMode

    from mstar.model.submodule_base import PerRequestState, iteration_columns

    class Node:
        ITERATION_INPUTS = {
            "cond": IterationInput("trailing", shape=(2,), pad="pad", capture_fill=-1.0),
            "floor": IterationInput("floor", pad=False, capture_fill=True, dtype=torch.bool),
            "t": IterationInput("ts", pad=None, offset=1, dtype=torch.long),
        }

        def __init__(self):
            self.request_states = {}

    node = Node()
    node.iteration_tables = IterationTables(CPU)
    for rid, (n, floor) in enumerate([(5, 3), (2, 0)], start=1):
        state = node.request_states[rid] = PerRequestState()
        state.add("trailing", torch.arange(2.0 * n).reshape(n, 2) + 100 * rid)
        state.add("pad", torch.tensor([[-7.0, -8.0], [-9.0, -10.0]]))
        state.add("floor", torch.ones(floor, dtype=torch.bool))
        state.add("ts", torch.arange(40))
    pad = torch.tensor([[-7.0, -8.0], [-9.0, -10.0]])
    for ks in ([0, 0], [3, 1], [6, 2], [30, 30]):
        out = iteration_columns(node, [(1, ks[0]), (2, ks[1]), None], repeats, CPU)
        for i, rid in enumerate((1, 2)):
            state = node.request_states[rid]
            assert torch.equal(out["cond"][i], _cut(state["trailing"], pad, ks[i], repeats))
            assert torch.equal(out["floor"][i], _cut(state["floor"], torch.zeros(1, dtype=torch.bool), ks[i], repeats))
            assert torch.equal(out["t"][i], state["ts"][ks[i] + 1:ks[i] + 1 + repeats])
        assert (out["cond"][2] == -1).all() and out["floor"][2].all()
        assert out["t"].dtype == torch.long and out["floor"].dtype == torch.bool
    with pytest.raises(ValueError, match="reads past"):
        iteration_columns(node, [(1, 39)], repeats, CPU)
    node.request_states[1].add("trailing", torch.zeros(5, 2))  # a walk change reassigns its source
    assert iteration_columns(node, [(1, 0)], repeats, CPU)["cond"][0].abs().sum() == 0

    class Ops(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.kernels = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if not func._schema.name.startswith(("aten::slice", "aten::view", "aten::reshape", "aten::_reshape",
                                                  "aten::alias", "aten::unsqueeze", "aten::expand")):
                self.kernels += 1
            return func(*args, **(kwargs or {}))

    counts = []
    for rows in ([(1, 2)], [(1, 2), (2, 2), None, None]):
        iteration_columns(node, rows, repeats, CPU)  # the tables are made once
        with Ops() as ops:
            iteration_columns(node, rows, repeats, CPU)
        counts.append(ops.kernels)
    if repeats == 1:
        assert counts[0] == 0, "one row's single iteration is a view, with no allocation or kernel"
    else:
        assert counts[0] == counts[1], "a step's kernels are one gather an input, whatever its rows"


def test_a_removed_requests_rows_of_an_iteration_table_go_to_the_next_request():
    """A table grows past its rows and never refuses, and what a request held
    is handed on (a long-running server's table stops growing)."""
    spec = {"cond": IterationInput("t", shape=(2,), pad=0.0, capture_fill=-1.0)}
    tables = IterationTables(CPU)
    states = {rid: {"t": torch.full((100, 2), float(rid))} for rid in (1, 2, 3)}
    out = tables.read(spec, [(1, 99), (2, 0), (3, 100), None], states, 1)["cond"]
    assert out[:, 0, 0].tolist() == [1.0, 2.0, 0.0, -1.0]
    size = tables._tables["cond"].used
    tables.remove_request(2)
    states[4] = {"t": torch.full((100, 2), 4.0)}
    assert tables.read(spec, [(4, 5), (1, 5)], states, 1)["cond"][:, 0, 0].tolist() == [4.0, 1.0]
    assert tables._tables["cond"].used == size


def test_a_loop_back_that_also_leaves_the_loop_is_not_kept_in_rows():
    """BAGEL's latents loop back and, when the Loop ends, go to the VAE: a step
    that kept them on the device would hand the VAE nothing."""
    from mstar.graph.base import GraphEdge, GraphNode, Loop, loop_edges
    from mstar.worker.step_repeats import StepRepeats

    node = GraphNode(name="n", input_names={"x", "t"}, outputs=[GraphEdge("n", "x"), GraphEdge("n", "t")])
    loop = Loop(section=node, max_iters=8, outputs=[GraphEdge("vae", "x")], name="loop")
    edges = loop_edges(loop, "n")
    assert edges.leaving == {"x"}
    repeats = StepRepeats.__new__(StepRepeats)
    repeats.edges = {("n", "w"): edges}
    repeats.limit = lambda node: 4
    assert repeats.kept_rows(resident=()) == {("n", "w"): {"t"}}


def test_a_walk_with_a_capture_key_that_cant_be_resident_is_refused_at_load_not_at_capture():
    """BAGEL's decode captures twice, without and with guidance, and only the
    guided step declares a KV sink: the walk is refused on the verdict line,
    where before the second capture stopped the worker."""
    from mstar.engine.engine import Engine

    segments = (Segment(1, "main", 1),)

    def step(sink):
        kv = KVStep(segments=segments, sink=frozenset({(1, "main")}) if sink else frozenset())
        return SubmoduleStep(steps={"kv": kv}, segments=list(segments))

    engine = Engine.__new__(Engine)
    engine._runner = SimpleNamespace(order=("kv",))
    mgmt = SimpleNamespace(resources={"kv": object()})
    runner = SimpleNamespace(keyed_template_steps=lambda walk: {False: step(False), True: step(True)})
    why = engine._keyed_resident_refusal(mgmt, runner, WALK, step(False))
    assert why is not None and why.startswith("its capture keyed True:") and "sink" in why
    runner.keyed_template_steps = lambda walk: {False: step(False)}
    assert engine._keyed_resident_refusal(mgmt, runner, WALK, step(False)) is None


@pytest.mark.parametrize("repeat_reason", [None, "resource refuses repeats"])
def test_requested_device_execution_refuses_unsupported_walk_at_load(repeat_reason):
    repeats = _repeats(4)
    repeats._configs["n"] = RepeatConfig(max=4, device_stop="skip")
    with pytest.raises(ValueError, match="n/decode.*device_stop.*no conditional graph"):
        repeats.settle(lambda *key: None, lambda *key: repeat_reason,
                       {"device stop": lambda *key: "backend: no conditional graph"})


def test_a_batched_input_window_does_not_read_past_each_rows_admitted_horizon():
    from mstar.model.submodule_base import iteration_columns

    node = SimpleNamespace(
        ITERATION_INPUTS={
            "t": IterationInput("ts", offset=1, dtype=torch.long, capture_fill=-9),
            "s": IterationInput("stream", streamed=True, dtype=torch.long, capture_fill=-8),
        },
        iteration_tables=IterationTables(CPU),
        request_states={1: {"ts": torch.arange(8)}, 2: {"ts": torch.arange(8)}},
    )
    rows = [(1, 1, {"stream": torch.arange(6)}), (2, 6, {"stream": torch.tensor([70])}), None]
    # A continuing row covers predecessor uncertainty plus its next run;
    # a fresh row near its fixed cap has only one valid input, in the same batch.
    out = iteration_columns(node, rows, 8, CPU, horizons=[6, 1, 0])
    assert out["t"].tolist() == [[2, 3, 4, 5, 6, 7, -9, -9], [7, -9, -9, -9, -9, -9, -9, -9], [-9] * 8]
    assert out["s"].tolist() == [[0, 1, 2, 3, 4, 5, -8, -8], [70, -8, -8, -8, -8, -8, -8, -8], [-8] * 8]
    with pytest.raises(ValueError, match="reads past"):
        iteration_columns(node, rows, 8, CPU, horizons=[6, 2, 0])


def test_an_engine_input_window_keeps_lower_origins_entry_keys_and_per_row_extents():
    from mstar.engine.device_run import IterationWindow
    from mstar.engine.engine import Engine, ForwardPassInfoWrapper
    from mstar.graph.loop_indices import LoopLayout
    from mstar.model.submodule_base import InputMetadata

    node = SimpleNamespace(
        ITERATION_INPUTS={"t": IterationInput("ts", dtype=torch.long, capture_fill=-9)},
        iteration_tables=IterationTables(CPU),
        request_states={1: {"ts": torch.arange(9)}, 2: {"ts": torch.arange(10)}},
    )
    loop = LoopLayout("inner", ("outer",), ())
    engine = Engine.__new__(Engine)
    engine._device, engine._loops = CPU, {("n", "decode"): loop}
    engine._runner = SimpleNamespace(request_generation=lambda rid: 100 + rid)
    ctx = StepContext(request_ids=(1, 2), graph_walk="decode", slot=0, capture=True, repeats=4)
    metadata = {
        1: InputMetadata(dynamic_loop_iter_counts={"inner": 5, "outer": 7}, graph_pass=3, iteration_origin=1),
        2: InputMetadata(dynamic_loop_iter_counts={"inner": 6, "outer": 4}, graph_pass=9, iteration_origin=6),
    }
    info = {rid: ForwardPassInfoWrapper(SimpleNamespace(), item) for rid, item in metadata.items()}
    window = engine._iteration_columns("n", node, ctx, [1, 2, -1], info, actual_loop=loop)
    assert isinstance(window, IterationWindow)
    assert window.origins.tolist() == [1, 6, 0]
    assert window.entries.tolist() == [[101, 3, 7], [102, 9, 4], [-1, -1, -1]]
    assert window.columns["t"].tolist() == [list(range(1, 9)), [6, 7, 8, 9, -9, -9, -9, -9], [-9] * 8]
    # A later assignment reuses the owner's arena, but cannot change a queued snapshot.
    node.request_states[1]["ts"] = torch.arange(9) + 100
    engine._iteration_columns("n", node, ctx, [1, 2, -1], info, actual_loop=loop)
    assert window.columns["t"][0].tolist() == list(range(1, 9))

    one = StepContext(request_ids=(1,), graph_walk="decode", slot=0, capture=True, repeats=1)
    info[1].metadata.dynamic_loop_iter_counts["inner"] = 1
    single = engine._iteration_columns("n", node, one, [1], info, actual_loop=loop)
    node.request_states[1]["ts"] = torch.arange(9) + 200
    engine._iteration_columns("n", node, one, [1], info, actual_loop=loop)
    assert single.columns["t"].item() == 101
