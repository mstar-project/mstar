"""Every rank of a TP instance tears a request down after its admit of the step rank 0
stamped the teardown with, before its admit of the next, and in the order rank 0 did.

A teardown releases pages, so the side of an admit it lands on decides what that admit finds
free, and once the pool is full a follower can refuse an admit rank 0 grants. Rank 0
forwards each teardown stamped with the last step it has broadcast. A follower holds it
until it has popped that step, and reads no message between popping a step and the admit
fence at the top of the next pass, so it lands after that admit. Rank 0 lands it there too.
When the follower drops a step it popped, it tears the held ones down before it takes the
next. A removal deferred behind step N is retried once N is post-processed, when N+1 is
speculated, broadcast and maybe not yet admitted; but ``_remove_request`` defers it again
while N is still the step in flight, and the next pass tears it down behind N+1's admit
fence. A removal that reaches rank 0 once N is done, with nothing in flight, lands before
N+1 is sent; a follower still running N defers it, and tears it down once N is done, before
it takes N+1. The order matters as well: the free list hands pages back out in the order
they were freed, so another order gives a follower other page ids than rank 0's.

Drives ``Worker.run`` over stubs. The GPU thread is a stub that holds N+1's admit back until
a request is torn down, or a while.
"""

from __future__ import annotations

import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.engine.resources import StepRunner, SubmoduleStep  # noqa: E402
from mstar.engine.resources.base import EngineResourceInfo  # noqa: E402
from mstar.engine.resources.kv.cache import PageAllocator  # noqa: E402
from mstar.engine.resources.recurrent.config import (  # noqa: E402
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool  # noqa: E402
from mstar.engine.resources.step import Segment, StepContext  # noqa: E402
from mstar.utils.containers import RecentSet  # noqa: E402
from mstar.utils.ipc_format import MessageSource, RemoveRequest, WorkerMessageType  # noqa: E402
from mstar.worker import worker as worker_mod  # noqa: E402
from mstar.worker.stream_buffers import StreamBufferManager  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402

NODE, WALK = "LLM", "decode"
# wire id -> handle. All but "idle" run in step N (seq 0); only "kept" goes on into N+1
# (seq 1)
HANDLES = {"gone": 1, "kept": 2, "also": 3, "idle": 4}
STEP_N = [HANDLES[rid] for rid in ("gone", "kept", "also")]
# how long the GPU thread holds N+1's admit back for a teardown that would jump it
HOLD_S = 0.25


class _Stop(BaseException):
    """Ends ``run`` at the pass after N+1 went out."""


def _batch(seq: int, rids: list[int]):
    """A step's scheduled and executing halves."""
    scheduled = SimpleNamespace(
        node_name=NODE, graph_walk=WALK, tp_seq=seq,
        request_to_worker_graph={rid: 0 for rid in rids},
        rids_by_walk=lambda: {WALK: list(rids)},
    )
    executing = SimpleNamespace(
        node_name=NODE, request_ids=list(rids), admit_error=None, failed_requests={},
        launch_started_event=None, read_check_rids=None,
    )
    return scheduled, executing


def _speculation(seq: int, rids: list[int]):
    scheduled, executing = _batch(seq, rids)
    return SimpleNamespace(
        scheduled_batch=scheduled, node_batch=executing, is_yield_away=False,
        plan_future=None, partition="p", is_new_iter=False, loop_name=None,
        tp_seq=seq, dropped=set(),
    )


class _Scheduler:
    """N comes off the serial path. N+1 is the speculative head rank 0 sends from N, or
    with ``serial`` a step it sends after N, off the serial path too."""

    def __init__(self, serial: bool = False):
        self.heads = [] if serial else [SimpleNamespace(
            node_name=NODE, graph_walk=WALK, speculative=True, spec_seq=1, spec_from_seq=0,
        )]
        self.steps = [STEP_N] + ([[HANDLES["kept"]]] if serial else [])
        self.last_consumed_tp_seq = -1

    def get_next_batch(self, request_state, **kwargs):
        if self.last_consumed_tp_seq + 1 >= len(self.steps):
            return None
        self.last_consumed_tp_seq += 1
        return _batch(self.last_consumed_tp_seq, self.steps[self.last_consumed_tp_seq])[0]

    def peek_tp_follow(self):
        return self.heads[0] if self.heads else None

    def pop_tp_follow_head(self):
        self.last_consumed_tp_seq = self.heads.pop(0).spec_seq

    def take_admit_errors(self):
        return {}

    def clear_batch_limit(self, node_name, graph_walk, rows, walks=()):
        # an admitted step lifts its refusal limit; this stub has none
        del node_name, graph_walk, rows, walks

    def clear_rid(self, rid, wire_rid):
        pass

    def clear_wire_rid(self, wire_rid):
        pass


def _rank(
    monkeypatch, *, tp_rank: int, inbox: list[list[RemoveRequest]], serial: bool = False,
):
    """A worker at TP rank ``tp_rank`` that runs N and then N+1, speculated or, with
    ``serial``, sent after rank 0 marked N as not speculated from. Each pass of
    ``_process_messages`` takes the next entry of ``inbox``. Returns the worker and its
    log of admits, teardowns and forwarded removals, in the order they happened."""
    monkeypatch.setattr(worker_mod, "gc", SimpleNamespace(
        collect=lambda: None, freeze=lambda: None, get_freeze_count=lambda: 0,
    ))
    log: list[tuple] = []
    torn_down = threading.Event()
    w = Worker.__new__(Worker)
    w.worker_id = f"rank{tp_rank}"
    w.is_tp_follower = tp_rank > 0
    w.enable_nvtx, w._phase_period, w._phase_buf = False, 0, {}
    w._max_consecutive_spec, w._launch_wait_s, w._metrics = 1024, 0.0, None
    w._in_flight_rids, w._pending_removes, w._removes_awaiting_step = set(), {}, {}
    w._pending_drains, w._draining_rids, w._reads_done_sent = set(), set(), set()
    w._removed_rids, w._unprocessed_messages = RecentSet(16), {}
    w.streaming_buffers, w._last_active = {}, {}
    w._tp_broadcast_seq = 0
    w._tp_nospec = RecentSet(16)
    if serial:
        w._tp_nospec.add(0)
    w.scheduler = _Scheduler(serial)

    group = SimpleNamespace(tp_size=2, _tp_rank=tp_rank, _workers=["rank0", "rank1"])
    wire = {handle: rid for rid, handle in HANDLES.items()}
    w._graph_runtime = SimpleNamespace(
        get_rid_handle=HANDLES.get, get_rid_string=wire.get,
        get_sharding_config=lambda rid: SimpleNamespace(groups=[group]),
        remove_request=lambda rid: None,
    )

    def _tear_down(handle):
        log.append(("teardown", wire[handle]))
        torn_down.set()

    def _send(worker, msg):
        if msg.message_type == WorkerMessageType.REMOVE_REQUEST:
            log.append(("forward", msg.body.request_id, msg.body.after_tp_seq))

    w.engine_manager = SimpleNamespace(
        warmup_all=lambda: None, remove_request=_tear_down, evictable_nodes=lambda: (),
    )
    w.communicator = SimpleNamespace(send=_send, wait_for_work=lambda timeout: None)
    w.request_state = SimpleNamespace(
        remove_request=lambda rid: None, get_partition_for_node=lambda node: "p",
    )
    w.tensor_manager = SimpleNamespace(force_cleanup_request=lambda rid: None)
    w.streams = StreamBufferManager(w.request_state, w.tensor_manager, [])
    w.profile_info = SimpleNamespace(pop_request=lambda rid: None)
    w.parallel_groups = SimpleNamespace(
        barrier_all=lambda: None, restore_default_timeout=lambda: None,
    )
    w.wakeup_event = SimpleNamespace(register_future=lambda future: None)

    def _broadcast(node_batch, **kwargs):
        if tp_rank:
            return -1
        w._tp_broadcast_seq += 1
        return w._tp_broadcast_seq - 1

    def _build_speculation(pending, consecutive, fair):
        if pending is not None and pending.tp_seq == 1:
            raise _Stop
        if pending is None or tp_rank:
            return None, None  # a follower takes N+1 off its FIFO instead
        spec = _speculation(1, [HANDLES["kept"]])
        spec.tp_seq = _broadcast(spec.node_batch, speculative=True, spec_from_seq=0)
        return spec, None

    def _follow(pending):
        w.scheduler.pop_tp_follow_head()
        w._in_flight_rids |= {HANDLES["kept"]}
        return _speculation(1, [HANDLES["kept"]])

    def _gpu(batch, node_batch, plan_future=None):
        if plan_future is not None:
            plan_future.result()
        if batch.tp_seq == 1:
            torn_down.wait(HOLD_S)
        log.append(("admit", batch.tp_seq))
        node_batch.launch_started_event.set()
        return {}

    passes = iter(inbox)

    def _process_messages():
        for body in next(passes, []):
            w._remove_request(body)

    w._process_messages = _process_messages
    w._build_executing_batch = lambda batch: _batch(
        batch.tp_seq, list(batch.request_to_worker_graph))[1]
    w.maybe_send_zmq_to_tp_followers = _broadcast
    w._build_speculation = _build_speculation
    w._try_follow_speculation = _follow
    w._is_tp_follow_pending = lambda pending: bool(tp_rank)
    w._execute_on_gpu_thread = _gpu
    w._preplan_spec = lambda pending, spec: False
    for name in (
        "_verify_tp_async_sched_agrees", "_init_engine_thread", "_fail_requests",
        "_check_ready_tensors", "_poll_stream_buffers", "_set_in_flight_flag",
        "_clear_in_flight_flag", "_thread_outputs_to_speculative", "_settle_speculation",
        "_postprocess_batch",
    ):
        setattr(w, name, lambda *a, **k: None)

    def _fail_loudly(error, *args):
        raise error

    w._handle_main_loop_error = _fail_loudly
    return w, log


def _run(w) -> None:
    try:
        w.run()
    except _Stop:
        return
    raise AssertionError("run returned")


def test_rank_0_tears_down_after_admitting_the_step_its_stamp_names(monkeypatch):
    """The removal arrives while N runs, so it waits on N, and N+1 goes out without the
    request. The teardown is stamped with N+1, so it lands after N+1's admit, though N is
    post-processed before that admit can have run."""
    w, log = _rank(monkeypatch, tp_rank=0, inbox=[[], [RemoveRequest(request_id="gone")]])
    _run(w)

    assert ("forward", "gone", 1) in log
    assert log.index(("admit", 1)) < log.index(("teardown", "gone")), log


def test_a_follower_tears_down_after_admitting_the_step_rank_0_stamped(monkeypatch):
    """Here before this rank has even popped N+1: held until it has, then torn down behind
    the admit fence of the next pass."""
    removal = RemoveRequest(request_id="gone", source=MessageSource.TP_RANK_0, after_tp_seq=1)
    w, log = _rank(monkeypatch, tp_rank=1, inbox=[[], [removal]])
    _run(w)

    assert log.index(("admit", 1)) < log.index(("teardown", "gone")), log


def test_a_follower_tears_down_in_the_order_rank_0_did(monkeypatch):
    """Rank 0 tore "gone" down, then "also", both after N+1. "gone" reaches this rank
    before it pops N+1 and is held; "also" after, when its step is already reached. The
    free list hands pages back out in the order they were freed, so "gone" goes first."""
    gone, also = (
        RemoveRequest(request_id=rid, source=MessageSource.TP_RANK_0, after_tp_seq=1)
        for rid in ("gone", "also")
    )
    w, log = _rank(monkeypatch, tp_rank=1, inbox=[[], [gone], [also]])
    _run(w)

    assert [entry[1] for entry in log if entry[0] == "teardown"] == ["gone", "also"], log


def test_a_follower_tears_down_what_it_deferred_in_the_order_rank_0_did(monkeypatch):
    """Rank 0 had nothing in flight and tore "also" down, then "gone", after N; here N
    still runs both, so both wait on it. Handles are this rank's own, so an order kept by
    handle is not rank 0's, and the free list would come out other than rank 0's."""
    pages = {HANDLES["gone"]: [1, 2], HANDLES["also"]: [3, 4], HANDLES["kept"]: [5]}
    also, gone = (
        RemoveRequest(request_id=rid, source=MessageSource.TP_RANK_0, after_tp_seq=0)
        for rid in ("also", "gone")
    )
    w, log = _rank(monkeypatch, tp_rank=1, inbox=[[], [also, gone]], serial=True)
    freed, on_rank_0 = PageAllocator(6), PageAllocator(6)
    for allocator in (freed, on_rank_0):
        allocator.allocate(6)
    tear_down = w.engine_manager.remove_request

    def _release(handle):
        tear_down(handle)
        freed.free(pages[handle])

    w.engine_manager.remove_request = _release
    _run(w)

    for rid in ("also", "gone"):
        on_rank_0.free(pages[HANDLES[rid]])
    assert [entry[1] for entry in log if entry[0] == "teardown"] == ["also", "gone"], log
    assert list(freed.free_pages.queue) == list(on_rank_0.free_pages.queue)


def _slot_pool() -> tuple[RecurrentStatePool, StepRunner]:
    """One TP rank's recurrent pool (GLM-5.3 keeps its KDA state in one): 5 slots and the
    sink, sharded over two ranks, so it decides nothing alone."""
    spec = RecurrentStateSpec("state", {NODE}, RecurrentStateConfig(
        num_layers=1,
        blocks={"state": RecurrentBlockConfig(shape=(2, 2), dtype=torch.float32, shard_dims=(0,))},
        max_slots=6,
    ))
    joint = SimpleNamespace(world_size=2)
    pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=joint))
    return pool, StepRunner({"state": pool}, node_resources={NODE: ["state"]})


def _admit_slots(run: StepRunner, rids: list[str]) -> None:
    """A prefill of ``rids``: each request not yet holding a slot takes one."""
    segments = [Segment(rid, "main", 1) for rid in rids]
    step = SubmoduleStep(segments=segments, steps={"state": RecurrentStep(segments=tuple(segments))})
    step.set_ctx(StepContext(request_ids=tuple(rids), graph_walk="prefill", slot=0, capture=False))
    for rid in rids:
        run.ingest_request(rid, {})
    assert run.admit(step).ok
    run.plan(step)
    run.commit(step)


def test_a_follower_admits_against_rank_0s_free_recurrent_slots(monkeypatch):
    """As above, on each rank's real recurrent pool, and with a new request in N+1 that
    takes a slot. Rank 0 had nothing in flight and tore "also" down, then "gone", before
    it sent N+1; here both wait on N. At each admit this rank's free slots are rank 0's, in
    rank 0's order, so N+1's new request gets the slot it got on rank 0 (the free list is
    a stack: the other order hands out also's slot)."""
    also, gone = (
        RemoveRequest(request_id=rid, source=MessageSource.TP_RANK_0, after_tp_seq=0)
        for rid in ("also", "gone")
    )
    w, log = _rank(monkeypatch, tp_rank=1, inbox=[[], [also, gone]], serial=True)
    w.scheduler.steps[1] = [HANDLES["kept"], HANDLES["idle"]]
    rid_of = {handle: rid for rid, handle in HANDLES.items()}
    pool, run = _slot_pool()
    rank_0, run_0 = _slot_pool()
    expected = [list(rank_0._free)]
    _admit_slots(run_0, ["gone", "kept", "also"])
    run_0.remove_request("also")
    run_0.remove_request("gone")
    expected.append(list(rank_0._free))
    _admit_slots(run_0, ["kept", "idle"])

    tear_down, gpu = w.engine_manager.remove_request, w._execute_on_gpu_thread
    at_admit = []

    def _release(handle):
        tear_down(handle)
        run.remove_request(rid_of[handle])

    def _admit(batch, node_batch, plan_future=None):
        out = gpu(batch, node_batch, plan_future)  # the admit, after any hold
        at_admit.append(list(pool._free))
        _admit_slots(run, [rid_of[h] for h in batch.request_to_worker_graph])
        return out

    w.engine_manager.remove_request, w._execute_on_gpu_thread = _release, _admit
    _run(w)

    assert [entry[1] for entry in log if entry[0] == "teardown"] == ["also", "gone"], log
    assert log.index(("teardown", "gone")) < log.index(("admit", 1)), log
    assert at_admit == expected
    assert pool._slots["idle"]["main"].index == rank_0._slots["idle"]["main"].index
    assert pool._free == rank_0._free and pool.num_free_slots == rank_0.num_free_slots


def test_a_follower_tears_down_nothing_ahead_of_one_it_deferred(monkeypatch):
    """Rank 0 tore "gone" down, then "idle", after N. Here N still runs "gone", so it waits
    on N; "idle" runs in no step, but waits behind it, as on rank 0."""
    gone, idle = (
        RemoveRequest(request_id=rid, source=MessageSource.TP_RANK_0, after_tp_seq=0)
        for rid in ("gone", "idle")
    )
    w, log = _rank(monkeypatch, tp_rank=1, inbox=[[], [gone, idle]], serial=True)
    _run(w)

    assert [entry[1] for entry in log if entry[0] == "teardown"] == ["gone", "idle"], log


def test_a_follower_tears_down_what_it_deferred_before_it_takes_the_next_step(monkeypatch):
    """Rank 0 had nothing in flight when it tore "gone" down, so it did before it sent
    N+1; here N still runs it. Once N is done the teardown goes, before this rank admits
    N+1, which rank 0 admitted against the pages "gone" let go."""
    gone = RemoveRequest(request_id="gone", source=MessageSource.TP_RANK_0, after_tp_seq=0)
    w, log = _rank(monkeypatch, tp_rank=1, inbox=[[], [gone]], serial=True)
    _run(w)

    assert log.index(("teardown", "gone")) < log.index(("admit", 1)), log


@pytest.mark.parametrize("cache_open", [False, True])
def test_a_follower_that_drops_a_step_tears_down_before_it_takes_the_next(monkeypatch, cache_open):
    """N fails "also", so both ranks drop the speculative N+1 built on it. Rank 0 stamped
    "gone" with N+1 and tore it down before it sent N+2. Here "gone" is held until this
    rank pops N+1, which it does as it drops it, after building it when the prefix cache is
    open, and goes before N+2's admit."""
    removal = RemoveRequest(request_id="gone", source=MessageSource.TP_RANK_0, after_tp_seq=1)
    w, log = _rank(monkeypatch, tp_rank=1, inbox=[[], [removal]])
    w.engine_manager.get_engine = lambda node: SimpleNamespace(
        prefix_cache_open=lambda node: cache_open, drop_cached_prefix=lambda head: None,
    )
    # N+1 is dropped, so the serial path's next step is N+2
    w.scheduler.steps += [None, [HANDLES["kept"]]]
    build_executing, build_speculation = w._build_executing_batch, w._build_speculation

    def _executing(batch):
        node_batch = build_executing(batch)
        if batch.tp_seq == 0:
            node_batch.failed_requests = {HANDLES["also"]: "failed"}
        return node_batch

    def _speculation_or_stop(pending, consecutive, fair):
        if pending is not None and pending.tp_seq == 2:
            raise _Stop
        return build_speculation(pending, consecutive, fair)

    follow = w._try_follow_speculation

    def _follow(pending):
        spec = follow(pending)
        spec.continuing_rids = {HANDLES["kept"]}
        return spec

    w._build_executing_batch = _executing
    w._build_speculation = _speculation_or_stop
    w._try_follow_speculation = _follow
    w._drop_failed_rids = lambda *args: None
    w._reset_skip_plan_flags = lambda node_batch: None
    w._graph_runtime.push_back_node = lambda *args: None
    _run(w)

    assert log.index(("teardown", "gone")) < log.index(("admit", 2)), log


def test_the_admit_fence_waits_out_a_slow_admit(monkeypatch):
    """The fence waits past any number of warning intervals; it never gives up.
    A lone-rank step (``tp_seq=-1``) is the case that used to."""
    tp_seq = -1
    monkeypatch.setattr(worker_mod, "_ADMIT_FENCE_WARN_S", 0.001)
    admitted = threading.Event()
    warnings = []

    def warn(*args):
        warnings.append(args)
        if len(warnings) == 3:
            admitted.set()  # the admit lands during the third interval

    monkeypatch.setattr(worker_mod.logger, "warning", warn)
    w = Worker.__new__(Worker)
    w.worker_id = "rank0"
    w._span = lambda name: nullcontext()
    _, executing = _batch(tp_seq, [HANDLES["kept"]])
    executing.launch_started_event = admitted
    pending = SimpleNamespace(
        tp_seq=tp_seq, node_name=NODE, graph_walk=WALK, node_batch=executing,
    )

    w._await_admit_settled(pending)

    assert admitted.is_set() and len(warnings) == 3
