"""TP async scheduling — a follower must always SETTLE the leader's decision
about step N before it post-processes N, and it must never guess.

The leader broadcasts its speculation DURING its forward N; a follower can
finish N first (faster rank, slow leader build). If a follower went serial
whenever the head had not arrived yet, a head landing later could reach the
serial path with a continuing rid whose loop ended at N — a batch the serial
readiness check can never build, while the leader runs it (one wasted
forward) and then hangs in the collective. So:

* the leader ALWAYS sends one of {speculative head from N, ``TPNoSpeculation``
  marker for N} for every lockstep step it looked at speculating from;
* the follower's ``_await_tp_follow_step`` — one loop that both waits for N
  and watches the FIFO — builds a head the moment it lands (during N, the
  point of the flag), and once N has finished does not return without a
  decision: builds the head from N's real outputs, or goes serial on the
  marker; N's verdict — ``admit_error`` / ``failed_requests`` on its
  ``ExecutingBatch``, written by the GPU thread before N's future resolves —
  drops a head without building it (the verdicts on which the leader clears
  its own speculation before threading), unless the prefix cache is open,
  where the leader pre-planned the head and the follower builds it too;
* a step whose forward raised is closed (``_close_tp_follow_step``): its head
  is dropped, queued or arriving.

Drives the real ``Worker`` methods unbound on a minimal stub with a real
``MicroScheduler`` for the FIFO, the pattern ``test_worker_message_handling``
uses. ``_try_follow_speculation`` is stubbed per test — its own build logic
is exercised against the graph API elsewhere; here we pin the protocol.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from test_loop_turns import _looping_worker, _Runtime, _scheduler, _StopLoopError  # noqa: E402

from mstar.engine.resources.step import FULL_ADMIT_OK  # noqa: E402
from mstar.model.submodule_base import StepOrder  # noqa: E402
from mstar.utils.containers import RecentSet  # noqa: E402
from mstar.utils.ipc_format import ScheduleTPNode, TPNoSpeculation  # noqa: E402
from mstar.worker.micro_scheduler import MicroScheduler  # noqa: E402
from mstar.worker.worker import (  # noqa: E402
    Worker,
    _parse_tp_async_sched,
    _tp_async_sched_setting,
)

NODE = "LLM"
WALK = "decode"


class _FakeEngineManager:
    def __init__(self, cache_open: bool = False):
        self.cache_open = cache_open

    def get_engine(self, node_name):
        return types.SimpleNamespace(
            check_ready=lambda *a: FULL_ADMIT_OK, drop_cached_prefix=lambda head: None,
            prefix_cache_open=lambda node: self.cache_open,
        )


class _Communicator:
    """wait_for_work delivers one queued message batch per call; the test
    pushes messages via ``inbox`` (drained by the next ``_process_messages``)
    or schedules them in ``deliveries`` (wait number -> messages that land on
    that wait). Raises if the follower keeps waiting after everything was
    delivered — that would be a hang."""

    def __init__(self, stub):
        self.stub = stub
        self.inbox = []
        self.deliveries = {}
        self.waits = 0
        self.max_waits = 50

    def wait_for_work(self, timeout_ms=50):
        self.waits += 1
        if self.waits > self.max_waits:
            raise TimeoutError("follower waited too long — would hang")
        self.inbox.extend(self.deliveries.pop(self.waits, []))


def _stub(cache_open: bool = False):
    stub = types.SimpleNamespace()
    stub.worker_id = "w1"
    stub.tp_async_sched = True
    stub.tp_async_nodes = None
    stub._tp_async_for = lambda n: Worker._tp_async_for(stub, n)
    stub._tp_nospec = RecentSet(Worker._TP_NOSPEC_KEEP)
    stub._tp_leader_gap_warned = False
    stub.engine_manager = _FakeEngineManager(cache_open)
    stub.scheduler = MicroScheduler(
        engine_manager=stub.engine_manager, parallel_leader_nodes=set(),
    )
    stub.communicator = _Communicator(stub)
    stub._check_ready_tensors = lambda: None
    stub._poll_stream_buffers = lambda: None
    stub.build_results = []   # what _try_follow_speculation returns, in order

    def _process_messages():
        while stub.communicator.inbox:
            msg = stub.communicator.inbox.pop(0)
            if isinstance(msg, TPNoSpeculation):
                stub._register_tp_nospec(msg)
            else:
                stub._register_tp_follow(msg)

    def _try_follow(pending):
        # like the real one: a successful build consumes the FIFO head
        if stub.build_results:
            res = stub.build_results.pop(0)
            if res is not None and stub.scheduler.peek_tp_follow() is not None:
                stub.scheduler.pop_tp_follow_head()
            return res
        return None

    stub._process_messages = _process_messages
    stub._try_follow_speculation = _try_follow
    stub._register_tp_follow = lambda m: Worker._register_tp_follow(stub, m)
    stub._register_tp_nospec = lambda m: Worker._register_tp_nospec(stub, m)
    stub._close_tp_follow_step = lambda p: Worker._close_tp_follow_step(stub, p)
    stub._await_tp_follow_step = (
        lambda p, arm: Worker._await_tp_follow_step(stub, p, arm)
    )
    stub._step_voids_head = Worker._step_voids_head
    stub._step_result = lambda p: Worker._step_result(stub, p)
    stub._prefix_cache_open = lambda node: Worker._prefix_cache_open(stub, node)
    return stub


def _leader_stub():
    stub = types.SimpleNamespace()
    stub.tp_async_sched = True
    stub.tp_async_nodes = None
    stub._tp_async_for = lambda n: Worker._tp_async_for(stub, n)
    stub.parallel_nodes = {NODE}
    stub.parallel_leader_nodes = {NODE}
    stub._is_tp_lead_pending = lambda p: Worker._is_tp_lead_pending(stub, p)
    stub._tp_lead_needs_marker = (
        lambda p, sp: Worker._tp_lead_needs_marker(stub, p, sp)
    )
    return stub


def _pending(tp_seq, future=None, admit_error=None, failed_requests=()):
    """``admit_error`` / ``failed_requests`` are N's verdict, on its
    ``ExecutingBatch`` the way the GPU thread leaves them."""
    return types.SimpleNamespace(
        node_name=NODE, graph_walk=WALK, tp_seq=tp_seq, future=future,
        batch=types.SimpleNamespace(request_to_worker_graph={"r0": "wg0"}),
        node_batch=types.SimpleNamespace(
            admit_error=admit_error,
            failed_requests={rid: "boom" for rid in failed_requests},
        ),
    )


def _future(stub, output, done_after=0):
    """N's future: done once the follower has waited ``done_after`` times."""
    return types.SimpleNamespace(
        done=lambda: stub.communicator.waits >= done_after,
        result=lambda: output,
    )


def _output():
    """N's outputs: rid -> name -> tensors. Carries no verdict (see _pending)."""
    return {"r0": {}}


def _head(rids, seq, from_seq):
    return ScheduleTPNode(
        node_name=NODE, graph_walk=WALK, request_ids=list(rids),
        speculative=True, spec_seq=seq, spec_from_seq=from_seq,
    )


def _nospec(from_seq):
    return TPNoSpeculation(node_name=NODE, graph_walk=WALK, spec_from_seq=from_seq)


# ------------------------------------------------------------- the switch

def test_env_parses_master_switch_and_node_list():
    assert _parse_tp_async_sched("0") == (False, None)
    assert _parse_tp_async_sched("") == (False, None)
    assert _parse_tp_async_sched("1") == (True, None)
    assert _parse_tp_async_sched("thinker, talker") == (
        True, frozenset({"thinker", "talker"}),
    )


@pytest.mark.parametrize("env, config, want", [
    (None, None, (True, None)),  # on by default
    (None, True, (True, None)),
    (None, False, (False, None)),
    (None, ["thinker", "talker"], (True, frozenset({"thinker", "talker"}))),
    (None, "LLM", (True, frozenset({"LLM"}))),
    ("0", True, (False, None)),  # the env var overrides the config
    ("1", ["talker"], (True, None)),
    ("", True, (True, None)),  # empty env falls through to the config
])
def test_config_key_and_env_override(env, config, want):
    assert _tp_async_sched_setting(env, config) == want


@pytest.mark.parametrize("env, config, want", [
    # a quoted yaml boolean is a string, not a node named "true"
    (None, "true", (True, None)),
    (None, "yes", (True, None)),
    (None, "false", (False, None)),
    (None, "No", (False, None)),
    ("off", True, (False, None)),
    ("True", None, (True, None)),
])
def test_boolean_words_are_the_switch_not_node_names(env, config, want):
    assert _tp_async_sched_setting(env, config) == want


def test_node_list_narrows_which_parallel_nodes_take_part():
    """Per-node control: a rank whose list excludes NODE behaves exactly as
    with the flag off for NODE — leader owes no marker, follower records
    nothing and queues heads verbatim for the serial path."""
    ld = _leader_stub()
    ld.tp_async_nodes = frozenset({"talker"})
    assert not ld._tp_lead_needs_marker(_pending(4), None)
    ld.tp_async_nodes = frozenset({NODE})
    assert ld._tp_lead_needs_marker(_pending(4), None)

    s = _stub()
    s.tp_async_nodes = frozenset({"talker"})
    s._register_tp_nospec(_nospec(4))
    assert 4 not in s._tp_nospec
    h = _head(["r0"], seq=5, from_seq=4)
    s._register_tp_follow(h)
    assert s.scheduler.peek_tp_follow() is h


# ------------------------------------------------------- the leader's duty

def test_leader_owes_a_marker_whenever_no_head_went_out():
    """A follower settles on {head from s, marker for s}; the leader must send
    one of them for EVERY lockstep step it looked at speculating from — also
    when it did speculate, but into a non-parallel node, which is never
    broadcast (tp_seq stays -1). Missing that marker hangs the followers."""
    ld = _leader_stub()
    p = _pending(4)
    assert ld._tp_lead_needs_marker(p, None)
    assert ld._tp_lead_needs_marker(p, types.SimpleNamespace(tp_seq=-1))
    assert not ld._tp_lead_needs_marker(p, types.SimpleNamespace(tp_seq=5))


def test_only_the_leader_of_a_lockstep_node_owes_a_marker():
    ld = _leader_stub()
    ld.parallel_leader_nodes = set()     # a follower shard of NODE
    assert not ld._tp_lead_needs_marker(_pending(4), None)
    ld = _leader_stub()
    ld.tp_async_sched = False
    assert not ld._tp_lead_needs_marker(_pending(4), None)


def _tp_loop(monkeypatch, follower=False, forward_raises=()):
    """The real ``Worker.run`` on one rank of a TP instance (``_looping_worker``),
    rows 0 and 1 ready and one row per step. Step 0's seq is 0. Records the
    rows each forward ran, the markers sent, the rows failed and what reached
    the loop's error handler. A forward of a row in ``forward_raises`` raises."""
    runtime = _Runtime()
    sched, state = _scheduler(runtime, StepOrder.TURNS)
    for rid in (0, 1):
        runtime.requests[rid] = None
        sched.note_arrival(rid)
    runtime.ready |= {0, 1}
    w = _looping_worker(monkeypatch, sched, state)
    node = next(iter(sched.parallel_leader_nodes))
    w.tp_async_sched, w.tp_async_nodes = True, None
    w.parallel_nodes, w.parallel_leader_nodes = {node}, set() if follower else {node}
    w.is_tp_follower = follower
    w._graph_runtime = types.SimpleNamespace(
        is_async_schedulable=lambda node, walk: True, push_back_node=lambda *a: None,
    )
    w._settle_speculation = lambda *a, **k: None
    w.maybe_send_zmq_to_tp_followers = lambda node_batch, **k: 0
    rec = types.SimpleNamespace(ran=[], markers=[], failed={}, errors=[])
    w._broadcast_tp_nospec = lambda pending: rec.markers.append(pending.tp_seq)
    w._handle_main_loop_error = lambda exc, *a: rec.errors.append(exc) or set()
    w._fail_requests = rec.failed.update
    forward = w._execute_on_gpu_thread

    def gpu(batch, node_batch, plan_future):
        rids = list(batch.request_to_worker_graph)
        rec.ran.append(rids)
        if set(rids) & set(forward_raises):
            raise RuntimeError("CUDA error")
        return forward(batch, node_batch, plan_future)

    w._execute_on_gpu_thread = gpu
    return w, sched, state, rec


def _builds_step_1(w, sched, state, head=False):
    """``_build_speculation`` on a leader: from step 0, step 1 (row 1), sent
    as seq 1, as a head or a yield-away; nothing after that."""
    def build(pending, consecutive, fair):
        if pending is None or pending.tp_seq != 0:
            return None, None
        spec = w._yield_away_speculation(sched.get_next_batch(state))
        spec.is_yield_away = not head
        spec.tp_seq = 1
        return spec, None
    return build


@pytest.mark.parametrize("raise_in", ["build", "finish_step"])
def test_a_leader_sends_the_marker_for_a_step_it_drops_before_its_decision(monkeypatch, raise_in):
    """The real run loop on a TP leader: step 0 runs and the yield-away step 1
    is built beside it. Then building step 1's successor raises (``build``),
    or post-processing step 0 raises with step 1 already submitted
    (``finish_step``). Either way no decision for step 1 went out, so its
    marker must, or the followers await step 1 for good (and a drain deferred
    behind it with them)."""
    w, sched, state, rec = _tp_loop(monkeypatch)
    build_step_1 = _builds_step_1(w, sched, state)

    def finish_step(pending, outputs):
        if raise_in == "finish_step":
            raise RuntimeError("boom")

    def build(pending, consecutive, fair):
        if pending is not None and pending.tp_seq == 1:
            raise RuntimeError("boom")
        return build_step_1(pending, consecutive, fair)

    w._finish_step, w._build_speculation = finish_step, build
    with pytest.raises(_StopLoopError):
        w.run()

    assert [str(e) for e in rec.errors] == ["boom"]
    assert rec.markers == [1]


def _raises(*a, **k):
    raise RuntimeError("boom")


def _raises_once(after=0):
    """Raises on call ``after + 1`` only."""
    calls = []

    def call(*a, **k):
        calls.append(None)
        if len(calls) == after + 1:
            raise RuntimeError("boom")
    return call


@pytest.mark.parametrize("head", [False, True], ids=["yield-away", "head"])
def test_a_forward_that_raised_fails_its_rows_and_the_loop_goes_on(monkeypatch, head):
    """Step 0's forward raises on every rank alike. Its rows fail as its
    verdict, which voids a head built on it on every rank, while a yield-away
    step (independent of step 0, and run by the followers whatever its
    verdict) still runs. Nothing reaches the error handler or stops."""
    w, sched, state, rec = _tp_loop(monkeypatch, forward_raises={0})
    w._build_speculation = _builds_step_1(w, sched, state, head=head)
    with pytest.raises(_StopLoopError):
        w.run()
    assert rec.errors == []
    assert set(rec.failed) == {0}
    assert rec.ran == ([[0]] if head else [[0], [1]])


@pytest.mark.parametrize("raise_at", ["threading", "clearing N's flag", "settling"])
def test_a_leader_that_drops_a_step_after_its_head_went_out_stops(monkeypatch, raise_at):
    """Head 1 went out during step 0, step 0 ran clean, and something on the
    leader alone raises before step 1 is submitted. The follower runs step 1,
    so the leader can't drop it and go on: its next step would wait on a rank
    that waits on step 1's collectives (#1171's live test). It stops."""
    w, sched, state, rec = _tp_loop(monkeypatch)
    w._build_speculation = _builds_step_1(w, sched, state, head=True)
    if raise_at == "threading":
        w._thread_outputs_to_speculative = _raises
    else:
        w._thread_outputs_to_speculative = lambda *a: None
    if raise_at == "clearing N's flag":
        w._clear_in_flight_flag = _raises_once()
    if raise_at == "settling":
        w._settle_speculation = _raises
    with pytest.raises(SystemExit, match="seq 1"):
        w.run()
    assert rec.ran == [[0]]


def test_a_leader_that_drops_a_step_it_broadcast_on_the_serial_path_stops(monkeypatch):
    w, sched, state, rec = _tp_loop(monkeypatch)
    w._build_speculation = lambda *a: (None, None)
    w._submit_step = _raises
    with pytest.raises(SystemExit, match="seq 0"):
        w.run()


@pytest.mark.parametrize("raise_in", ["post-processing", "clearing N's flag", "voiding the head"])
def test_a_leader_goes_on_after_a_raise_past_a_head_every_rank_voided(monkeypatch, raise_in):
    """Step 0 failed, so no rank runs head 1, and a raise that follows, even
    while the leader is still dropping the head, costs step 0's rows alone."""
    w, sched, state, rec = _tp_loop(monkeypatch, forward_raises={0})
    w._build_speculation = _builds_step_1(w, sched, state, head=True)
    if raise_in == "post-processing":
        w._finish_step = _raises
    elif raise_in == "clearing N's flag":
        w._clear_in_flight_flag = _raises_once()
    else:
        w._settle_speculation = _raises
    with pytest.raises(_StopLoopError):
        w.run()
    assert [str(e) for e in rec.errors] == ["boom"]


def test_a_leader_goes_on_after_a_raise_past_a_head_whose_rows_every_rank_dropped(monkeypatch):
    """Every row of head 1 was dropped on every rank, as a request's last step
    routinely does, so nothing runs: the raise that follows costs step 0 only."""
    w, sched, state, rec = _tp_loop(monkeypatch)
    w._build_speculation = _builds_step_1(w, sched, state, head=True)

    def thread(speculation, outputs):
        for rid in list(speculation.scheduled_batch.request_to_worker_graph):
            speculation.scheduled_batch.discard_rid(rid)

    w._thread_outputs_to_speculative = thread
    w._finish_step = _raises
    with pytest.raises(_StopLoopError):
        w.run()
    assert [str(e) for e in rec.errors] == ["boom"]


def test_a_leader_goes_on_after_a_raise_past_a_step_it_submitted(monkeypatch):
    """Step 0 runs on every rank, so a raise after its submit drops nothing the
    followers run: they get its marker instead."""
    w, sched, state, rec = _tp_loop(monkeypatch)
    w._build_speculation = lambda *a: (None, None)
    hold = w._hold_rows

    def hold_raising_once(lane, rids):
        hold(lane, rids)
        if rids and not rec.errors:
            raise RuntimeError("boom")

    w._hold_rows = hold_raising_once
    with pytest.raises(_StopLoopError):
        w.run()
    assert [str(e) for e in rec.errors] == ["boom"] and rec.markers == [0]


def test_a_followers_own_raise_does_not_stop_it(monkeypatch):
    """Only a leader knows the step it dropped was broadcast. A follower stays
    on the handler's path, however its step was taken."""
    w, sched, state, rec = _tp_loop(monkeypatch, follower=True)
    w._build_speculation = _builds_step_1(w, sched, state)
    w._settle_speculation = _raises
    w._close_tp_follow_step = lambda pending: None
    with pytest.raises(_StopLoopError):
        w.run()
    assert [str(e) for e in rec.errors] == ["boom"]


# ------------------------------------------------------------- registration

def test_nospec_marker_is_recorded_and_never_queued():
    s = _stub()
    s._register_tp_nospec(_nospec(4))
    assert 4 in s._tp_nospec
    assert s.scheduler.peek_tp_follow() is None


def test_head_for_a_closed_step_is_dropped_on_arrival():
    s = _stub()
    s._close_tp_follow_step(_pending(4))
    s._register_tp_follow(_head(["r0"], seq=5, from_seq=4))
    assert s.scheduler.peek_tp_follow() is None


def test_close_drops_an_already_queued_head_for_that_step_only():
    s = _stub()
    other = _head(["r9"], seq=9, from_seq=8)
    s.scheduler.register_tp_follow(_head(["r0"], seq=5, from_seq=4))
    s.scheduler.register_tp_follow(other)
    s._close_tp_follow_step(_pending(4))
    assert s.scheduler.peek_tp_follow() is other


def test_plain_and_unrelated_speculative_messages_queue_normally():
    s = _stub()
    plain = ScheduleTPNode(NODE, WALK, ["r0"], spec_seq=3)
    h = _head(["r0"], seq=5, from_seq=4)
    s._register_tp_follow(plain)
    s._register_tp_follow(h)
    assert list(s.scheduler.tp_batches_pending_schedule) == [plain, h]


def test_flag_off_ignores_markers_and_queues_heads_verbatim():
    """A rank running with the flag off records nothing and never
    second-guesses a head: the serial path runs whatever the leader sent."""
    s = _stub()
    s.tp_async_sched = False
    s._register_tp_nospec(_nospec(4))
    assert 4 not in s._tp_nospec
    h = _head(["r0"], seq=5, from_seq=4)
    s._register_tp_follow(h)
    assert s.scheduler.peek_tp_follow() is h


def test_empty_schedule_message_is_dropped_not_queued():
    """Defensive: the serial path indexes request_ids[0]."""
    s = _stub()
    s._register_tp_follow(ScheduleTPNode(NODE, WALK, [], spec_seq=3))
    assert s.scheduler.peek_tp_follow() is None


def test_nospec_store_is_bounded():
    s = _stub()
    for i in range(Worker._TP_NOSPEC_KEEP + 10):
        s._register_tp_nospec(_nospec(i))
    assert len(s._tp_nospec) == Worker._TP_NOSPEC_KEEP
    assert 0 not in s._tp_nospec and 9 not in s._tp_nospec
    assert Worker._TP_NOSPEC_KEEP + 9 in s._tp_nospec


# ------------------------------------------------------------------- await

def _await(s, pending, arm=None):
    return s._await_tp_follow_step(pending, arm or (lambda sp: None))


def test_await_builds_a_queued_head_and_arms_it():
    s = _stub()
    spec = object()
    s.build_results = [spec]
    s.scheduler.register_tp_follow(_head(["r0"], seq=5, from_seq=4))
    armed = []
    out = _output()
    assert _await(s, _pending(4, _future(s, out)), armed.append) == (out, spec)
    assert armed == [spec]
    assert s.communicator.waits == 0


def test_await_goes_serial_on_the_marker_without_waiting():
    s = _stub()
    s._register_tp_nospec(_nospec(4))
    out = _output()
    assert _await(s, _pending(4, _future(s, out))) == (out, None)
    assert s.communicator.waits == 0


def test_await_builds_a_head_that_lands_while_N_runs():
    """The point of the flag: the head is built DURING N. The call returns
    (output None) so the caller awaits N with the speculation already armed."""
    s = _stub()
    spec = object()
    s.build_results = [spec]
    s.communicator.deliveries = {1: [_head(["r0"], seq=5, from_seq=4)]}
    armed = []
    got = _await(s, _pending(4, _future(s, _output(), done_after=10)), armed.append)
    assert got == (None, spec) and armed == [spec]
    assert s.communicator.waits == 1


def test_await_returns_on_the_marker_while_N_runs():
    s = _stub()
    s.communicator.deliveries = {1: [_nospec(4)]}
    got = _await(s, _pending(4, _future(s, _output(), done_after=10)))
    assert got == (None, None)
    assert s.communicator.waits == 1


def test_await_keeps_waiting_after_N_finished_until_a_decision():
    """N finished first (fast rank, slow leader build). No guessing: wait for
    the head, then build it from N's real outputs."""
    s = _stub()
    spec = object()
    s.build_results = [spec]
    out = _output()
    s.communicator.deliveries = {4: [_head(["r0"], seq=5, from_seq=4)]}
    assert _await(s, _pending(4, _future(s, out, done_after=2))) == (out, spec)
    assert s.communicator.waits == 4
    assert s.scheduler.peek_tp_follow() is None   # consumed by the build


def test_await_waits_for_a_late_marker_then_goes_serial():
    s = _stub()
    out = _output()
    s.communicator.deliveries = {3: [_nospec(4)]}
    assert _await(s, _pending(4, _future(s, out))) == (out, None)
    assert s.communicator.waits == 3


def test_await_drops_the_head_on_allocation_failure_without_building():
    s = _stub()
    s.build_results = [object()]   # would build if asked — it must not be
    s.scheduler.register_tp_follow(_head(["r0"], seq=5, from_seq=4))
    out = _output()
    pending = _pending(4, _future(s, out), admit_error="kv cache: out of pages")
    assert _await(s, pending) == (out, None)
    assert s.scheduler.peek_tp_follow() is None
    assert s.build_results, "head must be dropped, not built"


def test_await_drops_the_head_on_failed_rids_without_building():
    """Same verdict class as allocation failure: the leader clears its whole
    speculation on ``failed_requests`` before threading, so the follower must
    not build (and then unwind) what the leader will not run."""
    s = _stub()
    s.build_results = [object()]
    s.scheduler.register_tp_follow(_head(["r0"], seq=5, from_seq=4))
    out = _output()
    pending = _pending(4, _future(s, out), failed_requests=["r0"])
    assert _await(s, pending) == (out, None)
    assert s.scheduler.peek_tp_follow() is None
    assert s.build_results, "head must be dropped, not built"


def test_await_has_no_verdict_to_apply_while_N_still_runs():
    """admit_error is N's verdict; while N runs there is none yet, so a
    head that lands early is built (the leader built it during N too — the
    shared clear-or-thread flow after N voids it on both ranks alike). The
    verdict is only read once ``future.done()``, never before."""
    s = _stub()
    spec = object()
    s.build_results = [spec]
    s.communicator.deliveries = {1: [_head(["r0"], seq=5, from_seq=4)]}
    pending = _pending(
        4, _future(s, _output(), done_after=10), admit_error="kv cache: out of pages",
    )
    assert _await(s, pending) == (None, spec)


@pytest.mark.parametrize("verdict", [
    {"admit_error": "kv cache: out of pages"}, {"failed_requests": ["r0"]},
])
def test_with_the_prefix_cache_open_await_builds_and_arms_a_head_its_step_voids(verdict):
    s = _stub(cache_open=True)
    spec = object()
    s.build_results = [spec]
    s.scheduler.register_tp_follow(_head(["r0"], seq=5, from_seq=4))
    armed = []
    out = _output()
    assert _await(s, _pending(4, _future(s, out), **verdict), armed.append) == (out, spec)
    assert armed == [spec], "the leader pre-planned this head, and this rank did not"


def test_await_retries_until_the_head_is_buildable():
    """Fresh rid not ready on this rank yet: keep polling readiness, do not
    go serial (the leader will run that composition)."""
    s = _stub()
    spec = object()
    s.build_results = [None, None, spec]
    s.scheduler.register_tp_follow(_head(["r0", "f1"], seq=5, from_seq=4))
    out = _output()
    assert _await(s, _pending(4, _future(s, out))) == (out, spec)
    assert s.communicator.waits == 2


def test_await_goes_serial_when_the_leader_moved_past_the_step():
    """The decision about s is sent before anything the leader broadcasts
    after s (per-peer FIFO). A later message at the FIFO front with no
    decision recorded therefore means none is coming — e.g. the leader runs
    with the flag off — so the follower goes serial instead of spinning."""
    s = _stub()
    later = ScheduleTPNode(NODE, WALK, ["r0"], spec_seq=6)   # plain, later
    s.scheduler.register_tp_follow(later)
    out = _output()
    assert _await(s, _pending(4, _future(s, out))) == (out, None)
    assert s.communicator.waits == 0
    assert s.scheduler.peek_tp_follow() is later   # left for the serial path
    assert 4 in s._tp_nospec


def test_await_treats_a_later_head_at_the_front_the_same_way():
    s = _stub()
    later = _head(["r9"], seq=9, from_seq=8)
    s.scheduler.register_tp_follow(later)
    got = _await(s, _pending(4, _future(s, _output(), done_after=10)))
    assert got == (None, None)                     # N still runs; caller awaits it
    assert s.scheduler.peek_tp_follow() is later


def test_a_forward_that_raised_is_the_steps_verdict_and_voids_its_head():
    """N's forward raised: every row of N fails and the head built on N is
    dropped, as the leader drops it on the same verdict, instead of the raise
    reaching the loop's handler."""
    s = _stub()
    s.scheduler.register_tp_follow(_head(["r0"], seq=5, from_seq=4))

    def raises():
        raise RuntimeError("CUDA error")

    p = _pending(4, types.SimpleNamespace(done=lambda: True, result=raises))
    outputs, spec = _await(s, p)
    assert spec is None and not outputs.per_rid_outputs
    assert set(p.node_batch.failed_requests) == {"r0"}
    assert s.scheduler.peek_tp_follow() is None


def test_await_never_returns_without_a_decision_once_N_finished():
    s = _stub()   # nothing ever arrives
    with pytest.raises(TimeoutError):
        _await(s, _pending(4, _future(s, _output())))
