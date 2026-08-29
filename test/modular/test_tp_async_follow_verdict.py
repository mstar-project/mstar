"""TP async scheduling — a follower must always SETTLE the leader's decision
about step N before it post-processes N, and it must never guess.

The leader broadcasts its speculation DURING its forward N; a follower can
finish N first (faster rank, slow leader build). If a follower went serial
whenever the head had not arrived yet, a head landing later could reach the
serial path with a continuing rid whose loop ended at N — a batch the serial
readiness check can never build, while the leader runs it (one wasted
forward) and then hangs in the collective. So:

* the leader ALWAYS sends one of {speculative head from N, empty "no-spec"
  marker for N} for every lockstep step it looked at speculating from;
* the follower's ``_resolve_follow_speculation`` waits (bounded by that
  send) for exactly one of them right after await(N): builds the head via the
  speculation flow (from N's real outputs), or goes serial on the marker;
  ``allocation_failed`` on N drops a head without building it (the verdict
  the leader reaches before threading);
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

from mstar.utils.containers import RecentSet  # noqa: E402
from mstar.utils.ipc_format import ScheduleTPNode  # noqa: E402
from mstar.worker.micro_scheduler import MicroScheduler  # noqa: E402
from mstar.worker.worker import Worker  # noqa: E402

NODE = "LLM"
WALK = "decode"


class _FakeEngineManager:
    def get_engine(self, node_name):
        return types.SimpleNamespace(check_ready=lambda *a: True)


class _Communicator:
    """wait_for_work delivers one queued message batch per call; the test
    pushes messages via ``inbox``. Raises if the follower keeps waiting after
    the inbox is dry — that would be a hang."""

    def __init__(self, stub):
        self.stub = stub
        self.inbox = []
        self.waits = 0
        self.max_waits = 50

    def wait_for_work(self, timeout_ms=50):
        self.waits += 1
        if self.waits > self.max_waits:
            raise TimeoutError("follower waited too long — would hang")


def _stub():
    stub = types.SimpleNamespace()
    stub.worker_id = "w1"
    stub.tp_async_sched = True
    stub._tp_nospec = RecentSet(Worker._TP_NOSPEC_KEEP)
    stub.scheduler = MicroScheduler(
        engine_manager=_FakeEngineManager(), parallel_leader_nodes=set(),
    )
    stub.communicator = _Communicator(stub)
    stub._check_ready_tensors = lambda: None
    stub._poll_stream_buffers = lambda: None
    stub.build_results = []   # what _try_follow_speculation returns, in order

    def _process_messages():
        while stub.communicator.inbox:
            stub._register_tp_follow(stub.communicator.inbox.pop(0))

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
    stub._close_tp_follow_step = lambda p: Worker._close_tp_follow_step(stub, p)
    stub._resolve_follow_speculation = (
        lambda p, o, arm: Worker._resolve_follow_speculation(stub, p, o, arm)
    )
    return stub


def _pending(tp_seq):
    return types.SimpleNamespace(
        node_name=NODE, graph_walk=WALK, tp_seq=tp_seq,
        batch=types.SimpleNamespace(node_objects={"r0": object()}),
    )


def _output(allocation_failed=False):
    return types.SimpleNamespace(allocation_failed=allocation_failed, failed_requests=[])


def _head(rids, seq, from_seq):
    return ScheduleTPNode(
        node_name=NODE, graph_walk=WALK, request_ids=list(rids),
        speculative=True, spec_seq=seq, spec_from_seq=from_seq,
    )


def _nospec(from_seq):
    return ScheduleTPNode(
        node_name=NODE, graph_walk=WALK, request_ids=[],
        speculative=True, spec_seq=-1, spec_from_seq=from_seq,
    )


# ------------------------------------------------------------- registration

def test_nospec_marker_is_recorded_and_never_queued():
    s = _stub()
    s._register_tp_follow(_nospec(4))
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


def test_flag_off_still_swallows_empty_markers_but_queues_heads_verbatim():
    """A rank running with the flag off must not crash on an empty marker
    (the serial path indexes request_ids[0]) and must not second-guess."""
    s = _stub()
    s.tp_async_sched = False
    s._register_tp_follow(_nospec(4))
    h = _head(["r0"], seq=5, from_seq=4)
    s._register_tp_follow(h)
    assert s.scheduler.peek_tp_follow() is h


def test_nospec_store_is_bounded():
    s = _stub()
    for i in range(Worker._TP_NOSPEC_KEEP + 10):
        s._register_tp_follow(_nospec(i))
    assert len(s._tp_nospec) == Worker._TP_NOSPEC_KEEP
    assert 0 not in s._tp_nospec and 9 not in s._tp_nospec
    assert Worker._TP_NOSPEC_KEEP + 9 in s._tp_nospec


# ----------------------------------------------------------------- resolve

def test_resolve_builds_a_queued_head_and_arms_it():
    s = _stub()
    spec = object()
    s.build_results = [spec]
    s.scheduler.register_tp_follow(_head(["r0"], seq=5, from_seq=4))
    armed = []
    got = s._resolve_follow_speculation(_pending(4), _output(), armed.append)
    assert got is spec and armed == [spec]
    assert s.communicator.waits == 0


def test_resolve_goes_serial_on_the_marker_without_waiting():
    s = _stub()
    s._register_tp_follow(_nospec(4))
    got = s._resolve_follow_speculation(_pending(4), _output(), lambda sp: None)
    assert got is None
    assert s.communicator.waits == 0


def test_resolve_waits_for_a_late_head_then_builds_it():
    s = _stub()
    spec = object()
    s.build_results = [spec]
    s.communicator.inbox = [_head(["r0"], seq=5, from_seq=4)]   # arrives after await
    got = s._resolve_follow_speculation(_pending(4), _output(), lambda sp: None)
    assert got is spec
    assert s.communicator.waits == 1
    assert s.scheduler.peek_tp_follow() is None   # consumed by the build (stubbed pop)


def test_resolve_waits_for_a_late_marker_then_goes_serial():
    s = _stub()
    s.communicator.inbox = [_nospec(4)]
    got = s._resolve_follow_speculation(_pending(4), _output(), lambda sp: None)
    assert got is None and s.communicator.waits == 1


def test_resolve_drops_the_head_on_allocation_failure_without_building():
    s = _stub()
    s.build_results = [object()]   # would build if asked — it must not be
    s.scheduler.register_tp_follow(_head(["r0"], seq=5, from_seq=4))
    got = s._resolve_follow_speculation(
        _pending(4), _output(allocation_failed=True), lambda sp: None,
    )
    assert got is None
    assert s.scheduler.peek_tp_follow() is None
    assert s.build_results, "head must be dropped, not built"


def test_resolve_retries_until_the_head_is_buildable():
    """Fresh rid not ready on this rank yet: keep polling readiness, do not
    go serial (the leader will run that composition)."""
    s = _stub()
    spec = object()
    s.build_results = [None, None, spec]
    s.scheduler.register_tp_follow(_head(["r0", "f1"], seq=5, from_seq=4))
    got = s._resolve_follow_speculation(_pending(4), _output(), lambda sp: None)
    assert got is spec
    assert s.communicator.waits == 2


def test_resolve_ignores_a_head_from_another_step():
    """A queued head speculated from a different step is neither built nor
    dropped; the decision for THIS step still has to arrive."""
    s = _stub()
    other = _head(["r9"], seq=9, from_seq=8)
    s.scheduler.register_tp_follow(other)
    s.communicator.inbox = [_nospec(4)]
    got = s._resolve_follow_speculation(_pending(4), _output(), lambda sp: None)
    assert got is None
    assert s.scheduler.peek_tp_follow() is other


def test_resolve_never_returns_without_a_decision():
    s = _stub()   # nothing ever arrives
    with pytest.raises(TimeoutError):
        s._resolve_follow_speculation(_pending(4), _output(), lambda sp: None)
