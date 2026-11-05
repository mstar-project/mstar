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
* a step whose forward raised is closed (``Lockstep.close_follow_step``): its head
  is dropped, queued or arriving.

Drives a real ``Worker`` and its real ``Lockstep``, which holds the FIFO.
``_try_follow_speculation`` is stubbed per test — its own build logic
is exercised against the graph API elsewhere; here we pin the protocol.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.engine.resources.step import FULL_ADMIT_OK  # noqa: E402
from mstar.testing_utils.worker import make_scheduler, make_worker  # noqa: E402
from mstar.utils.ipc_format import ScheduleTPNode, TPNoSpeculation, WorkerMessageType  # noqa: E402
from mstar.worker.lockstep import (  # noqa: E402
    _TP_NOSPEC_KEEP,
    parse_tp_async_sched,
    tp_async_sched_setting,
)
from mstar.worker.scheduled_batch import ScheduledBatch  # noqa: E402

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

    def __init__(self):
        self.inbox = []
        self.deliveries = {}
        self.waits = 0
        self.max_waits = 50

    def wait_for_work(self, timeout_ms=50):
        self.waits += 1
        if self.waits > self.max_waits:
            raise TimeoutError("follower waited too long — would hang")
        self.inbox.extend(self.deliveries.pop(self.waits, []))


def _stub(cache_open: bool = False, async_sched=(True, None)):
    """A follower of NODE: a real ``Worker``, ``Lockstep`` and ``MicroScheduler``."""
    engine_manager = _FakeEngineManager(cache_open)
    scheduler = make_scheduler(
        engine_manager=engine_manager, parallel_leader_nodes=set(), parallel_nodes={NODE},
        async_sched=async_sched,
    )
    stub = make_worker(
        worker_id="w1", scheduler=scheduler, lockstep=scheduler.lockstep,
        admission=scheduler.admission, engine_manager=engine_manager,
        communicator=_Communicator(),
    )
    stub._check_ready_tensors = lambda: None
    stub._poll_stream_buffers = lambda: None
    stub.build_results = []   # what _try_follow_speculation returns, in order

    def _process_messages():
        while stub.communicator.inbox:
            msg = stub.communicator.inbox.pop(0)
            if isinstance(msg, TPNoSpeculation):
                stub.lockstep.enqueue_no_spec(msg)
            else:
                stub.lockstep.enqueue_follow(msg)

    def _try_follow(pending):
        # like the real one: a successful build consumes the FIFO head
        if stub.build_results:
            res = stub.build_results.pop(0)
            if res is not None and stub.lockstep.follow_head() is not None:
                stub.lockstep.pop_follow_head()
            return res
        return None

    stub._process_messages = _process_messages
    stub._try_follow_speculation = _try_follow
    return stub


def _leader(leads: bool = True, async_nodes=None, async_sched: bool = True):
    """Rank 0 of NODE's instance, or with ``leads`` off a follower shard of it. Its
    ``Lockstep`` is real; what it sends lands in ``.sent``."""
    sent = []
    group = types.SimpleNamespace(_workers=["w0", "w1"])
    return types.SimpleNamespace(sent=sent, lockstep=make_worker(
        parallel_nodes={NODE}, parallel_leader_nodes={NODE} if leads else set(), is_tp_follower=not leads,
        tp_async_sched=async_sched, tp_async_nodes=async_nodes,
        _graph_runtime=types.SimpleNamespace(get_sharding_config=lambda rid: types.SimpleNamespace(
            get_sharding_group=lambda node, walk: group,
        )),
        communicator=types.SimpleNamespace(send=lambda dest, msg: sent.append((dest, msg))),
    ).lockstep)


def _markers(leader, pending, speculation) -> list[tuple[str, int]]:
    """Who the leader tells that no head follows ``pending``, and which step it names."""
    leader.sent.clear()
    leader.lockstep.maybe_broadcast_nospec(pending, speculation)
    assert all(msg.message_type == WorkerMessageType.TP_NO_SPEC for _, msg in leader.sent)
    return [(dest, msg.body.spec_from_seq) for dest, msg in leader.sent]


def _pending(tp_seq, future=None, admit_error=None, failed_requests=()):
    """``admit_error`` / ``failed_requests`` are N's verdict, on its
    ``ExecutingBatch`` the way the GPU thread leaves them."""
    return types.SimpleNamespace(
        node_name=NODE, graph_walk=WALK, tp_seq=tp_seq, future=future,
        batch=ScheduledBatch(node_name=NODE, graph_walk=WALK, request_to_worker_graph={"r0": "wg0"}),
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
    assert parse_tp_async_sched("0") == (False, None)
    assert parse_tp_async_sched("") == (False, None)
    assert parse_tp_async_sched("1") == (True, None)
    assert parse_tp_async_sched("thinker, talker") == (
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
    assert tp_async_sched_setting(env, config) == want


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
    assert tp_async_sched_setting(env, config) == want


def test_node_list_narrows_which_parallel_nodes_take_part():
    """Per-node control: a rank whose list excludes NODE behaves exactly as
    with the flag off for NODE — leader owes no marker, follower records
    nothing and queues heads verbatim for the serial path."""
    assert _markers(_leader(async_nodes=frozenset({"talker"})), _pending(4), None) == []
    assert _markers(_leader(async_nodes=frozenset({NODE})), _pending(4), None) == [("w1", 4)]

    s = _stub(async_sched=(True, frozenset({"talker"})))
    s.lockstep.enqueue_no_spec(_nospec(4))
    assert not s.lockstep.is_no_spec(4)
    h = _head(["r0"], seq=5, from_seq=4)
    s.lockstep.enqueue_follow(h)
    assert s.lockstep.follow_head() is h


# ------------------------------------------------------- the leader's duty

def test_leader_owes_a_marker_whenever_no_head_went_out():
    """A follower settles on {head from s, marker for s}; the leader must send
    one of them for EVERY lockstep step it looked at speculating from — also
    when it did speculate, but into a non-parallel node, which is never
    broadcast (tp_seq stays -1). Missing that marker hangs the followers."""
    ld = _leader()
    p = _pending(4)
    assert _markers(ld, p, None) == [("w1", 4)]
    assert _markers(ld, p, types.SimpleNamespace(tp_seq=-1)) == [("w1", 4)]
    assert _markers(ld, p, types.SimpleNamespace(tp_seq=5)) == []


def test_only_the_leader_of_a_lockstep_node_owes_a_marker():
    assert _markers(_leader(leads=False), _pending(4), None) == []
    assert _markers(_leader(async_sched=False), _pending(4), None) == []


# ------------------------------------------------------------- registration

def test_nospec_marker_is_recorded_and_never_queued():
    s = _stub()
    s.lockstep.enqueue_no_spec(_nospec(4))
    assert s.lockstep.is_no_spec(4)
    assert s.lockstep.follow_head() is None


def test_head_for_a_closed_step_is_dropped_on_arrival():
    s = _stub()
    s.lockstep.close_follow_step(_pending(4))
    s.lockstep.enqueue_follow(_head(["r0"], seq=5, from_seq=4))
    assert s.lockstep.follow_head() is None


def test_close_drops_an_already_queued_head_for_that_step_only():
    s = _stub()
    other = _head(["r9"], seq=9, from_seq=8)
    s.lockstep.enqueue_follow(_head(["r0"], seq=5, from_seq=4))
    s.lockstep.enqueue_follow(other)
    s.lockstep.close_follow_step(_pending(4))
    assert s.lockstep.follow_head() is other


def test_plain_and_unrelated_speculative_messages_queue_normally():
    s = _stub()
    plain = ScheduleTPNode(NODE, WALK, ["r0"], spec_seq=3)
    h = _head(["r0"], seq=5, from_seq=4)
    s.lockstep.enqueue_follow(plain)
    s.lockstep.enqueue_follow(h)
    assert list(s.lockstep._follow_fifo) == [plain, h]


def test_flag_off_ignores_markers_and_queues_heads_verbatim():
    """A rank running with the flag off records nothing and never
    second-guesses a head: the serial path runs whatever the leader sent."""
    s = _stub(async_sched=(False, None))
    s.lockstep.enqueue_no_spec(_nospec(4))
    assert not s.lockstep.is_no_spec(4)
    h = _head(["r0"], seq=5, from_seq=4)
    s.lockstep.enqueue_follow(h)
    assert s.lockstep.follow_head() is h


def test_empty_schedule_message_is_dropped_not_queued():
    """Defensive: the serial path indexes request_ids[0]."""
    s = _stub()
    s.lockstep.enqueue_follow(ScheduleTPNode(NODE, WALK, [], spec_seq=3))
    assert s.lockstep.follow_head() is None


def test_nospec_store_is_bounded():
    s = _stub()
    for i in range(_TP_NOSPEC_KEEP + 10):
        s.lockstep.enqueue_no_spec(_nospec(i))
    assert len(s.lockstep._no_spec) == _TP_NOSPEC_KEEP
    assert not s.lockstep.is_no_spec(0) and not s.lockstep.is_no_spec(9)
    assert s.lockstep.is_no_spec(_TP_NOSPEC_KEEP + 9)


# ------------------------------------------------------------------- await

def _await(s, pending, arm=None):
    return s._await_tp_follow_step(pending, arm or (lambda sp: None))


def test_await_builds_a_queued_head_and_arms_it():
    s = _stub()
    spec = object()
    s.build_results = [spec]
    s.lockstep.enqueue_follow(_head(["r0"], seq=5, from_seq=4))
    armed = []
    out = _output()
    assert _await(s, _pending(4, _future(s, out)), armed.append) == (out, spec)
    assert armed == [spec]
    assert s.communicator.waits == 0


def test_await_goes_serial_on_the_marker_without_waiting():
    s = _stub()
    s.lockstep.enqueue_no_spec(_nospec(4))
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
    assert s.lockstep.follow_head() is None   # consumed by the build


def test_await_waits_for_a_late_marker_then_goes_serial():
    s = _stub()
    out = _output()
    s.communicator.deliveries = {3: [_nospec(4)]}
    assert _await(s, _pending(4, _future(s, out))) == (out, None)
    assert s.communicator.waits == 3


def test_await_drops_the_head_on_allocation_failure_without_building():
    s = _stub()
    s.build_results = [object()]   # would build if asked — it must not be
    s.lockstep.enqueue_follow(_head(["r0"], seq=5, from_seq=4))
    out = _output()
    pending = _pending(4, _future(s, out), admit_error="kv cache: out of pages")
    assert _await(s, pending) == (out, None)
    assert s.lockstep.follow_head() is None
    assert s.build_results, "head must be dropped, not built"


def test_await_drops_the_head_on_failed_rids_without_building():
    """Same verdict class as allocation failure: the leader clears its whole
    speculation on ``failed_requests`` before threading, so the follower must
    not build (and then unwind) what the leader will not run."""
    s = _stub()
    s.build_results = [object()]
    s.lockstep.enqueue_follow(_head(["r0"], seq=5, from_seq=4))
    out = _output()
    pending = _pending(4, _future(s, out), failed_requests=["r0"])
    assert _await(s, pending) == (out, None)
    assert s.lockstep.follow_head() is None
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
    s.lockstep.enqueue_follow(_head(["r0"], seq=5, from_seq=4))
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
    s.lockstep.enqueue_follow(_head(["r0", "f1"], seq=5, from_seq=4))
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
    s.lockstep.enqueue_follow(later)
    out = _output()
    assert _await(s, _pending(4, _future(s, out))) == (out, None)
    assert s.communicator.waits == 0
    assert s.lockstep.follow_head() is later   # left for the serial path
    assert s.lockstep.is_no_spec(4)


def test_await_treats_a_later_head_at_the_front_the_same_way():
    s = _stub()
    later = _head(["r9"], seq=9, from_seq=8)
    s.lockstep.enqueue_follow(later)
    got = _await(s, _pending(4, _future(s, _output(), done_after=10)))
    assert got == (None, None)                     # N still runs; caller awaits it
    assert s.lockstep.follow_head() is later


def test_await_never_returns_without_a_decision_once_N_finished():
    s = _stub()   # nothing ever arrives
    with pytest.raises(TimeoutError):
        _await(s, _pending(4, _future(s, _output())))
