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
is exercised against the graph API elsewhere; here we pin the protocol. Every
test runs without the ranks' vote on N's verdict and with a one-rank one,
which leaves the verdict as this rank found it.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.engine.resources.step import FULL_ADMIT_OK  # noqa: E402
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
# Set per test by ``_vote_switch``: whether ``_stub`` installs a one-rank vote.
ONE_RANK_VOTE = False


@pytest.fixture(autouse=True, params=[False, True], ids=["no-vote", "one-rank-vote"])
def _vote_switch(request, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "ONE_RANK_VOTE", request.param)


def _one_rank_vote(pending):
    """What ``Worker._agree_on_step`` makes of N over a one-rank instance."""
    pending.voted = True
    pending.failed_on_a_rank = bool(pending.node_batch.failed_requests)


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
    stub._prefix_cache_open = lambda node: Worker._prefix_cache_open(stub, node)
    # The rank vote over N's verdict, off unless a test installs ``stub.vote``
    # (the vote itself, over real ranks, is test_tp_step_vote.py's).
    stub.vote = _one_rank_vote if ONE_RANK_VOTE else None

    def _agree_on_step(pending):
        stub.vote(pending)

    stub._votes_on_step = lambda p: stub.vote is not None
    stub._agree_on_step = _agree_on_step
    stub._step_outputs = lambda p: Worker._step_outputs(stub, p)
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
        voted=False, failed_on_a_rank=False,
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
    (None, None, (False, None)),
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


def test_await_never_returns_without_a_decision_once_N_finished():
    s = _stub()   # nothing ever arrives
    with pytest.raises(TimeoutError):
        _await(s, _pending(4, _future(s, _output())))


# ---------------------------------------------------------------- the vote

def _vote(log, *failed_elsewhere, failed_on_a_rank=False):
    """A vote over N in which another rank failed ``failed_elsewhere`` after
    the forward, or (``failed_on_a_rank``) a row only it held: the agreed
    verdict lands on this rank's step."""
    def vote(pending):
        log.append(("vote", pending.tp_seq))
        for rid in failed_elsewhere:
            pending.node_batch.failed_requests.setdefault(rid, "failed on another rank")
        pending.failed_on_a_rank = failed_on_a_rank or bool(failed_elsewhere)
    return vote


def _logged_builds(s, log):
    build = s._try_follow_speculation

    def _try(pending):
        log.append(("build",))
        return build(pending)
    s._try_follow_speculation = _try


def test_await_votes_once_N_is_done_before_the_head_can_be_built():
    """The vote waits for every rank to finish N and for nothing else: a head
    whose fresh rid has not landed here yet must not hold the leader in it."""
    s = _stub()
    log = []
    s.vote = _vote(log)
    _logged_builds(s, log)
    spec = object()
    s.build_results = [None, None, spec]
    s.scheduler.register_tp_follow(_head(["r0", "f1"], seq=5, from_seq=4))
    out = _output()
    assert _await(s, _pending(4, _future(s, out))) == (out, spec)
    assert log == [("vote", 4), ("build",), ("build",), ("build",)]


def test_await_leaves_the_vote_to_the_caller_while_N_runs():
    """A head built during N returns before N is done; the run loop votes once
    it has N's outputs."""
    s = _stub()
    log = []
    s.vote = _vote(log)
    spec = object()
    s.build_results = [spec]
    s.communicator.deliveries = {1: [_head(["r0"], seq=5, from_seq=4)]}
    assert _await(s, _pending(4, _future(s, _output(), done_after=10))) == (None, spec)
    assert log == []


def test_await_drops_a_head_the_agreed_verdict_voids_though_this_rank_ran_N_clean():
    """Another rank failed r0 after the forward. Rank 0 clears its speculation
    on the agreed verdict, so this rank must not run the head either."""
    s = _stub()
    log = []
    s.vote = _vote(log, "r0")
    s.build_results = [object()]
    s.scheduler.register_tp_follow(_head(["r0"], seq=5, from_seq=4))
    out = _output()
    pending = _pending(4, _future(s, out))
    assert _await(s, pending) == (out, None)
    assert s.scheduler.peek_tp_follow() is None
    assert s.build_results, "head must be dropped, not built"
    assert pending.node_batch.failed_requests == {"r0": "failed on another rank"}


def test_await_drops_a_head_another_rank_voided_on_a_row_only_it_held():
    """Rank 0 failed a row before it sent N (its input was gone) and clears its
    speculation on that; the row was never in this rank's N."""
    s = _stub()
    log = []
    s.vote = _vote(log, failed_on_a_rank=True)
    s.build_results = [object()]
    s.scheduler.register_tp_follow(_head(["r0"], seq=5, from_seq=4))
    out = _output()
    pending = _pending(4, _future(s, out))
    assert _await(s, pending) == (out, None)
    assert s.build_results, "head must be dropped, not built"
    assert pending.node_batch.failed_requests == {}
