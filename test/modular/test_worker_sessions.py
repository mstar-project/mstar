"""Worker-side sessions: what is freed when, and what is handed back.

A request in a session gives its resource state back to the session instead of
freeing it. Ending the session is a message of its own: the removal never
carries a flag for it, because ``_process_message_list`` parks a REMOVE whose
rid has no handle — a request drained before its NEW_REQUEST arrived — until an
admission that never comes, which would strand the teardown and its ACK behind
the barrier's TTL. A
TEARDOWN_SESSION is never parked, so it is the one that frees the state and
ACKs, and the worker only has to hold it while one of the session's requests is
still on its way out: that request's removal would otherwise hand state back to
a session that no longer exists.

The cases that turn on the parking are delivered through
``_process_message_list``, not by calling ``_remove_request`` directly, since
that is the hop the parking happens on.
"""

from __future__ import annotations

import sys
from collections import deque
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.sessions import RequestSession
from mstar.utils.ipc_format import (
    ConductorMessageType,
    DrainRequest,
    MessageSource,
    NewRequest,
    OffloadDelta,
    RemoveRequest,
    TeardownSession,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.sessions import WorkerSessionManager
from mstar.worker.worker import Worker


def _worker(known_rids=("X",), sessions=None, in_flight=(), is_follower=False):
    w = Worker.__new__(Worker)
    w.worker_id = "w0"
    w.is_tp_follower = is_follower
    w.sent = []
    w.removed = []
    w.admitted = []   # rids whose engine state was opened
    w.removed_sessions = []
    w.overshot = []
    w.idle_seqs = []
    w.session_errors = {}
    w.communicator = SimpleNamespace(
        send=lambda e, msg=None, m=None: w.sent.append((e, msg if msg is not None else m))
    )
    w._in_flight_rids = set(in_flight)
    w._pending_drains = set()
    w._draining_rids = set()
    w._reads_done_sent = set()
    w._pending_removes = set()
    w._handles_awaiting_admit = set()
    w._drains_awaiting_new = set()
    w._lockstep_followers = {}
    w.parallel_leader_nodes = set()
    w._deferred_idle_seq = {}
    w._session_idle_seq = 0
    w._tp_forwards_awaiting_step = deque()
    w._tp_broadcast_seq = 0
    w._tp_last_seq_sent = {}
    w._last_active = {}
    w._unprocessed_messages = {}
    w._my_consumer_connections = []
    w.streaming_buffers = {}
    w.wakeup_event = SimpleNamespace(register_futures=lambda f: None)
    w.scheduler = SimpleNamespace(
        clear_rid=lambda rid, wire_rid=None: None,
        clear_wire_rid=lambda wire_rid: None,
        fail_rids=lambda rids: None,
        pending_tp_follow_count={},
        last_consumed_tp_seq=0,
        replay_forwarded_deltas=lambda deltas: True,
        replay_owed_step_deltas=lambda: True,
    )
    # Identity interning, as main's worker tests do: the rid string is its own
    # handle, so the string/handle split is exercised without a real runtime.
    interned = set(known_rids)
    w.request_state = SimpleNamespace(
        per_request_info={
            rid: SimpleNamespace(sharding_config=None, stream_buffers={})
            for rid in known_rids
        },
        remove_request=lambda rid: w.request_state.per_request_info.pop(rid, None),
        add_request=lambda rid, info: w.request_state.per_request_info.setdefault(
            rid, SimpleNamespace(sharding_config=None, stream_buffers={}),
        ),
    )

    def _add(request_id, **kwargs):
        interned.add(request_id)
        return request_id

    w._graph_runtime = SimpleNamespace(
        add_request=_add,
        get_rid_handle=lambda r: r if r in interned else None,
        get_rid_string=lambda h: h,
        remove_request=lambda h: None,
        get_sharding_config=lambda h: SimpleNamespace(groups=[]),
        ingest_inputs_batch=lambda edges, can_buffer=False: None,
    )
    # the real bookkeeping, asking this worker what is still leaving
    w._sessions = WorkerSessionManager(is_leaving=w._rid_is_leaving)
    for session_id, rids in (sessions or {}).items():
        for rid in rids:
            interned.add(rid)
            w._sessions.bind(rid, session_id)
    w.engine_manager = SimpleNamespace(
        remove_request=lambda rid, overshot_nodes=frozenset(), **kw: (
            w.removed.append(rid),
            w.overshot.append((rid, overshot_nodes)),
            w.idle_seqs.append((rid, kw.get("session_idle_seq", -1))),
        ),
        remove_session=lambda sid: w.removed_sessions.append(sid) or [],
        add_request=lambda rid, cfgs, session=None: w.admitted.append(rid),
        take_session_error=lambda sid: w.session_errors.pop(sid, None),
        evictable_nodes=lambda: [],
    )
    w.profile_info = SimpleNamespace(pop_request=lambda rid: None)
    w.tensor_manager = SimpleNamespace(
        has_inflight_reads=lambda rid: False,
        force_cleanup_request=lambda rid: None,
        register_request=lambda rid, cfg: None,
        start_read_tensors=lambda rid, inputs, graph_walk: [],
    )
    return w


def _acks(w):
    return [
        m.body.session_id for e, m in w.sent
        if e == "conductor"
        and m.message_type == ConductorMessageType.SESSION_TORN_DOWN
    ]


def _reads_done(w):
    return [
        m.body.request_id for e, m in w.sent
        if e == "conductor"
        and m.message_type == ConductorMessageType.READS_DONE
    ]


def _forwarded_removes(w):
    return [
        m.body for e, m in w.sent
        if m.message_type == WorkerMessageType.REMOVE_REQUEST
    ]


def _deliver(w, *bodies):
    """Hand the worker messages the way the communicator does, so the
    out-of-order parking in ``_process_message_list`` is in the path."""
    types = {
        RemoveRequest: WorkerMessageType.REMOVE_REQUEST,
        TeardownSession: WorkerMessageType.TEARDOWN_SESSION,
    }
    Worker._process_message_list(w, [
        WorkerMessage(message_type=types[type(body)], body=body)
        for body in bodies
    ])


# ── removal inside a live session ───────────────────────────────────────────

def test_a_plain_removal_leaves_the_session_holding_the_state():
    w = _worker(sessions={"s": {"X"}})

    Worker._remove_request(w, RemoveRequest(request_id="X"))

    assert w.removed == ["X"]
    assert w.removed_sessions == []
    assert _acks(w) == []
    # the rid is gone, the session is not
    assert w._sessions.session_of("X") is None
    assert w._sessions.get_rids("s") == set()


def test_the_removal_then_the_teardown_frees_the_session_and_acks():
    # how the conductor ends a session whose last request carried end_session
    w = _worker(sessions={"s": {"X"}})

    _deliver(w, RemoveRequest(request_id="X"), TeardownSession(session_id="s"))

    assert w.removed == ["X"]
    assert w.removed_sessions == ["s"]
    assert _acks(w) == ["s"]
    assert w._sessions.get_rids("s") == set()


def test_a_sessionless_removal_is_unchanged():
    w = _worker()

    Worker._remove_request(w, RemoveRequest(request_id="X"))

    assert w.removed == ["X"]
    assert _acks(w) == []


def test_a_teardown_behind_a_deferred_removal_waits_for_it():
    # the removal is held by an in-flight GPU step, so the teardown that
    # follows it must wait too: freeing the session first would have the
    # removal hand state back to a session that is gone
    w = _worker(sessions={"s": {"X"}}, in_flight=("X",))

    _deliver(w, RemoveRequest(request_id="X"), TeardownSession(session_id="s"))

    assert w._pending_removes == {"X"}
    assert w.removed == []
    assert w._sessions.has_pending is True
    assert w.removed_sessions == [] and _acks(w) == []

    # the GPU step retires; the removal lands, then the held teardown
    w._in_flight_rids.clear()
    Worker._apply_pending_removes_safe_to_drop(w, set())
    Worker._apply_pending_sessions(w)

    assert w.removed == ["X"]
    assert w.removed_sessions == ["s"]
    assert _acks(w) == ["s"]


def test_a_follower_that_waits_for_step_symmetry_still_ends_the_session():
    # rank 0 stamps both messages with the step it released the pages after,
    # and a follower that has not reached it parks them; the session still
    # goes, with its ACK, once the follower gets there.
    w = _worker(sessions={"s": {"X"}}, is_follower=True)
    w.scheduler.last_consumed_tp_seq = 3

    _deliver(
        w,
        RemoveRequest(
            request_id="X", source=MessageSource.TP_RANK_0, after_tp_seq=7,
        ),
        TeardownSession(
            session_id="s", source=MessageSource.TP_RANK_0, after_tp_seq=7,
        ),
    )

    assert w.removed == [], "it should be waiting for step 7"
    assert w.removed_sessions == [] and _acks(w) == []

    w.scheduler.last_consumed_tp_seq = 7
    Worker._apply_forwards_whose_step_landed(w)

    assert w.removed == ["X"]
    assert w.removed_sessions == ["s"]
    assert _acks(w) == ["s"]


def test_the_tp_leader_forwards_a_removal_that_says_nothing_of_the_session():
    w = _worker(sessions={"s": {"X"}})
    w._graph_runtime.get_sharding_config = lambda h: SimpleNamespace(groups=[
        SimpleNamespace(
            tp_size=2, _tp_rank=0, _workers=["w0", "w1"], nodes={"LLM"},
        ),
    ])

    Worker._remove_request(w, RemoveRequest(request_id="X"))

    [forwarded] = _forwarded_removes(w)
    assert forwarded.source == MessageSource.TP_RANK_0
    # the session goes on its own message, which rank 0 forwards from
    # _teardown_session; nothing here says a session is ending
    assert not hasattr(forwarded, "end_session")
    assert w.removed_sessions == [] and _acks(w) == []


# ── standalone teardown ─────────────────────────────────────────────────────

def test_teardown_frees_the_session_and_acks():
    w = _worker(known_rids=(), sessions={"s": {"gone"}})

    Worker._teardown_session(w, TeardownSession(session_id="s"))

    assert w.removed_sessions == ["s"]
    assert _acks(w) == ["s"]
    assert w._sessions.session_of("gone") is None
    assert w._sessions.get_rids("s") == set()


def test_teardown_of_a_session_this_worker_never_ran_still_acks():
    w = _worker(known_rids=())

    Worker._teardown_session(w, TeardownSession(session_id="stranger"))

    assert _acks(w) == ["stranger"]


def test_teardown_waits_while_one_of_its_requests_is_still_leaving():
    w = _worker(known_rids=("X",), sessions={"s": {"X"}})

    Worker._teardown_session(w, TeardownSession(session_id="s"))

    assert w._sessions.has_pending is True
    assert w.removed_sessions == []
    assert _acks(w) == []

    # nothing changed yet: the request is still known
    Worker._apply_pending_sessions(w)
    assert w.removed_sessions == []

    Worker._remove_request(w, RemoveRequest(request_id="X"))
    Worker._apply_pending_sessions(w)

    assert w.removed_sessions == ["s"]
    assert _acks(w) == ["s"]


def test_teardown_waits_behind_an_in_flight_step_too():
    w = _worker(known_rids=(), sessions={"s": {"X"}}, in_flight=("X",))

    Worker._teardown_session(w, TeardownSession(session_id="s"))

    assert w._sessions.has_pending is True

    w._in_flight_rids.clear()
    Worker._apply_pending_sessions(w)

    assert w.removed_sessions == ["s"]


# ── ingest ──────────────────────────────────────────────────────────────────

def _new_request(rid="Y", session_id=None):
    return NewRequest(
        request_id=rid,
        partition_worker_graph_ids=[],
        worker_graph_to_workers={},
        initial_inputs=[],
        request_info=CurrentForwardPassInfo(
            request_id=rid, graph_walk="prefill", fwd_index=0, random_seed=0,
            max_tokens=1,
            session=None if session_id is None else RequestSession(session_id),
        ),
    )


def test_ingest_binds_the_request_to_its_session():
    w = _worker(known_rids=())

    Worker._add_new_request(w, _new_request(session_id="s"))

    assert w._sessions.get_rids("s") == {"Y"}
    assert w._sessions.session_of("Y") == "s"


def test_a_stale_teardown_does_not_free_a_reused_id_s_new_session():
    # the API server sends each session down under its own key, so a client id
    # reused after an unconfirmed teardown reaches this worker as a new session
    w = _worker(known_rids=("X",), sessions={"s.k1": {"X"}})
    Worker._teardown_session(w, TeardownSession(session_id="s.k1"))
    assert w._sessions.has_pending is True  # X never leaves

    Worker._add_new_request(w, _new_request(rid="Y", session_id="s.k2"))
    Worker._remove_request(w, RemoveRequest(request_id="Y"))
    Worker._apply_pending_sessions(w)

    # the new session's turn handed its state back, and nothing freed it
    assert w.removed == ["Y"]
    assert w.removed_sessions == [] and _acks(w) == []
    assert w._sessions.session_of("Y") is None
    # the old teardown is still held behind X alone
    assert w._sessions.get_rids("s.k1") == {"X"}
    assert w._sessions.has_pending is True


def test_ingest_fails_a_request_whose_session_went_over_its_budget():
    w = _worker(known_rids=())
    w.session_errors["s"] = "session s exceeded its state budget"
    failed = []
    w._fail_requests = lambda errors, statuses=None: failed.append((errors, statuses))

    Worker._add_new_request(w, _new_request(session_id="s"))

    # 410: the session's state is gone, not the request malformed
    assert failed == [({"Y": "session s exceeded its state budget"}, {"Y": 410})]


def test_a_resumed_ingest_waits_for_the_previous_request_to_hand_state_over():
    # the API server may admit the resume as soon as the request completes,
    # which can outrun the previous request's REMOVE_REQUEST; ingesting then
    # would adopt an empty stream instead of the session's context
    w = _worker(known_rids=("X",), sessions={"s": {"X"}})

    Worker._add_new_request(w, _new_request(session_id="s"))

    assert w._sessions.has_pending is True
    assert w._sessions.session_of("Y") is None

    Worker._apply_pending_sessions(w)
    assert w._sessions.has_pending is True, "it was admitted before its turn"

    Worker._remove_request(w, RemoveRequest(request_id="X"))
    Worker._apply_pending_sessions(w)

    assert w._sessions.has_pending is False
    assert w._sessions.session_of("Y") == "s"
    assert w._sessions.get_rids("s") == {"Y"}


def test_the_last_leaving_request_s_removal_admits_the_held_resume():
    w = _worker(known_rids=("X", "W"), sessions={"s": {"X", "W"}})
    Worker._add_new_request(w, _new_request(session_id="s"))

    Worker._remove_request(w, RemoveRequest(request_id="X"))
    assert w.admitted == [], "W is still leaving"

    Worker._remove_request(w, RemoveRequest(request_id="W"))
    assert w.admitted == ["Y"]


def test_a_follower_builds_a_head_naming_a_resume_freed_inside_its_follow_loop():
    # The follower holds Y behind X and gets rank 0's removal of X only inside
    # the follow loop, whose head names Y; step 0 alone would free Y too late
    from concurrent.futures import Future

    from mstar.worker.micro_scheduler import REMOVED_RID, MicroScheduler

    w = _worker(known_rids=("X",), sessions={"s": {"X"}}, is_follower=True)
    Worker._add_new_request(w, _new_request(session_id="s"))
    head = SimpleNamespace(
        speculative=True, spec_from_seq=0, spec_seq=1, request_ids=["Y"],
    )
    w.scheduler.peek_tp_follow = lambda: head
    w._tp_nospec = set()
    inbox = deque([[WorkerMessage(
        message_type=WorkerMessageType.REMOVE_REQUEST,
        body=RemoveRequest(request_id="X", source=MessageSource.TP_RANK_0),
    )]])

    def wait_for_work(_ms):
        # the first poll delivers X's removal; a second means it did not help
        assert inbox, "follower stuck on held Y"
    w.communicator.wait_for_work = wait_for_work
    w._process_messages = lambda: Worker._process_message_list(w, inbox.popleft())
    w._check_ready_tensors = w._poll_stream_buffers = lambda: None
    sched = SimpleNamespace(
        rid_of=w._graph_runtime.get_rid_handle,
        handles_awaiting_admit=w._handles_awaiting_admit,
    )
    # pop_ready_rids' rule: a head builds once every rid it names resolves
    w._try_follow_speculation = lambda pending: (
        REMOVED_RID not in MicroScheduler.tp_rids(sched, head) or None
    )
    done = Future()
    done.set_result({})
    pending = SimpleNamespace(
        tp_seq=0, future=done,
        node_batch=SimpleNamespace(admit_error=None, failed_requests=None),
    )

    _, spec = Worker._await_tp_follow_step(w, pending, arm=lambda spec: None)

    assert spec is True and w.admitted == ["Y"]


def test_the_second_new_request_for_one_rid_is_not_held_by_the_first():
    # the conductor sends a NEW_REQUEST per partition, so a worker serving two
    # of them ingests the same rid twice
    w = _worker(known_rids=())

    Worker._add_new_request(w, _new_request(session_id="s"))
    Worker._add_new_request(w, _new_request(session_id="s"))

    assert w._sessions.has_pending is False
    assert w._sessions.get_rids("s") == {"Y"}


def test_ingest_without_a_session_touches_no_session_state():
    w = _worker(known_rids=())

    Worker._add_new_request(w, _new_request())

    assert w._sessions.get_rids("s") == set()
    assert w._sessions.session_of("Y") is None


def test_a_request_awaiting_its_session_opens_no_engine_state():
    # the handle and everything keyed by it exist; the engine state does not,
    # because that is what has to wait for the session's handover
    w = _worker(known_rids=("X",), sessions={"s": {"X"}})

    Worker._add_new_request(w, _new_request(session_id="s"))

    assert w._handles_awaiting_admit == {"Y"}
    assert w.admitted == [], "engine state must wait for the handover"
    assert w._sessions.session_of("Y") is None, "not bound until admitted"
    # ...and the handle is usable: _rid resolves it, so messages keyed by one
    # reach it rather than being parked
    assert Worker._rid(w, "Y") == "Y"

    Worker._remove_request(w, RemoveRequest(request_id="X"))
    Worker._apply_pending_sessions(w)

    assert w._handles_awaiting_admit == set()
    assert w.admitted == ["Y"]
    assert w._sessions.session_of("Y") == "s"


def test_a_teardown_behind_the_resume_x_s_removal_admitted_waits_for_it():
    # X's removal admits Y at once, so the teardown finds a live request
    w = _worker(known_rids=("X",), sessions={"s": {"X"}})
    Worker._add_new_request(w, _new_request(session_id="s"))

    _deliver(w, RemoveRequest(request_id="X"), TeardownSession(session_id="s"))
    assert w.admitted == ["Y"] and w.removed_sessions == []

    _deliver(w, RemoveRequest(request_id="Y"))
    Worker._apply_pending_sessions(w)
    assert w.removed_sessions == ["s"] and _acks(w) == ["s"]


def test_a_teardown_fails_the_requests_still_waiting_to_adopt_too():
    # ingested, but not yet admitted into a step, so it has taken nothing
    # from the session and would resume from an empty context
    w = _worker(known_rids=(), sessions={"s": set()})
    failed = []
    w._fail_requests = lambda errors, statuses=None: failed.append((errors, statuses))
    w.engine_manager.remove_session = (
        lambda sid: w.removed_sessions.append(sid) or ["Z"]
    )

    _deliver(w, TeardownSession(session_id="s"))

    [(errors, statuses)] = failed
    assert set(errors) == {"Z"} and statuses == {"Z": 410}


def test_a_request_whose_session_dies_while_it_waits_is_failed_410():
    # admitting it would resume from an empty context, which is worse than an
    # error; 410 is what the route promises for state that is gone
    w = _worker(known_rids=("X",), sessions={"s": {"X"}})
    failed = []
    w._fail_requests = lambda errors, statuses=None: failed.append((errors, statuses))
    Worker._add_new_request(w, _new_request(session_id="s"))

    _deliver(w, TeardownSession(session_id="s"))  # held: X is still leaving
    _deliver(w, RemoveRequest(request_id="X"))
    Worker._apply_pending_sessions(w)

    [(errors, statuses)] = failed
    assert list(errors) == ["Y"] and statuses == {"Y": 410}
    assert w.admitted == [], "it must not open state in a session that is going"
    assert w._handles_awaiting_admit == set()
    assert w.removed_sessions == ["s"]


# ── the offload victim order is the leader's ───────────────────────────────

def test_the_leader_numbers_each_session_as_it_parks_its_state():
    w = _worker(known_rids=("X", "Y"), sessions={"s": {"X"}, "t": {"Y"}})

    Worker._remove_request(w, RemoveRequest(request_id="X"))
    Worker._remove_request(w, RemoveRequest(request_id="Y"))

    # increasing, so the resource offloads the one that idled first
    assert w.idle_seqs == [("X", 1), ("Y", 2)]


def test_a_removal_outside_a_session_is_not_numbered():
    w = _worker(known_rids=("X", "Z"), sessions={"s": {"X"}})

    Worker._remove_request(w, RemoveRequest(request_id="Z"))
    Worker._remove_request(w, RemoveRequest(request_id="X"))

    assert w.idle_seqs == [("Z", -1), ("X", 1)]


def test_the_leader_forwards_the_number_it_minted():
    w = _worker(sessions={"s": {"X"}})
    _leads_one_follower(w)

    Worker._remove_request(w, RemoveRequest(request_id="X"))

    [forwarded] = _forwarded_removes(w)
    assert forwarded.session_idle_seq == 1
    assert w.idle_seqs == [("X", 1)]


def test_a_follower_takes_the_leader_s_number_rather_than_counting():
    # it never admitted some of the leader's requests, so counting for itself
    # would order its parked sessions differently
    w = _worker(sessions={"s": {"X"}}, is_follower=True)

    Worker._remove_request(w, RemoveRequest(
        request_id="X", source=MessageSource.TP_RANK_0, session_idle_seq=42,
    ))

    assert w.idle_seqs == [("X", 42)]
    assert w._session_idle_seq == 0, "a follower mints nothing"


def test_a_deferred_removal_keeps_the_number_it_arrived_with():
    # _pending_removes holds only the handle, so the stamp has to be kept aside
    w = _worker(sessions={"s": {"X"}}, in_flight=("X",), is_follower=True)

    Worker._remove_request(w, RemoveRequest(
        request_id="X", source=MessageSource.TP_RANK_0, session_idle_seq=42,
    ))
    assert w.idle_seqs == []

    w._in_flight_rids.clear()
    Worker._apply_pending_removes_safe_to_drop(w, set())

    assert w.idle_seqs == [("X", 42)]
    assert w._deferred_idle_seq == {}


def test_a_follower_that_parks_a_removal_for_a_step_keeps_the_number():
    w = _worker(sessions={"s": {"X"}}, is_follower=True)
    w.scheduler.last_consumed_tp_seq = 3

    Worker._remove_request(w, RemoveRequest(
        request_id="X", source=MessageSource.TP_RANK_0, after_tp_seq=7,
        session_idle_seq=42,
    ))
    assert w.idle_seqs == []

    w.scheduler.last_consumed_tp_seq = 7
    Worker._apply_forwards_whose_step_landed(w)

    assert w.idle_seqs == [("X", 42)]


# ── a drain for a request still held here ──────────────────────────────────

def _drains(w):
    return [
        (e, m.body) for e, m in w.sent
        if m.message_type == WorkerMessageType.DRAIN_REQUEST
    ]


def _leads_one_follower(w):
    w._graph_runtime.get_sharding_config = lambda h: SimpleNamespace(groups=[
        SimpleNamespace(
            tp_size=2, _tp_rank=0, _workers=["w0", "w1"], nodes={"LLM"},
        ),
    ])


def test_a_drain_for_a_request_awaiting_its_session_needs_no_special_case():
    # The handle is minted before the wait, so the drain gates the scheduler
    # and reaches the followers exactly as it does for an admitted request.
    # Without the early handle this fanned nothing, and the follower's
    # READS_DONE never came.
    w = _worker(known_rids=("X",), sessions={"s": {"X"}})
    _leads_one_follower(w)
    Worker._add_new_request(w, _new_request(session_id="s"))  # Y, awaiting
    assert w._handles_awaiting_admit == {"Y"}

    Worker._drain_request(w, DrainRequest(request_id="Y"))

    assert "Y" in w._draining_rids
    assert _reads_done(w) == ["Y"]
    [(to, body)] = _drains(w)
    assert (to, body.request_id) == ("w1", "Y")
    assert body.source == MessageSource.TP_RANK_0


def test_a_request_drained_while_awaiting_its_session_is_not_admitted():
    w = _worker(known_rids=("X",), sessions={"s": {"X"}})
    Worker._add_new_request(w, _new_request(session_id="s"))
    Worker._drain_request(w, DrainRequest(request_id="Y"))

    Worker._remove_request(w, RemoveRequest(request_id="X"))
    Worker._apply_pending_sessions(w)

    # its turn came, but it is draining, so it never opened engine state
    assert w._sessions.session_of("Y") is None
    assert "Y" not in w.admitted
    # and the handle stays gated until its REMOVE arrives: failed_rids does
    # not gate the TP-follow path, so tp_rids answering REMOVED_RID is the
    # only thing stopping a head that names it from running against engine
    # state that was never ingested
    assert w._handles_awaiting_admit == {"Y"}

    Worker._remove_request(w, RemoveRequest(request_id="Y"))

    assert w._handles_awaiting_admit == set()


def test_a_drain_for_a_request_this_worker_never_saw_is_unchanged():
    w = _worker(known_rids=())

    Worker._drain_request(w, DrainRequest(request_id="Z"))

    assert "Z" in w._draining_rids
    assert _reads_done(w) == ["Z"]


# ── a held request removed before it was admitted ──────────────────────────

def test_a_held_request_removed_before_admission_still_ends_its_session():
    # The previous request is still leaving, so Y waits; the client aborts Y
    # before it is admitted, and Y was the session's last request. Y's handle
    # was minted before the wait, so _process_message_list does NOT park its
    # REMOVE -- it is handled like any other, which is what the early mint is
    # for. Before it, the REMOVE sat in _unprocessed_messages forever.
    w = _worker(known_rids=("X",), sessions={"s": {"X"}})
    Worker._add_new_request(w, _new_request(session_id="s"))
    assert w._handles_awaiting_admit == {"Y"}

    _deliver(
        w,
        RemoveRequest(request_id="X"),
        RemoveRequest(request_id="Y"),
        TeardownSession(session_id="s"),
    )

    assert "Y" not in w._unprocessed_messages, "the REMOVE should not be parked"
    assert "Y" in w.removed, "it should have been torn down, not left waiting"
    # and it is not left to be admitted into a session that is gone
    assert w._handles_awaiting_admit == set()
    assert w._sessions.has_pending is False
    assert w.removed_sessions == ["s"]
    assert _acks(w) == ["s"]


def test_a_held_request_drained_before_admission_still_ends_its_session():
    w = _worker(known_rids=("X",), sessions={"s": {"X"}})
    Worker._add_new_request(w, _new_request(session_id="s"))
    w._draining_rids.add("Y")  # the abort's DRAIN, while Y was held

    _deliver(w, RemoveRequest(request_id="X"))
    Worker._apply_pending_sessions(w)  # Y comes out of the hold, refused
    assert w._sessions.session_of("Y") is None

    _deliver(w, RemoveRequest(request_id="Y"), TeardownSession(session_id="s"))

    assert w.removed_sessions == ["s"]
    assert _acks(w) == ["s"]


def test_an_unadmitted_removal_on_a_live_session_drops_only_the_hold():
    w = _worker(known_rids=("X",), sessions={"s": {"X"}})
    Worker._add_new_request(w, _new_request(session_id="s"))

    Worker._remove_request(w, RemoveRequest(request_id="Y"))

    assert w._sessions.has_pending is False
    assert w.removed_sessions == []
    assert _acks(w) == []


def test_a_teardown_drops_a_request_still_held_for_the_session():
    # a standalone teardown (DELETE, TTL, eviction) landing while a NEW for the
    # session is held: _apply_pending_sessions admits held ingests first, so a
    # hold left behind would be admitted into a session that is gone and then
    # block every later teardown of the same id here
    w = _worker(known_rids=("X",), sessions={"s": {"X"}})
    Worker._add_new_request(w, _new_request(session_id="s"))  # Y, held behind X
    assert w._sessions.has_pending is True

    _deliver(w, TeardownSession(session_id="s"))
    assert w._sessions.has_pending is True, "X is still leaving"
    assert w.removed_sessions == []

    _deliver(w, RemoveRequest(request_id="X"))
    Worker._apply_pending_sessions(w)

    assert w.removed_sessions == ["s"]
    assert _acks(w) == ["s"]
    # Y was dropped, not admitted into a session that is going
    assert w._sessions.get_rids("s") == set()
    assert w._sessions.session_of("Y") is None
    assert w._sessions.has_pending is False


# ── a decode loop that ran past its stop ───────────────────────────────────

def test_an_overshoot_reaches_the_engine_with_its_session_request():
    w = _worker(sessions={"s": {"X"}})
    w._sessions.note_overshoot("X", "LLM")

    Worker._remove_request(w, RemoveRequest(request_id="X"))

    assert w.overshot == [("X", frozenset({"LLM"}))]


def test_a_removal_with_no_overshoot_names_no_node():
    w = _worker(sessions={"s": {"X"}})

    Worker._remove_request(w, RemoveRequest(request_id="X"))

    assert w.overshot == [("X", frozenset())]


def test_an_overshoot_outside_a_session_is_not_kept():
    # nothing continues from that request's KV
    w = _worker(known_rids=("X",))
    w._sessions.note_overshoot("X", "LLM")

    Worker._remove_request(w, RemoveRequest(request_id="X"))

    assert w.overshot == [("X", frozenset())]


# ── a session's teardown under TP lands where rank 0's did ──────────────────

def _forwarded_teardowns(w):
    return [
        (e, m.body) for e, m in w.sent
        if m.message_type == WorkerMessageType.TEARDOWN_SESSION
    ]


def test_a_follower_ignores_the_conductor_s_teardown():
    w = _worker(known_rids=(), is_follower=True)

    Worker._teardown_session(w, TeardownSession(session_id="s"))

    assert w.removed_sessions == []
    assert _acks(w) == []


def test_the_leader_forwards_the_teardown_stamped_with_its_last_step():
    w = _worker(known_rids=())
    w._sessions.bind("X", "s", tp_followers={"w1"})
    w._sessions.release("X")
    w._tp_last_seq_sent = {"w1": 7}

    Worker._teardown_session(w, TeardownSession(session_id="s"))

    [(to, body)] = _forwarded_teardowns(w)
    assert (to, body.session_id) == ("w1", "s")
    assert body.source == MessageSource.TP_RANK_0
    assert body.after_tp_seq == 7
    assert w.removed_sessions == ["s"]
    assert _acks(w) == ["s"]


def test_a_held_teardown_is_forwarded_when_it_runs_not_when_it_arrived():
    # the stamp has to be the step rank 0 frees the pages after
    w = _worker(sessions={"s": {"X"}}, in_flight=("X",))
    w._sessions.bind("X", "s", tp_followers={"w1"})

    Worker._teardown_session(w, TeardownSession(session_id="s"))
    assert _forwarded_teardowns(w) == []

    w._tp_last_seq_sent = {"w1": 4}
    w._in_flight_rids.clear()
    Worker._remove_request(w, RemoveRequest(request_id="X"))
    Worker._apply_pending_sessions(w)

    [(_, body)] = _forwarded_teardowns(w)
    assert body.after_tp_seq == 4
    assert _acks(w) == ["s"]


def test_a_follower_frees_the_session_once_it_reaches_rank_0_s_step():
    w = _worker(known_rids=(), is_follower=True)
    forwarded = TeardownSession(
        session_id="s", source=MessageSource.TP_RANK_0, after_tp_seq=3,
    )

    Worker._teardown_session(w, forwarded)
    assert w.removed_sessions == []

    w.scheduler.last_consumed_tp_seq = 3
    Worker._apply_forwards_whose_step_landed(w)

    assert w.removed_sessions == ["s"]
    assert _acks(w) == ["s"]
    # a follower leads no one, so nothing goes further
    assert _forwarded_teardowns(w) == []


def test_a_follower_ends_the_session_of_a_request_it_never_admitted():
    # rank 0 forwarded the REMOVE and then the teardown; this rank was still
    # waiting on the session, which the early handle makes unremarkable
    w = _worker(known_rids=("X",), sessions={"s": {"X"}}, is_follower=True)
    Worker._add_new_request(w, _new_request(session_id="s"))

    _deliver(
        w,
        RemoveRequest(request_id="X", source=MessageSource.TP_RANK_0),
        RemoveRequest(request_id="Y", source=MessageSource.TP_RANK_0),
        TeardownSession(session_id="s", source=MessageSource.TP_RANK_0),
    )

    assert "Y" not in w._unprocessed_messages
    assert "Y" in w.removed
    assert w._handles_awaiting_admit == set()
    assert w._sessions.has_pending is False
    assert w.removed_sessions == ["s"]
    assert _acks(w) == ["s"]



# ── what rank 0 forwards carries the moves it made before it ────────────────

def test_the_teardown_forward_carries_the_moves_made_before_it():
    # A teardown frees pages, so the moves rank 0 made since its last
    # broadcast were made against the state before it; a follower that met
    # them in the next broadcast would replay them after the teardown instead.
    w = _worker(known_rids=())
    w._sessions.bind("X", "s", tp_followers={"w1"})
    w._sessions.release("X")
    w.parallel_leader_nodes = {"LLM"}
    w._lockstep_followers = {"LLM": ("w1",)}
    journal = OffloadDelta.new()
    journal.add_offloaded("V")
    w.engine_manager.get_engine = lambda node: SimpleNamespace(
        take_resident_delta=lambda node_name: journal.take(),
    )

    Worker._teardown_session(w, TeardownSession(session_id="s"))

    [(_, body)] = _forwarded_teardowns(w)
    assert body.resident_deltas["LLM"].rids == ["V"]
    assert len(journal) == 0, "the next broadcast would carry them a second time"


# ── a drain rank 0 forwards ahead of the conductor's NEW_REQUEST ────────────

def test_a_drain_rank_0_forwarded_before_the_new_request_waits_for_it():
    # Rank 0's forward and the conductor's NEW come from different senders,
    # so the forward can win. Applied at once it would refuse the NEW, and this
    # rank would sit out every move and the removal rank 0 makes for it.
    w = _worker(known_rids=(), is_follower=True)
    Worker._drain_request(
        w, DrainRequest(request_id="Y", source=MessageSource.TP_RANK_0),
    )
    assert "Y" not in w._draining_rids
    assert _reads_done(w) == []

    Worker._add_new_request(w, _new_request())

    assert w.admitted == ["Y"], "the NEW was refused"
    assert "Y" in w._draining_rids
    assert _reads_done(w) == ["Y"]
    assert w._drains_awaiting_new == set()


def test_a_drain_that_overtook_a_request_held_for_its_session_still_lands():
    # held, it has a handle, and answers a drain as it answers a removal
    w = _worker(known_rids=("X",), sessions={"s": {"X"}}, is_follower=True)
    Worker._drain_request(
        w, DrainRequest(request_id="Y", source=MessageSource.TP_RANK_0),
    )

    Worker._add_new_request(w, _new_request(session_id="s"))

    assert w._handles_awaiting_admit == {"Y"} and w.admitted == []
    assert "Y" in w._draining_rids
    assert _reads_done(w) == ["Y"]


def test_a_follower_still_ignores_the_conductor_s_drain():
    w = _worker(known_rids=(), is_follower=True)
    Worker._drain_request(w, DrainRequest(request_id="Y"))

    assert w._drains_awaiting_new == set() and "Y" not in w._draining_rids
