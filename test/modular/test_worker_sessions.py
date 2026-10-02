"""Worker-side sessions: what is freed when, and what is handed back.

A request in a session gives its resource state back to the session instead of
freeing it, so the worker must get two things right: the ``end_session`` flag
has to survive every hop it takes (the TP fan-out, a remove deferred behind an
in-flight GPU step), and a TEARDOWN_SESSION must not land while one of the
session's requests is still on its way out — that request's removal would hand
state back to a session that no longer exists.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.model.sessions import RequestSession
from mstar.utils.ipc_format import (
    ConductorMessageType,
    MessageSource,
    RemoveRequest,
    TeardownSession,
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
    w.removed_sessions = []
    w.session_errors = {}
    w.communicator = SimpleNamespace(
        send=lambda e, msg=None, m=None: w.sent.append((e, msg if msg is not None else m))
    )
    w._in_flight_rids = set(in_flight)
    w._pending_drains = set()
    w._draining_rids = set()
    w._reads_done_sent = set()
    w._pending_removes = set()
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
        remove_request=lambda rid, end_session=False: w.removed.append(
            (rid, end_session)
        ),
        remove_session=w.removed_sessions.append,
        add_request=lambda rid, cfgs, session_id=None: None,
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


def _forwarded_removes(w):
    return [
        m.body for e, m in w.sent
        if m.message_type == WorkerMessageType.REMOVE_REQUEST
    ]


# ── removal inside a live session ───────────────────────────────────────────

def test_a_plain_removal_leaves_the_session_holding_the_state():
    w = _worker(sessions={"s": {"X"}})

    Worker._remove_request(w, RemoveRequest(request_id="X"))

    assert w.removed == [("X", False)]
    assert w.removed_sessions == []
    assert _acks(w) == []
    # the rid is gone, the session is not
    assert w._sessions.session_of("X") is None
    assert w._sessions.get_rids("s") == set()


def test_end_session_frees_the_session_and_acks():
    w = _worker(sessions={"s": {"X"}})

    Worker._remove_request(w, RemoveRequest(request_id="X", end_session=True))

    assert w.removed == [("X", True)]
    assert _acks(w) == ["s"]
    assert w._sessions.get_rids("s") == set()


def test_a_sessionless_removal_is_unchanged():
    w = _worker()

    Worker._remove_request(w, RemoveRequest(request_id="X"))

    assert w.removed == [("X", False)]
    assert _acks(w) == []


def test_a_deferred_removal_keeps_its_end_session_flag():
    w = _worker(sessions={"s": {"X"}}, in_flight=("X",))

    Worker._remove_request(w, RemoveRequest(request_id="X", end_session=True))

    assert w._pending_removes == {"X"}
    assert w._sessions.ends_session("X") is True
    assert w.removed == []

    # the GPU step retires and the deferred remove is applied
    w._in_flight_rids.clear()
    Worker._apply_pending_removes_safe_to_drop(w, set())

    assert w.removed == [("X", True)]
    assert _acks(w) == ["s"]


def test_the_tp_leader_forwards_end_session_to_its_followers():
    w = _worker(sessions={"s": {"X"}})
    w._graph_runtime.get_sharding_config = lambda h: SimpleNamespace(groups=[
        SimpleNamespace(tp_size=2, _tp_rank=0, _workers=["w0", "w1"]),
    ])

    Worker._remove_request(w, RemoveRequest(request_id="X", end_session=True))

    [forwarded] = _forwarded_removes(w)
    assert forwarded.end_session is True
    assert forwarded.source == MessageSource.TP_RANK_0


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
    return SimpleNamespace(
        request_id=rid,
        partition_worker_graph_ids=[],
        worker_graph_to_workers={},
        initial_inputs=[],
        request_info=SimpleNamespace(
            resource_configs={}, graph_walk="prefill", partition_name="default",
            rid_handle=-1,
            session=None if session_id is None else RequestSession(session_id),
        ),
    )


def test_ingest_binds_the_request_to_its_session():
    w = _worker(known_rids=())

    Worker._add_new_request(w, _new_request(session_id="s"))

    assert w._sessions.get_rids("s") == {"Y"}
    assert w._sessions.session_of("Y") == "s"


def test_ingest_fails_a_request_whose_session_overflowed_under_the_error_policy():
    w = _worker(known_rids=())
    w.session_errors["s"] = "session s exceeded its state budget"
    failed = {}
    w._fail_requests = failed.update

    Worker._add_new_request(w, _new_request(session_id="s"))

    assert failed == {"Y": "session s exceeded its state budget"}


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
