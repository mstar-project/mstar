"""Conductor-side sessions: the pinned replica pick and the teardown barrier.

Two things have to hold here. A session's requests must land on the workers
that hold its state, so the replica pick is made once and reused. And the API
server's tombstone must only lift once every worker has freed the state, so the
session ACK is a barrier over the workers the session ran on.
"""

from __future__ import annotations

import sys
import types
from collections import deque

sys.path.insert(0, ".")

from mstar.conductor.conductor import Conductor, RequestData, SessionData
from mstar.conductor.request_info import (
    DEFAULT_PARTITION,
    CurrentForwardConductorMetadata,
    PartitionDefinition,
)
from mstar.model.base import ForwardPassArgs
from mstar.model.sessions import RequestSession
from mstar.utils.ipc_format import (
    NewRequestConductor,
    ReadsDone,
    SessionTornDown,
    WorkerMessageType,
)

PREPROCESS = "api_server_preprocess_worker"


def _new_request(rid, session_id=None):
    return NewRequestConductor(
        request_id=rid,
        initial_signals={},
        initial_input_modalities=["text"],
        initial_output_modalities=["text"],
        input_metadata={},
        model_kwargs={},
        session_id=session_id,
    )


def _request_data(workers=("w0",)):
    return RequestData(
        persist_signals={},
        persist_signal_ref_cnt={},
        worker_graph_to_workers={"wg": list(workers)},
        all_worker_graph_ids={"wg"},
        max_output_tokens=1,
        random_seed=0,
        resource_configs={},
    )


def _conductor(requests=(), sessions=()):
    c = Conductor.__new__(Conductor)
    c.sent = []
    c.admits = 0
    c.communicator = types.SimpleNamespace(
        send=lambda entity_id, msg: c.sent.append((entity_id, msg))
    )
    c.requests = dict(requests)
    c.draining = {}
    c.sessions = {s.session_id: s for s in sessions}
    c.session_teardowns = {}
    c.request_sessions = {}
    c._session_teardown_deadlines = deque()
    c._drain_ttl_s = 120.0
    c._early_reads_done = {}
    c._early_abort_requests = set()
    c._draining_deadlines = deque()
    c._early_reads_done_deadlines = deque()
    c._early_abort_deadlines = deque()
    c.waiting_queue = []
    c.enable_prof = False
    c._try_admit_waiting = lambda: setattr(c, "admits", c.admits + 1)
    return c


def _sent(c, entity, message_type):
    return [m for e, m in c.sent if e == entity and m.message_type == message_type]


def _session_acks(c):
    return [
        m.body.session_id for e, m in c.sent
        if e == "api_server" and m.message_type == "session_torn_down"
    ]


# ── the replica pick is the session's for life ──────────────────────────────

def _wg(ranks, group_id=0, instance_ranks=()):
    wg = types.SimpleNamespace()
    wg.ranks = list(ranks)
    wg._group_id = group_id
    wg._instance_ranks = [list(r) for r in instance_ranks]
    return wg


def test_a_session_reuses_its_first_replica_pick():
    c = _conductor()
    # two DP replicas, so a fresh pick is a coin flip every time
    c.worker_graphs = {"wg": _wg(ranks=[0, 1])}

    first = c._assign_worker_graphs_to_workers()
    c._register_session_request("s", "r0", first)

    for rid in ("r1", "r2", "r3", "r4", "r5"):
        again = c._assign_worker_graphs_to_workers("s")
        assert again == first
        c._register_session_request("s", rid, again)


def test_the_pick_is_copied_not_shared_with_the_session():
    c = _conductor()
    c.worker_graphs = {"wg": _wg(ranks=[0])}
    c._register_session_request("s", "r0", c._assign_worker_graphs_to_workers())

    handed_out = c._assign_worker_graphs_to_workers("s")
    handed_out["wg"].append("worker_9")

    assert c.sessions["s"].worker_graph_to_workers == {"wg": ["worker_0"]}


def test_a_request_with_no_session_still_picks_freely():
    c = _conductor()
    c.worker_graphs = {"wg": _wg(ranks=[0, 1])}

    picks = {
        tuple(c._assign_worker_graphs_to_workers()["wg"]) for _ in range(50)
    }

    assert len(picks) == 2


def test_the_first_request_opens_the_session_and_later_ones_join_it():
    c = _conductor()

    c._register_session_request("s", "r0", {"wg": ["worker_0"]})
    c._register_session_request("s", "r1", {"wg": ["worker_9"]})

    session = c.sessions["s"]
    assert session.request_ids == {"r0", "r1"}
    # the second request's (differing) assignment does not move the session
    assert session.worker_graph_to_workers == {"wg": ["worker_0"]}
    assert c.request_sessions == {"r0": "s", "r1": "s"}


def test_end_session_on_any_request_marks_the_session_ending():
    c = _conductor()
    c._register_session_request("s", "r0", {"wg": ["worker_0"]})
    c._register_session_request("s", "r1", {"wg": ["worker_0"]}, end_session=True)

    assert c.sessions["s"].ending is True


# ── what the model and its submodules are handed ────────────────────────────

def _ingestable(c, model_session=None):
    """Enough conductor to run a real ingest, capturing what the model got."""
    walks = {"prefill"}
    wg = _wg(ranks=[0])
    wg.graph_walks = walks
    wg.section = types.SimpleNamespace(get_nodes=lambda: {"LLM"})
    c.worker_graphs = {1: wg}
    c._all_worker_graph_ids_to_graph_walks = {1: walks}
    c.max_concurrent_requests = None
    c.enable_nvtx = False
    c.model_config = {}
    c.streaming_consumers = set()
    c.default_sharding_config = types.SimpleNamespace(
        clone_empty=lambda: types.SimpleNamespace(
            groups=[], setup=lambda m: None,
            assert_stream_consumer_compatibility=lambda s: None,
        ),
    )
    partition = PartitionDefinition(
        name=DEFAULT_PARTITION, graph_walks=walks, initial_walk=None,
        producer_partitions=[],
    )
    fwd = ForwardPassArgs(
        full_metadata=CurrentForwardConductorMetadata(
            graph_walk="prefill", is_prefill=True,
        ),
        inputs=[],
        unpersist_tensors=[],
        step_metadata={},
    )

    def _initial(**kwargs):
        model_session["seen"] = kwargs.get("session")
        return fwd

    c.model = types.SimpleNamespace(
        get_initial_forward_pass_args=_initial,
        get_max_output_tokens=lambda **kw: 8,
        get_partitions=lambda: [partition],
        get_partition_topology=lambda: types.SimpleNamespace(connections=[]),
        get_request_resource_configs=lambda **kw: {},
        prefix_key_streams=lambda: {},
    )
    return c


def _new_requests_sent(c):
    return [
        m.body for e, m in c.sent
        if m.message_type == WorkerMessageType.NEW_REQUEST
    ]


def test_the_model_is_handed_the_request_s_session():
    # first-class, not riding on the client's model_kwargs
    seen = {}
    c = _ingestable(_conductor(), seen)

    body = _new_request("r0", session_id="s")
    body.resumed = True
    body.end_session = True
    c._do_ingest_request(body)

    assert seen["seen"] == RequestSession("s", resumed=True, end_session=True)


def test_a_sessionless_request_hands_the_model_no_session():
    seen = {}
    c = _ingestable(_conductor(), seen)

    c._do_ingest_request(_new_request("r0"))

    assert seen["seen"] is None


def test_the_forward_pass_the_worker_gets_carries_the_session():
    # a submodule reads it off the step, not off a kwarg
    seen = {}
    c = _ingestable(_conductor(), seen)
    body = _new_request("r0", session_id="s")
    body.resumed = True

    c._do_ingest_request(body)

    [sent] = _new_requests_sent(c)
    assert sent.request_info.session == RequestSession("s", resumed=True)
    assert not hasattr(sent, "session_id"), "one carrier, not two"


# ── standalone teardown (DELETE, TTL, a failed request) ─────────────────────

def test_teardown_asks_every_worker_and_waits_for_all_of_them():
    c = _conductor(sessions=[SessionData("s", {"wg": ["w0", "w1"]})])

    c._teardown_session("s")

    asked = _sent(c, "w0", WorkerMessageType.TEARDOWN_SESSION)
    assert [m.body.session_id for m in asked] == ["s"]
    assert _sent(c, "w1", WorkerMessageType.TEARDOWN_SESSION)
    assert _session_acks(c) == []  # nothing yet

    c._handle_session_torn_down(SessionTornDown("s", "w0"))
    assert _session_acks(c) == []

    c._handle_session_torn_down(SessionTornDown("s", "w1"))
    assert _session_acks(c) == ["s"]
    assert "s" not in c.sessions


def test_teardown_of_an_unknown_session_acks_immediately():
    c = _conductor()

    c._teardown_session("never-ingested")

    assert _session_acks(c) == ["never-ingested"]


def test_a_second_teardown_ask_does_not_open_a_second_barrier():
    c = _conductor(sessions=[SessionData("s", {"wg": ["w0"]})])

    c._teardown_session("s")
    c._teardown_session("s")

    assert len(_sent(c, "w0", WorkerMessageType.TEARDOWN_SESSION)) == 1
    c._handle_session_torn_down(SessionTornDown("s", "w0"))
    assert _session_acks(c) == ["s"]


def test_teardown_with_a_request_in_flight_tears_the_request_down_first():
    session = SessionData("s", {"wg": ["w0"]}, request_ids={"r0"})
    c = _conductor(requests={"r0": _request_data(("w0",))}, sessions=[session])
    c.request_sessions["r0"] = "s"

    c._teardown_session("s")

    # phase 1 of the request's own teardown, not a session message yet
    assert _sent(c, "w0", WorkerMessageType.DRAIN_REQUEST)
    assert _sent(c, "w0", WorkerMessageType.TEARDOWN_SESSION) == []
    assert session.ending is True

    # the request drains; the session goes with its removal
    c._handle_reads_done(ReadsDone("r0", "w0"))
    c._handle_reads_done(ReadsDone("r0", PREPROCESS))

    removes = _sent(c, "w0", WorkerMessageType.REMOVE_REQUEST)
    assert [m.body.end_session for m in removes] == [True]
    c._handle_session_torn_down(SessionTornDown("s", "w0"))
    assert _session_acks(c) == ["s"]


def test_teardown_drops_the_session_s_queued_requests():
    # admitting one after the teardown would re-open a session the API server
    # has already forgotten, and nothing would free the state it built
    c = _conductor(sessions=[SessionData("s", {"wg": ["w0"]})])
    c.waiting_queue = [_new_request("r9", session_id="s"), _new_request("r8")]

    c._teardown_session("s")

    assert [b.request_id for b in c.waiting_queue] == ["r8"]
    failed = [
        m.body for e, m in c.sent
        if e == "api_server" and m.message_type == "request_failed"
    ]
    assert [f.request_id for f in failed] == ["r9"]
    assert failed[0].status == 409
    removes = _sent(c, PREPROCESS, WorkerMessageType.REMOVE_REQUEST)
    assert [m.body.request_id for m in removes] == ["r9"]


def test_a_stalled_worker_cannot_hold_the_tombstone_forever():
    c = _conductor(sessions=[SessionData("s", {"wg": ["w0"]})])
    c._teardown_session("s")

    # the deadline passes with no ACK
    c._session_teardown_deadlines[0] = (0.0, "s")
    c._sweep_expiry()

    assert _session_acks(c) == ["s"]
    assert c.session_teardowns == {}


# ── teardown riding on the last request ────────────────────────────────────

def test_the_happy_path_removal_carries_end_session_and_opens_the_barrier():
    session = SessionData("s", {"wg": ["w0"]}, request_ids={"r0"}, ending=True)
    c = _conductor(requests={"r0": _request_data(("w0",))}, sessions=[session])
    c.request_sessions["r0"] = "s"
    c._register_draining(
        "r0", expected_acks={PREPROCESS}, participants={"w0", PREPROCESS},
    )

    c._handle_reads_done(ReadsDone("r0", PREPROCESS))

    worker_removes = _sent(c, "w0", WorkerMessageType.REMOVE_REQUEST)
    assert [m.body.end_session for m in worker_removes] == [True]
    # named, for a worker that never admitted the request to tear down
    assert [m.body.session_id for m in worker_removes] == ["s"]
    # the preprocess worker holds no session state, so it is not asked to
    # end one and is not part of the barrier
    pre_removes = _sent(c, PREPROCESS, WorkerMessageType.REMOVE_REQUEST)
    assert [m.body.end_session for m in pre_removes] == [False]
    assert [m.body.session_id for m in pre_removes] == [None]
    assert c.session_teardowns["s"].expected_acks == {"w0"}

    c._handle_session_torn_down(SessionTornDown("s", "w0"))
    assert _session_acks(c) == ["s"]


def test_a_removal_in_a_live_session_keeps_the_session_s_state():
    session = SessionData("s", {"wg": ["w0"]}, request_ids={"r0"})
    c = _conductor(requests={"r0": _request_data(("w0",))}, sessions=[session])
    c.request_sessions["r0"] = "s"
    c._register_draining(
        "r0", expected_acks={PREPROCESS}, participants={"w0", PREPROCESS},
    )

    c._handle_reads_done(ReadsDone("r0", PREPROCESS))

    removes = _sent(c, "w0", WorkerMessageType.REMOVE_REQUEST)
    assert [m.body.end_session for m in removes] == [False]
    assert "s" in c.sessions
    assert c.sessions["s"].request_ids == set()
    assert _session_acks(c) == []


def test_ending_waits_for_the_last_request_of_several():
    session = SessionData(
        "s", {"wg": ["w0"]}, request_ids={"r0", "r1"}, ending=True,
    )
    c = _conductor(
        requests={"r0": _request_data(("w0",)), "r1": _request_data(("w0",))},
        sessions=[session],
    )
    c.request_sessions.update({"r0": "s", "r1": "s"})
    for rid in ("r0", "r1"):
        c._register_draining(
            rid, expected_acks={PREPROCESS}, participants={"w0", PREPROCESS},
        )

    c._handle_reads_done(ReadsDone("r0", PREPROCESS))
    assert [m.body.end_session for m in _sent(c, "w0", WorkerMessageType.REMOVE_REQUEST)] == [False]

    c._handle_reads_done(ReadsDone("r1", PREPROCESS))
    assert [
        m.body.end_session for m in _sent(c, "w0", WorkerMessageType.REMOVE_REQUEST)
    ] == [False, True]
