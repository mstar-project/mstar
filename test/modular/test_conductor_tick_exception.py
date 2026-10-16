"""One request's exception must not take down the rest of a conductor tick.

``Conductor.run`` drains every message off the socket, then advances every
partition whose pass finished. Those messages are gone from the socket, so a
request whose handling raises must fail on its own: its client gets the
error, its concurrency slot is freed, and every other request in the same
tick carries on.
"""
from collections import deque
from types import SimpleNamespace

import pytest

import mstar.conductor.conductor as conductor_mod
from mstar.conductor.conductor import Conductor, RequestData
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    PartitionDefinition,
    PartitionState,
)
from mstar.model.base import ForwardPassArgs
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    ReadsDone,
    WorkerGraphsDone,
    WorkerMessageType,
)

PREPROCESS = Conductor.PREPROCESS_WORKER


class _StopLoop(BaseException):
    """Ends ``run`` after the scripted ticks; a BaseException, so the loop's
    error handling must let it through."""


class _Model:
    """Raises for a request whose metadata asks it to, as a model does on a
    kwarg it only reads at the second step or on a bug in its transition."""

    def get_partition_forward_pass_args(self, partition_name, partition_metadata,
                                        persist_signals, incoming_connections):
        raise_ = partition_metadata.kwargs.get("raise")
        if raise_ is not None:
            raise raise_("model rejected this request's step")
        return ForwardPassArgs(
            full_metadata=CurrentForwardConductorMetadata(
                graph_walk="decode", is_prefill=False,
            ),
            inputs=[], unpersist_tensors=[], request_done=True,
        )


def _request(raise_=None) -> RequestData:
    rd = RequestData(
        persist_signals={}, persist_signal_ref_cnt={},
        worker_graph_to_workers={0: ["worker_0"]}, all_worker_graph_ids={0},
        max_output_tokens=100, random_seed=0, resource_configs={},
    )
    rd.dispatched = True
    rd.partition_definitions = {
        "default": PartitionDefinition(name="default", graph_walks={"decode"}),
    }
    rd.partition_states = {
        "default": PartitionState(
            partition_name="default",
            metadata=CurrentForwardConductorMetadata(
                graph_walk="decode", is_prefill=False, kwargs={"raise": raise_},
            ),
            current_worker_graph_ids={0},
        ),
    }
    return rd


def _done(rid, wg_id=0):
    return ConductorMessage(
        message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
        body=WorkerGraphsDone(request_id=rid, worker_graph_ids=[wg_id], is_first_tp_rank=True),
    )


def _reads_done(rid, entity):
    return ConductorMessage(
        message_type=ConductorMessageType.READS_DONE,
        body=ReadsDone(request_id=rid, entity_id=entity),
    )


def _conductor(*ticks, requests):
    """A conductor whose socket yields one scripted batch per tick, then
    leaves ``run`` once the batches are spent."""
    c = Conductor.__new__(Conductor)
    c.sent = []
    batches = [list(t) for t in ticks]
    c.communicator = SimpleNamespace(
        send=lambda entity, msg: c.sent.append((entity, msg)),
        get_all_new_messages=lambda: batches.pop(0) if batches else [],
    )
    c.model = _Model()
    c.enable_nvtx = False
    c.enable_prof = False
    c._startup_message_backlog = []
    c.requests = dict(requests)
    c.draining = {}
    c.waiting_queue = []
    c.max_concurrent_requests = None
    c._early_reads_done = {}
    c._early_abort_requests = set()
    c._draining_deadlines = deque()
    c._early_reads_done_deadlines = deque()
    c._early_abort_deadlines = deque()
    c._unsent_teardown = []
    c._drain_ttl_s = 120.0
    c.worker_graphs = {0: SimpleNamespace(graph_walks={"decode"})}
    c.producer_triggered_partitions = set()
    c._all_worker_graph_ids_to_graph_walks = {0: {"decode"}}
    c._wait_for_workers_ready = lambda: None

    def _poll():
        if not batches:
            raise _StopLoop
    c._poll_worker_liveness = _poll
    return c


@pytest.fixture(autouse=True)
def _no_sleep_no_freeze(monkeypatch):
    monkeypatch.setattr(conductor_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(conductor_mod, "freeze_after_setup", lambda _n: None)


def _to_client(c, kind):
    return {
        m.body.request_id: m.body for entity, m in c.sent
        if entity == "api_server" and m.message_type == kind
    }


def _run(c):
    with pytest.raises(_StopLoop):
        c.run()


def test_two_healthy_requests_complete_in_one_tick():
    c = _conductor([_done("a"), _done("b")], requests={"a": _request(), "b": _request()})
    _run(c)
    assert set(_to_client(c, "request_complete")) == {"a", "b"}
    assert _to_client(c, "request_failed") == {}


def test_a_failing_pass_does_not_strand_the_other_requests_in_its_tick():
    c = _conductor(
        [_done("bad"), _done("good")],
        requests={"bad": _request(raise_=ValueError), "good": _request()},
    )
    _run(c)
    assert "good" in _to_client(c, "request_complete")
    assert "bad" not in _to_client(c, "request_complete")


def test_a_failing_message_does_not_drop_the_rest_of_the_batch():
    # The first done names a worker graph the request was never assigned:
    # handling it raises before the pass is even queued.
    c = _conductor(
        [_done("bad", wg_id=7), _done("good")],
        requests={"bad": _request(), "good": _request()},
    )
    _run(c)
    assert "good" in _to_client(c, "request_complete")
    assert "bad" in c.draining


def test_the_failing_request_gets_an_error_and_frees_its_slot():
    c = _conductor(
        [_done("bad")],
        [_reads_done("bad", "worker_0"), _reads_done("bad", PREPROCESS)],
        requests={"bad": _request(raise_=ValueError)},
    )
    admitted = []
    c.max_concurrent_requests = 1
    c.waiting_queue = [SimpleNamespace(request_id="queued")]
    c._admit_request = lambda body: admitted.append(body.request_id)
    _run(c)
    failed = _to_client(c, "request_failed")
    assert "model rejected this request's step" in failed["bad"].error_message
    assert failed["bad"].status == 500
    assert "bad" not in c.requests and "bad" not in c.draining
    # The freed slot goes to the request that was waiting for it.
    assert admitted == ["queued"]


def test_a_completion_that_raises_after_notifying_is_torn_down_quietly():
    c = _conductor(
        [_done("a")],
        [_reads_done("a", "worker_0"), _reads_done("a", PREPROCESS)],
        requests={"a": _request()},
    )
    calls = {"n": 0}
    request_workers = c._request_workers

    def _raise_once(request_data):
        calls["n"] += 1
        if calls["n"] == 1:  # inside _process_request_done, after request_complete
            raise RuntimeError("teardown bookkeeping failed")
        return request_workers(request_data)
    c._request_workers = _raise_once
    _run(c)
    # The client already has its result; it must not get a failure after it.
    assert "a" in _to_client(c, "request_complete")
    assert _to_client(c, "request_failed") == {}
    assert "a" not in c.requests and "a" not in c.draining


def _send_raises_once(c, kind):
    """The first send of ``kind`` raises, as a socket error would, and sends
    nothing."""
    send = c.communicator.send
    raised = []

    def _send(entity, msg):
        if msg.message_type == kind and not raised:
            raised.append(entity)
            raise OSError(f"could not send {kind}")
        send(entity, msg)
    c.communicator.send = _send
    return raised


def test_a_completed_request_whose_teardown_send_raises_is_not_failed():
    c = _conductor(
        [_done("a")],
        [_reads_done("a", PREPROCESS)],
        [_reads_done("a", "worker_0"), _reads_done("a", PREPROCESS)],
        requests={"a": _request()},
    )
    admitted = []
    c.max_concurrent_requests = 1
    c.waiting_queue = [SimpleNamespace(request_id="queued")]
    c._admit_request = lambda body: admitted.append(body.request_id)
    raised = _send_raises_once(c, WorkerMessageType.REMOVE_REQUEST)
    _run(c)
    assert "a" in _to_client(c, "request_complete")
    assert _to_client(c, "request_failed") == {}
    assert "a" not in c.requests and "a" not in c.draining
    assert admitted == ["queued"]
    # The send that raised is retried, and does not hold up the others.
    removed = [
        entity for entity, m in c.sent
        if m.message_type == WorkerMessageType.REMOVE_REQUEST
    ]
    assert sorted(removed) == sorted(["worker_0", PREPROCESS])
    assert removed[-1] == raised[0]


def test_a_completion_whose_send_raises_fails_its_client():
    c = _conductor(
        [_done("a")],
        [_reads_done("a", "worker_0"), _reads_done("a", PREPROCESS)],
        requests={"a": _request()},
    )
    admitted = []
    c.max_concurrent_requests = 1
    c.waiting_queue = [SimpleNamespace(request_id="queued")]
    c._admit_request = lambda body: admitted.append(body.request_id)
    _send_raises_once(c, "request_complete")
    _run(c)
    assert _to_client(c, "request_complete") == {}
    assert "could not send request_complete" in _to_client(c, "request_failed")["a"].error_message
    assert "a" not in c.requests and "a" not in c.draining
    assert admitted == ["queued"]


def test_a_failure_whose_send_raises_still_reaches_its_client():
    c = _conductor(
        [_done("bad")],
        [_reads_done("bad", "worker_0"), _reads_done("bad", PREPROCESS)],
        requests={"bad": _request(raise_=ValueError)},
    )
    admitted = []
    c.max_concurrent_requests = 1
    c.waiting_queue = [SimpleNamespace(request_id="queued")]
    c._admit_request = lambda body: admitted.append(body.request_id)
    _send_raises_once(c, "request_failed")
    _run(c)
    failed = [
        m.body for entity, m in c.sent
        if entity == "api_server" and m.message_type == "request_failed"
    ]
    assert [f.request_id for f in failed] == ["bad"]
    assert "model rejected this request's step" in failed[0].error_message
    assert "bad" not in c.requests and "bad" not in c.draining
    assert admitted == ["queued"]
    assert c._unsent_teardown == []


def test_an_unsendable_failure_is_dropped_after_the_ttl_and_failed_at_shutdown():
    c = _conductor([], requests={})
    c._drain_ttl_s = 0.0

    def _send(entity, msg):
        if msg.message_type == "request_failed":
            raise OSError("api server unreachable")
        c.sent.append((entity, msg))
    c.communicator.send = _send
    c.draining["bad"] = conductor_mod.DrainingRequest(
        expected_acks=set(), participants={"worker_0"}, failure_error="boom",
    )
    c._finalize_draining("bad")
    assert [rid for _, rid, entity, _ in c._unsent_teardown] == ["bad"]

    c.communicator.send = lambda entity, msg: c.sent.append((entity, msg))
    c._fail_all_requests("shutting down")
    assert _to_client(c, "request_failed")["bad"].error_message == "shutting down"

    c.communicator.send = _send
    c._retry_unsent_teardown()
    assert c._unsent_teardown == []


def test_a_base_exception_is_not_swallowed():
    c = _conductor(
        [_done("bad"), _done("good")],
        requests={"bad": _request(raise_=KeyboardInterrupt), "good": _request()},
    )
    with pytest.raises(KeyboardInterrupt):
        c.run()
