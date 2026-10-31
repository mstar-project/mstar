"""One request's exception must not take down the rest of a conductor tick.

``Conductor.run`` drains every message off the socket, then advances every
partition whose pass finished. Those messages are gone from the socket, so a
request whose handling raises must fail on its own: its client gets the
error, its concurrency slot is freed, and every other request in the same
tick carries on.
"""
import re
from collections import Counter, deque
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
    FailRequests,
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
    c._replica_load = Counter()
    c.waiting_queue = []
    c.max_concurrent_requests = None
    c._early_reads_done = {}
    c._early_abort_requests = set()
    c._draining_deadlines = deque()
    c._early_reads_done_deadlines = deque()
    c._early_abort_deadlines = deque()
    c._drain_ttl_s = 120.0
    c.worker_graphs = {0: SimpleNamespace(graph_walks={"decode"})}
    c.producer_triggered_partitions = set()
    c._stream_fed_partitions = set()
    c._client_streams = {}
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


def _fail_ab():
    return ConductorMessage(
        message_type=ConductorMessageType.FAIL_REQUESTS,
        body=FailRequests(errors={"a": "worker failed a", "b": "worker failed b"}),
    )


def _drained_ab():
    return [_reads_done(rid, entity) for rid in "ab" for entity in ("worker_0", PREPROCESS)]


def test_a_failure_batch_whose_handling_raises_fails_every_request_it_names(caplog):
    # A FAIL_REQUESTS body has no request_id, only ``errors``.
    c = _conductor([_fail_ab()], _drained_ab(), requests={"a": _request(), "b": _request()})

    def _raise(body):
        raise RuntimeError("bookkeeping failed")
    c._fail_requests = _raise
    _run(c)
    failed = _to_client(c, "request_failed")
    assert set(failed) == {"a", "b"}
    for rid in "ab":
        # the client gets an error id; the exception text stays in the log
        (error_id,) = re.fullmatch(
            r"Conductor error handling FAIL_REQUESTS: internal error \(error id ([0-9a-f]{12})\)",
            failed[rid].error_message,
        ).groups()
        (record,) = [r for r in caplog.records if f"[error id {error_id}]" in r.getMessage()]
        assert "bookkeeping failed" in record.getMessage()
    assert not c.requests and not c.draining


def test_one_request_that_raises_in_a_failure_batch_keeps_the_others_errors():
    c = _conductor([_fail_ab()], _drained_ab(), requests={"a": _request(), "b": _request()})
    remove_request = c._remove_request

    def _raise_once_for_a(rid, request_data, failure_error=None):
        if rid == "a" and failure_error == "worker failed a":
            raise RuntimeError("could not tear a down")
        remove_request(rid, request_data, failure_error=failure_error)
    c._remove_request = _raise_once_for_a
    _run(c)
    failed = [
        m.body for entity, m in c.sent
        if entity == "api_server" and m.message_type == "request_failed"
    ]
    assert sorted(f.request_id for f in failed) == ["a", "b"], "a request was answered twice"
    by_rid = {f.request_id: f.error_message for f in failed}
    assert re.search(r"^Conductor error .*: internal error \(error id [0-9a-f]{12}\)$", by_rid["a"])
    assert "could not tear a down" not in by_rid["a"]
    assert by_rid["b"] == "worker failed b"


def test_a_drain_send_that_raises_is_finished_by_the_drain_ttl(monkeypatch):
    # A dead peer makes a send block, not raise; the one raise seen from a
    # send is the Rust transport's EINTR on a signal.
    clock = [0.0]
    monkeypatch.setattr(conductor_mod.time, "perf_counter", lambda: clock[0])
    c = _conductor(
        [_done("bad")],
        [_reads_done("bad", PREPROCESS)],
        requests={"bad": _request(raise_=ValueError)},
    )
    send = c.communicator.send

    def _send(entity, msg):
        if entity == "worker_0" and msg.message_type == WorkerMessageType.DRAIN_REQUEST:
            raise RuntimeError("zmq: Interrupted system call")
        send(entity, msg)
    c.communicator.send = _send
    _run(c)
    # worker_0 never got its DRAIN_REQUEST, so its READS_DONE never comes
    assert "bad" in c.draining and _to_client(c, "request_failed") == {}

    clock[0] = c._drain_ttl_s
    c._sweep_expiry()
    failed = _to_client(c, "request_failed")
    assert "model rejected this request's step" in failed["bad"].error_message
    assert "bad" not in c.requests and "bad" not in c.draining
    assert ("worker_0", "bad") in [
        (entity, m.body.request_id) for entity, m in c.sent
        if m.message_type == WorkerMessageType.REMOVE_REQUEST
    ]


def test_a_base_exception_is_not_swallowed():
    c = _conductor(
        [_done("bad"), _done("good")],
        requests={"bad": _request(raise_=KeyboardInterrupt), "good": _request()},
    )
    with pytest.raises(KeyboardInterrupt):
        c.run()
