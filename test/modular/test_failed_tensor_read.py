"""A failed async tensor read (a Mooncake transfer error) fails only its own
request, and its buffers are freed once Mooncake reports that every transfer
into them stopped.

Before, ``get_ready_tensors`` re-raised the same failed future on every poll,
so no other request on that worker or data worker received its inputs until
the failed request was torn down (#418)."""

import collections
import ctypes
import itertools
import json
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
from test_request_failure_propagation import _preprocess_thread, _worker

from mstar.api_server.request_types import ResultChunk
from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import (
    _TRANSFER_COMPLETED,
    MOONCAKE_IMPORT_ERROR,
    AsyncMooncakeReader,
    AsyncReadError,
    FutureAndPointers,
    MooncakeCommunicationManager,
    SharedMemoryCommunicationManager,
    TensorCommunicationManager,
    TrackedMooncakeReader,
    TransferReadInfo,
    create_tensor_communication_manager,
)
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.utils.ipc_format import ConductorMessageType, WorkerMessageType


class _Communicator(BaseCommunicator):
    def __init__(self):
        self.sent = []

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))

    def get_all_new_messages(self):
        return []


def _acked(comm):
    return [
        (msg.body.request_id, dict(msg.body.successful_tensors))
        for _, msg in comm.sent
        if msg.message_type == WorkerMessageType.TENSOR_RECEIVED
    ]


# ── the read fix: one failed read fails one request ───────────────────────


def _shm_manager(tmp_path):
    comm = _Communicator()
    mgr = SharedMemoryCommunicationManager("worker_1", "host", "cpu", comm, shm_dir=str(tmp_path))
    return mgr, comm


def _edge(*uuids, name="x"):
    edge = GraphEdge(next_node="node", name=name)
    edge.tensor_info = [SimpleNamespace(uuid=u, source_entity="worker_0") for u in uuids]
    return edge


def _failed_future(message="Mooncake read failed. Status: -1"):
    f = Future()
    f.set_exception(RuntimeError(message))
    return f


def _read(rid, edge, future=None):
    return FutureAndPointers(future=future, graph_edges=[edge], request_id=rid)


def test_a_failed_read_does_not_block_other_requests(tmp_path):
    mgr, comm = _shm_manager(tmp_path)
    bad, healthy = _edge(11), _edge()
    mgr.pending = [_read(7, bad, _failed_future()), _read(8, healthy)]

    assert mgr.get_ready_tensors() == {8: [healthy]}
    assert mgr.pending == []
    (failed,) = mgr.take_failed_reads()
    assert failed.request_id == 7 and failed.graph_edges == [bad]
    assert "Mooncake read failed" in str(failed.error)
    # the transfer may not have stopped, so the producer keeps its copy
    assert _acked(comm) == []

    assert mgr.get_ready_tensors() == {}
    assert mgr.take_failed_reads() == []


def test_a_cancelled_read_counts_as_failed(tmp_path):
    mgr, _ = _shm_manager(tmp_path)
    cancelled = Future()
    cancelled.cancel()
    mgr.pending = [_read(7, _edge(11), cancelled)]

    assert mgr.get_ready_tensors() == {}
    (failed,) = mgr.take_failed_reads()
    assert failed.request_id == 7 and mgr.pending == []


def test_later_reads_of_a_failed_request_are_dropped_until_cleanup(tmp_path):
    mgr, comm = _shm_manager(tmp_path)
    in_flight = Future()
    mgr.pending = [
        _read(7, _edge(11), _failed_future()),
        _read(7, _edge(12), in_flight),
        _read(7, _edge(13)),
    ]

    assert mgr.get_ready_tensors() == {}
    first, later = mgr.take_failed_reads()
    assert first.error is later.error
    # a read still touching the producer keeps the drain waiting
    assert [ep.future for ep in mgr.pending] == [in_flight]
    assert mgr.has_inflight_reads(7)

    in_flight.set_result(None)
    assert mgr.get_ready_tensors() == {}
    (dropped,) = mgr.take_failed_reads()
    assert dropped.error is first.error
    assert mgr.pending == [] and not mgr.has_inflight_reads(7)
    # the reads that completed are acked; the failed one is not
    assert _acked(comm) == [("7", {13: 1}), ("7", {12: 1})]

    mgr.cleanup_request(7)
    fresh = _edge()
    mgr.pending = [_read(7, fresh)]
    assert mgr.get_ready_tensors() == {7: [fresh]}
    assert mgr.take_failed_reads() == []


class _CountingFuture(Future):
    """A read future that counts the queries made on it."""

    def __init__(self):
        super().__init__()
        self.done_calls = 0
        self.result_calls = 0

    def done(self):
        self.done_calls += 1
        return super().done()

    def result(self, timeout=None):
        self.result_calls += 1
        return super().result(timeout)


def test_a_finished_read_is_queried_once_while_it_waits_behind_an_earlier_one(tmp_path):
    mgr, _ = _shm_manager(tmp_path)
    earlier, later = _CountingFuture(), _CountingFuture()
    later.set_result(None)
    first, second = _edge(), _edge()
    mgr.pending = [_read(7, first, earlier), _read(7, second, later)]

    for _ in range(3):
        assert mgr.get_ready_tensors() == {}
    assert (earlier.done_calls, earlier.result_calls) == (3, 0)
    assert (later.done_calls, later.result_calls) == (1, 1)

    earlier.set_result(None)
    assert mgr.get_ready_tensors() == {7: [first, second]}
    assert (later.done_calls, later.result_calls) == (1, 1)


class _EventFuture:
    """Shaped like the SHM arena's ``_CudaEventFuture``: ``done()`` and
    ``result()`` only."""

    def __init__(self, done):
        self.finished = done

    def done(self):
        return self.finished

    def result(self):
        return None


def test_a_future_with_only_done_and_result_is_polled_like_any_other(tmp_path):
    """The arena attaches such a future to every host-to-device read."""
    mgr, _ = _shm_manager(tmp_path)
    running, finished = _EventFuture(False), _EventFuture(True)
    first, second = _edge(), _edge()
    mgr.pending = [_read(7, first, running), _read(8, second, finished)]

    assert mgr.get_ready_tensors() == {8: [second]}
    assert mgr.has_inflight_reads(7) and not mgr.has_inflight_reads(8)
    running.finished = True
    assert not mgr.has_inflight_reads(7)
    assert mgr.get_ready_tensors() == {7: [first]}
    assert mgr.take_failed_reads() == [] and mgr.pending == []


def test_the_worker_fails_a_request_once_when_its_input_read_fails(tmp_path):
    w = _worker()
    w.enable_nvtx = False
    w.wakeup_event = SimpleNamespace(drain=lambda: None)
    w.tensor_manager, _ = _shm_manager(tmp_path)
    w.tensor_manager.pending = [_read("r1", _edge(11), _failed_future())]

    w._check_ready_tensors()
    w._check_ready_tensors()

    (sent,) = w.sent
    assert sent[1].message_type == ConductorMessageType.FAIL_REQUESTS
    assert list(sent[1].body.errors) == ["r1"]
    assert "input tensor read failed: RuntimeError: Mooncake read failed" in sent[1].body.errors["r1"]
    assert w.scheduler.failed_rids == {"r1"}


def test_the_data_worker_answers_a_failed_output_read_in_its_slot(tmp_path, caplog):
    """The read of output 0 failed while output 1 waits in the reorder buffer:
    the error takes slot 0, so 1 follows it instead of waiting for the TTL,
    and the API server gets one chunk per tensor it counted. The failure is
    logged in one line each, without a traceback, as an input read's is."""
    wt = _preprocess_thread(model=None)
    held = ResultChunk(request_id="r1", modality="text", data=b"later", metadata={})
    wt.request_output_state["r1"].pending[1] = held
    wt.request_output_state["r1"].next_sequence = 2
    wt.tensor_manager, _ = _shm_manager(tmp_path)
    wt.tensor_manager.pending = [_read("r1", _edge("u1", name="text_output"), _failed_future())]

    with caplog.at_level("ERROR"):
        assert wt._process_read_tensors() is True
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert [r.exc_info for r in errors] == [None, None]
    assert "output tensor read failed for request r1: RuntimeError" in errors[1].getMessage()
    chunks = [wt.out_queue.get_nowait(), wt.out_queue.get_nowait()]
    assert [c.modality for c in chunks] == ["error", "text"]
    assert b"output tensor read failed: RuntimeError: Mooncake read failed" in chunks[0].data
    assert chunks[0].metadata["status"] == 500
    assert wt.out_queue.empty()
    assert wt.request_output_state["r1"].order == {}
    assert wt.tensor_uuid_to_metadata_per_request["r1"] == {}

    assert wt._process_read_tensors() is False
    assert wt.out_queue.empty()


# ── releasing a failed read's buffers ─────────────────────────────────────


class _Engine:
    """Mooncake's async read API, one task per batch, with scripted statuses.

    ``plan[remote_ptr]`` lists, per submit of that tensor, the statuses its
    batch reports to successive ``transfer_check_status`` calls; the last one
    repeats. ``"submit-fails"`` makes that submit return ``submit_failure``.
    COMPLETED copies the bytes; COMPLETED and FAILED free the batch, as in
    Mooncake, and a freed batch must never be polled again."""

    def __init__(self, hcas=0):
        self.plan: dict[int, list] = {}
        self.hcas = hcas
        self.topology_calls = 0
        self.topology_delay = 0
        self.submit_failure = 0
        self.unregister_result = 0
        self.unregistered: list[int] = []
        self.polls: list[tuple[int, int, str]] = []  # (batch, remote, thread)
        self.after_free: list[int] = []
        self._batches: dict[int, list] = {}
        self._freed: set[int] = set()
        self._submits = collections.Counter()
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def get_local_topology(self, device_name=""):
        self.topology_calls += 1
        time.sleep(self.topology_delay)
        hcas = [f"mlx5_{i}" for i in range(self.hcas)]
        if device_name:
            hcas = [hca for hca in hcas if hca in device_name.split(",")]
        return json.dumps({"cpu:0": [hcas, []]})

    def register_memory(self, ptr, nbytes):
        return 0

    def unregister_memory(self, ptr):
        self.unregistered.append(ptr)
        return self.unregister_result

    def batch_transfer_async_read(self, session, local, remote, lengths):
        (local,), (remote,), (nbytes,) = local, remote, lengths
        with self._lock:
            attempts = self.plan[remote]
            statuses = attempts[min(self._submits[remote], len(attempts) - 1)]
            self._submits[remote] += 1
            if statuses == "submit-fails":
                return self.submit_failure
            batch_id = next(self._ids)
            self._batches[batch_id] = [local, remote, nbytes, list(statuses)]
            return batch_id

    def transfer_check_status(self, batch_id):
        with self._lock:
            if batch_id in self._freed:
                self.after_free.append(batch_id)
                return 0
            local, remote, nbytes, statuses = self._batches[batch_id]
            self.polls.append((batch_id, remote, threading.current_thread().name))
            status = statuses.pop(0) if len(statuses) > 1 else statuses[0]
            if status == 1:
                ctypes.memmove(local, remote, nbytes)
            if status in (1, -1):
                self._freed.add(batch_id)
            return status

    def report(self, remote, status, wait=True):
        """From now on, the live batch reading ``remote`` reports ``status``.
        ``wait`` waits for the reader thread to submit it, which a loaded host
        delays."""
        deadline = time.monotonic() + 5
        while True:
            with self._lock:
                live = [
                    batch for batch_id, batch in self._batches.items()
                    if batch[1] == remote and batch_id not in self._freed
                ]
                for batch in live:
                    batch[3] = [status]
            if live or not wait:
                return
            assert time.monotonic() < deadline, f"no live batch reads {remote:#x}"
            time.sleep(0.001)

    def submits(self, remote):
        return self._submits[remote]

    def polled_by(self, remote):
        return {thread for _, r, thread in self.polls if r == remote}


@pytest.fixture
def engine():
    engine = _Engine()
    managers = []

    def make():
        mgr = MooncakeCommunicationManager.__new__(MooncakeCommunicationManager)
        TensorCommunicationManager.__init__(mgr, "receiver", "receiver", "cpu", _Communicator(), engine)
        mgr.protocol = CommProtocol.RDMA
        mgr._async_reader = TrackedMooncakeReader(engine, "cpu")
        mgr._transfer_status = mgr._async_reader.transfer_status
        managers.append(mgr)
        return mgr

    engine.manager = make
    yield engine
    # let any reader still polling a running batch finish
    for remote in list(engine.plan):
        engine.report(remote, 1, wait=False)
    for mgr in managers:
        mgr._async_reader._executor.shutdown(wait=True)
    assert engine.after_free == [], "a freed batch was polled"


def _sources(n, nbytes=16):
    return [torch.full((nbytes,), i + 1, dtype=torch.uint8) for i in range(n)]


def _remote_edge(sources, first_uuid=100):
    infos = [
        TensorPointerInfo(
            list(src.shape), src.dtype, src.nbytes, src.data_ptr(),
            list(src.stride()), first_uuid + i, "producer:1", "producer",
        )
        for i, src in enumerate(sources)
    ]
    return GraphEdge(next_node="node", name="x", tensor_info=infos)


def _start(mgr, rid, sources):
    edge = _remote_edge(sources)
    (future,) = mgr.start_read_tensors(rid, [edge])
    buffers = [mgr.get_tensor(info.uuid) for info in edge.tensor_info]
    return future, [b.data_ptr() for b in buffers]


def _kept(mgr):
    return [b.data_ptr() for kept in mgr._kept_reads for b in kept.buffers]


def _never_freed(mgr):
    return [b.data_ptr() for b in mgr._unreleasable_buffers]


@pytest.mark.parametrize("final", [1, -1], ids=["completed", "failed"])
def test_a_failed_reads_buffers_are_freed_once_its_last_transfer_stops(engine, final):
    """Tensor 0 fails while tensor 1 is still moving. Its buffers stay through
    teardown while tensor 1 reports WAITING (0) or TIMEOUT (-2), and are freed
    once it reports COMPLETED or FAILED."""
    mgr = engine.manager()
    src = _sources(2)
    engine.plan = {src[0].data_ptr(): [[0, 0, -1]], src[1].data_ptr(): [[0]]}
    future, ptrs = _start(mgr, "bad", src)

    error = future.exception(timeout=5)
    assert isinstance(error, AsyncReadError)
    assert list(error.running) == [ptrs[1]] and error.unknown == set()
    assert mgr.get_ready_tensors() == {}
    assert [f.request_id for f in mgr.take_failed_reads()] == ["bad"]
    mgr.force_cleanup_request("bad")
    assert not mgr.tensor_store.get_all_uuids("bad")

    for status in (0, -2, 0):
        engine.report(src[1].data_ptr(), status)
        mgr.get_ready_tensors()
        assert _kept(mgr) == ptrs and engine.unregistered == []

    engine.report(src[1].data_ptr(), final)
    mgr.get_ready_tensors()
    assert mgr._kept_reads == [] and _never_freed(mgr) == []
    assert sorted(engine.unregistered) == sorted(ptrs)
    assert _acked(mgr.communicator) == []
    # the reader polled only tensor 0; tensor 1 was polled here, after hand-over
    assert engine.polled_by(src[0].data_ptr()) != {threading.current_thread().name}
    assert engine.polled_by(src[1].data_ptr()) == {threading.current_thread().name}


@pytest.mark.parametrize("first", [1, -1], ids=["completes", "fails"])
def test_a_read_torn_down_mid_transfer_is_freed_once_it_stops(engine, first):
    """Teardown finds the read running. The reader owns its batches until it
    finishes; the manager never polls them before that, and frees the buffers
    once every transfer stopped, whether the read completed or failed."""
    mgr = engine.manager()
    src = _sources(2)
    engine.plan = {s.data_ptr(): [[0]] for s in src}
    future, ptrs = _start(mgr, "r", src)

    mgr.force_cleanup_request("r")
    assert _kept(mgr) == ptrs and _acked(mgr.communicator) == []
    mgr.get_ready_tensors()
    assert _kept(mgr) == ptrs

    engine.report(src[0].data_ptr(), first)
    if first == -1:
        assert list(future.exception(timeout=5).running) == [ptrs[1]]
        mgr.get_ready_tensors()
        assert _kept(mgr) == ptrs, "tensor 1 is still moving"
    engine.report(src[1].data_ptr(), 1)
    if first == 1:
        assert future.exception(timeout=5) is None
    mgr.get_ready_tensors()
    assert mgr._kept_reads == [] and sorted(engine.unregistered) == sorted(ptrs)
    me = threading.current_thread().name
    assert me not in engine.polled_by(src[0].data_ptr())
    if first == 1:
        assert me not in engine.polled_by(src[1].data_ptr())


def test_a_read_that_fails_while_teardown_polls_it_keeps_its_buffers(engine, monkeypatch):
    """Teardown polls the read just before the reader fails and hands over a
    running transfer: the buffers wait for that transfer, not for the poll."""
    mgr = engine.manager()
    src = _sources(2)
    engine.plan = {src[0].data_ptr(): [[-1]], src[1].data_ptr(): [[0]]}
    future, ptrs = _start(mgr, "r", src)
    assert list(future.exception(timeout=5).running) == [ptrs[1]]
    poll = type(mgr)._poll_read
    monkeypatch.setattr(mgr, "_poll_read", lambda ep: (False, None))
    mgr.force_cleanup_request("r")
    monkeypatch.setattr(mgr, "_poll_read", poll)

    mgr.get_ready_tensors()
    assert _kept(mgr) == ptrs and engine.unregistered == []
    engine.report(src[1].data_ptr(), 1)
    mgr.get_ready_tensors()
    assert mgr._kept_reads == [] and sorted(engine.unregistered) == sorted(ptrs)


def test_the_drain_waits_for_a_failed_reads_running_transfers(engine):
    """READS_DONE lets the producer free what the read is reading, so a
    failed read counts as in flight until its last transfer stops, and not
    after its request is torn down."""
    mgr = engine.manager()
    src = _sources(2)
    engine.plan = {src[0].data_ptr(): [[0, -1]], src[1].data_ptr(): [[0]]}
    future, _ = _start(mgr, "r", src)
    assert mgr.has_inflight_reads("r")
    future.exception(timeout=5)
    assert mgr.has_inflight_reads("r"), "the reader is done, the transfer is not"
    mgr.get_ready_tensors()
    assert mgr.take_failed_reads() and mgr.has_inflight_reads("r")
    engine.report(src[1].data_ptr(), 1)
    mgr.get_ready_tensors()
    assert not mgr.has_inflight_reads("r")


def test_a_torn_down_requests_kept_read_no_longer_holds_a_drain(engine):
    mgr = engine.manager()
    src = _sources(2)
    engine.plan = {src[0].data_ptr(): [[-1]], src[1].data_ptr(): [[0]]}
    future, ptrs = _start(mgr, "r", src)
    future.exception(timeout=5)
    mgr.get_ready_tensors()
    assert mgr.has_inflight_reads("r")
    mgr.force_cleanup_request("r")
    assert not mgr.has_inflight_reads("r") and _kept(mgr) == ptrs


def test_the_data_worker_holds_reads_done_for_a_failed_reads_running_transfer(tmp_path):
    """The failed read's error chunks count as delivered, so the API server
    reports the request drained; READS_DONE still waits for the transfer."""
    from mstar.utils.ipc_format import ConductorMessageType as T

    wt = _preprocess_thread(model=None)
    wt._live_requests = {"r1": 1}
    wt._draining_rids, wt._reads_done_sent = set(), set()
    wt.sent = []
    wt.communicator = SimpleNamespace(send=lambda entity, msg: wt.sent.append(msg))
    running = {"r1"}
    wt.tensor_manager = SimpleNamespace(has_inflight_reads=lambda rid: rid in running)

    wt._finish_reading("r1", drained=True)
    assert [m.message_type for m in wt.sent] == [] and "r1" in wt._draining_rids
    running.clear()
    wt._complete_drain_if_ready("r1")
    assert [m.message_type for m in wt.sent] == [T.READS_DONE]


@pytest.mark.parametrize("failure", [0, 2**64 - 1], ids=["zero", "legacy-uint64-minus-1"])
@pytest.mark.parametrize("when", ["first-submit", "retry-submit"])
def test_a_failed_submit_keeps_the_buffers_for_good(engine, failure, when):
    """A failed submit may have posted part of its transfer and gives no batch
    id to ask about, so nothing of the read is freed, and nothing is polled
    for it after the read fails."""
    engine.hcas = 1  # two attempts per tensor
    engine.submit_failure = failure
    mgr = engine.manager()
    src = _sources(2)
    second = [[0, -1], "submit-fails"] if when == "retry-submit" else ["submit-fails"]
    engine.plan = {src[0].data_ptr(): [[0, 0, 1]], src[1].data_ptr(): second}
    future, ptrs = _start(mgr, "bad", src)

    error = future.exception(timeout=5)
    assert isinstance(error, AsyncReadError) and "submit failed" in str(error)
    assert error.unknown == {ptrs[1]}
    mgr.get_ready_tensors()
    mgr.force_cleanup_request("bad")
    for _ in range(3):
        mgr.get_ready_tensors()
    assert mgr._kept_reads == [] and _never_freed(mgr) == ptrs
    assert engine.unregistered == []
    assert threading.current_thread().name not in engine.polled_by(src[0].data_ptr())


def test_an_unknown_status_keeps_the_buffers_for_good(engine):
    """Only COMPLETED and FAILED say a transfer stopped. Any other status may
    or may not have freed its batch: the buffers stay and it is never polled
    again, in the reader or in the manager."""
    mgr = engine.manager()
    src = _sources(3)
    engine.plan = {
        src[0].data_ptr(): [[-1]], src[1].data_ptr(): [[0]], src[2].data_ptr(): [[0]],
    }
    future, ptrs = _start(mgr, "a", src)
    assert set(future.exception(timeout=5).running) == {ptrs[1], ptrs[2]}
    mgr.get_ready_tensors()
    engine.report(src[1].data_ptr(), 7)
    mgr.get_ready_tensors()
    mgr.get_ready_tensors()
    assert mgr._kept_reads == [] and _never_freed(mgr) == ptrs
    assert [r for _, r, _ in engine.polls].count(src[1].data_ptr()) == 1
    assert engine.unregistered == []

    reader_src = _sources(1, nbytes=8)
    engine.plan[reader_src[0].data_ptr()] = [[0, 9]]
    future, reader_ptrs = _start(mgr, "b", reader_src)
    error = future.exception(timeout=5)
    assert "unknown status 9" in str(error) and error.unknown == set(reader_ptrs)
    mgr.get_ready_tensors()
    assert _never_freed(mgr) == ptrs + reader_ptrs


def test_buffers_whose_unregister_fails_are_kept(engine):
    engine.unregister_result = -3
    mgr = engine.manager()
    src = _sources(1)
    engine.plan = {src[0].data_ptr(): [[-1]]}
    future, ptrs = _start(mgr, "a", src)
    future.exception(timeout=5)
    mgr.get_ready_tensors()
    mgr.get_ready_tensors()
    assert engine.unregistered == ptrs
    assert mgr._kept_reads == [] and _never_freed(mgr) == ptrs


@pytest.mark.parametrize("hcas, outcome", [(1, "read"), (0, "failed")])
def test_a_failed_tensor_is_retried_as_often_as_the_sync_read_retries(engine, hcas, outcome):
    engine.hcas = hcas
    mgr = engine.manager()
    src = _sources(1)
    engine.plan = {src[0].data_ptr(): [[-1], [1]]}
    future, _ = _start(mgr, "a", src)
    error = future.exception(timeout=5)
    assert engine.submits(src[0].data_ptr()) == hcas + 1
    if outcome == "read":
        assert error is None
        (edge,) = mgr.get_ready_tensors()["a"]
        assert torch.equal(mgr.get_tensor(edge.tensor_info[0].uuid), src[0])
    else:
        assert "Status: -1" in str(error)


@pytest.mark.parametrize("device_filter, attempts", [("", 3), ("mlx5_1", 2)])
def test_retries_count_only_the_hcas_the_engines_device_filter_keeps(engine, device_filter, attempts):
    engine.hcas = 2
    reader = TrackedMooncakeReader(engine, "cpu", device_filter=device_filter)
    assert reader._read_attempts == attempts
    reader.shutdown()


def test_reads_that_succeed_never_count_the_hcas(engine):
    """Counting starts a second Mooncake engine, so neither startup nor a
    successful read pays for it."""
    mgr = engine.manager()
    src = _sources(2)
    engine.plan = {s.data_ptr(): [[0, 1]] for s in src}
    future, _ = _start(mgr, "ok", src)
    assert future.exception(timeout=5) is None
    assert engine.topology_calls == 0


def test_the_first_failed_tensor_counts_the_hcas_once(engine):
    """Reads failing at once on the reader's three threads count once."""
    engine.hcas, engine.topology_delay = 1, 0.05
    mgr = engine.manager()
    futures = []
    for i in range(3):
        src = _sources(1)
        engine.plan[src[0].data_ptr()] = [[-1], [1]]
        (future,) = mgr.start_read_tensors(f"r{i}", [_remote_edge(src, first_uuid=100 + i)])
        futures.append((future, src))
    for future, src in futures:
        assert future.exception(timeout=5) is None
        assert engine.submits(src[0].data_ptr()) == 2
    assert engine.topology_calls == 1


def test_a_read_past_its_deadline_hands_over_its_running_transfers(engine):
    mgr = engine.manager()
    mgr._async_reader._timeout_s = 0
    src = _sources(1)
    engine.plan = {src[0].data_ptr(): [[0]]}
    future, ptrs = _start(mgr, "slow", src)
    error = future.exception(timeout=5)
    assert "timed out" in str(error) and list(error.running) == ptrs

    mgr.get_ready_tensors()
    mgr.force_cleanup_request("slow")
    mgr.get_ready_tensors()
    assert _kept(mgr) == ptrs
    engine.report(src[0].data_ptr(), 1)
    mgr.get_ready_tensors()
    assert mgr._kept_reads == [] and engine.unregistered == ptrs


def test_kv_reads_keep_the_batched_sync_read():
    calls = []
    engine = SimpleNamespace(
        batch_transfer_sync_read=lambda session, local, remote, n: calls.append(len(local)) or 0,
    )
    reader = AsyncMooncakeReader(engine, "cpu")
    try:
        infos = [TransferReadInfo("peer", 1000 + i, 2000 + i, 1) for i in range(3)]
        assert reader.submit(infos).result(timeout=5) is None
        assert calls == [3]
    finally:
        reader.shutdown()


# ── real Mooncake TCP ─────────────────────────────────────────────────────


class _Recording:
    """Records what ``transfer_check_status`` returned."""

    def __init__(self, engine):
        self._engine = engine
        self.statuses = []

    def transfer_check_status(self, batch_id):
        status = self._engine.transfer_check_status(batch_id)
        self.statuses.append(status)
        return status

    def __getattr__(self, name):
        return getattr(self._engine, name)


@pytest.mark.skipif(MOONCAKE_IMPORT_ERROR is not None, reason=f"mooncake unavailable: {MOONCAKE_IMPORT_ERROR}")
def test_real_mooncake_tcp_frees_a_failed_reads_buffer():
    """A read of producer memory that was never registered: Mooncake's TCP
    transport rejects it, ``transfer_check_status`` reports FAILED, the
    request fails alone, and its buffer is unregistered and freed at the
    next poll. A good read on the same pair still delivers its bytes."""
    def make(entity):
        return create_tensor_communication_manager(
            protocol=CommProtocol.TCP, my_entity_id=entity, hostname="localhost",
            device="cpu", communicator=_Communicator(), tcp_transfer_device="lo",
        )

    producer, consumer = make("producer"), make("consumer")
    recording = _Recording(consumer._async_reader._engine)
    consumer._async_reader._engine = recording
    unregistered = []
    unregister = consumer.transfer_engine.unregister_memory
    consumer.transfer_engine.unregister_memory = lambda ptr: unregistered.append(ptr) or unregister(ptr)
    try:
        good, bad = torch.arange(4096, dtype=torch.int32), torch.ones(4096, dtype=torch.int32)
        good_edge = GraphEdge(next_node="n", name="x")
        good_edge.tensor_info = producer.store_and_return_tensor_info("good", {"x": [good]})["x"]
        producer.register_for_send("good", good_edge.tensor_info)
        bad_edge = GraphEdge(next_node="n", name="x")
        bad_edge.tensor_info = producer.store_and_return_tensor_info("bad", {"x": [bad]})["x"]

        (bad_future,) = consumer.start_read_tensors("bad", [bad_edge])
        buffer_ptr = consumer.get_tensor(bad_edge.tensor_info[0].uuid).data_ptr()
        (good_future,) = consumer.start_read_tensors("good", [good_edge])
        assert bad_future.exception(timeout=30) is not None
        assert good_future.exception(timeout=30) is None
        assert -1 in recording.statuses

        ready = consumer.get_ready_tensors()
        assert list(ready) == ["good"]
        assert torch.equal(consumer.get_tensor(ready["good"][0].tensor_info[0].uuid), good)
        assert [f.request_id for f in consumer.take_failed_reads()] == ["bad"]
        consumer.force_cleanup_request("bad")
        consumer.get_ready_tensors()
        assert buffer_ptr in unregistered
        assert consumer._kept_reads == [] and consumer._unreleasable_buffers == []
    finally:
        for mgr in (producer, consumer):
            mgr._async_reader._executor.shutdown(wait=True)


@pytest.mark.skipif(MOONCAKE_IMPORT_ERROR is not None, reason=f"mooncake unavailable: {MOONCAKE_IMPORT_ERROR}")
def test_real_mooncake_tcp_keeps_a_failed_reads_buffers_until_its_running_transfer_stops():
    """One read of two tensors over real Mooncake TCP: the first was never
    registered on the producer and fails at once, the second (2 GiB) is still
    moving when the read gives up. The request fails at once, but its buffers
    stay registered and the request keeps a read in flight (READS_DONE waits)
    until Mooncake reports the large transfer COMPLETED; then both buffers
    are unregistered and freed."""
    def make(entity):
        return create_tensor_communication_manager(
            protocol=CommProtocol.TCP, my_entity_id=entity, hostname="localhost",
            device="cpu", communicator=_Communicator(), tcp_transfer_device="lo",
        )

    producer, consumer = make("producer"), make("consumer")
    recording = _Recording(consumer._async_reader._engine)
    consumer._async_reader._engine = recording
    unregistered = []
    unregister = consumer.transfer_engine.unregister_memory
    consumer.transfer_engine.unregister_memory = lambda ptr: unregistered.append(ptr) or unregister(ptr)
    try:
        bad = torch.ones(4096, dtype=torch.int32)
        big = torch.ones(2 << 30, dtype=torch.uint8)
        bad_info = producer.store_and_return_tensor_info("r", {"bad": [bad]})["bad"]
        big_info = producer.store_and_return_tensor_info("r", {"big": [big]})["big"]
        producer.register_for_send("r", big_info)
        edge = GraphEdge(next_node="n", name="x")
        edge.tensor_info = bad_info + big_info

        (future,) = consumer.start_read_tensors("r", [edge])
        ptrs = [consumer.get_tensor(info.uuid).data_ptr() for info in edge.tensor_info]
        error = future.exception(timeout=60)
        assert isinstance(error, AsyncReadError), error
        assert ptrs[1] in error.running, "the 2 GiB transfer finished before the read gave up"

        assert consumer.get_ready_tensors() == {}
        assert [f.request_id for f in consumer.take_failed_reads()] == ["r"]
        assert consumer.has_inflight_reads("r")
        assert not set(ptrs) & set(unregistered)
        deadline = time.monotonic() + 120
        while consumer.has_inflight_reads("r"):
            assert not set(ptrs) & set(unregistered), "a buffer was freed while its transfer ran"
            assert time.monotonic() < deadline, "the large transfer never reported that it stopped"
            consumer.get_ready_tensors()
            time.sleep(0.01)
        assert _TRANSFER_COMPLETED in recording.statuses
        assert set(ptrs) <= set(unregistered)
        assert consumer._kept_reads == [] and consumer._unreleasable_buffers == []
    finally:
        for mgr in (producer, consumer):
            mgr._async_reader._executor.shutdown(wait=True)
