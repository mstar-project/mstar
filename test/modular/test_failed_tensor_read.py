"""A failed async tensor read (a Mooncake/RDMA transfer error) fails only its
own request. Before, ``get_ready_tensors`` re-raised the same failed future on
every poll, so no other request on that worker or data worker received its
inputs until the failed request was torn down (the repeated "Mooncake read
failed" of #332)."""

import ctypes
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import (
    READ_QUARANTINE_BUDGET_ENV,
    AsyncMooncakeReader,
    FutureAndPointers,
    MooncakeCommunicationManager,
    SharedMemoryCommunicationManager,
    TensorCommunicationManager,
    read_quarantine_budget_bytes,
)
from mstar.graph.base import GraphEdge, TensorPointerInfo
from mstar.profile.format import ReadQuarantine
from mstar.utils.ipc_format import WorkerMessageType


class _Communicator(BaseCommunicator):
    def __init__(self):
        self.sent = []

    def send(self, entity_id, msg):
        self.sent.append((entity_id, msg))

    def get_all_new_messages(self):
        return []


def _manager(tmp_path):
    comm = _Communicator()
    mgr = SharedMemoryCommunicationManager(
        "worker_1", "host", "cpu", comm, shm_dir=str(tmp_path),
    )
    return mgr, comm


def _edge(*uuids):
    edge = GraphEdge(next_node="node", name="x")
    edge.tensor_info = [SimpleNamespace(uuid=u, source_entity="worker_0") for u in uuids]
    return edge


def _failed_future(message="Mooncake read failed. Status: -1"):
    f = Future()
    f.set_exception(RuntimeError(message))
    return f


def _read(rid, edge, future=None):
    return FutureAndPointers(future=future, graph_edges=[edge], request_id=rid)


def _acked(comm):
    return [
        (msg.body.request_id, dict(msg.body.successful_tensors))
        for _, msg in comm.sent
        if msg.message_type == WorkerMessageType.TENSOR_RECEIVED
    ]


def test_a_failed_read_does_not_block_other_requests(tmp_path):
    mgr, comm = _manager(tmp_path)
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
    mgr, _ = _manager(tmp_path)
    cancelled = Future()
    cancelled.cancel()
    mgr.pending = [_read(7, _edge(11), cancelled)]

    assert mgr.get_ready_tensors() == {}
    (failed,) = mgr.take_failed_reads()
    assert failed.request_id == 7 and mgr.pending == []


def test_later_reads_of_a_failed_request_are_dropped_until_cleanup(tmp_path):
    mgr, comm = _manager(tmp_path)
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


class _Engine:
    """Copies like a transfer engine; session "timed-out" returns -1 the way
    Mooncake's sync read does when it gives up on a batch it could not free."""

    def __init__(self):
        self.unregistered = []
        self.started = threading.Event()
        self.release = threading.Event()

    def register_memory(self, ptr, nbytes):
        return 0

    def unregister_memory(self, ptr):
        self.unregistered.append(ptr)
        return 0

    def batch_transfer_sync_read(self, session, local, remote, lengths):
        if session == "timed-out":
            return -1
        if session == "slow-timeout":
            self.started.set()
            assert self.release.wait(5)
            return -1
        for dst, src, n in zip(local, remote, lengths, strict=True):
            ctypes.memmove(dst, src, n)
        return 0


def _mooncake_manager(engine):
    mgr = MooncakeCommunicationManager.__new__(MooncakeCommunicationManager)
    TensorCommunicationManager.__init__(mgr, "receiver", "receiver", "cpu", _Communicator(), engine)
    mgr.protocol = CommProtocol.RDMA
    mgr._async_reader = AsyncMooncakeReader(engine, "cpu")
    return mgr


def _remote_edge(uuid, src, session):
    info = TensorPointerInfo(
        list(src.shape), src.dtype, src.nbytes, src.data_ptr(),
        list(src.stride()), uuid, session, "producer",
    )
    return GraphEdge(next_node="node", name="x", tensor_info=[info])


def test_a_failed_transfer_keeps_its_buffers_through_teardown():
    engine = _Engine()
    mgr = _mooncake_manager(engine)
    src = torch.tensor([42, 43])
    try:
        (bad,) = mgr.start_read_tensors("bad", [_remote_edge(31, src, "timed-out")])
        good_edge = _remote_edge(32, src, "peer")
        (good,) = mgr.start_read_tensors("good", [good_edge])
        buffer = mgr.get_tensor(31)
        for f in (bad, good):
            f.exception(timeout=5)

        assert mgr.get_ready_tensors() == {"good": [good_edge]}
        assert torch.equal(mgr.get_tensor(32), src)
        assert [f.request_id for f in mgr.take_failed_reads()] == ["bad"]

        mgr.force_cleanup_request("bad")
        # still registered and alive: the transfer may write into it
        assert buffer.data_ptr() not in engine.unregistered
        assert any(t is buffer for t in mgr._quarantined_buffers)
        assert _acked(mgr.communicator) == [("good", {32: 1})]

        good_ptr = mgr.get_tensor(32).data_ptr()
        mgr.dereference(32)
        mgr.force_cleanup_request("good")
        assert engine.unregistered == [good_ptr]
        assert not mgr.tensor_store.get_all_uuids("good")
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_teardown_before_the_next_poll_still_keeps_a_failed_transfers_buffers():
    """The read fails after the last poll, so only teardown sees it."""
    engine = _Engine()
    mgr = _mooncake_manager(engine)
    try:
        (future,) = mgr.start_read_tensors("bad", [_remote_edge(41, torch.tensor([1]), "slow-timeout")])
        buffer = mgr.get_tensor(41)
        assert engine.started.wait(5)
        assert mgr.get_ready_tensors() == {}
        engine.release.set()
        assert future.exception(timeout=5) is not None
        assert not mgr.has_inflight_reads("bad")

        mgr.force_cleanup_request("bad")

        assert buffer.data_ptr() not in engine.unregistered
        assert any(t is buffer for t in mgr._quarantined_buffers)
        assert _acked(mgr.communicator) == [] and mgr.pending == []
    finally:
        engine.release.set()
        mgr._async_reader._executor.shutdown(wait=True)


def test_kept_buffers_are_counted_and_flagged_past_the_budget(monkeypatch, caplog):
    """Kept buffers are never freed, so they are counted (reads and bytes) and
    the process reports itself over budget once the bytes pass it, logging
    that once."""
    monkeypatch.setenv(READ_QUARANTINE_BUDGET_ENV, str(20 / (1 << 20)))  # 20 bytes
    engine = _Engine()
    mgr = _mooncake_manager(engine)
    try:
        assert mgr.read_quarantine() == ReadQuarantine("receiver", 0, 0, 20)
        seen = []
        for i, rid in enumerate(("a", "b", "c")):
            uuid = 50 + i
            mgr.uuid_to_edge_name[uuid] = "x"
            (future,) = mgr.start_read_tensors(rid, [_remote_edge(uuid, torch.tensor([1, 2]), "timed-out")])
            future.exception(timeout=5)
            with caplog.at_level("WARNING", logger="mstar.communication.tensors"):
                mgr.get_ready_tensors()
            assert uuid not in mgr.uuid_to_edge_name and uuid not in mgr.uuid_to_shard_dim
            seen.append((mgr.read_quarantine(), mgr.quarantine_over_budget()))

        assert seen == [
            (ReadQuarantine("receiver", 1, 16, 20), False),
            (ReadQuarantine("receiver", 2, 32, 20), True),
            (ReadQuarantine("receiver", 3, 48, 20), True),
        ]
        assert "(3 reads, 48 bytes kept so far; budget 20 bytes)" in caplog.text
        over = [r for r in caplog.records if "reporting unhealthy" in r.getMessage()]
        assert len(over) == 1 and over[0].levelname == "ERROR"
    finally:
        mgr._async_reader._executor.shutdown(wait=True)


def test_read_quarantine_budget_setting(monkeypatch):
    monkeypatch.delenv(READ_QUARANTINE_BUDGET_ENV, raising=False)
    assert read_quarantine_budget_bytes() == 1024 << 20
    monkeypatch.setenv(READ_QUARANTINE_BUDGET_ENV, "0.5")
    assert read_quarantine_budget_bytes() == 512 << 10
    for bad in ("lots", "-1", "nan", "inf", "-inf"):
        monkeypatch.setenv(READ_QUARANTINE_BUDGET_ENV, bad)
        with pytest.raises(ValueError, match=READ_QUARANTINE_BUDGET_ENV):
            read_quarantine_budget_bytes()


class _CountingFuture:
    """A read future that counts the event queries made on it."""

    def __init__(self):
        self.finished = False
        self.done_calls = 0
        self.result_calls = 0

    def done(self):
        self.done_calls += 1
        return self.finished

    def result(self):
        self.result_calls += 1


def test_a_finished_read_is_queried_once_per_poll_and_its_result_checked_once(tmp_path):
    """A read that finished behind an unfinished read of its request waits
    for it; while it waits, polling it costs nothing more."""
    mgr, _ = _manager(tmp_path)
    earlier, later = _CountingFuture(), _CountingFuture()
    later.finished = True
    first, second = _edge(), _edge()
    mgr.pending = [_read(7, first, earlier), _read(7, second, later)]

    for _ in range(3):
        assert mgr.get_ready_tensors() == {}
    assert earlier.done_calls == 3 and earlier.result_calls == 0
    assert later.done_calls == 1 and later.result_calls == 1

    earlier.finished = True
    assert mgr.get_ready_tensors() == {7: [first, second]}
    assert earlier.done_calls == 4 and earlier.result_calls == 1
    assert later.done_calls == 1 and later.result_calls == 1


def test_log_stats_report_shows_kept_read_buffers(tmp_path):
    from mstar.profile.display import pretty_print_profile
    from mstar.profile.format import RequestProfile

    out = tmp_path / "stats.txt"
    pretty_print_profile(RequestProfile(rid="r"), str(out))
    assert "Read quarantine" not in out.read_text()

    pretty_print_profile(RequestProfile(rid="r", read_quarantine=[
        ReadQuarantine("worker_0", 1229, 117 << 20, 1 << 30),
        ReadQuarantine("api_server_preprocess_worker", 2, 2048, 1024),
    ]), str(out))
    text = out.read_text()
    assert "Read quarantine (buffers kept from failed reads, process totals):" in text
    assert "worker_0" in text and "117.0 MiB  (x1229)  budget 1.0 GiB\n" in text
    assert "2.0 KiB  (x2)  budget 1.0 KiB  OVER BUDGET" in text


def test_teardown_during_a_running_transfer_keeps_its_buffers():
    engine = _Engine()
    mgr = _mooncake_manager(engine)
    try:
        mgr.start_read_tensors("r", [_remote_edge(42, torch.tensor([1]), "slow-timeout")])
        buffer = mgr.get_tensor(42)
        assert engine.started.wait(5)

        mgr.force_cleanup_request("r")

        assert buffer.data_ptr() not in engine.unregistered
        assert any(t is buffer for t in mgr._quarantined_buffers)
        assert _acked(mgr.communicator) == []
    finally:
        engine.release.set()
        mgr._async_reader._executor.shutdown(wait=True)
