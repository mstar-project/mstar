"""The data worker thread's idle wait: it blocks on the communicator's poll
and is woken by a producer's signal, a finished tensor read or shutdown,
instead of sleeping 1 ms and sweeping its queues."""
from __future__ import annotations

import os
import queue
import select
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from mstar.api_server.data_worker import InlineToken, PreprocessWorkerThread, ResultQueue
from mstar.communication.event import EventWakeup
from mstar.graph.loop_indices import NestedLoopIndices


class _Model:
    def get_partition_topology(self):
        from mstar.streaming.topology import PartitionTopology
        return PartitionTopology(partitions=["default"])

    def postprocess(self, output, modality, request_kwargs=None):
        return f"<{int(output[0])}>".encode()


class _PollingComm:
    """A communicator whose idle wait really blocks on the wakeup's fd."""

    def __init__(self):
        self.event = None
        self.waits = 0

    def register_event_for_poll(self, event):
        self.event = event

    def wait_for_work(self, timeout_ms):
        self.waits += 1
        ready, _, _ = select.select([self.event.fd], [], [], timeout_ms / 1000.0)
        if ready:
            self.event.drain()

    def get_all_new_messages(self):
        return []

    def send(self, entity, msg):
        pass

    # the data worker's outbox encodes with this: identity, so sends arrive as objects
    codec = SimpleNamespace(encode=lambda msg: msg)

    def try_send_bytes(self, entity, msg):
        self.send(entity, msg)
        return True


class _BareComm:
    def get_all_new_messages(self):
        return []

    def send(self, entity, msg):
        pass

    # the data worker's outbox encodes with this: identity, so sends arrive as objects
    codec = SimpleNamespace(encode=lambda msg: msg)

    def try_send_bytes(self, entity, msg):
        self.send(entity, msg)
        return True


class _TM:
    def __init__(self):
        self.futures: list[Future] = []

    def start_read_tensors(self, request_id, graph_edges, graph_walk=None):
        return list(self.futures)

    def take_failed_reads(self):

        return []


    def get_ready_tensors(self):
        return {}

    def has_inflight_reads(self, request_id):
        return False

    def cleanup_request(self, request_id):
        pass

    def force_cleanup_request(self, request_id):
        pass

    def ack_unread_tensors(self, request_id, graph_edges):
        pass


def _thread(comm, wakeup, idle_wait_ms=2000, out_queue=None):
    stop = threading.Event()
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(),
        out_queue=out_queue if out_queue is not None else queue.Queue(),
        profile_queue=queue.Queue(), cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=stop, communicator=comm,
        tensor_manager=_TM(), model=_Model(), wakeup=wakeup,
    )
    worker.idle_wait_ms = idle_wait_ms
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    return worker, stop, t


def _loop(i):
    return NestedLoopIndices(
        loop_name_order=["decode_loop"], loop_indices={"decode_loop": i}, wg_fwd_pass_idx=i + 1,
    )


def _wait_until(pred, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.001)
    return pred()


def test_a_signalled_producer_ends_the_idle_wait_at_once():
    comm, wakeup = _PollingComm(), EventWakeup()
    worker, stop, t = _thread(comm, wakeup, idle_wait_ms=2000)
    assert _wait_until(lambda: comm.waits >= 1, 1.0)  # the thread is parked
    t0 = time.monotonic()
    worker.result_tensor_queue.put(InlineToken("req", 7, _loop(0), "text"))
    wakeup.signal()
    chunk = worker.out_queue.get(timeout=1.0)
    assert time.monotonic() - t0 < 0.5  # far below the 2 s backstop
    assert chunk.request_id == "req" and chunk.data == b"<7>"
    stop.set()
    wakeup.signal()
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_shutdown_wakes_a_parked_thread():
    comm, wakeup = _PollingComm(), EventWakeup()
    worker, stop, t = _thread(comm, wakeup, idle_wait_ms=5000)
    assert _wait_until(lambda: comm.waits >= 1, 1.0)
    t0 = time.monotonic()
    stop.set()
    wakeup.signal()
    t.join(timeout=2.0)
    assert not t.is_alive() and time.monotonic() - t0 < 1.0


def test_an_idle_thread_does_not_spin():
    comm, wakeup = _PollingComm(), EventWakeup()
    worker, stop, t = _thread(comm, wakeup, idle_wait_ms=50)
    time.sleep(0.3)
    stop.set()
    wakeup.signal()
    t.join(timeout=1.0)
    assert not t.is_alive()
    # ~6 waits of 50 ms in 0.3 s, not ~300 sweeps of 1 ms
    assert comm.waits <= 12


def test_a_finished_read_wakes_the_thread():
    comm, wakeup = _PollingComm(), EventWakeup()
    worker, stop, t = _thread(comm, wakeup, idle_wait_ms=2000)
    assert _wait_until(lambda: comm.waits >= 1, 1.0)
    fut: Future = Future()
    worker.tensor_manager.futures = [fut]
    worker.wakeup.register_futures(worker.tensor_manager.start_read_tensors("r", []))
    waits = comm.waits
    time.sleep(0.05)
    assert comm.waits == waits  # still parked
    fut.set_result(None)  # the reader thread finishing wakes the poll
    assert _wait_until(lambda: comm.waits > waits, 1.0)
    stop.set()
    wakeup.signal()
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_without_a_wakeup_the_thread_keeps_the_sleep():
    worker, stop, t = _thread(_BareComm(), None)
    worker.result_tensor_queue.put(InlineToken("req", 3, _loop(0), "text"))
    chunk = worker.out_queue.get(timeout=1.0)
    assert chunk.data == b"<3>"
    stop.set()
    t.join(timeout=1.0)
    assert not t.is_alive()


# -- the other direction: the data worker waking the API message thread --------


def _signals(wakeup) -> int:
    """How many signals ``wakeup`` took since it was last drained (the
    eventfd adds them up), draining it."""
    try:
        return os.eventfd_read(wakeup.fd)
    except BlockingIOError:
        return 0


def test_a_flush_signals_once_for_every_put_since_the_last():
    out = ResultQueue(EventWakeup())
    out.flush()
    assert _signals(out.wakeup) == 0
    for i in range(16):
        out.put(i)
    out.flush()
    out.flush()
    assert _signals(out.wakeup) == 1 and out.qsize() == 16


def test_a_step_of_results_wakes_the_message_thread_once():
    comm, wakeup = _PollingComm(), EventWakeup()
    out = ResultQueue(EventWakeup())
    worker, stop, t = _thread(comm, wakeup, out_queue=out)
    assert _wait_until(lambda: comm.waits >= 1, 1.0)
    # one step's frame: every row's token handed over, then one wake
    for i in range(16):
        worker.result_tensor_queue.put(InlineToken(f"r{i}", i, _loop(0), "text"))
    wakeup.signal()
    assert _wait_until(lambda: out.qsize() == 16, 1.0)
    assert _wait_until(lambda: comm.waits >= 2, 1.0)  # the pass is over
    assert _signals(out.wakeup) == 1
    stop.set()
    wakeup.signal()
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_a_queue_that_refills_while_draining_wakes_once_a_batch():
    """The next step's tokens landing mid-drain are a batch of their own:
    the chunks drained so far go out first instead of waiting for the
    queue to run dry."""
    comm, wakeup = _PollingComm(), EventWakeup()
    out = ResultQueue(EventWakeup())
    flushed_at = []
    flush = out.flush

    def recording_flush():
        if out._unsignaled:
            flushed_at.append(out.qsize())
        flush()

    out.flush = recording_flush
    worker, stop, t = _thread(comm, wakeup, out_queue=out)
    assert _wait_until(lambda: comm.waits >= 1, 1.0)
    late = [InlineToken(f"late{i}", i, _loop(1), "text") for i in range(4)]

    class _Refilling(_Model):
        def postprocess(self, output, modality, request_kwargs=None):
            while late:  # the message thread hands over the next step
                worker.result_tensor_queue.put(late.pop())
            return super().postprocess(output, modality, request_kwargs)

    worker.model = _Refilling()
    for i in range(4):
        worker.result_tensor_queue.put(InlineToken(f"r{i}", i, _loop(0), "text"))
    wakeup.signal()
    assert _wait_until(lambda: out.qsize() == 8, 1.0)
    assert _wait_until(lambda: comm.waits >= 2, 1.0)
    assert flushed_at == [4, 8]
    stop.set()
    wakeup.signal()
    t.join(timeout=1.0)
    assert not t.is_alive()


def _communicator(kind, tmp_path):
    if kind == "rust":
        pytest.importorskip("mstar_rust")
        from mstar.communication.rust_communicator import RustZMQCommunicator as Comm
    else:
        from mstar.communication.communicator import ZMQCommunicator as Comm
    return Comm(f"api_{kind}", push_ids=[], ipc_socket_path_prefix=str(tmp_path) + "/")


@pytest.mark.parametrize("kind", ["python", "rust"])
def test_the_message_loop_waits_on_the_results_wakeup(kind, tmp_path):
    """The message thread's wait between passes ends on the data worker's
    signal and drains it, on either transport; unsignalled, it is the 1 ms
    backstop the loop's nap was."""
    from api_server_helpers import make_test_api_server

    results = EventWakeup()
    server = make_test_api_server(
        communicator=_communicator(kind, tmp_path),
        preprocess_worker=SimpleNamespace(results_wakeup=results),
    )
    wait = server._pass_wait()
    t0 = time.monotonic()
    for _ in range(20):
        wait()
    assert 0.015 < time.monotonic() - t0 < 1.0
    results.signal()
    wait()
    assert _signals(results) == 0, "the wait drained the signal"
    # a stand-in without the poll keeps the nap
    bare = make_test_api_server(communicator=SimpleNamespace(), preprocess_worker=SimpleNamespace())
    t0 = time.monotonic()
    bare._pass_wait()()
    assert time.monotonic() - t0 >= 0.001
