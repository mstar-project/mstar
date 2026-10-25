"""The data worker thread's idle wait: it blocks on the communicator's poll
and is woken by a producer's signal, a finished tensor read or shutdown,
instead of sleeping 1 ms and sweeping its queues."""
from __future__ import annotations

import queue
import select
import threading
import time
from concurrent.futures import Future

from mstar.api_server.data_worker import InlineToken, PreprocessWorkerThread
from mstar.communication.event import EventWakeup
from mstar.graph.loop_indices import NestedLoopIndices


class _Model:
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


class _BareComm:
    def get_all_new_messages(self):
        return []

    def send(self, entity, msg):
        pass


class _TM:
    def __init__(self):
        self.futures: list[Future] = []

    def start_read_tensors(self, request_id, graph_edges, graph_walk=None):
        return list(self.futures)

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


def _thread(comm, wakeup, idle_wait_ms=2000):
    stop = threading.Event()
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=queue.Queue(),
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
