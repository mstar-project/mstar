"""The data worker thread's idle wait: it blocks on the communicator's poll
and is woken by a producer's signal, a finished tensor read or shutdown,
instead of sleeping 1 ms and sweeping its queues. And the other direction:
the API message thread's wait, which the data worker's flushes, a request
handler and the timed checks end."""
from __future__ import annotations

import os
import queue
import select
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from mstar.api_server.data_worker import InlineTokens, PreprocessWorkerThread, ResultQueue
from mstar.communication.event import EventWakeup


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


def _thread(comm, wakeup, idle_wait_ms=2000, out_queue=None, profile_queue=None):
    stop = threading.Event()
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(),
        out_queue=out_queue if out_queue is not None else queue.Queue(),
        profile_queue=profile_queue if profile_queue is not None else queue.Queue(),
        cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=stop, communicator=comm,
        tensor_manager=_TM(), model=_Model(), wakeup=wakeup,
    )
    worker.idle_wait_ms = idle_wait_ms
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    return worker, stop, t


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
    worker.result_tensor_queue.put(InlineTokens(["req"], [7], "text"))
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
    worker.result_tensor_queue.put(InlineTokens(["req"], [3], "text"))
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
    # one step's RESULT_TOKENS frame is one item
    worker.result_tensor_queue.put(InlineTokens([f"r{i}" for i in range(16)], list(range(16)), "text"))
    wakeup.signal()
    assert _wait_until(lambda: out.qsize() == 16, 1.0)
    assert _wait_until(lambda: comm.waits >= 2, 1.0)  # the pass is over
    assert _signals(out.wakeup) == 1
    stop.set()
    wakeup.signal()
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_each_step_s_frame_wakes_the_message_thread_once():
    """Frames that queued up while the thread was busy, and one handed over
    while it postprocesses, go out a step at a time: each frame's chunks with
    one wake, as soon as they are ready."""
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
    late = [InlineTokens([f"late{i}" for i in range(4)], [0, 1, 2, 3], "text")]

    class _Refilling(_Model):
        def postprocess(self, output, modality, request_kwargs=None):
            while late:  # the message thread hands over the next step
                worker.result_tensor_queue.put(late.pop())
            return super().postprocess(output, modality, request_kwargs)

    worker.model = _Refilling()
    for step in range(2):
        worker.result_tensor_queue.put(InlineTokens([f"s{step}r{i}" for i in range(4)], [0, 1, 2, 3], "text"))
    wakeup.signal()
    assert _wait_until(lambda: out.qsize() == 12, 1.0)
    assert _wait_until(lambda: comm.waits >= 2, 1.0)
    assert flushed_at == [4, 8, 12]
    stop.set()
    wakeup.signal()
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_a_profile_update_wakes_the_message_thread():
    comm, wakeup = _PollingComm(), EventWakeup()
    profiles = ResultQueue(EventWakeup())
    worker, stop, t = _thread(comm, wakeup, profile_queue=profiles)
    assert _wait_until(lambda: comm.waits >= 1, 1.0)
    # a preprocessed request leaves its profile update for the message thread
    worker._process_waiting_input = worker.profile_queue.put
    worker.in_queue.put("profile")
    wakeup.signal()
    assert _wait_until(lambda: profiles.qsize() == 1, 1.0)
    assert _wait_until(lambda: comm.waits >= 2, 1.0)
    assert _signals(profiles.wakeup) == 1
    stop.set()
    wakeup.signal()
    t.join(timeout=1.0)
    assert not t.is_alive()


# -- the API message thread's wait ---------------------------------------------


def _communicator(kind, tmp_path):
    if kind == "rust":
        pytest.importorskip("mstar_rust")
        from mstar.communication.rust_communicator import RustZMQCommunicator as Comm
    else:
        from mstar.communication.communicator import ZMQCommunicator as Comm
    return Comm(f"api_{kind}", push_ids=[], ipc_socket_path_prefix=str(tmp_path) + "/")


class _Results:
    """The message thread's view of the data worker: counts its passes."""

    def __init__(self):
        self.results_wakeup = EventWakeup()
        self.passes = 0

    def get_profile_updates(self):
        return []

    def get_result_chunks(self):
        self.passes += 1
        return []


def _message_thread(server):
    server.conductor_proc = None
    server.running = True
    thread = threading.Thread(target=server._process_messages, daemon=True)
    thread.start()
    return thread


@pytest.mark.parametrize("kind", ["python", "rust"])
def test_the_message_loop_sleeps_until_signalled(kind, tmp_path):
    """Between passes the thread waits on the communicator's poll, on either
    transport: idle it runs a pass per liveness check, and the data worker's
    signal ends the wait at once and is drained."""
    from api_server_helpers import make_test_api_server

    results = _Results()
    server = make_test_api_server(communicator=_communicator(kind, tmp_path), preprocess_worker=results)
    thread = _message_thread(server)
    try:
        assert _wait_until(lambda: results.passes >= 1, 1.0)
        time.sleep(0.2)
        assert results.passes == 1  # the next liveness check is 0.5 s away
        t0 = time.monotonic()
        results.results_wakeup.signal()
        assert _wait_until(lambda: results.passes == 2, 1.0)
        assert time.monotonic() - t0 < 0.1
        time.sleep(0.05)
        assert results.passes == 2 and _signals(results.results_wakeup) == 0, "the wait drained it"
    finally:
        server.running = False
        server._wake_message_loop()
        thread.join(timeout=2.0)
    assert not thread.is_alive()


def _completed(server, rid, counts):
    from mstar.api_server.request_types import APIServerMessage, RequestComplete

    return APIServerMessage("request_complete", RequestComplete(
        request_id=rid, output_chunk_counts=counts, conductor_ingest_time=0.0, conductor_finish_time=0.0,
    ))


def _pending(server, rid):
    from mstar.api_server.entrypoint import PendingRequest
    from mstar.profile.format import RequestProfile

    req = PendingRequest(streaming=True, input_modalities=["text"], output_modalities=["text"],
                         profile=RequestProfile(rid=rid))
    server.pending_requests[rid] = req
    return req


class _Delivery(_Results):
    """Every chunk is in: the request is drained once it completes."""

    def __init__(self):
        super().__init__()
        self.read_done = []

    def has_pending_tensors(self, rid):
        return False

    def received_final_chunks(self, rid, counts):
        return bool(counts)

    def finished_reading(self, rid, drained=True):
        self.read_done.append((rid, drained))


def test_a_completed_request_ends_in_the_pass_that_completes_it():
    """The completion and the release share a pass: the stream doesn't wait
    for the next one, which nothing may signal for a whole liveness interval."""
    from api_server_helpers import StandInSocket, make_test_api_server

    passes = []
    worker = _Delivery()
    server = make_test_api_server(preprocess_worker=worker)
    req = _pending(server, "r")

    def one_pass():
        passes.append(req.event.is_set())
        server.running = False
        return [_completed(server, "r", {"text": 1})]

    server.communicator = StandInSocket(one_pass)
    server.conductor_proc = None
    server.running = True
    server._process_messages()
    assert passes == [False] and req.event.is_set()
    assert worker.read_done == [("r", True)] and not server.recently_completed


def test_the_wait_ends_by_the_next_timed_check():
    """With nothing signalled, the poll waits until the liveness check, or
    until a parked request's delivery TTL runs out if that comes first."""
    from api_server_helpers import StandInSocket, make_test_api_server

    worker = _Delivery()
    server = make_test_api_server(preprocess_worker=worker)
    server._liveness_interval_s = 0.5
    server._recently_completed_ttl = 0.2
    _pending(server, "r")
    inbox = [[], [_completed(server, "r", {})]]

    def next_pass():
        if not inbox:
            server.running = False
            return []
        return inbox.pop(0)

    server.communicator = StandInSocket(next_pass)
    server.conductor_proc = None
    server.running = True
    server._process_messages()
    first, parked, _ = server.communicator.timeouts
    assert 490 <= first <= 500  # the liveness check
    assert 190 <= parked <= 200  # the TTL of the request that completed without its outputs
    assert "r" in server.recently_completed


def test_a_handler_letting_go_of_a_parked_request_wakes_the_loop():
    """A failed request stays parked until its handler pops it; then only the
    message thread's prune releases its outputs, so the pop ends its wait.
    A rejected append parks the call, which starts a TTL the wait must see."""
    from api_server_helpers import make_test_api_server

    worker = _Delivery()
    worker.abort_request = worker.forward_abort = lambda rid: None
    server = make_test_api_server(preprocess_worker=worker)
    req = _pending(server, "r")
    req.event.set()
    server.recently_completed["r"] = time.time()
    server._live["r"], server._client_ids["r"] = "r", "r"
    server.release_submission("r")
    assert _signals(worker.results_wakeup) == 1
    server.release_submission("r")  # already gone: nothing to do
    assert _signals(worker.results_wakeup) == 0

    call = _pending(server, "call")
    call.open_streams = frozenset({"audio_in"})
    assert server.reject_input("call", "append rejected", 400)
    assert _signals(worker.results_wakeup) == 1
