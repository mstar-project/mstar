"""The data worker thread's idle wait: it blocks on the communicator's poll
and is woken by a producer's signal, a finished tensor read or shutdown,
instead of sleeping 1 ms and sweeping its queues."""
from __future__ import annotations

import os
import queue
import select
import threading
import time
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from mstar.api_server.data_worker import InlineTokens, PreprocessWorkerThread
from mstar.communication.event import EventWakeup
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.model.base import Model


class _Model:
    def get_partition_topology(self):
        from mstar.streaming.topology import PartitionTopology
        return PartitionTopology(partitions=["default"])
    # the inline path's per-value default, over the stub's postprocess
    postprocess_inline = Model.postprocess_inline
    text_output = Model.text_output

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


class _SlowModel(_Model):
    """process_prompt of request "slow" blocks until released."""

    PREPROCESS_WORKERS = 2

    def __init__(self):
        self.release = threading.Event()
        self.started: list[str] = []

    def process_prompt(self, text, input_modalities, output_modalities, tensors=None, **kwargs):
        self.started.append(text)
        if text == "slow":
            self.release.wait(5.0)


class _StoreTM(_TM):
    def store_and_return_tensor_info(self, request_id, tensors):
        return {}

    def register_for_send(self, request_id, tensor_infos):
        pass


class _SentComm(_PollingComm):
    def __init__(self):
        super().__init__()
        self.sent: list[str] = []

    def send(self, entity, msg):
        self.sent.append(msg.body.request_id)


def test_pooled_preprocessing_overlaps_requests_and_keeps_arrival_order():
    from mstar.api_server.request_types import PreprocessInput

    comm, wakeup, model = _SentComm(), EventWakeup(), _SlowModel()
    stop = threading.Event()
    worker = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=queue.Queue(),
        profile_queue=queue.Queue(), cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=stop, communicator=comm,
        tensor_manager=_StoreTM(), model=model, wakeup=wakeup, preprocess_workers=2,
    )
    t = threading.Thread(target=worker.run, daemon=True)
    t.start()
    for rid in ("slow", "fast"):
        worker.in_queue.put(PreprocessInput(rid, rid, None, [], ["text"], {}))
    wakeup.signal()
    # "fast" is prepared while "slow" still holds its worker ...
    assert _wait_until(lambda: model.started == ["slow", "fast"], 2.0)
    time.sleep(0.05)
    # ... but reaches the conductor only after it, in arrival order
    assert comm.sent == []
    model.release.set()
    assert _wait_until(lambda: comm.sent == ["slow", "fast"], 2.0)
    stop.set()
    wakeup.signal()
    t.join(timeout=2.0)
    assert not t.is_alive()


@pytest.mark.parametrize(("declared", "workers"), [(2, 0), (2, "2"), (1, 2), (0, 4)])
def test_a_bad_preprocess_workers_fails_startup(declared, workers):
    """Not a dead data-worker thread under a server that comes up; nor a pool
    for a model class that does not declare its prepares thread-safe."""
    from mstar.api_server.data_worker import PreprocessWorker
    from mstar.communication.communicator import CommProtocol

    model = _SlowModel()
    model.PREPROCESS_WORKERS = declared
    with pytest.raises(ValueError, match="(?i)preprocess_workers"):
        PreprocessWorker(
            model=model, model_config={"preprocess_workers": workers},
            tensor_comm_protocol=CommProtocol.SHM, socket_path_prefix=f"/tmp/mstar-pw{os.getpid()}",
        )


def test_a_deployment_may_size_a_thread_safe_pool_past_the_class_value():
    from mstar.api_server.data_worker import preprocess_workers

    assert preprocess_workers(_SlowModel(), {"preprocess_workers": 16}) == 16


def test_a_prepared_request_waits_while_a_peer_cannot_take_more():
    """As on the inline path, nothing is handed on to the conductor while
    earlier sends are still queued for a peer that is backed up, including a
    send made by the hand-off just before it. The queue gauge, read during a
    hand-off, counts only the inputs behind it."""
    from mstar.api_server.data_worker import PreparingInput
    from mstar.api_server.request_types import PreprocessInput

    worker = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    worker._prep_workers, worker.in_queue = 2, queue.Queue()
    done = Future()
    done.set_result(({}, {}, {}))
    worker._preparing = deque(
        PreparingInput(PreprocessInput(rid, rid, None, [], ["text"], {}), None, done) for rid in ("r", "s")
    )
    taken, waiting = [], []

    def take_input(pre_input, profile=None, prepared=None):
        taken.append(pre_input.request_id)
        waiting.append(worker.waiting_inputs)

    worker._take_input = take_input
    # each hand-off's send fills the peer until the next pass
    worker.communicator = SimpleNamespace(flush=lambda: bool(taken))

    PreprocessWorkerThread._take_new_input(worker, sends_pending=True)
    assert taken == [], "a prepared request was sent to a peer that can't take what it was already sent"
    PreprocessWorkerThread._take_new_input(worker, sends_pending=False)
    assert taken == ["r"], "handed on behind a send the peer had not taken"
    PreprocessWorkerThread._take_new_input(worker, sends_pending=False)
    assert taken == ["r", "s"]
    assert waiting == [1, 0], "counted the input being handed on as waiting"


def test_the_pool_skips_the_prepare_of_a_request_the_api_server_let_go_of():
    """A request aborted while queued costs no media load or process_prompt;
    it is dropped as it is handed on, as on the inline path."""
    from mstar.api_server.request_types import PreprocessInput

    worker = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    worker._prep_workers, worker.in_queue, worker._preparing = 2, queue.Queue(), deque()
    submitted = []
    worker._prep_pool = SimpleNamespace(submit=lambda fn, pre_input: submitted.append(pre_input.request_id))
    worker._is_pending = lambda rid: False
    worker.timing_requested, worker.wakeup = threading.Event(), None
    worker.communicator = SimpleNamespace(flush=lambda: False)
    taken = []
    worker._take_input = lambda pre_input, profile=None, prepared=None: taken.append((pre_input.request_id, prepared))
    worker.in_queue.put(PreprocessInput("gone", "gone", None, [], ["text"], {}))

    PreprocessWorkerThread._take_new_input(worker, sends_pending=False)
    assert submitted == [], "prepared a request nothing will read"
    assert taken == [("gone", None)]
