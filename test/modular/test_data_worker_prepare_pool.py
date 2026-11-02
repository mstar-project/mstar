"""A model that declares PREPROCESS_PARALLEL prepares requests on the data worker's pool:
they reach the conductor in arrival order, output delivery runs meanwhile, and a failed or
abandoned request ends as it does inline. A model that doesn't declare it runs as before."""
from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from mstar.api_server.data_worker import InlineTokens, PreprocessWorkerThread, ResultQueue
from mstar.api_server.request_types import PreprocessInput
from mstar.communication.event import EventWakeup
from mstar.model.base import ProcessPromptOutput
from mstar.streaming.topology import PartitionTopology
from mstar.utils import cpus
from mstar.utils.ipc_format import ConductorMessageType


class _Model:
    """process_prompt blocks on the gate named by the prompt, so a test decides
    when each preparation finishes."""
    PREPROCESS_PARALLEL = True

    def __init__(self):
        self.gates: dict[str, threading.Event] = {}
        self.threads: dict[str, str] = {}
        self.overlap = 0
        self._running = 0
        self._lock = threading.Lock()

    def get_partition_topology(self):
        return PartitionTopology(partitions=["default"])

    def kv_labels(self):
        return {}

    def gate(self, text: str) -> threading.Event:
        return self.gates.setdefault(text, threading.Event())

    def process_prompt(self, text, input_modalities, output_modalities, **kwargs):
        self.threads[text] = threading.current_thread().name
        with self._lock:
            self._running += 1
            self.overlap = max(self.overlap, self._running)
        try:
            if not self.gate(text).wait(timeout=10):
                raise TimeoutError(text)
        finally:
            with self._lock:
                self._running -= 1
        if text.startswith("bad"):
            raise ValueError("bad prompt")
        return ProcessPromptOutput({"text_inputs": [torch.tensor([1])]}, {"seen": text})


class _Serial(_Model):
    PREPROCESS_PARALLEL = False


class _Tensors:
    def __init__(self):
        self.stored: list[tuple[str, str]] = []
        self.dropped: list[str] = []

    def store_and_return_tensor_info(self, request_id, tensors):
        self.stored.append((request_id, threading.current_thread().name))
        return {}

    def register_for_send(self, request_id, tensor_infos):
        pass

    def cleanup_request(self, request_id):
        pass

    def force_cleanup_request(self, request_id):
        self.dropped.append(request_id)

    def has_inflight_reads(self, request_id):
        return False

    def get_ready_tensors(self):
        return {}

    def take_failed_reads(self):
        return []


class _Comm:
    # the outbox the run loop wraps this in encodes with it: identity, so sends arrive as objects
    codec = SimpleNamespace(encode=lambda msg: msg)

    def __init__(self):
        self.sent: list = []

    def get_all_new_messages(self):
        return []

    def send(self, entity, msg):
        self.sent.append(msg)

    def try_send_bytes(self, entity, msg):
        self.sent.append(msg)
        return True


def _thread(model, config=None, pending=None) -> PreprocessWorkerThread:
    return PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=ResultQueue(EventWakeup()),
        profile_queue=ResultQueue(EventWakeup()), cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=threading.Event(), communicator=_Comm(),
        tensor_manager=_Tensors(), model=model, model_config=config, is_pending=pending,
    )


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(cpus, "_plan", cpus.CpuPlan(frozenset(range(8)), usable=8))
    wt = _thread(_Model())
    yield wt
    for gate in wt.model.gates.values():
        gate.set()
    wt._prepare_pool.shutdown(wait=True)
    wt._encode_pool.shutdown(wait=True)


def _input(request_id: str, text: str | None = None) -> PreprocessInput:
    return PreprocessInput(
        request_id=request_id, text=text or request_id, file_paths=None,
        input_modalities=["text"], output_modalities=["text"], model_kwargs={},
    )


def _sent(wt) -> list[str]:
    comm = getattr(wt.communicator, "_inner", wt.communicator)  # run() wraps it in an outbox
    return [
        m.body.request_id for m in comm.sent
        if m.message_type == ConductorMessageType.NEW_REQUEST
    ]


def _release_until(wt, n: int, timeout_s: float = 5.0) -> list[str]:
    deadline = threading.Event()
    timer = threading.Timer(timeout_s, deadline.set)
    timer.start()
    try:
        while len(_sent(wt)) < n and not deadline.is_set():
            wt._release_prepared()
    finally:
        timer.cancel()
    return _sent(wt)


@pytest.mark.parametrize(("usable", "workers"), [(1, 1), (3, 3), (8, 4), (16, 4), (40, 8), (256, 8)])
def test_the_pool_times_its_intra_op_threads_fits_the_cpus(usable, workers):
    plan = cpus.CpuPlan(frozenset(range(256)), usable)
    assert plan.preprocess_workers == workers
    assert workers * plan.preprocess_threads <= usable


@pytest.mark.parametrize(("usable", "threads", "workers"), [(8, 4, 2), (8, 8, 1), (16, 1, 8), (40, 4, 8), (3, 0, 3)])
def test_a_higher_thread_count_shrinks_the_pool(usable, threads, workers):
    plan = cpus.CpuPlan(frozenset(range(256)), usable)
    assert plan.prepare_workers(threads) == workers
    assert workers * max(1, threads) <= usable


def test_only_a_model_that_declares_it_gets_a_pool(monkeypatch):
    monkeypatch.setattr(cpus, "_plan", cpus.CpuPlan(frozenset(range(8)), usable=8))
    monkeypatch.setattr(torch, "get_num_threads", lambda: 2)
    serial, parallel = _thread(_Serial()), _thread(_Model())
    assert serial._prepare_pool is None
    assert parallel._prepare_workers == 4 and parallel._prepare_pool is not None
    parallel._prepare_pool.shutdown()


def test_the_pool_follows_the_threads_this_process_runs(monkeypatch):
    # ``preprocess_torch_threads`` may set the intra-op pool above the plan's default
    monkeypatch.setattr(cpus, "_plan", cpus.CpuPlan(frozenset(range(8)), usable=8))
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    wt = _thread(_Model())
    assert wt._prepare_workers == 2
    wt._prepare_pool.shutdown()
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    assert _thread(_Model())._prepare_pool is None


def test_one_cpu_prepares_inline(monkeypatch):
    monkeypatch.setattr(cpus, "_plan", cpus.CpuPlan(frozenset({0}), usable=1))
    assert _thread(_Model())._prepare_pool is None


def test_a_deployment_sets_the_pool_or_turns_it_off(monkeypatch):
    monkeypatch.setattr(cpus, "_plan", cpus.CpuPlan(frozenset(range(8)), usable=8))
    assert _thread(_Model(), {"preprocess_workers": 1})._prepare_pool is None
    wt = _thread(_Model(), {"preprocess_workers": 2})
    assert wt._prepare_workers == 2
    wt._prepare_pool.shutdown()


def test_a_model_with_client_streams_prepares_inline(monkeypatch):
    class Streams(_Model):
        def get_partition_topology(self):
            topology = super().get_partition_topology()
            topology.client_connections = lambda: [
                type("Conn", (), {"edge_name": "mic", "to_partition": "default"})(),
            ]
            return topology

    monkeypatch.setattr(cpus, "_plan", cpus.CpuPlan(frozenset(range(8)), usable=8))
    assert _thread(Streams())._prepare_pool is None


def test_a_model_without_the_declaration_runs_on_the_thread(monkeypatch):
    monkeypatch.setattr(cpus, "_plan", cpus.CpuPlan(frozenset(range(8)), usable=8))
    wt = _thread(_Serial())
    wt.model.gate("a").set()
    wt._process_waiting_input(_input("a"))
    assert wt.model.threads["a"] == threading.current_thread().name
    assert _sent(wt) == ["a"]


def test_requests_overlap_and_reach_the_conductor_in_arrival_order(worker):
    worker._process_waiting_input(_input("a"))
    worker._process_waiting_input(_input("b"))
    worker._process_waiting_input(_input("c"))
    # c and b finish first
    worker.model.gate("c").set()
    worker.model.gate("b").set()
    worker._preparing[2].future.result(timeout=5)
    worker._preparing[1].future.result(timeout=5)
    assert worker._release_prepared() is False and _sent(worker) == []
    worker.model.gate("a").set()
    assert _release_until(worker, 3) == ["a", "b", "c"]
    assert all(name.startswith("mstar-prepare") for name in worker.model.threads.values())


def test_the_request_is_stored_and_sent_on_the_data_workers_thread(worker):
    worker.model.gate("a").set()
    worker._process_waiting_input(_input("a"))
    _release_until(worker, 1)
    assert worker.tensor_manager.stored == [("a", threading.current_thread().name)]
    assert worker.request_model_kwargs["a"]["seen"] == "a"


def test_a_full_pool_leaves_requests_in_the_queue_undecoded(worker):
    for i in range(6):
        worker.in_queue.put(_input(str(i)))
    while (batch := worker._next_inputs()):
        worker._process_waiting_input(batch[0])
    assert len(worker._preparing) == worker._prepare_workers == 4
    assert worker.in_queue.qsize() == 2
    for i in range(6):
        worker.model.gate(str(i)).set()
    _release_until(worker, 1)
    assert len(worker._next_inputs()) == 1


def test_a_failed_preparation_answers_in_its_turn_and_leaves_nothing(worker):
    worker.model.gate("bad").set()
    worker.model.gate("b").set()
    worker._process_waiting_input(_input("a", "bad"))
    worker._process_waiting_input(_input("b"))
    assert _release_until(worker, 1) == ["b"]
    chunk = worker.out_queue.get_nowait()
    assert (chunk.request_id, chunk.modality, chunk.metadata["status"]) == ("a", "error", 400)
    assert "a" not in worker._live_requests and "a" not in worker.in_flight_requests
    assert worker.tensor_manager.dropped == ["a"]


def test_a_request_abandoned_while_it_prepares_is_never_sent(monkeypatch):
    monkeypatch.setattr(cpus, "_plan", cpus.CpuPlan(frozenset(range(8)), usable=8))
    let_go: set[str] = set()
    wt = _thread(_Model(), pending=lambda key: key not in let_go)
    wt._process_waiting_input(_input("a"))
    # the API server lets go first, then queues the abort
    let_go.add("a")
    wt.abort_request_queue.put("a")
    wt._process_releases()
    wt.model.gate("a").set()
    wt._preparing[0].future.result(timeout=5)
    assert wt._release_prepared() is True
    assert _sent(wt) == []
    assert "a" not in wt._live_requests and "a" not in wt._draining_rids and "a" not in wt._reads_done_sent
    assert wt.tensor_manager.stored == [] and wt.tensor_manager.dropped == ["a"]
    wt._prepare_pool.shutdown(wait=True)


def test_a_release_queued_before_the_request_applies_to_the_earlier_one(worker):
    worker._live_requests["a"] = 1  # an earlier request under the rid
    worker.reads_done_queue.put(("a", True))
    worker._process_releases()
    worker._process_waiting_input(_input("a"))
    assert worker._live_requests["a"] == 2
    assert "a" in worker._reads_done_sent


def test_the_run_loop_delivers_outputs_while_a_request_prepares(monkeypatch):
    monkeypatch.setattr(cpus, "_plan", cpus.CpuPlan(frozenset(range(8)), usable=8))
    wt = _thread(_Model())
    delivered = threading.Event()
    wt._emit_inline_tokens = lambda result: delivered.set()
    wt.in_queue.put(_input("slow"))
    thread = threading.Thread(target=wt.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while "slow" not in wt.model.threads and time.monotonic() < deadline:
            time.sleep(0.001)
        assert "slow" in wt.model.threads
        wt.result_tensor_queue.put(InlineTokens(["other"], [1], "text"))
        assert delivered.wait(timeout=5), "an output waited behind a request's preparation"
        assert _sent(wt) == []
        wt.model.gate("slow").set()
        deadline = time.monotonic() + 5
        while not _sent(wt) and time.monotonic() < deadline:
            time.sleep(0.001)
        assert _sent(wt) == ["slow"]
    finally:
        wt.model.gate("slow").set()
        wt.stop_event.set()
        thread.join(timeout=5)
    assert not thread.is_alive()


def test_a_stopping_thread_refuses_a_request_it_can_no_longer_prepare(worker):
    worker._prepare_pool.shutdown(wait=True)
    worker._process_waiting_input(_input("a"))
    chunk = worker.out_queue.get_nowait()
    assert (chunk.request_id, chunk.modality) == ("a", "error")
    assert "a" not in worker._live_requests


def test_a_thread_built_without_init_prepares_inline():
    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    assert wt._prepare_pool is None and wt._release_prepared() is False
