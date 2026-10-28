"""Image outputs encode on the data worker's pool: each is released to its request
when its own encode finishes, in the request's output order, while every other
modality is postprocessed inline as before."""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace

import pytest
import torch

from mstar.api_server.data_worker import PreprocessWorkerThread, RequestOutputState, encode_workers
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.streaming.topology import PartitionTopology


class _Model:
    """postprocess blocks on the gate named by the output's value, so a test
    decides when each encode finishes."""

    def __init__(self):
        self.gates: dict[int, threading.Event] = {}
        self.threads: dict[int, str] = {}

    def get_partition_topology(self):
        return PartitionTopology(partitions=["default"])

    def stop_token_ids(self):
        return None  # the base Model's: every text reply reads as stopped

    def gate(self, value: int) -> threading.Event:
        return self.gates.setdefault(value, threading.Event())

    def postprocess(self, output, modality, request_kwargs=None):
        value = int(output.flatten()[0])
        self.threads[value] = threading.current_thread().name
        if not self.gate(value).wait(timeout=10):
            raise TimeoutError(value)
        if value < 0:
            raise ValueError("bad pixels")
        return f"{modality}:{value}".encode()


class _TensorManager:
    def __init__(self):
        self.ready: dict = {}
        self.tensors: dict[str, torch.Tensor] = {}
        self.dereferenced: list[str] = []

    def get_ready_tensors(self):
        ready, self.ready = self.ready, {}
        return ready

    def take_failed_reads(self):
        return []

    def get_tensor(self, uuid):
        return self.tensors[uuid]

    def dereference(self, uuid, n=1):
        self.dereferenced.append(uuid)

    def has_inflight_reads(self, request_id):
        return False

    def cleanup_request(self, request_id):
        pass

    def force_cleanup_request(self, request_id):
        pass


class _Comm:
    def get_all_new_messages(self):
        return []

    def send(self, entity, msg):
        pass


@pytest.fixture
def worker(monkeypatch):
    # The tests hold one encode on a gate while another finishes, so two
    # threads whatever the host's CPUs.
    monkeypatch.setattr("mstar.api_server.data_worker.encode_workers", lambda: 2)
    wt = PreprocessWorkerThread(
        in_queue=queue.Queue(), result_tensor_queue=queue.Queue(), out_queue=queue.Queue(),
        profile_queue=queue.Queue(), cleanup_request_queue=queue.Queue(),
        abort_request_queue=queue.Queue(), reads_done_queue=queue.Queue(),
        discard_tensor_queue=queue.Queue(), stop_event=threading.Event(), communicator=_Comm(),
        tensor_manager=_TensorManager(), model=_Model(),
    )
    yield wt
    for gate in wt.model.gates.values():
        gate.set()
    wt._encode_pool.shutdown(wait=True)


def _output(wt, request_id: str, modality: str, values: list[int], looped: bool = False):
    """``values`` arrive as ``request_id``'s next outputs, one tensor each;
    ``looped`` ones as a loop emits them, one per iteration."""
    state = wt.request_output_state.setdefault(request_id, RequestOutputState())
    infos = []
    for i, value in enumerate(values):
        uuid = f"{request_id}/{state.next_sequence}"
        wt.tensor_manager.tensors[uuid] = torch.tensor([value])
        wt.tensor_uuid_to_metadata_per_request.setdefault(request_id, {})[uuid] = {"v": value}
        indices = NestedLoopIndices(["rollout"], {"rollout": i}, 0) if looped else None
        state.order[uuid] = (state.next_sequence, indices)
        state.next_sequence += 1
        infos.append(SimpleNamespace(uuid=uuid))
    edge = SimpleNamespace(name=f"{modality}_output", tensor_info=infos)
    wt.tensor_manager.ready.setdefault(request_id, []).append(edge)


def _drain(wt) -> list:
    chunks = []
    while not wt.out_queue.empty():
        chunks.append(wt.out_queue.get_nowait())
    return chunks


def _pass_until(wt, n: int, timeout_s: float = 5.0) -> list:
    """Run the thread's read and release steps until ``n`` chunks are out."""
    chunks, deadline = [], threading.Event()
    timer = threading.Timer(timeout_s, deadline.set)
    timer.start()
    try:
        while len(chunks) < n and not deadline.is_set():
            wt._process_read_tensors()
            wt._release_encoded()
            chunks += _drain(wt)
    finally:
        timer.cancel()
    return chunks


@pytest.mark.parametrize(("cpus", "workers"), [(1, 1), (8, 1), (16, 2), (40, 4), (256, 4)])
def test_the_pool_times_a_pngs_own_threads_fits_the_cpus(cpus, workers):
    assert encode_workers(cpus, png_workers=8) == workers
    assert workers * 8 <= max(cpus, 8)


def test_a_whole_video_is_pooled_and_a_loops_frames_stay_inline(worker):
    _output(worker, "clip", "video", [1])
    _output(worker, "rollout", "video", [2, 3], looped=True)
    worker.model.gate(2).set()
    worker.model.gate(3).set()
    worker._process_read_tensors()
    assert worker.model.threads[2] == threading.current_thread().name
    assert worker.model.threads[3] == threading.current_thread().name
    assert [c.data for c in _drain(worker)] == [b"video:2", b"video:3"]
    worker.model.gate(1).set()
    (chunk,) = _pass_until(worker, 1)
    assert chunk.data == b"video:1" and worker.model.threads[1].startswith("mstar-encode")


def test_an_image_is_encoded_off_the_thread_and_released_when_done(worker):
    _output(worker, "a", "image", [1])
    worker._process_read_tensors()
    assert _drain(worker) == [] and worker.tensor_manager.dereferenced == []
    worker.model.gate(1).set()
    (chunk,) = _pass_until(worker, 1)
    assert (chunk.request_id, chunk.modality, chunk.data, chunk.metadata) == ("a", "image", b"image:1", {"v": 1})
    assert worker.model.threads[1].startswith("mstar-encode")
    # the transport keeps its reference until the encode is released
    assert worker.tensor_manager.dereferenced == ["a/0"]
    assert worker.request_output_state["a"].pending == {}


def test_a_finished_image_does_not_wait_for_another_requests_encode(worker):
    _output(worker, "slow", "image", [1])
    _output(worker, "fast", "image", [2])
    worker.model.gate(2).set()
    (chunk,) = _pass_until(worker, 1)
    assert chunk.request_id == "fast"
    worker.model.gate(1).set()
    (chunk,) = _pass_until(worker, 1)
    assert chunk.request_id == "slow"


def test_one_requests_images_keep_their_order(worker):
    _output(worker, "a", "image", [1, 2, 3])
    worker._process_read_tensors()
    worker.model.gate(3).set()
    worker.model.gate(2).set()
    assert _pass_until(worker, 1, timeout_s=0.3) == [], "a later image overtook the first"
    worker.model.gate(1).set()
    assert [c.data for c in _pass_until(worker, 3)] == [b"image:1", b"image:2", b"image:3"]


def test_a_streamed_chunk_after_an_image_waits_for_it_and_stays_inline(worker):
    _output(worker, "a", "image", [1])
    _output(worker, "a", "text", [2])
    worker.model.gate(2).set()
    worker._process_read_tensors()
    assert worker.model.threads[2] == threading.current_thread().name
    assert _drain(worker) == []
    worker.model.gate(1).set()
    assert [c.modality for c in _pass_until(worker, 2)] == ["image", "text"]


def test_a_failed_encode_answers_in_its_slot(worker):
    _output(worker, "a", "image", [-1])
    _output(worker, "a", "text", [2])
    worker.model.gate(2).set()
    worker.model.gate(-1).set()
    chunks = _pass_until(worker, 2)
    assert [c.modality for c in chunks] == ["error", "text"]
    assert chunks[0].metadata == {"status": 400}


def test_an_encode_finishing_after_teardown_delivers_nothing(worker):
    _output(worker, "a", "image", [1])
    worker._process_read_tensors()
    worker._drop_request_state("a")
    # the rid comes back as a new request before the old encode finishes
    worker.request_output_state["a"] = RequestOutputState()
    worker.model.gate(1).set()
    worker._encode_pool.shutdown(wait=True)
    worker._release_encoded()
    assert _drain(worker) == []
    assert worker.request_output_state["a"].next_emit == 0
    assert worker.tensor_manager.dereferenced == ["a/0"]


def test_a_thread_built_without_init_encodes_inline():
    wt = PreprocessWorkerThread.__new__(PreprocessWorkerThread)
    assert wt._encode_pool is None and wt._release_encoded() is False
