"""A stream buffer keeps copies of the tensors a read brought in. On the shared
memory transport the copies are made on the stream the read's host-to-device
copies ran on, and handed to the reading stream once: a read on another stream
could see a received tensor's memory as it was before the copy (a Qwen3-TTS
codec frame then indexed the codebook with garbage). Drives the real
``SharedMemoryCommunicationManager.clone_received`` with the stream calls
recorded, and the real ``Worker._check_ready_tensors`` routing."""
from __future__ import annotations

import contextlib
from types import SimpleNamespace

import torch

from mstar.communication.tensors import SharedMemoryCommunicationManager, TensorCommunicationManager
from mstar.graph.base import GraphEdge
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.worker.worker import Worker


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))


class _DeviceTensor:
    """A stand-in for a received device tensor (or a block of them): records where
    its copy ran and which stream it was marked for."""

    is_cuda = True
    device = torch.device("cuda", 0)
    dtype = torch.long

    def __init__(self, value, log, current, shape=(1, 16)):
        self.value, self.log, self.current, self.shape = value, log, current, shape

    def clone(self):
        self.log.append(("clone", self.value, self.current[0]))
        return _DeviceTensor(self.value, self.log, self.current, self.shape)

    def __getitem__(self, row):
        return _DeviceTensor(self.value[row], self.log, self.current)

    def record_stream(self, stream):
        self.log.append(("record", self.value, stream.name))


def _manager(monkeypatch, log, current):
    manager = SharedMemoryCommunicationManager.__new__(SharedMemoryCommunicationManager)
    manager.device = "cuda:0"
    manager._h2d_stream = _Stream("copy", log)
    consumer = _Stream("consumer", log)

    @contextlib.contextmanager
    def stream(s):
        before, current[0] = current[0], s.name
        try:
            yield
        finally:
            current[0] = before

    def stack(tensors):
        values = tuple(t.value for t in tensors)
        log.append(("stack", values, current[0]))
        return _DeviceTensor(values, log, current)

    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: consumer)
    monkeypatch.setattr(torch, "stack", stack)
    return manager


def test_received_frames_are_copied_on_the_copy_stream_and_handed_over_once(monkeypatch):
    log, current = [], ["consumer"]
    manager = _manager(monkeypatch, log, current)
    frames = [_DeviceTensor(v, log, current) for v in (1, 2, 3)]
    other = _DeviceTensor(9, log, current, shape=(1, 4))

    copies = manager.clone_received([frames[0], other, frames[1], frames[2]])

    assert [c.value for c in copies] == [1, 9, 2, 3]
    # one stack of the frames and the odd one cloned, both behind the reads' own
    # copies; then one hand-over; then each block marked as read by the consumer,
    # so the copy stream's pool keeps it while the consumer reads it
    assert log == [
        ("stack", (1, 2, 3), "copy"), ("clone", 9, "copy"),
        ("wait", "consumer", "copy"),
        ("record", (1, 2, 3), "consumer"), ("record", 9, "consumer"),
    ]
    assert current == ["consumer"]


def test_host_tensors_and_transports_without_a_copy_stream_clone_in_place():
    cpu = [torch.tensor([1, 2]), torch.tensor([3])]
    manager = SharedMemoryCommunicationManager.__new__(SharedMemoryCommunicationManager)
    manager._h2d_stream = None
    out = manager.clone_received(cpu)
    assert [t.tolist() for t in out] == [[1, 2], [3]]
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(out, cpu, strict=True))
    base = TensorCommunicationManager.clone_received(manager, cpu)
    assert [t.tolist() for t in base] == [[1, 2], [3]]


def test_a_ready_pass_routes_every_streaming_frame_through_one_clone_received():
    calls, derefs, ingested = [], [], []
    buffers = {rid: StreamBuffer(request_id=rid, edge_name="codec_tokens", from_partition="Talker",
                                 policy=SimpleNamespace()) for rid in (1, 2)}
    for rid, buf in buffers.items():
        buf.pre_read_register(10 * rid)
        buf.pre_read_register(10 * rid + 1)
    tensors = {u: torch.tensor([u]) for u in (10, 11, 20, 21)}

    def clone_received(ts):
        calls.append([int(t) for t in ts])
        return [t + 1000 for t in ts]

    w = Worker.__new__(Worker)
    w.enable_nvtx = False
    w._lockstep_led = set()
    w._hop_keys = None
    w.wakeup_event = SimpleNamespace(drain=lambda: None)
    w.scheduler = SimpleNamespace(read_failed_rids=set(), failed_rids=set())
    w.request_state = SimpleNamespace(per_request_info={
        rid: SimpleNamespace(stream_buffers={"codec_tokens": buf}) for rid, buf in buffers.items()
    })
    def edge(rid):
        return GraphEdge(next_node="Codec", name="codec_tokens", is_streaming=True,
                         tensor_info=[SimpleNamespace(uuid=10 * rid), SimpleNamespace(uuid=10 * rid + 1)])

    text = GraphEdge(next_node="Codec", name="text", tensor_info=[])
    w.tensor_manager = SimpleNamespace(
        get_ready_tensors=lambda: {1: [edge(1)], 2: [edge(2), text]},
        take_failed_reads=lambda: [],
        get_tensor=lambda uuid: tensors[uuid],
        clone_received=clone_received,
        dereference_batch_uniform=lambda uuids: derefs.append(list(uuids)),
    )
    w._graph_runtime = SimpleNamespace(ingest_inputs_batch=lambda edges, can_buffer: ingested.append(edges))
    w._one_rids_edges = lambda rid, edges: (rid, [e.name for e in edges])

    Worker._check_ready_tensors(w)

    assert calls == [[10, 11, 20, 21]]
    assert derefs == [[10, 11, 20, 21]]
    for rid, buf in buffers.items():
        held = {k: int(v) for k, v in buf._id_to_tensor.items()}
        assert held == {10 * rid: 10 * rid + 1000, 10 * rid + 1: 10 * rid + 1001}
    assert ingested == [(2, ["text"])]
