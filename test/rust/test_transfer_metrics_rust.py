"""``test/modular/test_transfer_metrics.py`` on the Rust pieces: the edge
metrics must come out the same when the descriptors were stored by the Rust
bookkeeper and the frame was built and sent by the Rust graph runtime, when
the bytes move through the Rust SHM arena, and when the worker's status
travels over the Rust communicator.

The consumer always counts in the Python transport manager; what Rust
decides is what reaches it (the source node on each descriptor), so each case
reads a frame Rust actually sent off a real inbox.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
# the shared helpers live beside the Python-side test; "test" itself can
# resolve to the standard library's package, so import it by directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modular"))

import pytest

pytest.importorskip("mstar_rust")

import torch
from test_rust_wire_frames import PEER, WALK, _Mesh  # noqa: E402
from test_transfer_metrics import _read, _series, _Sink  # noqa: E402

import mstar.communication.wire_types  # noqa: E402,F401  (registers the tags)
from mstar.communication import wire  # noqa: E402
from mstar.communication.arena import ArenaShmCommunicationManager  # noqa: E402
from mstar.communication.rust_communicator import RustZMQCommunicator  # noqa: E402
from mstar.communication.tensor_store import (  # noqa: E402
    PythonTensorBookkeeping,
    RustTensorBookkeeping,
    TensorStore,
)
from mstar.communication.tensors import SharedMemoryCommunicationManager  # noqa: E402
from mstar.conductor.request_info import CurrentForwardPassInfo  # noqa: E402
from mstar.graph.base import GraphEdge  # noqa: E402
from mstar.observability.recorders import EdgeTransfers  # noqa: E402
from mstar.utils.containers import ParallelList  # noqa: E402
from mstar.utils.ipc_format import (  # noqa: E402
    ConductorMessage,
    ConductorMessageType,
    InputSignals,
    WorkerMessage,
    WorkerMessageType,
    WorkerStatus,
)

ME = "worker_0"
REMOTE = [{"wg_id": 1, "graph_walks": [WALK], "nodes": ["remote"], "dyn_loops": []}]


def _manager(cls, entity, shm_dir, bookkeeping):
    if cls is ArenaShmCommunicationManager:
        os.environ.setdefault("MSTAR_SHM_ARENA_SEGMENT_MB", "1")
        os.environ.setdefault("MSTAR_SHM_ARENA_MAX_SEGMENTS", "4")
    mgr = cls(my_entity_id=entity, hostname="localhost", device="cpu",
              communicator=_Sink(), shm_dir=str(shm_dir))
    mgr.tensor_store = TensorStore(bookkeeping=bookkeeping)
    return mgr


def _store_and_stage(producer, rid, tensors):
    """The worker's output path: one batch for node ``only``, stored through
    the bookkeeper's columnar put, then staged for remote readers."""
    stored = producer.store_and_return_tensor_info_batch(
        [rid], {rid: {"out": tensors}}, ["out"], node_name="only", graph_walk=WALK,
    )
    # the safety hold the worker puts on a stored batch until routing settles it
    producer.increment_ref_batch_uniform(stored.flat_uuids, 1)
    producer.register_for_send_uuids(ParallelList([rid], [stored.flat_uuids]))
    return stored.flat_uuids


def _rust_frame(tmp_path, cls, tensors):
    """The descriptors as the Rust runtime sends them: stored by the Rust
    bookkeeper, routed and framed in Rust, read off the peer's inbox."""
    (tmp_path / "zmq").mkdir()
    mesh = _Mesh(tmp_path / "zmq", outs=[("out", "remote", False)], remote=REMOTE)
    producer = _manager(cls, ME, tmp_path, mesh.book)
    rid = mesh.admit(wgs=(0, 1), workers=(ME, PEER))
    uuids = _store_and_stage(producer, rid, tensors)
    (frame,) = mesh.run(rid, uuids=uuids)[PEER]
    assert frame.message_type == WorkerMessageType.INPUT_SIGNALS
    return producer, frame.body.inputs


def _python_frame(tmp_path, cls, tensors):
    """The same descriptors as the Python runtime sends them: the store's own
    objects on the edge, through the wire codec."""
    producer = _manager(cls, ME, tmp_path, PythonTensorBookkeeping())
    uuids = _store_and_stage(producer, "r1", tensors)
    edge = GraphEdge(next_node="remote", name="out",
                     tensor_info=[producer.tensor_store.get_info(u) for u in uuids])
    message = WorkerMessage(WorkerMessageType.INPUT_SIGNALS, InputSignals(
        request_id="r1", inputs=[edge], partition_name="default",
        request_info=CurrentForwardPassInfo(
            request_id="r1", graph_walk=WALK, fwd_index=0, random_seed=0, max_tokens=8,
            partition_name="default"),
    ))
    return producer, wire.decode(wire.encode(message)).body.inputs


def _consume(cls, tmp_path, edges):
    consumer = _manager(cls, PEER, tmp_path, PythonTensorBookkeeping())
    stats = consumer.enable_transfer_metrics()
    _read(consumer, "r1", edges)
    return consumer, stats


@pytest.mark.parametrize("cls", [SharedMemoryCommunicationManager, ArenaShmCommunicationManager],
                         ids=["files", "arena"])
def test_a_rust_sent_edge_counts_like_a_python_sent_one(tmp_path, cls):
    tensors = [torch.randn(4, 8), torch.arange(3)]
    nbytes = sum(t.nbytes for t in tensors)
    views = {}
    for path, frame in (("rust", _rust_frame), ("python", _python_frame)):
        (tmp_path / path).mkdir()
        producer, edges = frame(tmp_path / path, cls, [t.clone() for t in tensors])
        assert all(info._source_node_name == "only" for info in edges[0].tensor_info), path
        consumer, stats = _consume(cls, tmp_path / path, edges)
        got = consumer.get_tensor(edges[0].tensor_info[0].uuid)
        assert torch.equal(got, tensors[0]), path
        ((labels, (b, n, reads, seconds)),) = _series(stats.snapshot()).items()
        assert seconds > 0, path
        views[path] = (labels, b, n, reads)

    assert views["rust"] == views["python"] == (("only", "remote", "SHM"), nbytes, 2, 1)


def test_an_arena_spill_is_still_one_shm_read(tmp_path, monkeypatch):
    """A full arena spills a tensor to its own file; the read is the same
    transfer on the same series."""
    monkeypatch.setenv("MSTAR_SHM_ARENA_SEGMENT_MB", "1")
    monkeypatch.setenv("MSTAR_SHM_ARENA_MAX_SEGMENTS", "1")
    producer = _manager(ArenaShmCommunicationManager, ME, tmp_path, RustTensorBookkeeping())
    big = [torch.zeros(200_000, dtype=torch.float32) for _ in range(2)]  # 1.6 MB > one 1 MiB segment
    uuids = _store_and_stage(producer, 1, big)
    infos = [producer.tensor_store.get_info(u) for u in uuids]
    assert any(i.shm_segment is None for i in infos), "nothing spilled; the case is not exercised"

    _, stats = _consume(ArenaShmCommunicationManager, tmp_path, [
        GraphEdge(next_node="remote", name="out", tensor_info=infos)])
    ((labels, (b, n, reads, _)),) = _series(stats.snapshot()).items()
    assert (labels, b, n, reads) == (("only", "remote", "SHM"), 2 * big[0].nbytes, 2, 1)


def test_the_worker_status_crosses_the_rust_communicator(tmp_path):
    prefix = str(tmp_path) + "/"
    conductor = RustZMQCommunicator("conductor", [], ipc_socket_path_prefix=prefix)
    worker = RustZMQCommunicator(ME, ["conductor"], ipc_socket_path_prefix=prefix)
    stats = EdgeTransfers("SHM")
    stats.record("only", "remote", 64, 1, 0.001)
    sent = ConductorMessage(ConductorMessageType.WORKER_STATUS, WorkerStatus(ME, stats.snapshot()))
    worker.send("conductor", sent)

    (got,) = conductor.get_all_new_messages(blocking=True, timeout_s=5.0)
    assert got == sent
    assert _series(got.body.metrics) == {("only", "remote", "SHM"): (64, 1, 1, 0.001)}

