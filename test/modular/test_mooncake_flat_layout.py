"""On Mooncake (RDMA / TCP) the consumer has the transfer engine copy a stored
tensor's ``nbytes`` as one flat range from its ``address`` into a buffer it
views with the descriptor's strides. So an output with gaps (``hidden[:, -1, :]``,
a row of a transposed row output) must be stored as a contiguous copy, while a
contiguous tensor or a dense permutation is stored as is.

The transfer is emulated on CPU: the consumer is the real
``MooncakeCommunicationManager.start_read_tensors``, with the engine's batch
read replaced by the flat byte copy it performs.
"""
import ctypes

import pytest
import torch

from mstar.communication import tensors
from mstar.communication.communicator import CommProtocol
from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore
from mstar.communication.tensors import (
    LocalTransferEngine,
    MooncakeCommunicationManager,
    SharedMemoryCommunicationManager,
)
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.engine.engine import Engine
from mstar.graph.base import GraphEdge, NodeAndGraphWalk
from mstar.model.submodule_base import BatchedModelOutput, NodeSubmodule


class _NullCommunicator:
    protocol = CommProtocol.TCP

    def _endpoint(self, entity):
        return f"tcp://127.0.0.1:1/{entity}"

    def send(self, *args, **kwargs):
        pass


def _bookkeeping(kind):
    if kind == "python":
        return PythonTensorBookkeeping()
    try:
        from mstar.communication.tensor_store import RustTensorBookkeeping
        return RustTensorBookkeeping()
    except ImportError:
        pytest.skip("the columnar tensor store needs mstar_rust")


class _NoEngine(LocalTransferEngine):
    """Registration is a no-op; there is no Mooncake engine to hand the reader."""

    def __init__(self, hostname, **kwargs):
        super().__init__(hostname)
        self._engine = None


class _FlatCopyReader:
    """The transfer engine's batch read: ``nbytes`` from ``remote_ptr``."""

    def __init__(self, engine, device, enable_prof=False, device_filter=""):
        self.device_filter = device_filter

    def submit(self, reads):
        for r in reads:
            ctypes.memmove(r.local_ptr, r.remote_ptr, r.nbytes)

    def transfer_status(self, batch_id):
        raise AssertionError("no read fails here, so no transfer is handed over")


@pytest.fixture(autouse=True)
def _no_transfer_engine(monkeypatch):
    monkeypatch.setattr(tensors, "MooncakeTransferEngine", _NoEngine)
    monkeypatch.setattr(tensors, "TrackedMooncakeReader", _FlatCopyReader)


def _mooncake(entity, bookkeeping="python"):
    """A TCP ``MooncakeCommunicationManager`` whose reads are the flat copy."""
    mgr = MooncakeCommunicationManager(
        entity, "host", "cpu", _NullCommunicator(), protocol=CommProtocol.TCP,
    )
    mgr.tensor_store = TensorStore(_bookkeeping(bookkeeping))
    return mgr


def _transfer(prod, cons, rid, infos):
    """Register ``infos`` on the producer, read them on the consumer, and
    return what arrived, in order."""
    prod.register_for_send(rid, infos)
    cons.start_read_tensors(rid, [GraphEdge(name="x", next_node="dst", tensor_info=infos)])
    [edge] = cons.get_ready_tensors()[rid]
    return [cons.get_tensor(info.uuid) for info in edge.tensor_info]


def _gapped_views():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    return {
        "last_position": hidden[:, -1, :],
        "narrowed": x[:, 1:4],
        "stepped": x[::2],
    }


@pytest.mark.parametrize("bookkeeping", ["python", "rust"])
@pytest.mark.parametrize("case", list(_gapped_views()))
def test_gapped_view_arrives_intact(case, bookkeeping):
    sent = _gapped_views()[case]
    assert not sent.is_contiguous()
    prod, cons = _mooncake("worker_0", bookkeeping), _mooncake("worker_1", bookkeeping)
    [info] = prod.store_and_return_tensor_info(1, {"x": [sent]})["x"]

    [received] = _transfer(prod, cons, 2, [info])
    assert torch.equal(received, sent)

    stored = prod.get_tensor(info.uuid)
    assert stored.is_contiguous()
    assert info.address == stored.data_ptr()
    assert tuple(info.stride) == stored.stride()
    assert info.nbytes == stored.nbytes


@pytest.mark.parametrize("bookkeeping", ["python", "rust"])
def test_batch_store_gapped_views_arrive_intact(bookkeeping):
    """``store_and_return_tensor_info_batch``: the descriptor form (Python
    bookkeeper) and the columnar form (Rust bookkeeper)."""
    prod, cons = _mooncake("worker_0", bookkeeping), _mooncake("worker_1", bookkeeping)
    views = _gapped_views()
    dense = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    outputs = {
        1: {"h": [views["last_position"]], "y": [dense]},
        2: {"h": [views["narrowed"], views["stepped"]]},
    }
    stored = prod.store_and_return_tensor_info_batch([1, 2], outputs, ["h", "y"])
    assert prod.tensor_store.has_put_tensor_batch_columns == (bookkeeping == "rust")
    sent = [views["last_position"], dense, views["narrowed"], views["stepped"]]
    infos = [prod.tensor_store.get_info(u) for u in stored.flat_uuids]
    assert prod.get_tensor(stored.flat_uuids[1]) is dense

    received = _transfer(prod, cons, 3, infos)
    for got, want in zip(received, sent, strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize("protocol, device_filter", [(CommProtocol.TCP, "eth1"), (CommProtocol.RDMA, "")])
def test_the_reader_gets_the_device_filter_the_engine_was_initialized_with(protocol, device_filter):
    """The engine filters its topology by the TCP device only, so the reader
    counts retries under the same filter."""
    mgr = MooncakeCommunicationManager(
        "e", "host", "cpu", _NullCommunicator(), protocol=protocol, tcp_transfer_device="eth1",
    )
    assert mgr._async_reader.device_filter == device_filter


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_flat_copy_waits_on_the_ready_event():
    """The copy runs on its own stream, so a producer still writing on
    another stream is waited on through the step's completion event."""
    prod = _mooncake("worker_0")
    x = torch.zeros(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    producer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(200_000_000)
        x.fill_(1.0)
        done = torch.cuda.Event()
        done.record(producer)
    stored = prod.store_and_return_tensor_info_batch(
        [1], {1: {"h": [x[:, :512]]}}, ["h"], skip_cuda_sync=True, ready_event=done,
    )
    copy = prod.get_tensor(stored.flat_uuids[0])
    assert copy.is_contiguous() and copy.data_ptr() != x.data_ptr()
    assert bool((copy == 1).all())


class _UnfilteredSubmodule:
    filter_batched_output = NodeSubmodule.filter_batched_output


@pytest.mark.parametrize("bookkeeping", ["python", "rust"])
def test_engine_row_output_arrives_intact(bookkeeping):
    """A row output whose batch clone is transposed: the engine hands out
    rows with gaps, and the store path the worker takes (the row store, or
    the general one when that declines) must still send each row intact."""
    rids = [1, 2, 3, 4]
    raw = BatchedModelOutput(
        row_outputs={"x": torch.arange(12.0).reshape(3, 4).t()},
    )
    per_rid: dict = {}
    _, row_views = Engine._merge_per_rid(
        None, per_rid, raw, rids, rids, _UnfilteredSubmodule(), {},
    )
    assert not row_views["x"][0].is_contiguous()
    outputs = BatchedModelOutput(
        per_rid_outputs=per_rid, row_views=row_views, row_request_ids=tuple(rids),
    )
    prod, cons = _mooncake("worker_0", bookkeeping), _mooncake("worker_1", bookkeeping)

    stored = prod.store_row_outputs_batch(
        rids, outputs, ["x"], outputs.row_views, outputs.row_request_ids,
    )
    if stored is None:
        stored = prod.store_and_return_tensor_info_batch(rids, outputs, ["x"])
    infos = [prod.tensor_store.get_info(u) for u in stored.flat_uuids]

    received = _transfer(prod, cons, 5, infos)
    for got, rid in zip(received, rids, strict=True):
        assert torch.equal(got, per_rid[rid]["x"][0])


def test_contiguous_rows_keep_the_row_store():
    rids = [1, 2, 3]
    clone = torch.arange(6.0).reshape(3, 2)
    views = clone.split(1)
    outputs = BatchedModelOutput(
        per_rid_outputs={rid: {"x": [views[i]]} for i, rid in enumerate(rids)},
        row_views={"x": views}, row_request_ids=tuple(rids),
    )
    prod, cons = _mooncake("worker_0", "rust"), _mooncake("worker_1", "rust")
    stored = prod.store_row_outputs_batch(
        rids, outputs, ["x"], outputs.row_views, outputs.row_request_ids,
    )
    assert stored is not None
    for uuid, view in zip(stored.flat_uuids, views, strict=True):
        assert prod.get_tensor(uuid) is view
    infos = [prod.tensor_store.get_info(u) for u in stored.flat_uuids]
    for got, view in zip(_transfer(prod, cons, 4, infos), views, strict=True):
        assert torch.equal(got, view)


def test_sharded_dense_permutation_slice_arrives_intact():
    """A sharded output's slices are byte ranges of rows, so a transposed one
    must be stored contiguous even though a transpose alone reads fine."""
    cfg = ShardingConfig(
        groups=[ShardingGroup(nodes={"src"}, tp_size=1), ShardingGroup(nodes={"a"}, tp_size=2)],
        tp_enabled_nodes={"a"}, shard_dim={"x": 0},
    )
    cfg.setup({
        NodeAndGraphWalk("src", "w"): ["worker_0"],
        NodeAndGraphWalk("a", "w"): ["worker_1", "worker_2"],
    })
    cfg.groups[0]._tp_rank = 0
    prod, cons = _mooncake("worker_0"), _mooncake("worker_2")
    prod.register_request(1, cfg)
    cons.register_request(2, cfg)
    sent = torch.arange(12.0).reshape(3, 4).t()
    [info] = prod.store_and_return_tensor_info(
        1, {"x": [sent]}, node_name="src", graph_walk="w",
    )["x"]
    edge = cfg.fanout_graph_edges(
        GraphEdge(name="x", next_node="a", tensor_info=[info]),
        source_node="src", source_graph_walk="w", dest_graph_walk="w",
    )["worker_2"]
    prod.register_for_send(1, [info])
    cons.start_read_tensors(2, [edge], graph_walk="w")
    [got_edge] = cons.get_ready_tensors()[2]
    assert torch.equal(cons.get_tensor(got_edge.tensor_info[0].uuid), sent[2:])


def test_contiguous_and_dense_permutation_are_stored_as_is():
    prod, cons = _mooncake("worker_0"), _mooncake("worker_1")
    contiguous = torch.arange(12.0).reshape(3, 4)
    transposed = torch.arange(12.0).reshape(3, 4).t()
    infos = prod.store_and_return_tensor_info(1, {"x": [contiguous, transposed]})["x"]
    for info, sent in zip(infos, (contiguous, transposed), strict=True):
        assert prod.get_tensor(info.uuid) is sent
        assert tuple(info.stride) == sent.stride()
    for got, sent in zip(_transfer(prod, cons, 2, infos), (contiguous, transposed), strict=True):
        assert torch.equal(got, sent)


def test_overlapping_view_is_copied():
    prod = _mooncake("worker_0")
    expanded = torch.arange(3.0).expand(4, 3)
    [info] = prod.store_and_return_tensor_info(1, {"x": [expanded]})["x"]
    stored = prod.get_tensor(info.uuid)
    assert stored.is_contiguous() and torch.equal(stored, expanded)
    assert info.nbytes == stored.untyped_storage().nbytes()


def test_shm_transport_stores_a_gapped_view_as_is(tmp_path):
    """The SHM transports serialize a contiguous copy at send time, so the
    store does not copy for them."""
    mgr = SharedMemoryCommunicationManager(
        my_entity_id="worker_0", hostname="host", device="cpu",
        communicator=_NullCommunicator(), shm_dir=str(tmp_path),
    )
    sent = _gapped_views()["last_position"]
    [info] = mgr.store_and_return_tensor_info(1, {"x": [sent]})["x"]
    assert mgr.get_tensor(info.uuid) is sent
