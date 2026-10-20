"""Edge fanout and the same-worker re-slice on tensors of awkward shape.

The fanout used to read ``dims[0]`` of every tensor before asking whether
the signal is sharded at all, so a replicated 0-dim tensor raised
``IndexError`` on every routed edge (the Python graph runtime and the
conductor's input split both go through it). The row size was taken as
``nbytes // prod(dims)``, which divides by zero for a sharded tensor with an
empty trailing dim, and the same-worker re-slice divided by the byte size of
a row. The Rust fanout handles all of these; ``test/rust`` checks the two
agree.
"""
import pytest
import torch

from mstar.communication.communicator import CommProtocol
from mstar.communication.tensors import SharedMemoryCommunicationManager
from mstar.conductor.conductor import Conductor
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.graph.base import GraphEdge, NodeAndGraphWalk, TensorPointerInfo

WALK = "w"


def _info(dims, dtype=torch.float32, uuid=1):
    dims = list(dims)
    numel = 1
    for d in dims:
        numel *= d
    return TensorPointerInfo(
        dims=dims, dtype=dtype, nbytes=numel * dtype.itemsize, address=0,
        stride=[1] * len(dims), uuid=uuid, source_session_id="s",
        source_entity="worker_0",
    )


def _cfg(shard_dim=None, dest_workers=("worker_1", "worker_2")):
    src = ShardingGroup(nodes={"src"}, tp_size=1)
    dst = ShardingGroup(nodes={"dst"}, tp_size=len(dest_workers))
    cfg = ShardingConfig(
        groups=[src, dst], tp_enabled_nodes={"dst"},
        shard_dim={} if shard_dim is None else {"x": shard_dim},
    )
    cfg.setup({
        NodeAndGraphWalk("src", WALK): ["worker_0"],
        NodeAndGraphWalk("dst", WALK): list(dest_workers),
    })
    src._tp_rank = 0
    dst._tp_rank = 0
    return cfg


def _fanout(cfg, info):
    edge = GraphEdge(next_node="dst", name="x", tensor_info=[info])
    return cfg.fanout_graph_edges(
        edge, source_node="src", source_graph_walk=WALK, dest_graph_walk=WALK,
    )


class TestReplicated:
    def test_a_0_dim_tensor_reaches_every_destination_whole(self):
        out = _fanout(_cfg(), _info([]))
        assert set(out) == {"worker_1", "worker_2"}
        for edge in out.values():
            assert list(edge.tensor_info[0].dims) == []
            assert edge.tensor_info[0].nbytes == 4

    def test_the_conductor_splits_a_0_dim_input(self):
        # ``_split_inputs_to_workers`` uses no conductor state.
        edge = GraphEdge(next_node="dst", name="x", tensor_info=[_info([])])
        per_worker = Conductor._split_inputs_to_workers(None, _cfg(), [edge], WALK)
        assert set(per_worker) == {"worker_1", "worker_2"}


class TestSharded:
    def test_rows_of_zero_bytes_split_by_row_count(self):
        out = _fanout(_cfg(shard_dim=0), _info([4, 0]))
        got = {w: (tuple(e.tensor_info[0].dims), e.tensor_info[0].nbytes,
                   e.tensor_info[0].offset) for w, e in out.items()}
        assert got == {"worker_1": ((2, 0), 0, 0), "worker_2": ((2, 0), 0, 0)}

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
    def test_the_row_size_is_the_trailing_dims_times_the_element_size(self, dtype):
        out = _fanout(_cfg(shard_dim=0), _info([4, 2, 3], dtype=dtype))
        row = 6 * dtype.itemsize
        assert (out["worker_2"].tensor_info[0].offset,
                out["worker_2"].tensor_info[0].nbytes) == (2 * row, 2 * row)


class _Comm:
    def __init__(self, path):
        self.protocol = CommProtocol.IPC
        self.ipc_socket_path_prefix = str(path)

    def _endpoint(self, entity):
        return f"ipc://{self.ipc_socket_path_prefix}/{entity}.ipc"


class TestSameWorkerReslice:
    """A destination rank on the producer's own worker slices the stored
    tensor instead of reading it back over the transport."""

    def _manager(self, tmp_path):
        tm = SharedMemoryCommunicationManager(
            my_entity_id="worker_0", hostname="h", device="cpu",
            communicator=_Comm(tmp_path), shm_dir=str(tmp_path),
        )
        tm.register_request(1, _cfg(shard_dim=0, dest_workers=("worker_0", "worker_1")))
        return tm

    def _reslice(self, tm, tensor, rank):
        stored = tm.store_and_return_tensor_info(
            1, {"x": [tensor]}, node_name="src", graph_walk=WALK,
        )["x"][0]
        (mine,) = [
            e for w, e in _fanout(tm.sharding_configs[1], stored).items()
            if w == f"worker_{rank}"
        ]
        info = mine.tensor_info[0].clone()
        tm._slice_existing_tensor(1, "x", "dst", WALK, info)
        return tm.get_tensor(info.uuid)

    def test_rows_of_zero_bytes(self, tmp_path):
        tm = self._manager(tmp_path)
        got = self._reslice(tm, torch.empty(4, 0), rank=0)
        assert tuple(got.shape) == (2, 0)

    def test_ordinary_rows_are_unchanged(self, tmp_path):
        tm = self._manager(tmp_path)
        x = torch.arange(12.0).reshape(4, 3)
        assert torch.equal(self._reslice(tm, x, rank=0), x[:2])
