"""End-to-end sanity tests for the cross-worker tensor transport path
(store -> register -> read -> consolidate), simulated by spinning up
multiple TensorCommunicationManager instances in a single process.

Each test covers a different fanout / fanin pattern. The new fan-in
buffering in get_ready_tensors is the main thing being exercised.

Parametrized over SHM and TCP-via-Mooncake; the latter skips when
mooncake isn't installed.
"""
from copy import deepcopy

import pytest
import torch

from mstar.communication.communicator import BaseCommunicator, CommProtocol
from mstar.communication.tensors import (
    MOONCAKE_IMPORT_ERROR,
    SharedMemoryCommunicationManager,
    create_tensor_communication_manager,
)
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.graph.base import GraphEdge, NodeAndGraphWalk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockCommunicator(BaseCommunicator):
    """Records sent messages, returns nothing on receive."""

    def __init__(self):
        self.sent: list[tuple[str, object]] = []

    def send(self, entity_id: str, msg):
        self.sent.append((entity_id, msg))

    def get_all_new_messages(self):
        return []


_TCP_AVAILABLE = MOONCAKE_IMPORT_ERROR is None
_PROTOCOLS = [
    pytest.param(CommProtocol.SHM, id="SHM"),
    pytest.param(
        CommProtocol.TCP, id="TCP",
        marks=[pytest.mark.skipif(
            not _TCP_AVAILABLE, reason="mooncake not installed",
        )],
    ),
]


@pytest.fixture
def make_manager(tmp_path):
    """Factory returning a fresh TensorCommunicationManager per entity."""
    def _factory(entity_id: str, protocol: CommProtocol):
        if protocol == CommProtocol.SHM:
            return SharedMemoryCommunicationManager(
                my_entity_id=entity_id, hostname="localhost",
                device="cpu", communicator=_MockCommunicator(),
                shm_dir=str(tmp_path),
            )
        return create_tensor_communication_manager(
            protocol=protocol, my_entity_id=entity_id,
            hostname="localhost", device="cpu",
            communicator=_MockCommunicator(),
            tcp_transfer_device="lo",
        )
    return _factory


def _setup_cfg(*, groups, shard_dim, node_to_worker) -> ShardingConfig:
    cfg = ShardingConfig(tp_enabled_nodes=set(), groups=list(groups), shard_dim=shard_dim)
    cfg.setup(node_to_worker)
    return cfg


def _drain_reads(manager, request_id, graph_edges, graph_walk):
    """Issue reads and block until any async futures complete. SHM returns
    an empty list (sync read); Mooncake returns futures we have to wait on
    before calling get_ready_tensors.
    """
    futures = manager.start_read_tensors(
        request_id, graph_edges, graph_walk=graph_walk,
    )
    for f in futures or []:
        f.result()


def _send(
    manager, request_id: str, signal: str, dest_node: str,
    tensor: torch.Tensor, source_tp_rank: int, source_tp_size: int,
    fanin: int, shard_dim: int
) -> GraphEdge:
    """Producer side: store the tensor, stamp source TP info into the edge's
    tensor_info, and register for send. Returns the edge for the receiver
    to consume.
    """
    edge = GraphEdge(
        next_node=dest_node, name=signal,
        _total_fanin=fanin,
        _shard_dim=shard_dim
    )
    manager.store_and_populate_graph_edges(
        request_id, {signal: [tensor]}, [edge],
    )
    for info in edge.tensor_info:
        info.source_tp_rank = source_tp_rank
        info.source_tp_size = source_tp_size
    manager.register_for_send(
        request_id, [info.uuid for info in edge.tensor_info],
    )
    return edge


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", _PROTOCOLS)
def test_basic_replicated_roundtrip(make_manager, protocol):
    """Single sender -> single receiver, replicated signal (no buffering)."""
    sender = make_manager("w0", protocol)
    receiver = make_manager("w1", protocol)
    cfg = _setup_cfg(
        groups=[], shard_dim={"x": None},
        node_to_worker={NodeAndGraphWalk("A", "decode"): ["w0"], NodeAndGraphWalk("B", "decode"): ["w1"]},
    )
    sender.register_request("req1", cfg)
    receiver.register_request("req1", cfg)

    original = torch.randn(4, 8)
    edge = _send(sender, "req1", "x", "B", original,
                 source_tp_rank=0, source_tp_size=1,
                 fanin=1, shard_dim=None)

    _drain_reads(receiver, "req1", [deepcopy(edge)], graph_walk="decode")
    ready = receiver.get_ready_tensors(graph_walk="decode")

    assert "req1" in ready and len(ready["req1"]) == 1
    out_uuid = ready["req1"][0].tensor_info[0].uuid
    assert torch.equal(receiver.get_tensor("req1", out_uuid), original)


@pytest.mark.parametrize("protocol", _PROTOCOLS)
def test_sharded_tp2_to_tp1_fanin_consolidates(make_manager, protocol):
    """Two TP=2 producers -> one TP=1 consumer. Consumer consolidates."""
    sender0 = make_manager("w0", protocol)
    sender1 = make_manager("w1", protocol)
    receiver = make_manager("w2", protocol)

    # Producers need an explicit src group so setup() doesn't try to auto-
    # generate a TP=1 singleton for a multi-worker node. Consumer's dst is
    # auto-singleton via setup() with _tp_rank=0.
    a_grp = ShardingGroup(nodes={"A"}, tp_size=2, graph_walks={"decode"})
    a_grp.register_workers(["w0", "w1"], my_tp_rank=0)
    cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[a_grp], shard_dim={"x": 1})
    cfg.setup({
        NodeAndGraphWalk("A", "decode"): ["w0", "w1"],
        NodeAndGraphWalk("B", "decode"): ["w2"],
    })
    for m in (sender0, sender1, receiver):
        m.register_request("req1", cfg)

    shard0 = torch.randn(4, 8)
    shard1 = torch.randn(4, 8)
    edge0 = _send(sender0, "req1", "x", "B", shard0, 0, 2, fanin=2, shard_dim=1)
    edge1 = _send(sender1, "req1", "x", "B", shard1, 1, 2, fanin=2, shard_dim=1)

    _drain_reads(
        receiver, "req1", [deepcopy(edge0), deepcopy(edge1)], graph_walk="decode",
    )
    ready = receiver.get_ready_tensors(graph_walk="decode")

    assert "req1" in ready and len(ready["req1"]) == 1
    out_uuid = ready["req1"][0].tensor_info[0].uuid
    consolidated = receiver.get_tensor("req1", out_uuid)
    assert torch.equal(consolidated, torch.cat([shard0, shard1], dim=1))


@pytest.mark.parametrize("protocol", _PROTOCOLS)
def test_fanin_buffers_partial_then_completes(make_manager, protocol):
    """Receiver emits nothing while shards are partial; emits a consolidated
    edge once the final shard arrives.
    """
    sender0 = make_manager("w0", protocol)
    sender1 = make_manager("w1", protocol)
    receiver = make_manager("w2", protocol)
    a_grp = ShardingGroup(nodes={"A"}, tp_size=2, graph_walks={"decode"})
    a_grp.register_workers(["w0", "w1"], my_tp_rank=0)
    cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[a_grp], shard_dim={"x": 0})
    cfg.setup({
        NodeAndGraphWalk("A", "decode"): ["w0", "w1"],
        NodeAndGraphWalk("B", "decode"): ["w2"],
    })
    for m in (sender0, sender1, receiver):
        m.register_request("req1", cfg)

    shard0 = torch.randn(3, 4)
    shard1 = torch.randn(3, 4)
    edge0 = _send(sender0, "req1", "x", "B", shard0, 0, 2, fanin=2, shard_dim=0)
    edge1 = _send(sender1, "req1", "x", "B", shard1, 1, 2, fanin=2, shard_dim=0)

    # First poll: only shard 0 has arrived — buffered, nothing emitted.
    _drain_reads(receiver, "req1", [deepcopy(edge0)], graph_walk="decode")
    first = receiver.get_ready_tensors(graph_walk="decode")
    assert first.get("req1", []) == []

    # Second poll: shard 1 arrives — consolidation runs and edge is emitted.
    _drain_reads(receiver, "req1", [deepcopy(edge1)], graph_walk="decode")
    second = receiver.get_ready_tensors(graph_walk="decode")
    assert "req1" in second and len(second["req1"]) == 1
    out_uuid = second["req1"][0].tensor_info[0].uuid
    assert torch.equal(
        receiver.get_tensor("req1", out_uuid),
        torch.cat([shard0, shard1], dim=0),
    )


@pytest.mark.parametrize("protocol", _PROTOCOLS)
def test_sharded_tp4_to_tp2_fanin(make_manager, protocol):
    """TP=4 -> TP=2 gather: each receiver consolidates 2 source shards.

    Senders 0,1 -> receiver 0 (which holds dst rank 0)
    Senders 2,3 -> receiver 1 (which holds dst rank 1)
    """
    senders = [make_manager(f"w{i}", protocol) for i in range(4)]
    receivers = [make_manager(f"w{i}", protocol) for i in (4, 5)]

    node_to_worker = {
        NodeAndGraphWalk("A", "decode"): ["w0", "w1", "w2", "w3"],
        NodeAndGraphWalk("B", "decode"): ["w4", "w5"],
    }

    # Receiver configs differ only in dst group's _tp_rank.
    def _dst_cfg(my_rank: int) -> ShardingConfig:
        src = ShardingGroup(nodes={"A"}, tp_size=4, graph_walks={"decode"})
        src.register_workers(["w0", "w1", "w2", "w3"], my_tp_rank=0)
        dst = ShardingGroup(nodes={"B"}, tp_size=2, graph_walks={"decode"})
        dst.register_workers(["w4", "w5"], my_tp_rank=my_rank)
        cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[src, dst], shard_dim={"x": 0})
        cfg.setup(node_to_worker)
        return cfg

    a_grp = ShardingGroup(nodes={"A"}, tp_size=4, graph_walks={"decode"})
    a_grp.register_workers(["w0", "w1", "w2", "w3"], my_tp_rank=0)
    b_grp = ShardingGroup(nodes={"B"}, tp_size=2, graph_walks={"decode"})
    b_grp.register_workers(["w4", "w5"], my_tp_rank=0)
    sender_cfg = ShardingConfig(tp_enabled_nodes=set(), groups=[a_grp, b_grp], shard_dim={"x": 0})
    sender_cfg.setup(node_to_worker)
    for s in senders:
        s.register_request("req1", sender_cfg)
    receivers[0].register_request("req1", _dst_cfg(my_rank=0))
    receivers[1].register_request("req1", _dst_cfg(my_rank=1))

    shards = [torch.randn(2, 6) for _ in range(4)]
    edges = [
        _send(senders[i], "req1", "x", "B", shards[i],
              source_tp_rank=i, source_tp_size=4,
              fanin=2, shard_dim=0)
        for i in range(4)
    ]

    _drain_reads(
        receivers[0], "req1",
        [deepcopy(edges[0]), deepcopy(edges[1])], graph_walk="decode",
    )
    _drain_reads(
        receivers[1], "req1",
        [deepcopy(edges[2]), deepcopy(edges[3])], graph_walk="decode",
    )

    r0_ready = receivers[0].get_ready_tensors(graph_walk="decode")
    r1_ready = receivers[1].get_ready_tensors(graph_walk="decode")
    assert len(r0_ready["req1"]) == 1
    assert len(r1_ready["req1"]) == 1

    r0_uuid = r0_ready["req1"][0].tensor_info[0].uuid
    r1_uuid = r1_ready["req1"][0].tensor_info[0].uuid
    assert torch.equal(
        receivers[0].get_tensor("req1", r0_uuid),
        torch.cat([shards[0], shards[1]], dim=0),
    )
    assert torch.equal(
        receivers[1].get_tensor("req1", r1_uuid),
        torch.cat([shards[2], shards[3]], dim=0),
    )


@pytest.mark.parametrize("protocol", _PROTOCOLS)
def test_sharded_with_nonzero_shard_dim_roundtrips(make_manager, protocol):
    """Single producer / single consumer, shard_dim=1: exercises the
    canonical-layout rearrange + undo on a sharded signal.
    """
    sender = make_manager("w0", protocol)
    receiver = make_manager("w1", protocol)
    cfg = _setup_cfg(
        groups=[], shard_dim={"x": 1},
        node_to_worker={
            NodeAndGraphWalk("A", "decode"): ["w0"], NodeAndGraphWalk("B", "decode"): ["w1"],
        },
    )
    sender.register_request("req1", cfg)
    receiver.register_request("req1", cfg)

    original = torch.randn(3, 5, 7)
    edge = _send(sender, "req1", "x", "B", original,
                 source_tp_rank=0, source_tp_size=1,
                 fanin=1, shard_dim=1)
    _drain_reads(receiver, "req1", [deepcopy(edge)], graph_walk="decode")
    ready = receiver.get_ready_tensors(graph_walk="decode")
    assert "req1" in ready and len(ready["req1"]) == 1
    out_uuid = ready["req1"][0].tensor_info[0].uuid
    assert torch.equal(receiver.get_tensor("req1", out_uuid), original)


@pytest.mark.parametrize("protocol", _PROTOCOLS)
def test_colocated_replicated_to_sharded_slices_locally(make_manager, protocol):
    """Colocated encoder (TP=1) + LLM (TP=2) on the same worker. The signal
    is sharded (shard_dim=1). The LLM consumer should get its own slice
    under a fresh local UUID rather than aliasing the producer's UUID.
    """
    mgr = make_manager("w0", protocol)

    llm_group = ShardingGroup(nodes={"LLM"}, tp_size=2, graph_walks={"decode"})
    llm_group.register_workers(["w0", "w1"], my_tp_rank=0)
    cfg = _setup_cfg(
        groups=[llm_group], shard_dim={"hidden": 1},
        node_to_worker={
            NodeAndGraphWalk("encoder", "decode"): ["w0"],
            NodeAndGraphWalk("LLM", "decode"): ["w0", "w1"],
        },
    )
    mgr.register_request("req1", cfg)

    # Producer: encoder makes a [4, 8] float tensor. Canonical form (shard
    # dim 1 → leading) is [8, 4]. LLM rank 0 wants dim-1 [0, 4) of the
    # original, which in canonical layout is the first 4 rows.
    original = torch.randn(4, 8)
    edge = _send(mgr, "req1", "hidden", "LLM", original,
                 source_tp_rank=0, source_tp_size=1,
                 fanin=1, shard_dim=1)
    producer_uuid = edge.tensor_info[0].uuid

    # Simulate the cross-worker LLM rank 1's pending read by pre-bumping the
    # producer's refcount. _send only registered one outgoing edge (rank 0),
    # so without this the colocated slice's dereference would drop the
    # producer's ref to 0 and GC the canonical tensor.
    mgr.tensor_store.increment_ref("req1", producer_uuid, 1)

    # Worker-pre-computed per-receiver slice metadata.
    half_nbytes = edge.tensor_info[0].nbytes // 2
    edge.tensor_info[0].offset = 0
    edge.tensor_info[0].nbytes = half_nbytes
    edge.tensor_info[0].dims = torch.Size([4, 4])
    edge.tensor_info[0].stride = (4, 1)

    _drain_reads(mgr, "req1", [edge], graph_walk="decode")
    ready = mgr.get_ready_tensors(graph_walk="decode")

    assert "req1" in ready and len(ready["req1"]) == 1
    slice_uuid = ready["req1"][0].tensor_info[0].uuid
    # A fresh UUID — producer's UUID stays available for the other LLM rank.
    assert slice_uuid != producer_uuid
    assert mgr.tensor_store.check_uuid_presence("req1", producer_uuid)

    received = mgr.get_tensor("req1", slice_uuid)
    expected = original[:, :4]
    assert torch.equal(received, expected)
