"""``WorkerLayout.derive``: what the worker reads off its graphs, the model and
its process groups at startup, computed without building a worker."""
from types import MappingProxyType, SimpleNamespace

import pytest

from mstar.conductor.request_info import PartitionDefinition
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.model.base import WorkerGraph
from mstar.model.submodule_base import chunk_loop_name
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge
from mstar.worker.chunk_loops import ChunkLoops
from mstar.worker.services import WorkerLayout

ALL_WG_NODES = {0: {"producer"}, 1: {"consumer", "helper"}, 2: {"codec", "post"}}
CHUNK_LOOP = chunk_loop_name("consumer", "consume")


def _walks():
    producer = GraphNode(
        name="producer", input_names=["text"],
        outputs=[StreamingGraphEdge(next_node="consumer", name="chunk", target_partition="B")],
    )
    consume = Sequential([
        Loop(
            name=CHUNK_LOOP,
            section=GraphNode(name="consumer", input_names=["chunk"],
                              outputs=[GraphEdge(next_node="helper", name="hidden")]),
            max_iters=8, outputs=[],
        ),
        GraphNode(
            name="helper", input_names=["hidden"],
            outputs=[StreamingGraphEdge(next_node="codec", name="codes", target_partition="C")],
        ),
    ])
    decode = Sequential([
        GraphNode(name="codec", input_names=["codes"], outputs=[GraphEdge(next_node="post", name="pcm")]),
        GraphNode(name="post", input_names=["pcm"], outputs=[]),
    ])
    return {"produce": producer, "consume": consume, "decode_audio": decode}


def _connection(source, dest, edge):
    return Connection(
        from_partition=source, to_partition=dest, edge_name=edge,
        chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1),
    )


def _model(topology=True, combined=None):
    return SimpleNamespace(
        get_graph_walk_graphs=_walks,
        get_partitions=lambda: [
            PartitionDefinition(name="A", graph_walks={"produce"}),
            PartitionDefinition(name="B", graph_walks={"consume"}),
            PartitionDefinition(name="C", graph_walks={"decode_audio"}),
        ],
        get_partition_topology=lambda: PartitionTopology(
            partitions=["A", "B", "C"],
            connections=[_connection("A", "B", "chunk"), _connection("B", "C", "codes")],
        ) if topology else None,
        chunking_policy_of=lambda: {("consumer", "consume"): object()},
        combined_walk_of=lambda: combined or {},
    )


def _groups(sizes=None, ranks=None):
    """Instance world size and this worker's rank per node: 1 and 0 by default."""
    return SimpleNamespace(
        get_instance_world_size_for_node=lambda node: (sizes or {}).get(node, 1),
        get_instance_rank_for_node=lambda node: (ranks or {}).get(node, 0),
    )


def _derive(model=None, groups=None, section=None, walks=("consume",), model_config=None, sharding=None):
    section = _walks()["consume"] if section is None else section
    return WorkerLayout.derive(
        model, [WorkerGraph(section=section, graph_walks=set(walks))], ALL_WG_NODES,
        sharding, groups or _groups(), model_config,
    )


def test_a_worker_with_no_parallel_node_leads_every_node_and_follows_none():
    layout = _derive()
    assert layout.parallel_nodes == frozenset()
    assert layout.parallel_leader_nodes == {"consumer", "helper"}
    assert not layout.is_tp_follower
    assert layout.followed_nodes == frozenset()


def test_instance_rank_zero_leads_its_parallel_nodes():
    layout = _derive(groups=_groups(sizes={"consumer": 2}))
    assert layout.parallel_nodes == {"consumer"}
    assert layout.parallel_leader_nodes == {"consumer", "helper"}
    assert not layout.is_tp_follower
    assert layout.followed_nodes == frozenset()


def test_another_rank_of_the_instance_follows_it():
    layout = _derive(groups=_groups(sizes={"consumer": 2}, ranks={"consumer": 1}))
    assert layout.parallel_nodes == {"consumer"}
    assert layout.parallel_leader_nodes == {"helper"}
    assert layout.is_tp_follower
    assert layout.followed_nodes == {"consumer"}


def test_the_nodes_and_walks_are_this_workers_across_its_graphs():
    walks = _walks()
    layout = WorkerLayout.derive(
        None,
        [WorkerGraph(section=walks["produce"], graph_walks={"produce"}),
         WorkerGraph(section=walks["consume"], graph_walks={"consume", "again"})],
        ALL_WG_NODES, None, _groups(), None,
    )
    assert layout.node_names == {"producer", "consumer", "helper"}
    assert layout.my_walks == {
        ("producer", "produce"),
        ("consumer", "consume"), ("helper", "consume"),
        ("consumer", "again"), ("helper", "again"),
    }
    assert layout.all_wg_nodes is ALL_WG_NODES


def test_node_to_partition_covers_the_whole_model_and_is_read_only():
    layout = _derive(_model())
    assert layout.node_to_partition == {
        "producer": "A", "consumer": "B", "helper": "B", "codec": "C", "post": "C",
    }
    assert isinstance(layout.node_to_partition, MappingProxyType)


def test_innermost_loop_maps_the_nodes_inside_a_loop():
    assert _derive(_model()).innermost_loop_by_node == {("consume", "consumer"): CHUNK_LOOP}


def test_consumer_connections_are_those_streaming_into_this_workers_nodes():
    layout = _derive(_model())
    assert [c.edge_name for c in layout.consumer_connections] == ["chunk"]
    assert layout.streaming_edge_names == {"chunk"}

    downstream = _derive(_model(), section=_walks()["decode_audio"], walks=("decode_audio",))
    assert [c.edge_name for c in downstream.consumer_connections] == ["codes"]
    assert downstream.streaming_edge_names == {"codes"}

    producer = _derive(_model(), section=_walks()["produce"], walks=("produce",))
    assert producer.consumer_connections == ()
    assert producer.streaming_edge_names == frozenset()


def test_a_model_with_no_topology_has_no_consumer_connections():
    layout = _derive(_model(topology=False))
    assert layout.consumer_connections == ()
    assert layout.streaming_edge_names == frozenset()
    assert layout.node_to_partition["consumer"] == "B"


def test_no_model_still_derives_from_the_graphs_and_groups():
    layout = _derive(groups=_groups(sizes={"helper": 4}, ranks={"helper": 3}))
    assert layout.followed_nodes == {"helper"}
    assert layout.chunk_loops.walks == {("consumer", "consume")}
    assert not layout.node_to_partition
    assert not layout.innermost_loop_by_node
    assert layout.consumer_connections == ()
    assert not layout.combined_walk_of


def test_chunk_loops_read_the_speculate_mid_prefill_setting():
    on = _derive(_model())
    assert isinstance(on.chunk_loops, ChunkLoops)
    assert on.chunk_loops.speculated == {("consumer", "consume")}

    off = _derive(_model(), model_config={"speculate_mid_prefill": {"consumer": False}})
    assert off.chunk_loops.walks == {("consumer", "consume")}
    assert off.chunk_loops.speculated == frozenset()


def test_combined_walks_come_from_the_model():
    combined = {("consumer", "consume"): "mixed"}
    assert _derive(_model(combined=combined), sharding=SimpleNamespace(groups=[])).combined_walk_of == combined


COMBINED = {("consumer", "consume"): "mixed", ("consumer", "again"): "mixed"}
# each walk of the combined one in a TP group of its own
SPLIT = SimpleNamespace(groups=[
    SimpleNamespace(nodes={"consumer"}, graph_walks={"consume"}, tp_size=2),
    SimpleNamespace(nodes={"consumer"}, graph_walks={"again"}, tp_size=4),
])


def test_a_combined_walk_spanning_tp_groups_is_refused():
    with pytest.raises(ValueError, match="spans TP groups"):
        _derive(_model(combined=COMBINED), walks=("consume", "again"), sharding=SPLIT)


def test_a_combined_walk_half_on_another_worker_is_not_refused():
    assert _derive(_model(combined=COMBINED), sharding=SPLIT).combined_walk_of == COMBINED
