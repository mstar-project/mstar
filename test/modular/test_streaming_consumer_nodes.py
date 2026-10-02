"""A streaming consumer nested inside a ``Loop`` must be found by the worker's
partition-node and consumer-node lookups.

Both used to inspect only a walk's top-level section: a walk that is itself a
single ``GraphNode`` matched, but a decode ``Loop`` whose inner node consumes
the streamed edge did not. Its chunks were then routed to ``next_node=""`` and
its ``StreamBuffer`` was never created (``KeyError`` when routing the edge).
"""
from types import SimpleNamespace

from mstar.conductor.request_info import PartitionDefinition
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import Connection, StreamingGraphEdge
from mstar.worker.worker import Worker


def _model():
    producer = GraphNode(
        name="producer",
        input_names=["text_inputs"],
        outputs=[StreamingGraphEdge(next_node="consumer", name="chunk", target_partition="B")],
    )
    consumer_loop = Loop(
        name="consumer_loop",
        section=GraphNode(
            name="consumer",
            input_names=["chunk"],
            outputs=[StreamingGraphEdge(next_node="codec", name="codes", target_partition="C")],
        ),
        max_iters=8,
        outputs=[],
    )
    codec_chunk = Sequential([
        GraphNode(name="codec", input_names=["codes"],
                  outputs=[GraphEdge(next_node="post", name="pcm")]),
        GraphNode(name="post", input_names=["pcm"],
                  outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio", output_modality="audio")]),
    ])
    walks = {"produce": producer, "consume": consumer_loop, "decode_audio": codec_chunk}
    partitions = [
        PartitionDefinition(name="A", graph_walks={"produce"}, initial_walk="produce"),
        PartitionDefinition(name="B", graph_walks={"consume"}, initial_walk="consume", producer_partitions=["A"]),
        PartitionDefinition(name="C", graph_walks={"decode_audio"}, initial_walk="decode_audio",
                            producer_partitions=["B"]),
    ]
    return SimpleNamespace(
        get_graph_walk_graphs=lambda: walks,
        get_partitions=lambda: partitions,
    )


def _connections():
    return [
        Connection(from_partition="A", to_partition="B", edge_name="chunk",
                   chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1)),
        Connection(from_partition="B", to_partition="C", edge_name="codes",
                   chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1)),
    ]


def test_partition_nodes_recurse_into_loops_and_sequentials():
    model = _model()
    # ``self`` is unused by the lookup, so call it unbound on a stub.
    assert Worker._get_node_names_for_partition(None, "A", model) == ["producer"]
    # the Loop's inner node, not the Loop's own name
    assert Worker._get_node_names_for_partition(None, "B", model) == ["consumer"]
    assert set(Worker._get_node_names_for_partition(None, "C", model)) == {"codec", "post"}
    assert Worker._get_node_names_for_partition(None, "missing", model) == []


def test_consumer_node_cache_finds_nested_consumers():
    cache = Worker._build_consumer_node_cache(_connections(), _model().get_graph_walk_graphs())
    assert cache == {"chunk": "consumer", "codes": "codec"}


def test_innermost_loop_cache_maps_only_nodes_inside_a_loop():
    cache = Worker._build_innermost_loop_cache(_model().get_graph_walk_graphs())
    assert cache == {("consume", "consumer"): "consumer_loop"}
    # Nodes outside any loop get no entry, so no stop is ever synthesised.
    assert ("produce", "producer") not in cache
    assert ("decode_audio", "codec") not in cache


def test_innermost_loop_cache_picks_the_inner_loop_when_nested():
    inner = Loop(
        name="inner_loop",
        section=GraphNode(name="deep", input_names=["chunk"],
                          outputs=[GraphEdge(next_node="out", name="y")]),
        max_iters=4, outputs=[],
    )
    outer = Loop(name="outer_loop", section=inner, max_iters=2, outputs=[])
    cache = Worker._build_innermost_loop_cache({"w": outer})
    assert cache == {("w", "deep"): "inner_loop"}
