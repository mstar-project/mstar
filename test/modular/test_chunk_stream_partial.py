"""A chunked prefill's stream, sent per chunk or gathered.

A ``per_chunk_outputs`` edge stays on the chunk Loop's body, so every chunk
routes its piece as it completes; a gathered one leaves on the Loop at the
end. A consumer's ChunkPolicy counts items either way, so a stream it would
misread (each piece taken for the whole input) is refused at load.
"""

import sys

sys.path.insert(0, ".")

from types import SimpleNamespace

import pytest

from mstar.graph.base import GraphEdge, GraphNode
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import _wrap_node_in_loop
from mstar.model.submodule_base import ChunkingPolicy, chunk_tick_name
from mstar.streaming.chunk_policy import FixedChunkPolicy
from mstar.streaming.topology import Connection, PartitionTopology, StreamingGraphEdge
from mstar.validation.placement import _chunked_streams

NODE, WALK = "Thinker", "prefill"


def _node(states_streaming=True):
    states = (
        StreamingGraphEdge(next_node="Talker", name="states", target_partition="Talker")
        if states_streaming else GraphEdge(next_node="Talker", name="states")
    )
    return GraphNode(
        name=NODE, input_names={"text_inputs"},
        outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="new_token", persist=True), states],
    )


def _policy(**kwargs):
    return ChunkingPolicy(walks=(WALK,), chunkable=True, **kwargs)


def test_a_per_chunk_edge_stays_on_the_body_with_its_destination():
    loop = _wrap_node_in_loop(_node(), NODE, WALK, _policy(per_chunk_outputs=frozenset({"states"})))

    body = {e.name: e for e in loop.section.outputs}
    assert body["states"].next_node == "Talker" and body["states"].is_streaming
    assert body["new_token"].next_node == EMPTY_DESTINATION, "a final-chunk edge still only ticks"
    assert chunk_tick_name(NODE, WALK) in body
    assert [e.name for e in loop.outputs] == ["new_token"]
    assert loop.accumulated_outputs == []


def test_an_output_cannot_be_both_gathered_and_sent_per_chunk():
    with pytest.raises(ValueError, match="both gathered and sent per chunk"):
        _policy(
            accumulated_outputs=frozenset({"states"}), per_chunk_outputs=frozenset({"states"}),
        )


def test_a_consumer_takes_no_partial_input_unless_it_says_so():
    assert not FixedChunkPolicy(chunk_size=1).allow_partial_input()
    assert FixedChunkPolicy(chunk_size=1, allow_partial=True).allow_partial_input()


def _problems(policy, *, allow_partial=False, join_dim=None, states_streaming=True):
    walk_graphs = {WALK: _wrap_node_in_loop(_node(states_streaming), NODE, WALK, policy)}
    topology = PartitionTopology(partitions=["Thinker", "Talker"], connections=[Connection(
        from_partition="Thinker", to_partition="Talker", edge_name="states",
        chunk_policy_factory=lambda: FixedChunkPolicy(chunk_size=1, allow_partial=allow_partial),
        join_dim=join_dim,
    )])
    model = SimpleNamespace(
        get_partition_topology=lambda: topology,
        chunking_policy_of=lambda: {(NODE, WALK): policy},
    )
    return list(_chunked_streams(model, walk_graphs))


PER_CHUNK = _policy(per_chunk_outputs=frozenset({"states"}))
GATHERED = _policy(accumulated_outputs=frozenset({"states"}))


def test_a_per_chunk_stream_into_a_partial_consumer_is_accepted():
    assert _problems(PER_CHUNK, allow_partial=True) == []


@pytest.mark.parametrize("join_dim", [None, 0], ids=["no join", "join_dim"])
def test_a_per_chunk_stream_into_a_whole_input_consumer_is_refused(join_dim):
    """``join_dim`` joins the pieces one pass delivers, and a per-chunk stream
    delivers one per pass, so it cannot stand in for partial input."""
    [problem] = _problems(PER_CHUNK, join_dim=join_dim)
    assert "does not allow partial input" in problem


@pytest.mark.parametrize(("allow_partial", "join_dim"), [(False, 0), (True, None)])
def test_a_gathered_stream_the_consumer_reads_whole_or_in_pieces_is_accepted(allow_partial, join_dim):
    assert _problems(GATHERED, allow_partial=allow_partial, join_dim=join_dim) == []


def test_a_gathered_stream_the_consumer_would_read_piece_by_piece_is_refused():
    [problem] = _problems(GATHERED)
    assert "would read each chunk's piece as an item of its own" in problem


def test_a_per_chunk_edge_must_be_a_stream():
    [problem] = _problems(PER_CHUNK, allow_partial=True, states_streaming=False)
    assert "not a streaming edge" in problem


def test_a_per_chunk_name_that_is_no_output_is_refused():
    [problem] = _problems(_policy(per_chunk_outputs=frozenset({"stats"})), allow_partial=True)
    assert "'stats', which is not one of its outputs" in problem


def test_an_unchunkable_policy_is_not_checked():
    """No Loop wraps it, so its edges route as declared."""
    policy = ChunkingPolicy(walks=(WALK,), per_chunk_outputs=frozenset({"states"}))
    model = SimpleNamespace(
        get_partition_topology=lambda: PartitionTopology(partitions=["Thinker"]),
        chunking_policy_of=lambda: {(NODE, WALK): policy},
    )
    assert list(_chunked_streams(model, {WALK: _node()})) == []

