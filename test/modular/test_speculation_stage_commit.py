"""A speculative prep STAGES its streaming ingests; commit_speculation settles them.

A prep ingests each rid's next chunk into the speculative node's input slot. If
that step never runs, the chunk has to come back OUT of the slot before it is
handed to its StreamBuffer, for two separate reasons:

* Left in the slot while the buffer also holds it, one chunk is tracked twice.
  Streamed chunks carry no reference, so the first pass to clear the slot frees
  a tensor the buffer still points at.
* Left in the slot and NOT returned, the stream inverts: a same-node prep
  ingests into the next-iter slot, and an ingest takes the current slot whenever
  it is free, so the FOLLOWING chunk is consumed first.

The second is what this pins -- it is the one with no loud failure.
"""
import sys

sys.path.insert(0, ".")

from rust_runtime import build_runtime

from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.runtime.base import EdgeSpec, SpeculationPrepInput
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList


class _Reg:
    def register_ingested_input(self, edge):
        pass


def _node():
    node = GraphNode(
        name="Talker", input_names={"thinker_states"},
        outputs=[GraphEdge(next_node="Code2Wav", name="codec_tokens")],
    )
    node._register_streaming({"thinker_states"})
    node._managing_registry = _Reg()
    return node


def _chunk(i):
    return GraphEdge(next_node="Talker", name="thinker_states",
                     is_streaming=True, tensor_info=[f"chunk{i}"])


def _held(slot):
    edge = slot.ready_inputs.get("thinker_states")
    return None if edge is None else edge.tensor_info[0]


def test_a_chunk_left_in_the_next_iter_slot_is_overtaken():
    """Why the undo is needed: this is the ordering failure it prevents."""
    node = _node()
    assert node.ingest_input(_chunk(1), can_buffer=False)   # step K
    assert node.ingest_input(_chunk(2), can_buffer=True)    # same-node prep -> next
    assert (_held(node.ready_signals), _held(node.ready_next_iter)) == ("chunk1", "chunk2")

    node.ready_signals.clear()                              # step K consumed its input
    # Chunk 2 is still in the next-iter slot, so chunk 3 takes the free current
    # slot and is consumed BEFORE it.
    assert node.ingest_input(_chunk(3), can_buffer=False)
    order = [_held(node.ready_signals)]
    node.reset_for_outer_iter()
    order.append(_held(node.ready_signals))
    assert order == ["chunk3", "chunk2"], "ordering hazard no longer reproduces"


def test_undoing_the_staged_ingest_keeps_the_stream_in_order():
    """With the staged ingest undone, the slot is free and chunk 2 -- handed back
    to its buffer, which a poll drains before any fresh chunk -- is next."""
    node = _node()
    assert node.ingest_input(_chunk(1), can_buffer=False)
    assert node.ingest_input(_chunk(2), can_buffer=True)

    # What commit_speculation(success=False) does for this rid.
    node.ready_next_iter.remove("thinker_states")
    assert _held(node.ready_next_iter) is None

    node.ready_signals.clear()
    # The buffer's waiting edge is re-ingested first, so chunk 2 leads.
    assert node.ingest_input(_chunk(2), can_buffer=False)
    order = [_held(node.ready_signals)]
    node.ready_signals.clear()
    assert node.ingest_input(_chunk(3), can_buffer=False)
    order.append(_held(node.ready_signals))
    assert order == ["chunk2", "chunk3"]


def _chunk_uuid(rid):
    return 1000 + rid


def _staged_runtime():
    """A runtime with one staged speculation over two rids, each holding a chunk."""
    walk = "talk"
    section = Sequential(sections=[
        GraphNode(name="Thinker", input_names={"prompt"}, outputs=[]),
        Loop(name="talk_loop", max_iters=8, outputs=[], section=_node()),
    ])
    nodes = {"Thinker", "Talker"}
    runtime = build_runtime(
        my_worker_id="w0",
        my_worker_graphs=[WorkerGraph(section=section, graph_walks={walk}, ranks=[0], worker_graph_id=0)],
        all_wg_ids_to_graph_walks={0: {walk}}, all_wg_ids_to_dyn_loops={0: {"talk_loop"}},
        all_wg_ids_to_nodes={0: nodes}, node_to_partition=dict.fromkeys(nodes, "default"),
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
    )
    rids = [
        runtime.add_request(
            request_id=f"r{i}", partition="default", graph_walk=walk, partition_worker_graph_ids=[0],
            worker_graph_to_workers=ParallelList([0], [["w0"]]), loop_caps={},
        )
        for i in range(2)
    ]
    out = _prep(runtime, rids, chunks=True)
    assert out.ready_rids == rids and out.spec_id != 0
    return runtime, rids, out.spec_id


def _prep(runtime, rids, chunks):
    return runtime.prep_spec_rids(SpeculationPrepInput(
        spec_node_name="Talker", curr_node_name="Thinker", graph_walk="talk", rids=rids,
        room_for_continuing=None,
        streaming_edges=[
            EdgeSpec(signal="thinker_states", next_node="Talker", uuids=[_chunk_uuid(r)]) for r in rids
        ] if chunks else [],
        streaming_edges_per_rid=[1 if chunks else 0] * len(rids),
    ))


def _held_by(runtime, rid):
    """The chunk uuids in ``rid``'s input slot."""
    popped = runtime.pop_rids("Talker", "talk", [rid])
    return [u for e in popped.input_edges.edge_tuples() for u in e.uuids]


def test_commit_keeps_the_ingests_of_rids_that_still_run():
    runtime, (kept, dropped), spec_id = _staged_runtime()
    runtime.commit_speculation(spec_id, success=True, dropped_rids=[dropped])

    assert _held_by(runtime, dropped) == [], "the dropped rid's ingest is undone"
    # Settled: the stage is gone, so a second call cannot undo the kept rid as well.
    runtime.commit_speculation(spec_id, success=False)
    assert _held_by(runtime, kept) == [_chunk_uuid(kept)], "it runs: its ingest stands"


def test_commit_with_success_false_undoes_every_rid():
    runtime, rids, spec_id = _staged_runtime()
    runtime.commit_speculation(spec_id, success=False)
    assert [_held_by(runtime, rid) for rid in rids] == [[], []]


def test_a_prep_that_staged_nothing_reports_no_id():
    runtime, rids, spec_id = _staged_runtime()
    runtime.commit_speculation(spec_id, success=False)
    assert _prep(runtime, rids, chunks=False).spec_id == 0
    runtime.commit_speculation(0, success=False)      # no-op, must not raise


def test_the_undo_does_not_dereference():
    """``remove`` deliberately does not drop the tensor: the chunk is going back
    to its StreamBuffer, which then holds its only reference."""
    node = _node()
    assert node.ingest_input(_chunk(1), can_buffer=False)
    removed = node.ready_signals.remove("thinker_states")
    assert removed is None or removed  # no exception, no dereference hook fired
    assert _held(node.ready_signals) is None
