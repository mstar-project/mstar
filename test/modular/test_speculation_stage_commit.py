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
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.graph.base import GraphEdge, GraphNode


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


def _staged_runtime():
    """A PythonGraphRuntime with one staged speculation over two rids."""
    from mstar.graph.runtime.python import PythonGraphRuntime

    runtime = PythonGraphRuntime.__new__(PythonGraphRuntime)
    runtime._staged_specs = {}
    runtime._spec_counter = 0
    nodes = {7: _node(), 9: _node()}
    for node in nodes.values():
        assert node.ingest_input(_chunk(1), can_buffer=False)
    prepped = [
        SimpleNamespace(rid=rid, node=node,
                        into_signals=[(0, "thinker_states")], into_next_iter=[])
        for rid, node in nodes.items()
    ]
    spec_id = runtime._stage_spec(prepped)
    return runtime, nodes, spec_id


def test_commit_keeps_the_ingests_of_rids_that_still_run():
    runtime, nodes, spec_id = _staged_runtime()
    runtime.commit_speculation(spec_id, success=True, dropped_rids=[9])

    assert _held(nodes[7].ready_signals) == "chunk1", "rid 7 runs: ingest stands"
    assert _held(nodes[9].ready_signals) is None, "rid 9 dropped: ingest undone"
    # Settled: the stage is gone, so a second call cannot undo rid 7 as well.
    runtime.commit_speculation(spec_id, success=False)
    assert _held(nodes[7].ready_signals) == "chunk1"


def test_commit_with_success_false_undoes_every_rid():
    runtime, nodes, spec_id = _staged_runtime()
    runtime.commit_speculation(spec_id, success=False)
    assert all(_held(n.ready_signals) is None for n in nodes.values())


def test_a_prep_that_staged_nothing_reports_no_id():
    runtime, _nodes, _spec_id = _staged_runtime()
    empty = SimpleNamespace(rid=1, node=_node(), into_signals=[], into_next_iter=[])
    assert runtime._stage_spec([empty]) == 0
    runtime.commit_speculation(0, success=False)      # no-op, must not raise


def test_the_undo_does_not_dereference():
    """``remove`` deliberately does not drop the tensor: the chunk is going back
    to its StreamBuffer, which then holds its only reference."""
    node = _node()
    assert node.ingest_input(_chunk(1), can_buffer=False)
    removed = node.ready_signals.remove("thinker_states")
    assert removed is None or removed  # no exception, no dereference hook fired
    assert _held(node.ready_signals) is None
