"""``ColumnarEdgeSpecs``: the shape a pop and a speculation prep report in.

Load-bearing beyond the crossing it exists for -- the scheduler's backlog
re-slices these blocks when a batch is split or a rid is dropped -- so the
column arithmetic gets its own tests rather than only being exercised through
a runtime.
"""
import sys

sys.path.insert(0, ".")

import torch

from mstar.communication.tensor_store import TensorStore
from mstar.graph.base import TensorPointerInfo
from mstar.graph.runtime.base import ColumnarEdgeSpecs


def _block(*edges) -> ColumnarEdgeSpecs:
    """``edges`` are (rid, signal, uuids, is_final_streaming_chunk)."""
    block = ColumnarEdgeSpecs.empty()
    for rid, signal, uuids, final in edges:
        block.add(rid, signal, uuids, final)
    return block


def test_the_builder_carries_each_signal_name_once():
    block = _block(
        (1, "token", [10], False),
        (1, "kv", [11, 12], False),
        (2, "token", [20], False),
        (2, "kv", [21], False),
    )
    assert block.signal_names == ["token", "kv"], "one entry per DISTINCT name"
    assert block.signal_name_idxs == [0, 1, 0, 1]
    assert block.rids == [1, 1, 2, 2]
    assert block.tensors_per_edge == [1, 2, 1, 1]
    assert block.uuids == [10, 11, 12, 20, 21]
    assert len(block) == 4, "len is EDGES, not rids"


def test_to_input_tensors_groups_per_rid_and_names_the_final_chunks():
    block = _block(
        (1, "token", [10], False),
        (1, "kv", [11, 12], True),
        (2, "token", [20], False),
    )
    out = block.to_input_tensors(lambda u: f"t{u}")
    assert out.by_rid == {
        1: {"token": ["t10"], "kv": ["t11", "t12"]},
        2: {"token": ["t20"]},
    }
    assert out.final_stream_rids == {1}


def test_a_rid_with_nothing_ready_still_gets_an_entry_when_seeded():
    """The batch build keys per_request_info off these, and the engine then
    indexes it by rid -- so a rid the pop returned with no ready input has to
    appear, as slicing a zero-length run used to give it."""
    block = _block((1, "token", [10], False))
    seeded = block.to_input_tensors(lambda u: f"t{u}", [1, 2]).by_rid
    assert seeded == {1: {"token": ["t10"]}, 2: {}}
    # Unseeded, only the rids that have edges appear.
    bare = block.to_input_tensors(lambda u: f"t{u}").by_rid
    assert bare == {1: {"token": ["t10"]}}


def test_select_rids_keeps_each_survivors_tensors_intact():
    block = _block(
        (1, "token", [10], False),
        (2, "kv", [20, 21], True),
        (3, "token", [30], False),
    )
    kept = block.select_rids({1, 3})
    assert kept.edge_tuples() == [
        (1, "token", [10], False),
        (3, "token", [30], False),
    ]
    # The dropped rid's tensors go with it, not just its row.
    assert kept.uuids == [10, 30]
    # The original is untouched -- split_off_first returns two new blocks.
    assert len(block) == 3


def test_select_rids_of_nothing_is_empty_not_partial():
    block = _block((1, "token", [10], False))
    empty = block.select_rids(set())
    assert len(empty) == 0 and empty.uuids == []


def test_extend_remaps_signal_names_that_disagree_on_order():
    """Two blocks for the same node do agree in practice -- the order comes
    from the node's input spec -- but nothing enforces it, and taking the
    other block's indices as-is would rename every one of its edges."""
    first = _block((1, "token", [10], False))
    second = _block((2, "kv", [20], False), (2, "token", [21], True))
    assert second.signal_names == ["kv", "token"], "the orders differ"

    first.extend(second)
    assert first.edge_tuples() == [
        (1, "token", [10], False),
        (2, "kv", [20], False),
        (2, "token", [21], True),
    ]
    assert first.uuids == [10, 20, 21]


def test_extend_onto_an_empty_block_is_the_other_block():
    block = ColumnarEdgeSpecs.empty()
    block.extend(_block((7, "token", [70], True)))
    assert block.edge_tuples() == [(7, "token", [70], True)]


def test_to_edges_rebuilds_real_graph_edges():
    store = TensorStore()
    for uuid in (10, 11):
        store.put_tensor(1, uuid, torch.zeros(4), TensorPointerInfo(
            dims=[4], dtype=torch.float16, nbytes=8, address=0, stride=(1,),
            uuid=uuid, source_session_id="h:1", source_entity="worker_0",
        ))
    block = _block((1, "token", [10, 11], True))
    edges = block.to_edges(store, is_streaming=True, next_node="ar_decode")
    assert len(edges) == 1
    edge = edges[0]
    assert edge.name == "token" and edge.next_node == "ar_decode"
    assert [i.uuid for i in edge.tensor_info] == [10, 11]
    assert edge.is_streaming and edge._final_stream_chunk


def test_to_edges_reads_the_destination_column_when_not_overridden():
    block = _block((1, "token", [], False), (1, "kv", [], False))
    block.next_nodes = ["a", "b"]
    block.next_node_idxs = [1, 0]
    dests = [e.next_node for e in block.to_edges(TensorStore(), False)]
    assert dests == ["b", "a"]


def test_adding_to_a_block_that_arrived_without_its_name_cache():
    """``_signal_idx_of`` is a cache, and most blocks are built without one --
    Rust fills the columns itself, and ``select_rids`` copies the names across.
    Appending blind would add a DUPLICATE name and silently re-label every edge
    pointing at the original.
    """
    # As select_rids leaves it: names present, cache empty.
    sliced = _block(
        (1, "token", [10], False), (2, "kv", [20], False),
    ).select_rids({2})
    assert sliced.signal_names == ["token", "kv"]

    sliced.add(3, "token", [30], False)
    assert sliced.signal_names == ["token", "kv"], "no duplicate name"
    assert sliced.edge_tuples() == [
        (2, "kv", [20], False),
        (3, "token", [30], False),
    ]


def test_extending_a_block_that_arrived_without_its_name_cache():
    """Same hazard through ``extend``, which the backlog merge uses."""
    sliced = _block(
        (1, "token", [10], False), (2, "kv", [20], False),
    ).select_rids({1})
    sliced.extend(_block((4, "kv", [40], True)))
    assert sliced.signal_names == ["token", "kv"]
    assert sliced.edge_tuples() == [
        (1, "token", [10], False),
        (4, "kv", [40], True),
    ]


def test_the_name_cache_is_not_part_of_the_value():
    """Two blocks with the same columns compare equal whether or not either has
    populated its cache, and the cache stays out of the repr."""
    built = _block((1, "token", [10], False))          # cache populated
    bare = ColumnarEdgeSpecs(
        signal_names=["token"], uuids=[10], tensors_per_edge=[1],
        signal_name_idxs=[0], rids=[1], is_final_streaming_chunk=[False],
    )                                                   # as the adapter builds
    assert built == bare
    assert "_signal_idx_of" not in repr(built)
