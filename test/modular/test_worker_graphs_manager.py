"""Tests for ``WorkerGraphsManager``.

Covers:
- Inverted ``walk_node_to_worker_graph_id`` index built in __post_init__
- ``get_worker_graph_id_for_node`` uses the index (O(1) lookup, no scan)
- ``mark_node_complete`` returns the registry's ``NodeCompletionOutput``
- ``process_new_inputs`` returns leftover edges that no wg claimed
- ``stop_loops`` returns the loop-back ``set[(name, dest)]``
"""

import pytest

from mstar.conductor.request_info import (
    CurrentForwardPassInfo,
)
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.model.base import WorkerGraph
from mstar.worker.node_manager_utils import (
    WorkerGraphQueues,
    WorkerGraphsManager,
)

# --- minimal stubs for tensor manager + fwd info -----------------------------

class StubTensorManager:
    """Records ref/deref calls so we can assert reference balance."""

    def __init__(self):
        self.refs: dict[tuple[str, str], int] = {}

    def increment_ref(self, request_id: str, uuid: str, n: int = 1):
        key = (request_id, uuid)
        self.refs[key] = self.refs.get(key, 0) + n

    def dereference(self, request_id: str, uuid: str, n: int = 1):
        key = (request_id, uuid)
        self.refs[key] = self.refs.get(key, 0) - n


def _fwd_info(graph_walk: str, partition: str = "default", fwd_index: int = 0):
    return CurrentForwardPassInfo(
        request_id="rid",
        graph_walk=graph_walk,
        requires_cfg=False,
        fwd_index=fwd_index,
        random_seed=0,
        max_tokens=128,
        sampling_config={},
        partition_name=partition,
    )


# --- fixtures ----------------------------------------------------------------

def _make_ar_walk_graph():
    """Single-worker, single-walk AR-shaped graph: prefill → ar_loop(ar_decode).

    The loop body's "token" output drives the loop-back AND (by name match in
    ``Loop.__post_init__``) the loop's terminal output to ``post_processor``
    on done. ``Loop.outputs`` entries whose names don't match a section-
    produced name are filtered out at construction.
    """
    return Sequential(sections=[
        GraphNode(
            name="prefill",
            input_names={"prompt"},
            outputs=[
                GraphEdge(name="token", next_node="ar_decode"),
                GraphEdge(name="kv_cache", next_node="ar_decode"),
            ],
        ),
        Loop(
            name="ar_loop",
            section=GraphNode(
                name="ar_decode",
                input_names={"token", "kv_cache"},
                outputs=[
                    GraphEdge(name="token", next_node="ar_decode"),
                    GraphEdge(name="kv_cache", next_node="ar_decode"),
                ],
            ),
            outputs=[GraphEdge(name="token", next_node="post_processor")],
            max_iters=10,
        ),
    ])


def _make_manager(wg_id="wg0", graph_walk="decode", worker_id="worker0"):
    """Build a WorkerGraphsManager with one WorkerGraphQueues + one request."""
    graph = _make_ar_walk_graph()
    worker_graph = WorkerGraph(
        section=graph,
        graph_walks={graph_walk},
        ranks=[0],
        worker_graph_id=wg_id,
    )
    queues = {
        wg_id: WorkerGraphQueues(
            worker_graph_id=wg_id,
            graph_walks={graph_walk},
            worker_graph=worker_graph,
            per_request_queues={},
            tensor_manager=StubTensorManager(),
        )
    }
    mgr = WorkerGraphsManager(
        queues=queues,
        per_request_info={},
        all_worker_graph_ids_to_graph_walks={wg_id: {graph_walk}},
        all_worker_graph_ids_to_nodes={wg_id: {"prefill", "ar_decode"}},
        all_worker_graph_ids_to_dyn_loops={wg_id: {"ar_loop"}},
        node_to_partition={"prefill": "default", "ar_decode": "default"},
    )
    mgr.add_request(
        request_id="rid",
        partition_worker_graph_ids=[wg_id],
        worker_graph_to_worker={wg_id: worker_id},
        current_fwd_info=_fwd_info(graph_walk),
    )
    return mgr, wg_id, graph_walk


# --- tests -------------------------------------------------------------------

def test_inverted_index_populated_at_init():
    mgr, wg_id, walk = _make_manager()
    # Both nodes should be indexed under the walk.
    assert mgr.walk_node_to_worker_graph_id[(walk, "prefill")] == wg_id
    assert mgr.walk_node_to_worker_graph_id[(walk, "ar_decode")] == wg_id
    # Unknown (walk, node) pairs should not be in the index.
    assert ("other_walk", "prefill") not in mgr.walk_node_to_worker_graph_id


def test_get_worker_graph_id_uses_inverted_index():
    mgr, wg_id, _ = _make_manager()
    assert mgr.get_worker_graph_id_for_node("rid", "prefill") == wg_id
    assert mgr.get_worker_graph_id_for_node("rid", "ar_decode") == wg_id


def test_get_worker_graph_id_raises_for_unknown_node():
    mgr, _, walk = _make_manager()
    # An unknown node either has no partition (KeyError on fwd_info lookup) or
    # an unknown (walk, node) in the inverted index (RuntimeError below). Both
    # paths surface as exceptions — what matters is that the manager doesn't
    # silently return a wrong wg id.
    # Add a fake partition mapping so we reach the inverted-index check.
    mgr.node_to_partition["mystery_node"] = "default"
    with pytest.raises(RuntimeError, match="Could not find worker graph"):
        mgr.get_worker_graph_id_for_node("rid", "mystery_node")


def test_mark_node_complete_returns_node_completion_output():
    mgr, wg_id, _ = _make_manager()
    # Ingest prompt → prefill, then complete prefill.
    leftovers = mgr.process_new_inputs("rid", [
        GraphEdge(name="prompt", next_node="prefill"),
    ])
    assert leftovers == []  # prefill is in this wg, edge claimed

    completion = mgr.mark_node_complete("rid", wg_id, "prefill")
    # Top-level GraphNode completion returns its outputs (token + kv_cache → ar_decode)
    # with no filtered signals (prefill isn't loop-managed).
    names = sorted((e.name, e.next_node) for e in completion.output_edges)
    assert names == [("kv_cache", "ar_decode"), ("token", "ar_decode")]
    assert completion.filtered_signals == set()


def test_process_new_inputs_leftovers_when_destination_unknown():
    mgr, _, _ = _make_manager()
    leftovers = mgr.process_new_inputs("rid", [
        GraphEdge(name="prompt", next_node="prefill"),
        GraphEdge(name="some_other_input", next_node="not_in_this_wg"),
    ])
    # The unknown-destination edge isn't claimed by any wg on this manager.
    assert len(leftovers) == 1
    assert leftovers[0].next_node == "not_in_this_wg"


def test_stop_loops_returns_loop_back_signal_set():
    mgr, wg_id, _ = _make_manager()
    # Drive prefill → ar_decode so the loop is active.
    mgr.process_new_inputs("rid", [GraphEdge(name="prompt", next_node="prefill")])
    mgr.mark_node_complete("rid", wg_id, "prefill")

    stopped = mgr.stop_loops(
        request_id="rid",
        partition="default",
        loop_names={"ar_loop"},
    )
    # ar_loop has two loop-back inputs: (token, ar_decode) and (kv_cache, ar_decode).
    assert stopped == {("token", "ar_decode"), ("kv_cache", "ar_decode")}
    # _finish_signal should be set on the live loop.
    wgio = mgr.queues[wg_id].per_request_queues["rid"]
    assert wgio.loops["ar_loop"]._finish_signal is True


def test_stop_loops_snapshots_loop_stop_times_when_req_info_provided():
    mgr, wg_id, walk = _make_manager()
    mgr.process_new_inputs("rid", [GraphEdge(name="prompt", next_node="prefill")])
    mgr.mark_node_complete("rid", wg_id, "prefill")
    fwd_info = mgr.get_fwd_info("rid", "default")

    mgr.stop_loops(
        request_id="rid",
        partition="default",
        loop_names={"ar_loop"},
        req_info=fwd_info,
        last_node_run="ar_decode",
    )
    # NestedLoopIndices snapshot should be in loop_stop_times.
    snapshot = fwd_info.loop_stop_times.get("ar_loop")
    assert snapshot is not None
    assert snapshot.wg_fwd_pass_idx == fwd_info.fwd_index
    assert snapshot.loop_name_order == ["ar_loop"]


def test_complete_loops_shim_drops_loop_back_on_loop_done():
    """End-to-end: drive prefill + 2 ar_decode iters with EOS on the 2nd;
    complete_loops on the final ar_decode should drop the loop-back signals
    from kept and append the loop's terminal outputs."""
    mgr, wg_id, _ = _make_manager()
    mgr.process_new_inputs("rid", [GraphEdge(name="prompt", next_node="prefill")])
    mgr.mark_node_complete("rid", wg_id, "prefill")
    # Route prefill's outputs back in.
    mgr.process_new_inputs("rid", [
        GraphEdge(name="token", next_node="ar_decode"),
        GraphEdge(name="kv_cache", next_node="ar_decode"),
    ])
    mgr.mark_node_complete("rid", wg_id, "ar_decode")  # advance: iter 0 done

    # Now request a stop on ar_loop, then complete the next iter.
    mgr.process_new_inputs("rid", [
        GraphEdge(name="token", next_node="ar_decode"),
        GraphEdge(name="kv_cache", next_node="ar_decode"),
    ])
    mgr.stop_loops("rid", "default", {"ar_loop"})
    ar_decode_outputs = list(
        mgr.queues[wg_id].per_request_queues["rid"].nodes["ar_decode"].outputs
    )
    result = mgr.complete_loops("rid", wg_id, ar_decode_outputs, "ar_decode")

    # Loop-back edges should be filtered out.
    filtered_names = sorted((e.name, e.next_node) for e in result.filtered_out)
    assert filtered_names == [("kv_cache", "ar_decode"), ("token", "ar_decode")]
    # Kept should include the loop's terminal output (token → post_processor)
    # since the loop's done branch returns ar_loop.outputs and the shim appends
    # any output edges whose (name, dest) is not in the caller's set.
    kept_dests = {e.next_node for e in result.kept}
    assert "post_processor" in kept_dests


def test_mark_node_complete_on_empty_outputs_node_flips_is_done():
    """Regression: nodes with no declared outputs (BAGEL prefill_text,
    vae_encoder) must still drive ``is_done`` to True via mark_node_complete.

    The worker's _store_outputs_and_finish_loops used to ``continue`` over
    empty-output nodes BEFORE calling complete_loops, so the registry never
    flipped → WORKER_GRAPHS_DONE never fired → t2i hung on prefill_text.
    """
    empty_outputs_graph = GraphNode(
        name="prefill_text",
        input_names={"text_inputs"},
        outputs=[],  # no declared outputs — KV-cache-only step
    )
    wg_id = "wg_empty"
    worker_graph = WorkerGraph(
        section=empty_outputs_graph,
        graph_walks={"prefill_text"},
        ranks=[0],
        worker_graph_id=wg_id,
    )
    mgr = WorkerGraphsManager(
        queues={wg_id: WorkerGraphQueues(
            worker_graph_id=wg_id,
            graph_walks={"prefill_text"},
            worker_graph=worker_graph,
            per_request_queues={},
            tensor_manager=StubTensorManager(),
        )},
        per_request_info={},
        all_worker_graph_ids_to_graph_walks={wg_id: {"prefill_text"}},
        all_worker_graph_ids_to_nodes={wg_id: {"prefill_text"}},
        all_worker_graph_ids_to_dyn_loops={wg_id: set()},
        node_to_partition={"prefill_text": "default"},
    )
    mgr.add_request(
        request_id="rid",
        partition_worker_graph_ids=[wg_id],
        worker_graph_to_worker={wg_id: "worker0"},
        current_fwd_info=_fwd_info("prefill_text"),
    )
    mgr.process_new_inputs("rid", [GraphEdge(name="text_inputs", next_node="prefill_text")])
    assert not mgr.queues[wg_id].is_done("rid")  # not done before complete
    mgr.mark_node_complete("rid", wg_id, "prefill_text")
    assert mgr.queues[wg_id].is_done("rid"), \
        "mark_node_complete on a no-output node must flip is_done"


def test_process_node_outputs_marks_wg_done_with_all_external_outputs():
    """Regression: a wg whose just-completed node emits only special-destination
    edges (EMPTY_DESTINATION / EMIT_TO_CLIENT / streaming) must still flip to
    ``completed_worker_graph_ids`` so the worker can fire WORKER_GRAPHS_DONE.

    Previously the inverted-index routing only checked ``is_done`` on wgs
    that ingested an edge in this call, so prefill walks whose outputs all
    leave the local wg (Orpheus prefill -> EMPTY_DESTINATION + streaming to
    SNAC) never reported done and the conductor hung.
    """
    # Build a 1-node wg whose single output goes to EMPTY_DESTINATION.
    from mstar.graph.special_destinations import EMPTY_DESTINATION
    single_node_graph = GraphNode(
        name="prefill",
        input_names={"prompt"},
        outputs=[
            GraphEdge(name="new_token", next_node=EMPTY_DESTINATION,
                      conductor_new_token=True, persist=True),
        ],
    )
    wg_id = "wg_external"
    worker_graph = WorkerGraph(
        section=single_node_graph,
        graph_walks={"prefill"},
        ranks=[0],
        worker_graph_id=wg_id,
    )
    mgr = WorkerGraphsManager(
        queues={wg_id: WorkerGraphQueues(
            worker_graph_id=wg_id,
            graph_walks={"prefill"},
            worker_graph=worker_graph,
            per_request_queues={},
            tensor_manager=StubTensorManager(),
        )},
        per_request_info={},
        all_worker_graph_ids_to_graph_walks={wg_id: {"prefill"}},
        all_worker_graph_ids_to_nodes={wg_id: {"prefill"}},
        all_worker_graph_ids_to_dyn_loops={wg_id: set()},
        node_to_partition={"prefill": "default"},
    )
    mgr.add_request(
        request_id="rid",
        partition_worker_graph_ids=[wg_id],
        worker_graph_to_worker={wg_id: "worker0"},
        current_fwd_info=_fwd_info("prefill"),
    )
    mgr.process_new_inputs("rid", [GraphEdge(name="prompt", next_node="prefill")])
    mgr.mark_node_complete("rid", wg_id, "prefill")

    routing = mgr.process_node_outputs(
        "rid",
        outputs=list(mgr.queues[wg_id].per_request_queues["rid"].nodes["prefill"].outputs),
        graph_walk="prefill",
    )
    assert wg_id in routing.completed_worker_graph_ids, \
        "prefill wg with only EMPTY_DESTINATION outputs must still report done"
