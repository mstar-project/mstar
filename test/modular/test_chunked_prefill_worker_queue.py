"""Phase 2 chunked-prefill regression: worker must re-queue popped GraphNodes
for non-terminal rids whose per-rid output is empty.

Reproduces the production-stack hang where text-to-text requests with
``scheduler_owns_chunking=true`` get stuck server-side because the
non-terminal chunk's GraphNode is consumed from the ready queue but never
re-added — the rid's queue ends up empty, the scheduler can't find a ready
node, and the SSE response stream never closes (client sees aiohttp
TransferEncodingError after timeout).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from mminf.engine.base import NodeOutput
from mminf.graph.base import GraphNode
from mminf.worker.micro_scheduler import ScheduledBatch
from mminf.worker.worker import Worker


def _make_worker_with_mocks():
    """Construct a Worker shell with the dependencies _store_outputs_and_finish_loops
    actually touches. We bypass __init__ because it spawns conductor + workers."""
    worker = Worker.__new__(Worker)
    worker.enable_nvtx = False
    worker.tensor_manager = MagicMock()
    worker.tensor_manager.store_and_populate_graph_edges.return_value = []
    worker.worker_graphs_manager = MagicMock()
    worker.worker_graphs_manager.get_worker_graph_id_for_node.return_value = "wg0"
    worker.worker_graphs_manager.get_waiting_node.return_value = None
    worker.worker_graphs_manager.complete_loops.return_value = MagicMock(
        kept=[], filtered_out=[]
    )
    worker._queue = MagicMock()
    worker.worker_graphs_manager.queues = {"wg0": worker._queue}
    return worker


def _make_batch(is_terminal_per_request: dict[str, bool]) -> ScheduledBatch:
    graphnode = GraphNode(name="Thinker", input_ids=["text_inputs"], outputs=[])
    rids = list(is_terminal_per_request.keys())
    return ScheduledBatch(
        node_name="Thinker",
        graph_walk="thinker_step",
        node_objects={rid: graphnode for rid in rids},
        request_to_worker_graph={rid: "wg0" for rid in rids},
        is_terminal_per_request=is_terminal_per_request,
        prefill_chunk_sizes={},
    )


def test_non_terminal_rid_with_empty_output_re_queues_node():
    """Non-terminal rid + empty per-rid output (text_to_text postprocess
    drops everything) ⇒ popped GraphNode must be pushed back so next chunk
    can run.

    Without this, the rid's queue stays empty after the popped node, the
    scheduler can't find ready nodes for the rid, and the request hangs.
    """
    worker = _make_worker_with_mocks()
    batch = _make_batch(
        is_terminal_per_request={"rid_term": True, "rid_nonterm": False}
    )
    output = NodeOutput(per_request_output_tensors={
        "rid_term": {"new_token": [object()]},  # terminal: has token
        "rid_nonterm": {},  # non-terminal text-to-text: postprocess dropped everything
    })
    filtered_outputs_per_request = {"rid_term": [], "rid_nonterm": []}

    worker._store_outputs_and_finish_loops(
        batch, output, filtered_outputs_per_request
    )

    # The non-terminal rid's GraphNode must have been pushed back.
    push_back_calls = worker._queue.push_back_node.call_args_list
    pushed_rids = [call.args[0] for call in push_back_calls]
    assert "rid_nonterm" in pushed_rids, (
        "Non-terminal rid's GraphNode was not re-queued. The rid's ready "
        "queue is now empty and the scheduler can't pick it up next step. "
        f"push_back_node calls: {push_back_calls}"
    )
    # Sanity: terminal rid is NOT pushed back (its node advanced via complete_loops)
    assert "rid_term" not in pushed_rids, (
        "Terminal rid was incorrectly re-queued; it should advance via complete_loops."
    )


def test_terminal_rid_with_output_advances_normally():
    """Sanity: terminal rid with non-empty output goes through complete_loops
    and is NOT pushed back."""
    worker = _make_worker_with_mocks()
    batch = _make_batch(is_terminal_per_request={"rid_term": True})
    output = NodeOutput(per_request_output_tensors={
        "rid_term": {"new_token": [object()]},
    })
    filtered_outputs_per_request = {"rid_term": []}

    worker._store_outputs_and_finish_loops(
        batch, output, filtered_outputs_per_request
    )

    worker.worker_graphs_manager.complete_loops.assert_called_once()
    worker._queue.push_back_node.assert_not_called()


def test_empty_is_terminal_dict_preserves_legacy_behavior():
    """Sanity: when is_terminal_per_request is empty (Phase 1 / single-walk
    batches), all rids are treated as terminal — no push_back_node fires
    even for empty-output rids (preserves Talker non-last-prefill /
    KV-cache-only-step behavior)."""
    worker = _make_worker_with_mocks()
    batch = _make_batch(is_terminal_per_request={})
    # rid still in node_objects via _make_batch defaulting empty dict
    batch = ScheduledBatch(
        node_name="Talker_LLM",
        graph_walk="talker_prefill",
        node_objects={"rid_legacy": GraphNode(name="Talker_LLM", input_ids=[], outputs=[])},
        request_to_worker_graph={"rid_legacy": "wg0"},
        is_terminal_per_request={},  # legacy: empty dict ⇒ all terminal
        prefill_chunk_sizes={},
    )
    output = NodeOutput(per_request_output_tensors={"rid_legacy": {}})
    filtered_outputs_per_request = {"rid_legacy": []}

    worker._store_outputs_and_finish_loops(
        batch, output, filtered_outputs_per_request
    )

    # Empty output + legacy (treated as terminal) ⇒ existing skip-path,
    # no push_back fires.
    worker._queue.push_back_node.assert_not_called()
