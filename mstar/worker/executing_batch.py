"""Building a step's ``ExecutingBatch`` from its rows' inputs: shared by the
worker's own batches and its speculative ones."""
from __future__ import annotations

from collections.abc import Mapping

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import StepContext
from mstar.graph.loop_indices import LoopLayout
from mstar.graph.runtime.base import GraphRuntime
from mstar.model.submodule_base import InputMetadata
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.stream_buffers import StreamBufferManager


def input_metadata_and_fwd_info(
    request_state: RequestStateManager,
    streams: StreamBufferManager,
    node_name: str,
    batch_partition: str,
    per_request_inputs: dict[int, NameToTensorList],
    increment_loop_rids: list[int] | None = None,
    increment_loop: LoopLayout | None = None,
) -> tuple[dict[int, InputMetadata], dict[int, CurrentForwardPassInfo]]:
    """Per row, its forward-pass info and an ``InputMetadata`` with its own
    copy of the loop counters (one iteration on for ``increment_loop_rids``)
    and the stream chunks it consumes."""
    meta = {}
    fwd_info = {}
    possible = streams.possible()
    continuing = set(increment_loop_rids or ())
    for rid, inputs in per_request_inputs.items():
        req_info = request_state.get_fwd_info(rid, batch_partition)
        fwd_info[rid] = req_info
        stream_chunks = {}
        if possible and (chunks := streams.chunks_for(rid, node_name, inputs)):
            stream_chunks = chunks
        counts = req_info.dynamic_loop_iter_counts
        projected = (
            increment_loop.advance(counts)
            if increment_loop is not None and rid in continuing
            else counts.copy()
        )
        meta[rid] = InputMetadata(stream_chunks=stream_chunks, dynamic_loop_iter_counts=projected)
    return meta, fwd_info


def stamp_iteration_context(
    runtime: GraphRuntime, engine: Engine, node: str, walk: str,
    infos: Mapping[int, CurrentForwardPassInfo], metadata: dict[int, InputMetadata],
    request_to_worker_graph: Mapping[int, int],
    request_walks: Mapping[int, str] | None = None,
) -> None:
    """Snapshot the graph's actual pass before an immutable step is submitted.

    A speculative ancestor transition already projected the Loop counters;
    it remains in the same worker-graph pass. Ordinary and speculative input
    preparation share this boundary. Nodes without actual cursors query no state.
    """
    layouts = engine.iteration_layouts(node)
    if not layouts:
        return
    walks = request_walks or {}
    selected = {
        rid: request_to_worker_graph[rid] for rid in metadata
        if walks.get(rid, walk) in layouts
    }
    if selected:
        for rid, graph_pass in zip(selected, runtime.graph_passes(selected), strict=True):
            loop = layouts[walks.get(rid, walk)]
            before = infos[rid].dynamic_loop_iter_counts
            projected = metadata[rid].dynamic_loop_iter_counts
            entering = any(projected.get(name, 0) != before.get(name, 0) for name in loop.ancestors)
            origin = projected.get(loop.name, 0) if entering else before.get(loop.name, 0)
            metadata[rid] = metadata[rid]._replace(graph_pass=graph_pass, iteration_origin=origin)


def make_executing_batch(
    streams: StreamBufferManager,
    node_name: str,
    graph_walk: str,
    request_ids: list[int],
    per_request_input_tensors: dict[int, NameToTensorList],
    per_request_info: dict[int, CurrentForwardPassInfo],
    final_edges: dict[int, set[str]] | None = None,
    per_request_input_metadata: dict[int, InputMetadata] | None = None,
    request_walks: dict[int, str] | None = None,
    engine: Engine | None = None,
    chunk_ranges: dict[int, ChunkRange] | None = None,
    repeats: int = 1,
) -> ExecutingBatch:
    """One step's batch, with the step context the engine drives it through.

    ``final_edges`` maps a rid to the streams whose final chunk the step
    consumes. ``engine`` orders a split walk's rows (``Engine.order_split_rows``).
    The context starts unleased and eager; a slot is reserved later, once the
    real token count is known.
    """
    final_stream_rids, partition_done_rids = streams.settle_final(
        node_name, final_edges or {},
    )
    if request_walks and engine is not None:
        # Every path to a mixed step comes through here, a TP follower's too,
        # so each rank orders the leader's rows alike.
        request_ids = engine.order_split_rows(node_name, graph_walk, request_ids, request_walks)
    return ExecutingBatch(
        node_name=node_name,
        per_request_info=per_request_info,
        per_request_input_tensors=per_request_input_tensors,
        final_stream_rids=final_stream_rids,
        partition_done_rids=partition_done_rids,
        per_request_input_metadata=per_request_input_metadata or {},
        chunk_ranges=chunk_ranges or {},
        step_context=StepContext(
            request_ids=tuple(request_ids),
            graph_walk=graph_walk,
            slot=0,
            capture=False,
            request_walks=request_walks or {},
            # the ask: a TP follower takes its leader's, which admit narrows alike
            repeats=repeats,
        ),
    )
