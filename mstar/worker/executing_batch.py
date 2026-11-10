"""Building a step's ``ExecutingBatch`` from its rows' inputs: shared by the
worker's own batches and its speculative ones."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import StepContext
from mstar.model.submodule_base import InputMetadata
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.scheduled_batch import ScheduledBatch
from mstar.worker.services import WorkerServices
from mstar.worker.stream_buffers import StreamBufferManager

if TYPE_CHECKING:
    from mstar.worker.rebuild import Rebuilds

logger = logging.getLogger(__name__)


def input_metadata_and_fwd_info(
    request_state: RequestStateManager,
    streams: StreamBufferManager,
    node_name: str,
    batch_partition: str,
    per_request_inputs: dict[int, NameToTensorList],
    increment_loop_rids: list[int] | None = None,
    increment_loop_name: str | None = None,
) -> tuple[dict[int, InputMetadata], dict[int, CurrentForwardPassInfo]]:
    """Per row, its forward-pass info and an ``InputMetadata`` with its own
    copy of the loop counters (one iteration on for ``increment_loop_rids``)
    and the stream chunks it consumes."""
    meta = {}
    fwd_info = {}
    possible = streams.possible()
    for rid, inputs in per_request_inputs.items():
        req_info = request_state.get_fwd_info(rid, batch_partition)
        fwd_info[rid] = req_info
        stream_chunks = {}
        if possible and (chunks := streams.chunks_for(rid, node_name, inputs)):
            stream_chunks = chunks
        meta[rid] = InputMetadata(
            stream_chunks=stream_chunks,
            dynamic_loop_iter_counts=req_info.dynamic_loop_iter_counts.copy()
        )

    if increment_loop_rids is not None and increment_loop_name is not None:
        for rid in increment_loop_rids:
            meta[rid].dynamic_loop_iter_counts[increment_loop_name] = \
                meta[rid].dynamic_loop_iter_counts.get(increment_loop_name, 0) + 1
    return meta, fwd_info


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
    rebuild_rids: frozenset[int] = frozenset(),
    rebuilds: Rebuilds | None = None,
) -> ExecutingBatch:
    """One step's batch, with the step context the engine drives it through.

    ``final_edges`` maps a rid to the streams whose final chunk the step
    consumes. ``engine`` orders a split walk's rows (``Engine.order_split_rows``).
    ``rebuilds`` gives the ``rebuild_rids`` rows their walk and inputs.
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
    input_ready = ()
    if rebuild_rids:
        input_ready = rebuilds.fill(
            node_name, rebuild_rids, chunk_ranges or {}, per_request_input_tensors, per_request_info,
        )
    return ExecutingBatch(
        node_name=node_name,
        per_request_info=per_request_info,
        per_request_input_tensors=per_request_input_tensors,
        final_stream_rids=final_stream_rids,
        partition_done_rids=partition_done_rids,
        per_request_input_metadata=per_request_input_metadata or {},
        chunk_ranges=chunk_ranges or {},
        input_ready=input_ready,
        step_context=StepContext(
            request_ids=tuple(request_ids),
            graph_walk=graph_walk,
            slot=0,
            capture=False,
            request_walks=request_walks or {},
            rebuild_rids=rebuild_rids,
        ),
    )


def build_executing_batch(services: WorkerServices, batch: ScheduledBatch) -> ExecutingBatch:
    """Gather input tensors from tensor_manager for all requests in the batch."""
    batch_partition = services.request_state.get_partition_for_node(batch.node_name)

    # One walk of the columns for both: each rid's inputs as tensors, and
    # per rid, the edges that carried a stream's final chunk. Only this side can
    # turn a uuid back into a tensor, which is why the runtime reports uuids
    # and the resolution happens here.
    #
    # Seeded with the batch's rids rather than just the ones that have
    # edges: a rid with nothing ready still needs an entry, because
    # per_request_info is keyed off these and the engine indexes it by rid.
    # skip_missing: a backstop, not a fix. A live rid holding a freed uuid
    # is a bug (an abandoned speculation handing back a still-ingested
    # chunk is one), but resolving the batch in one call means one such rid
    # would otherwise fail every other request in the step -- 64 duplex
    # sessions for one bad chunk. Take out just that rid and report it, so
    # the failure is attributed and loud.
    per_request_inputs, final_edges, unresolved = (
        services.streams.input_tensors(
            batch.input_edges, batch.node_name, batch.request_to_worker_graph,
            skip_missing=True,
        )
    )
    for rid in unresolved:
        # Before the get_fwd_info loop below, so a rid that is going to be
        # failed anyway is not asked for forward-pass state as well.
        per_request_inputs.pop(rid, None)
        final_edges.pop(rid, None)
    per_request_input_metadata, per_request_info = input_metadata_and_fwd_info(
        services.request_state, services.streams,
        node_name=batch.node_name, batch_partition=batch_partition,
        per_request_inputs=per_request_inputs,
    )

    node_batch = make_executing_batch(
        services.streams,
        node_name=batch.node_name,
        graph_walk=batch.graph_walk,
        request_ids=[
            rid for rid in batch.request_to_worker_graph
            if rid not in unresolved
        ],
        per_request_input_tensors=per_request_inputs,
        per_request_info=per_request_info,
        final_edges=final_edges,
        per_request_input_metadata=per_request_input_metadata,
        request_walks=batch.request_walks,
        chunk_ranges=batch.chunk_ranges,
        engine=services.engine_manager.get_engine(batch.node_name) if batch.request_walks else None,
        rebuild_rids=batch.rebuild_rids, rebuilds=services.rebuilds,
    )
    node_batch.read_check_rids = batch.read_check_rids
    for rid in unresolved:
        # Reported the way a per-rid stage reports: left in
        # ``failed_requests`` for the run loop to drop and fail, so the
        # client gets an error instead of a session that silently stops.
        logger.error(
            "Worker %s: request %s has an input tensor the store no longer "
            "holds; failing it and running %s without it",
            services.worker_id, rid, batch.node_name,
        )
        node_batch.register_failure(
            rid, KeyError(f"input tensor missing for {batch.node_name}"),
        )
    return node_batch
