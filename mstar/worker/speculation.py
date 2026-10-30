"""Preparing and building the worker's speculative step.

While step N runs, the worker speculates step N+1 (see the async-worker
skill). ``Worker._try_speculate_next`` decides what to speculate and merges in
fresh rows from the scheduler; the stages on either side of that live here:

* ``SpeculationPreparer`` asks the graph runtime which of N's rows continue
  and what inputs they already hold.
* ``SpeculationBuilder`` turns the prepared rows into the ``Speculation`` the
  main loop runs, per row or as columns (``spec_rows``).

Both work through a ``SpeculationContext``: the worker's runtime, request
state, scheduler and stream buffers.
"""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mstar.communication.tensors import NameToTensorList, TensorCommunicationManager
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import StepContext
from mstar.graph.runtime.base import (
    EdgeSpec,
    GraphRuntime,
    SpeculationOutput,
    SpeculationPrepInput,
)
from mstar.streaming.stream_buffer import StreamingEdge
from mstar.utils.profiler import PhaseLaps
from mstar.worker.executing_batch import input_metadata_and_fwd_info, make_executing_batch
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.spec_rows import DeferredRowInputs, loop_counter_snapshots
from mstar.worker.stream_buffers import StreamBufferManager

if TYPE_CHECKING:
    from mstar.worker.worker import PendingBatch


@dataclass
class Speculation:
    """A speculative step N+1, built while step N runs.

    ``consumed_edges`` are the ``(name, next_node)`` pairs of N's outputs that
    it takes, spliced in by ``Worker._thread_outputs_to_speculative`` once N
    has produced them. ``spec_id`` names the runtime's staged stream ingests,
    settled exactly once by ``Worker._settle_speculation``. ``submitted``: the
    GPU thread owns its rows now, so an error fails them rather than giving
    them back. ``tp_seq`` is its TP broadcast seq (the leader's, assigned at
    broadcast; a follower's, from the head).
    """

    scheduled_batch: ScheduledBatch
    node_batch: ExecutingBatch
    consumed_edges: set[tuple[str, str]]
    continuing_rids: set[int]
    partition: str
    is_new_iter: bool
    is_same_node: bool
    # rid -> the stream chunks its rows took
    consumed_streaming_edges: dict[int, list[StreamingEdge]] = field(default_factory=dict)
    spec_id: int = 0
    is_yield_away: bool = False
    loop_name: str | None = None
    dropped: set[int] = field(default_factory=set)
    plan_future: Future | None = None
    submitted: bool = False
    tp_seq: int = -1


@dataclass(frozen=True)
class SpeculationContext:
    """What preparing and building a speculation reads: the worker's."""

    graph_runtime: GraphRuntime
    tensor_manager: TensorCommunicationManager
    request_state: RequestStateManager
    scheduler: MicroScheduler
    streams: StreamBufferManager


@dataclass(slots=True)
class PreparedSpeculation:
    """The continuing rows of a speculation, as ``SpeculationPreparer`` found
    them. The caller adds fresh rows to the two per-row dicts."""

    # N's rows that run the speculative step, in N's order
    ready_rids: list[int]
    spec_id: int
    # Set when the continuing rows are carried as columns: the worker graph
    # they share, and no per-row entries for them below
    rows_wg_id: int | None
    per_request_inputs: dict[int, NameToTensorList] = field(default_factory=dict)
    request_to_worker_graph: dict[int, int] = field(default_factory=dict)
    consumed_streaming_edges: dict[int, list[StreamingEdge]] = field(default_factory=dict)


class SpeculationPreparer:
    """Finds the continuing rows of a speculation and the inputs they already
    hold, in one runtime call.

    As columns (``prep_spec_rows``) whenever no streams are open, since then
    there are no chunks to poll and ingest; otherwise per row, with the
    stream buffers polled first (``prep_spec_rids``). A TP follower never comes
    here: it rebuilds whatever rows the leader's head names.
    """

    def __init__(self, ctx: SpeculationContext):
        self._ctx = ctx

    def rows_eligible(self) -> bool:
        return not self._ctx.streams.possible()

    def prepare(
        self, batch_N: ScheduledBatch, spec_node_name: str, graph_walk: str,
        candidates: list[int], max_continuing: int | None, laps: PhaseLaps,
    ) -> PreparedSpeculation:
        if self.rows_eligible():
            return self._prepare_rows(
                batch_N, spec_node_name, graph_walk, candidates, max_continuing, laps,
            )
        return self._prepare_dicts(
            batch_N, spec_node_name, graph_walk, candidates, max_continuing, laps,
        )

    def _prepare_rows(
        self, batch_N: ScheduledBatch, spec_node_name: str, graph_walk: str,
        candidates: list[int], max_continuing: int | None, laps: PhaseLaps,
    ) -> PreparedSpeculation:
        ctx = self._ctx
        rows = ctx.graph_runtime.prep_spec_rows(
            spec_node_name, batch_N.node_name, graph_walk, candidates, max_continuing,
        )
        laps.lap("prep_spec_rows")
        prep = PreparedSpeculation(rows.ready_rids, rows.spec_id, rows.wg_id)
        if rows.input_edges is not None and rows.ready_rids:
            # an input that is not the in-flight step's: per-row dicts
            prep.per_request_inputs.update(rows.input_edges.to_input_tensors(
                ctx.tensor_manager.get_tensor, rows.ready_rids,
            ).by_rid)
            prep.request_to_worker_graph.update(dict.fromkeys(rows.ready_rids, rows.wg_id))
            prep.rows_wg_id = None
        return prep

    def _prepare_dicts(
        self, batch_N: ScheduledBatch, spec_node_name: str, graph_walk: str,
        candidates: list[int], max_continuing: int | None, laps: PhaseLaps,
    ) -> PreparedSpeculation:
        """``prep_spec_rids`` with the stream polls, filling the per-row dicts
        for the ready rids."""
        ctx = self._ctx
        streams = ctx.streams
        # Polling the StreamBuffers stays on this side: they hold real tensors.
        polled: list[tuple[int, StreamingEdge]] = []
        per_rid_counts: list[int] = []
        if streams.possible():
            for r in candidates:
                edges = streams.poll_for(r, spec_node_name)
                polled.extend((r, e) for e in edges)
                per_rid_counts.append(len(edges))
        else:
            per_rid_counts = [0] * len(candidates)

        out = ctx.graph_runtime.prep_spec_rids(SpeculationPrepInput(
            spec_node_name=spec_node_name,
            curr_node_name=batch_N.node_name,
            graph_walk=graph_walk,
            rids=candidates,
            room_for_continuing=max_continuing,
            streaming_edges=[
                EdgeSpec(
                    signal=e.name, next_node=e.next_node,
                    uuids=[i.uuid for i in e.tensor_info],
                    is_final_streaming_chunk=e._final_stream_chunk,
                ) for _r, (e, _chunk) in polled
            ],
            streaming_edges_per_rid=per_rid_counts,
        ))
        laps.lap("prep_spec_rids")

        prep = PreparedSpeculation(out.ready_rids, out.spec_id, None)
        # Anything not consumed goes back to its StreamBuffer, so a later
        # scheduling of this node picks it up normally.
        consumed = set(out.consumed_streaming_edge_idxs)
        for i, (r, se) in enumerate(polled):
            if i not in consumed:
                streams.return_edge(r, se)
            else:
                streams.mark_ingested(r, se)
                prep.consumed_streaming_edges.setdefault(r, []).append(se)

        # `ready_rids` seeds it, so a ready rid with no prepped edge still gets
        # an empty mapping -- what slicing a zero-length run used to give.
        prepped_inputs = out.input_edges.to_input_tensors(
            ctx.tensor_manager.get_tensor, out.ready_rids,
        ).by_rid
        for i, r in enumerate(out.ready_rids):
            prep.per_request_inputs[r] = prepped_inputs[r]
            prep.request_to_worker_graph[r] = out.wg_ids[i]
        return prep


class SpeculationBuilder:
    """Packages prepared rows into the ``Speculation`` the main loop runs:
    per row (``assemble``, which a TP follower's rebuilt head also takes), or
    as columns (``assemble_rows``)."""

    def __init__(self, ctx: SpeculationContext):
        self._ctx = ctx
        # (source node, dest node, walk) -> ``get_consumed_edges``, which is
        # structural
        self._consumed_edges_cache: dict[tuple[str, str, str], frozenset] = {}
        # (node, walk) -> its output signals, also structural
        self._walk_signals_cache: dict[tuple[str, str], list[str]] = {}

    def build(
        self, pending: PendingBatch, spec_target: SpeculationOutput,
        prep: PreparedSpeculation, other_walks: dict[int, str],
        fresh_final_edges: dict[int, set[str]],
    ) -> Speculation:
        """The speculation from ``prep`` plus the fresh rows merged into its
        per-row dicts; ``other_walks`` maps the fresh rows of another walk (a
        mixing chain) to theirs, and ``fresh_final_edges`` the fresh rows whose
        final stream chunk was ingested before this step to its edges."""
        if prep.rows_wg_id is not None and not other_walks and not fresh_final_edges:
            # the dicts hold only the fresh rows here; a fresh row's final
            # stream chunk takes the general assembly, which carries it
            return self.assemble_rows(
                pending, spec_target, prep.ready_rids, prep.rows_wg_id,
                fresh_request_to_worker_graph=prep.request_to_worker_graph,
                fresh_inputs=prep.per_request_inputs,
            )
        per_request_inputs = prep.per_request_inputs
        request_to_worker_graph = prep.request_to_worker_graph
        if prep.rows_wg_id is not None:
            # a mixing chain took fresh rows of another walk: the general
            # assembly, with the continuing rows ahead of the fresh ones
            per_request_inputs = {**{r: {} for r in prep.ready_rids}, **per_request_inputs}
            request_to_worker_graph = {
                **dict.fromkeys(prep.ready_rids, prep.rows_wg_id), **request_to_worker_graph,
            }
        return self.assemble(
            pending, spec_target, request_to_worker_graph, per_request_inputs,
            prep.consumed_streaming_edges, prep.ready_rids, fresh_final_edges,
            is_same_node=spec_target.node_name == pending.node_name, spec_id=prep.spec_id,
            # a per-row map only when the step mixes walks
            request_walks={
                **dict.fromkeys(request_to_worker_graph, spec_target.graph_walk), **other_walks,
            } if other_walks else None,
        )

    def assemble(
        self,
        pending: PendingBatch,
        spec_target: SpeculationOutput,
        request_to_worker_graph: dict[int, int],
        per_request_inputs: dict[int, NameToTensorList],
        consumed_streaming_edges: dict[int, list[StreamingEdge]],
        continuing: list[int],
        fresh_final_edges: dict[int, set[str]],
        *,
        is_same_node: bool,
        spec_id: int = 0,
        tp_seq: int = -1,
        request_walks: dict[int, str] | None = None,
    ) -> Speculation:
        """Package prepared rids (batch order) into the ``Speculation`` the main
        loop runs. Leader and follower differ only in how they pick the rids.

        ``fresh_final_edges`` is ``final_edges`` (see ``make_executing_batch``)
        for the rids that did not come from batch N: their final chunks were
        ingested before this step, not polled by it."""
        ctx = self._ctx
        spec_node = spec_target.node_name
        request_ids = list(request_to_worker_graph)
        # rows of one walk (the usual chain) need no per-row map
        label, request_walks = (
            ctx.scheduler.label_for(spec_node, request_walks) if request_walks
            else (spec_target.graph_walk, {})
        )
        spec_batch = ScheduledBatch(
            node_name=spec_node,
            graph_walk=label,
            request_to_worker_graph=request_to_worker_graph,
            request_walks=request_walks,
            # Resolved here, on the main thread, so the GPU thread never asks
            # the runtime, which may be mid-call with the GIL released.
            walk_output_signals={
                walk: self._walk_output_signals(spec_node, walk, spec_target)
                for walk in set(request_walks.values())
            },
            # Reported by whichever step chose the target -- speculate_node on
            # the leader, get_spec_target on a follower -- so the hot path does
            # not cross back into the runtime once per forward pass.
            output_signals=spec_target.output_signals,
            tp_seq=tp_seq,
        )
        input_meta, per_request_info = input_metadata_and_fwd_info(
            ctx.request_state, ctx.streams, node_name=spec_node, batch_partition=pending.partition,
            per_request_inputs=per_request_inputs,
            increment_loop_rids=continuing if spec_target.is_new_loop_iter else None,
            increment_loop_name=spec_target.loop_name
        )
        spec_node_batch = make_executing_batch(
            ctx.streams, node_name=spec_node,
            graph_walk=label,
            request_ids=request_ids,
            per_request_input_tensors=per_request_inputs,
            per_request_info=per_request_info,
            final_edges={
                rid: {se.edge.name for se in edges if se.chunk.is_final}
                for rid, edges in consumed_streaming_edges.items()
                if any(se.chunk.is_final for se in edges)
            } | fresh_final_edges,
            per_request_input_metadata=input_meta,
            request_walks=request_walks,
        )
        return Speculation(
            scheduled_batch=spec_batch,
            node_batch=spec_node_batch,
            # Outputs of batch_N the spec batch consumes: every edge of the
            # in-flight node whose destination is the spec node.
            consumed_edges=ctx.graph_runtime.get_consumed_edges(
                pending.node_name, spec_node, spec_target.graph_walk,
            ),
            continuing_rids=set(continuing),
            partition=pending.partition,
            is_new_iter=spec_target.is_new_loop_iter,
            is_same_node=is_same_node,
            loop_name=spec_target.loop_name,
            consumed_streaming_edges=consumed_streaming_edges,
            spec_id=spec_id,
            tp_seq=tp_seq,
        )

    def assemble_rows(
        self,
        pending: PendingBatch,
        spec_target: SpeculationOutput,
        continuing: list[int],
        wg_id: int,
        *,
        fresh_request_to_worker_graph: dict[int, int],
        fresh_inputs: dict[int, NameToTensorList],
    ) -> Speculation:
        """``assemble`` for a step prepped as columns (``prep_spec_rows``): the
        continuing rows' inputs all come from the in-flight step, so they need
        no per-row work beyond the loop-counter snapshot until N's outputs are
        threaded in. Fresh rows of the same walk follow them with the inputs
        their pop gathered.

        Builds the same batch as ``assemble``, with C-level maps: every rid's
        ``per_request_info``, ``per_request_input_metadata`` (its own copy of
        the loop counters) and wrapped info; the inputs as a
        ``DeferredRowInputs``, which threading fills in once N's outputs are
        known (``Worker._thread_outputs_to_speculative``).
        """
        spec_node, walk = spec_target.node_name, spec_target.graph_walk
        request_ids = continuing + list(fresh_request_to_worker_graph)
        request_to_worker_graph = dict.fromkeys(continuing, wg_id)
        request_to_worker_graph.update(fresh_request_to_worker_graph)
        infos = self._ctx.request_state.fwd_infos(request_ids, pending.partition)
        metadata = loop_counter_snapshots(
            infos,
            spec_target.loop_name if spec_target.is_new_loop_iter else None,
            n_next_iter=len(continuing),
        )
        inputs = DeferredRowInputs(request_ids)
        for rid, rid_inputs in fresh_inputs.items():
            inputs[rid] = rid_inputs
        spec_batch = ScheduledBatch(
            node_name=spec_node,
            graph_walk=walk,
            request_to_worker_graph=request_to_worker_graph,
            output_signals=spec_target.output_signals,
        )
        spec_node_batch = ExecutingBatch(
            node_name=spec_node,
            per_request_info=dict(zip(request_ids, infos, strict=True)),
            per_request_input_tensors=inputs,
            per_request_input_metadata=dict(zip(request_ids, metadata, strict=True)),
            step_context=StepContext(
                request_ids=tuple(request_ids), graph_walk=walk, slot=0,
                capture=False, request_walks={},
            ),
        )
        return Speculation(
            scheduled_batch=spec_batch,
            node_batch=spec_node_batch,
            consumed_edges=self._consumed_edges(pending.node_name, spec_node, walk),
            continuing_rids=set(continuing),
            partition=pending.partition,
            is_new_iter=spec_target.is_new_loop_iter,
            is_same_node=spec_node == pending.node_name,
            loop_name=spec_target.loop_name,
        )

    def _walk_output_signals(
        self, node_name: str, walk: str, spec_target: SpeculationOutput,
    ) -> list[str]:
        if walk == spec_target.graph_walk:
            return list(spec_target.output_signals)
        key = (node_name, walk)
        signals = self._walk_signals_cache.get(key)
        if signals is None:
            signals = self._walk_signals_cache[key] = list(
                self._ctx.graph_runtime.get_output_signals(node_name, walk)
            )
        return list(signals)

    def _consumed_edges(
        self, source_node: str, dest_node: str, graph_walk: str,
    ) -> frozenset[tuple[str, str]]:
        key = (source_node, dest_node, graph_walk)
        edges = self._consumed_edges_cache.get(key)
        if edges is None:
            edges = self._consumed_edges_cache[key] = frozenset(
                self._ctx.graph_runtime.get_consumed_edges(*key)
            )
        return edges
