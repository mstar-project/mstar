"""Preparing and building the worker's speculative step.

While step N runs, the worker speculates step N+1 (see the async-worker
skill). ``Speculator`` decides what to speculate and merges in fresh rows from
the scheduler, with a stage on either side of that:

* ``SpeculationPreparer`` asks the graph runtime which of N's rows continue
  and what inputs they already hold.
* ``SpeculationBuilder`` turns the prepared rows into the ``Speculation`` the
  main loop runs, per row or as columns (``spec_rows``).

Both work through a ``SpeculationContext``: the worker's runtime, request
state, scheduler and stream buffers.
"""
from __future__ import annotations

import logging
import time as _time
from collections.abc import Collection
from concurrent.futures import Future
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

from mstar.communication.tensors import NameToTensorList, TensorCommunicationManager
from mstar.engine.chunking import ChunkRange
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import StepContext
from mstar.graph.runtime.base import (
    GraphRuntime,
    SpeculationOutput,
    SpeculationPrepInput,
)
from mstar.streaming.stream_buffer import StreamingEdge
from mstar.utils.profiler import PhaseLaps, range_pop, range_push
from mstar.worker.admission import Admission
from mstar.worker.executing_batch import build_executing_batch, input_metadata_and_fwd_info, make_executing_batch
from mstar.worker.lockstep import Lockstep
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.phase_timer import PhaseTimer
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.scheduled_batch import tp_chunk_ranges, unfinished_chunk_rids
from mstar.worker.services import WorkerLayout, WorkerServices
from mstar.worker.spec_rows import DeferredRowInputs, loop_counter_snapshots
from mstar.worker.stream_buffers import StreamBufferManager

if TYPE_CHECKING:
    from mstar.worker.engine_manager import EngineManager
    from mstar.worker.step_executor import PendingBatch


logger = logging.getLogger(__name__)


class SpeculationTarget(NamedTuple):
    info: SpeculationOutput
    walk_rids: list[int]  # the step's rids on ``info.graph_walk``


@dataclass
class Speculation:
    """A speculative step N+1, built while step N runs.

    ``consumed_edges`` are the ``(name, next_node)`` pairs of N's outputs that
    it takes, spliced in by ``Speculator.thread_outputs`` once N
    has produced them. ``spec_id`` names the runtime's staged stream ingests,
    settled exactly once by ``Speculator.settle``. ``submitted``: the
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
    # A yield-away the chain built from the rows waiting at its own target
    # when it gave them the whole step; for fairness it is still the chain's.
    gave_way: bool = False
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
    # orders a mixed step's rows for its split kernels
    engine_manager: EngineManager | None = None


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
        loop_strides: list[int] | None = None,
    ) -> PreparedSpeculation:
        """``loop_strides``: ``candidates``' in N (``RouteInput.loop_strides``),
        which decide whose loop N ends."""
        if self.rows_eligible():
            return self._prepare_rows(
                batch_N, spec_node_name, graph_walk, candidates, max_continuing, laps,
                loop_strides,
            )
        return self._prepare_dicts(
            batch_N, spec_node_name, graph_walk, candidates, max_continuing, laps,
            loop_strides,
        )

    def _prepare_rows(
        self, batch_N: ScheduledBatch, spec_node_name: str, graph_walk: str,
        candidates: list[int], max_continuing: int | None, laps: PhaseLaps,
        loop_strides: list[int] | None,
    ) -> PreparedSpeculation:
        ctx = self._ctx
        rows = ctx.graph_runtime.prep_spec_rows(
            spec_node_name, batch_N.node_name, graph_walk, candidates, max_continuing,
            loop_strides,
        )
        laps.lap("prep_spec_rows")
        prep = PreparedSpeculation(rows.ready_rids, rows.spec_id, rows.wg_id)
        if rows.input_edges is not None and rows.ready_rids:
            # an input that is not the in-flight step's: per-row dicts
            prep.per_request_inputs.update(ctx.streams.input_tensors(
                rows.input_edges, spec_node_name, rows.ready_rids,
            ).by_rid)
            prep.request_to_worker_graph.update(dict.fromkeys(rows.ready_rids, rows.wg_id))
            prep.rows_wg_id = None
        return prep

    def _prepare_dicts(
        self, batch_N: ScheduledBatch, spec_node_name: str, graph_walk: str,
        candidates: list[int], max_continuing: int | None, laps: PhaseLaps,
        loop_strides: list[int] | None,
    ) -> PreparedSpeculation:
        """``prep_spec_rids`` with the stream polls, filling the per-row dicts
        for the ready rids."""
        ctx = self._ctx
        streams = ctx.streams
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
            streaming_edges=streams.edge_specs(polled),
            streaming_edges_per_rid=per_rid_counts,
            loop_strides=loop_strides,
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
        prepped_inputs = streams.input_tensors(
            out.input_edges, spec_node_name, out.ready_rids,
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
        chunk_ranges: dict[int, ChunkRange] | None = None,
        incomplete_node_rids: set[int] | None = None,
        rebuild_rids: frozenset[int] = frozenset(),
    ) -> Speculation:
        """The speculation from ``prep`` plus the fresh rows merged into its
        per-row dicts; ``other_walks`` maps the fresh rows of another walk (a
        mixing chain) to theirs, and ``fresh_final_edges`` the fresh rows whose
        final stream chunk was ingested before this step to its edges."""
        if (
            prep.rows_wg_id is not None and not other_walks and not fresh_final_edges
            and not chunk_ranges and not rebuild_rids
        ):
            # the dicts hold only the fresh rows here; a fresh row's final
            # stream chunk, a cut chunk, or a rebuild row's inputs from the log
            # take the general assembly, which carries them
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
            chunk_ranges=chunk_ranges, incomplete_node_rids=incomplete_node_rids,
            rebuild_rids=rebuild_rids,
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
        chunk_ranges: dict[int, ChunkRange] | None = None,
        incomplete_node_rids: set[int] | None = None,
        rebuild_rids: frozenset[int] = frozenset(),
    ) -> Speculation:
        """Package prepared rids (batch order, which ``make_executing_batch``
        reorders for a split walk) into the ``Speculation`` the main loop runs.
        Leader and follower differ only in how they pick the rids.

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
        if not request_walks and label != spec_target.graph_walk:
            # Every row of the target's walk ended at N, so this is a step of
            # its rows' own walk: it routes that walk's outputs and continues
            # no row of the target's loop, like a step built for those rows.
            spec_target = SpeculationOutput(
                node_name=spec_node, graph_walk=label,
                output_signals=tuple(self._walk_output_signals(spec_node, label, spec_target)),
            )
        spec_batch = ScheduledBatch(
            node_name=spec_node,
            graph_walk=label,
            request_to_worker_graph=request_to_worker_graph,
            request_walks=request_walks,
            chunk_ranges=dict(chunk_ranges) if chunk_ranges else {},
            incomplete_node_rids=set(incomplete_node_rids) if incomplete_node_rids else set(),
            rebuild_rids=rebuild_rids,
            # Resolved here, on the main thread, so the GPU thread never asks
            # the runtime, which may be mid-call with the GIL released.
            walk_output_signals={
                walk: self._walk_output_signals(spec_node, walk, spec_target)
                for walk in set(request_walks.values())
            },
            # Reported by whichever step chose the target -- speculate_node on
            # the leader, get_spec_target on a follower, or the rebuild above
            # for a step that collapsed to another walk -- so the hot path does
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
                rid: {se.name for se in edges if se.chunk.is_final}
                for rid, edges in consumed_streaming_edges.items()
                if any(se.chunk.is_final for se in edges)
            } | fresh_final_edges,
            per_request_input_metadata=input_meta,
            request_walks=request_walks,
            engine=ctx.engine_manager.get_engine(spec_node) if request_walks else None,
            chunk_ranges=chunk_ranges,
            rebuild_rids=rebuild_rids, rebuilds=ctx.scheduler.rebuilds,
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
        known (``Speculator.thread_outputs``).
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


class Speculator:
    """Builds step N+1 while N runs, chaining on N's node or yielding to another,
    threads N's outputs in and settles its staged ingests; on a TP follower, it
    rebuilds the batch the leader's head names. Tracks how many steps in a row the
    current chain has run: at MSTAR_MAX_CONSECUTIVE_SPEC_STEPS the next step yields."""

    def __init__(
        self, services: WorkerServices, layout: WorkerLayout, scheduler: MicroScheduler,
        lifecycle: RequestLifecycle, lockstep: Lockstep, admission: Admission,
        timer: PhaseTimer, *, max_consecutive: int = 1024, peek: bool = True,
        enable_nvtx: bool = False,
    ):
        self.worker_id = services.worker_id
        self._services = services
        self._graph_runtime = services.graph_runtime
        self.tensor_manager = services.tensor_manager
        self.request_state = services.request_state
        self.streams = services.streams
        self.engine_manager = services.engine_manager
        self.chunk_loops = layout.chunk_loops
        self.parallel_nodes = layout.parallel_nodes
        self.scheduler = scheduler
        self.lifecycle = lifecycle
        self.lockstep = lockstep
        self.admission = admission
        self.timer = timer
        self.enable_nvtx = enable_nvtx
        # MSTAR_SPEC_PEEK_FOR_FAIRNESS: only break the chain when another (node,
        # walk) is ready right now. Single-walk workers always speculate.
        self._peek = peek
        self._max_consecutive = max_consecutive
        self._consecutive = 0

    def note_launched(self, speculation: Speculation) -> None:
        """A speculated step went to the GPU thread: a step that continues the
        chain counts toward the cap, and a real yield starts it over."""
        if speculation.is_yield_away and not speculation.gave_way:
            self._consecutive = 0
        else:
            self._consecutive += 1

    def chain_ended(self) -> None:
        self._consecutive = 0

    def _async_schedulable(self, batch: ScheduledBatch) -> bool:
        """Any walk of the batch may speculate; the runtime skips the others."""
        return any(
            self._graph_runtime.is_async_schedulable(batch.node_name, walk)
            for walk in batch.rids_by_walk()
        )

    def can_speculate(self, batch: ScheduledBatch) -> bool:
        if not self._async_schedulable(batch):
            return False
        if batch.node_name in self.parallel_nodes:
            # Only the leader initiates, only under TP async scheduling.
            return (
                self.lockstep.leads_async(batch.node_name)
            )
        return True

    def plans_ahead(self, spec: Speculation) -> bool:
        """With the prefix cache open, a TP leader admits a yield-away step at exec, as its followers
        do. A pre-plan refused on rank 0 alone puts its pages back at the end of rank 0's free list,
        and the ranks then hand out different page ids, which only the cache reads."""
        node = spec.node_batch.node_name
        if spec.node_batch.read_check_rids is not None:
            # every rank admits a read-checked step inline, after the check: a follower
            # that could not read a row pre-plans without it, rank 0 with it
            return False
        return not (spec.is_yield_away and node in self.parallel_nodes and self.prefix_cache_open(node))

    def prefix_cache_open(self, node_name: str) -> bool:
        return self.engine_manager.get_engine(node_name).prefix_cache_open(node_name)

    def is_tp_follow_pending(self, pending: PendingBatch) -> bool:
        """The leader will send a head or a marker for ``pending``. Mirrors ``can_speculate``."""
        return self.lockstep.follows_async(pending.node_name) and self._async_schedulable(pending.batch)

    def settle(
        self, speculation: Speculation, success: bool,
        dropped_rids: set[int] = frozenset(),
    ) -> None:
        """Settle the runtime's staged streaming ingests, then give back the
        chunks of every rid that will not run.

        The order matters: the undo takes the chunk out of the node's input
        slot, and only then is it handed back to its StreamBuffer. Returning it
        while it is still in the slot is what tracked one chunk twice (a freed
        tensor behind a stale buffered edge) and, because a same-node prep
        ingests into the next-iter slot, let the FOLLOWING chunk take the
        current slot and be consumed first.

        Settles once per speculation: ``spec_id`` and the consumed edges are
        cleared here, so a second call is a no-op on both sides. Without that,
        a failed settle after a successful one would return the kept chunks to
        their buffers while they are still in the node's slot.

        On ``success`` the same call marks the batch speculatively scheduled, so
        "this speculation is now real" is ONE crossing into the runtime rather
        than two. The two halves address DIFFERENT rids: the flag goes on
        ``scheduled_batch``, already pruned of the dropped rids, while the undo
        targets exactly those dropped rids. The flag is cleared elsewhere
        (``StepExecutor.clear_in_flight``), on paths with no stage to settle.
        """
        batch = speculation.scheduled_batch
        rid_to_wg = batch.graph_rows() if success else {}
        self._graph_runtime.commit_speculation(
            speculation.spec_id, success, list(dropped_rids),
            # Keeps these nodes off the ready queue while the step is in flight.
            node=batch.node_name if rid_to_wg else None,
            wg_ids=batch.wg_ids() if rid_to_wg else [],
            scheduled_rids=list(rid_to_wg),
        )
        give_back = (
            speculation.consumed_streaming_edges.items() if not success
            else [
                (rid, speculation.consumed_streaming_edges.get(rid, []))
                for rid in dropped_rids
            ]
        )
        for rid, edges in list(give_back):
            for se in edges:
                self.streams.return_edge(rid, se)
            speculation.consumed_streaming_edges.pop(rid, None)
        # The kept rids' chunks now belong to the step; nothing is left to undo.
        speculation.spec_id = 0
        speculation.consumed_streaming_edges.clear()
        if not success:
            self.engine_manager.get_engine(batch.node_name).release_cached_prefix(speculation.node_batch)

    @staticmethod
    def speculated_rids(speculation: Speculation | None) -> frozenset[int]:
        """The rids a built speculation will run, which a teardown or eviction
        on the current step's behalf must leave alone."""
        if speculation is None:
            return frozenset()
        return frozenset(speculation.scheduled_batch.request_to_worker_graph)

    def _choose_spec_target(self, batch: ScheduledBatch) -> SpeculationTarget | None:
        """The (node, walk) the step after ``batch`` speculates into, with the
        rids of ``batch`` on that walk; None when nothing can be speculated."""
        # Every row, grouped by real walk: the runtime filters eligibility per
        # walk and drops a loop-back no row can take; prep_spec_rids the loop
        # completion per rid.
        groups = batch.rids_by_walk()
        if batch.rebuild_rids:
            rebuild = batch.rebuild_rids
            groups = {walk: kept for walk, rids in groups.items() if (kept := [r for r in rids if r not in rebuild])}
            if not groups:
                return None
        if len(groups) == 1:
            rows = next(iter(groups.values()))
            counts = None if len(rows) == 1 else [len(rows)]
        else:
            rows = [rid for walk_rids in groups.values() for rid in walk_rids]
            counts = [len(walk_rids) for walk_rids in groups.values()]
        ready_for_spec = self._graph_runtime.speculate_node(
            batch.node_name, list(groups), rows, counts,
            batch.loop_strides(rows),
        )
        if not ready_for_spec:
            return None
        # TODO: use the microscheduler to break ties when ready_for_spec
        # contains multiple ready nodes or walks
        info = ready_for_spec[0]
        if len(ready_for_spec) > 1 and (info.node_name, info.graph_walk) in self.chunk_loops.walks:
            # After a mixed step, prefer another walk's loop-back (decode) to
            # a chunk Loop's tick: the chain continues only its target walk's
            # rows, a mixing chain takes the rows mid-prefill along too, and
            # a prompt otherwise lets other ready work go first after each step.
            info = next(
                (i for i in ready_for_spec if (i.node_name, i.graph_walk) not in self.chunk_loops.walks),
                info,
            )
        return SpeculationTarget(info, groups[info.graph_walk])

    def _try_speculate_next(
        self,
        pending: PendingBatch,
        target: SpeculationTarget,
    ) -> Speculation | None:
        """Build a speculative N+1 batch + node_batch, by checking which nodes
        will become ready after the current batch's outputs are ingested.

        The speculated batch is a merge of:
          * **continuing** rids (subset of batch_N still alive, not
            pending-stop / pending-remove) — placeholder inputs are gathered
            from the registry now and the entries
            tied to ``consumed_edges`` are overwritten with batch_N's outputs
            after await by ``thread_outputs``.
          * **fresh** rids — newly-arrived requests whose spec-target node
            is ready in the queue right now. Their inputs come from the
            usual tensor_manager path (same as ``build_executing_batch``).
            Without this merge, new rids have to wait for the entire
            current speculation chain to drain before they can be scheduled.

        Which of them run is the scheduler's (``pick_chain_step``). When the
        rows waiting there take the whole step, it comes back as a yield-away
        of those rows (``Speculation.gave_way``) instead.
        """
        laps = PhaseLaps("speculate.")
        batch_N = pending.batch
        spec_target_info, walk_rids = target
        graph_walk = spec_target_info.graph_walk
        rid = walk_rids[0]
        spec_node_name = spec_target_info.node_name
        spec_target = (spec_node_name, graph_walk)
        # The merge below takes only this chain's capture group; another
        # group's backlog runs only once the chain yields to it. The chain's
        # rows run this step one loop iteration on, and their group is the
        # one they have there.
        next_iter_of = spec_target_info.loop_name if spec_target_info.is_new_loop_iter else None
        if self.scheduler.backlog_splits_from(spec_target, rid, next_iter_of):
            return None
        # Removes, aborts and failures are filtered here; prep_spec_rids
        # assumes that. A rid is also filtered when the target node isn't ready
        # to run it, since the target's admit would otherwise skip the readiness
        # check. The teardown filter comes first, since the readiness check would
        # queue a torn-down rid or reload its pages.
        # N's rows mid-prefill take their next chunk in this step, not the one
        # after; their nodes stay off the ready queue until N's chunk lands.
        candidates, mid_prefill = batch_N.get_continuing_rows(
            spec_node_name, walk_rids,
            self.lifecycle.is_tearing_down if self.lifecycle.teardowns_pending() else None,
            self.chunk_loops.speculated,
        )
        if spec_node_name != batch_N.node_name:
            partition = self.request_state.get_partition_for_node(spec_node_name)
            candidates = [
                r for r in candidates
                if self.admission.check_ready(
                    spec_node_name, r, self.request_state.get_fwd_info(r, partition),
                )
            ]
        # Who runs the step: the chain only ever continues its own rows, so
        # rows waiting at its target (backlogged, or on its queues) get the
        # step here or not at all while it runs. Nothing is popped yet.
        chain = self.scheduler.pick_chain_step(
            spec_target, candidates, mid_prefill, next_iter_of,
        )
        laps.lap("pick_step")
        mid_prefill = chain.mid_prefill
        if chain.pick.room == 0 and not mid_prefill:
            # the waiting rows take the whole step, as a step of their own
            batch = self.scheduler.take_chain_step(chain)
            return None if batch is None else self._yield_away_speculation(batch, gave_way=True)
        # prep keeps the first `room` of the chain's rows, in the pick's order
        prep = self._spec_preparer.prepare(
            batch_N, spec_node_name, graph_walk, chain.pick.continuing, chain.pick.room, laps,
            batch_N.loop_strides(chain.pick.continuing),
        )
        laps.lap("continuing_inputs")
        continuing = prep.ready_rids

        if not continuing and not mid_prefill:
            return None
        # fresh rows of another walk (a mixing chain): rid -> walk
        other_walks: dict[int, str] = {}
        fresh_final_edges: dict[int, set[str]] = {}

        # Merge in the waiting rows the pick gave the rest of the step.
        # Speculation only consumes work compatible with the spec target. In
        # partitioned models, unrelated ready work stays queued for
        # the normal scheduler path.
        fresh_batch = self.scheduler.take_chain_step(
            chain, len(continuing),
            # estimated from their spans in this step, if a token budget needs them
            lambda: self._continuing_seq_lens(
                pending, spec_node_name, graph_walk, continuing,
                spec_node_name == batch_N.node_name,
            ),
        )
        laps.lap("fresh_schedule")

        if not continuing and (
            fresh_batch is None
            or fresh_batch.request_to_worker_graph.keys().isdisjoint(batch_N.request_to_worker_graph)
        ):
            # The token budget left out every row of N mid-prefill, so the step
            # holds no row of N. It does not follow N: run it as a step of its
            # own, under its own walk. Sent as N's head, a TP follower finds no
            # row of N to follow it by.
            # With no continuing row the prep staged nothing to undo, as for
            # the return above.
            return None if fresh_batch is None else self._yield_away_speculation(fresh_batch, gave_way=True)

        if fresh_batch is not None:
            # The merge below relabels these node objects with the spec
            # target's name/walk, so a batch for any other node must not be
            # merged in.
            assert fresh_batch.node_name == spec_node_name, (
                f"Speculation asked for {spec_node_name!r} but the "
                f"scheduler returned {fresh_batch.node_name!r}"
            )
            fresh_inputs, fresh_final, _ = self.streams.input_tensors(
                fresh_batch.input_edges, spec_node_name,
                fresh_batch.request_to_worker_graph,
            )
            in_flight = set(continuing)
            for rid in fresh_batch.request_to_worker_graph:
                if rid in in_flight:
                    # Shouldn't happen — continuing rids are held by the
                    # in-flight step and shouldn't be in ready queues —
                    # but if it does, the in-flight rid wins.
                    #
                    # Never a TP-follow batch: targeted calls don't pop the
                    # TP-follow FIFO (a rejected ScheduleTPNode can't re-queue).
                    self._graph_runtime.push_back_node(
                        fresh_batch.node_name, [rid],
                        [fresh_batch.request_to_worker_graph[rid]],
                    )
                    continue

                # a row mid-prefill has no popped inputs: its chunk is cut from what the engine holds
                prep.per_request_inputs[rid] = fresh_inputs.get(rid, {})
                prep.request_to_worker_graph[rid] = (
                    fresh_batch.request_to_worker_graph[rid]
                )
                if rid in fresh_final:
                    fresh_final_edges[rid] = fresh_final[rid]
                walk = fresh_batch.walk_of(rid)
                if walk != graph_walk:
                    other_walks[rid] = walk

        laps.lap("fresh_merge")
        logger.debug("Speculating: %s %s", spec_node_name, continuing)
        speculation = self._spec_builder.build(
            pending, spec_target_info, prep, other_walks, fresh_final_edges,
            chunk_ranges=None if fresh_batch is None else fresh_batch.chunk_ranges,
            incomplete_node_rids=None if fresh_batch is None else fresh_batch.incomplete_node_rids,
            rebuild_rids=frozenset() if fresh_batch is None else fresh_batch.rebuild_rids,
        )
        # N's rows, not fresh ones: a cleared speculation must not ready their
        # nodes while N's chunk is in flight; N's settle does that
        spec_rows = speculation.scheduled_batch.request_to_worker_graph
        speculation.continuing_rids.update(
            row.request_id for row in mid_prefill if row.request_id in spec_rows
        )
        if chain.pick.ordered:
            speculation.scheduled_batch.turn_order = chain.pick.taken
        laps.lap("assemble")
        return speculation

    def _continuing_seq_lens(
        self, pending: PendingBatch, node_name: str, graph_walk: str,
        continuing: Collection[int], same_node: bool,
    ) -> list[int]:
        """The spans the continuing rows will run, for the merge's token
        budget: the submodule's estimate, -1 where it has none."""
        engine = self.engine_manager.get_engine(node_name)
        prev, info = pending.node_batch, pending.node_batch.per_request_info
        lens = []
        for rid in continuing:
            span = engine.speculative_input_seq_len(
                node_name, graph_walk, info.get(rid),
                prev.seq_len_of(rid) if same_node else -1,
            )
            lens.append(-1 if span is None else span)
        return lens

    @cached_property
    def _spec_context(self) -> SpeculationContext:
        return SpeculationContext(
            graph_runtime=self._graph_runtime, tensor_manager=self.tensor_manager,
            request_state=self.request_state, scheduler=self.scheduler,
            streams=self.streams, engine_manager=self.engine_manager,
        )

    @cached_property
    def _spec_preparer(self) -> SpeculationPreparer:
        return SpeculationPreparer(self._spec_context)

    @cached_property
    def _spec_builder(self) -> SpeculationBuilder:
        return SpeculationBuilder(self._spec_context)

    def thread_outputs(
        self, speculation: Speculation,
        outputs_N: dict[int, NameToTensorList],
    ):
        """Splice batch N's outputs into the spec batch's inputs.

        Runs on the main thread right after N's future resolves, which is
        when the GPU thread has enqueued N, so it must not read a tensor
        VALUE: it only moves tensor lists and tests which keys are present.
        N's kernels may still be running.
        """
        threaded_continuing: set[int] = set()
        dropped: set[int] = set()
        # A rows-only step kept every output on the device: a rid that ran is
        # present, and its loop-back signal arrives empty, the way every
        # fallback batch of such a walk does (``device_loopback_signals``).
        rows_only = getattr(outputs_N, "rows_only", False)
        row_inputs = speculation.node_batch.per_request_input_tensors
        if isinstance(row_inputs, DeferredRowInputs):
            if rows_only and speculation.continuing_rids.issubset(
                outputs_N.row_request_ids or (),
            ):
                # Every continuing row ran: what the loop below would do row by
                # row, for all of them at once.
                row_inputs.set_signals(name for name, _ in speculation.consumed_edges)
                speculation.dropped = set()
                return
            # Anything else edits rows one by one.
            speculation.node_batch.per_request_input_tensors = row_inputs.materialize()
        ran = frozenset(outputs_N.row_request_ids or ()) if rows_only else None
        # a mixed N's rows that looped back on the device arrive empty too
        loopback = getattr(outputs_N, "loopback_rids", frozenset())
        chunked = speculation.scheduled_batch.chunk_ranges
        for rid in list(speculation.node_batch.request_ids):
            if rid not in speculation.continuing_rids:
                continue  # fresh rid — inputs already gathered.
            if rid in chunked:
                # mid-prefill: its chunk is cut from the input the engine holds, not N's outputs
                threaded_continuing.add(rid)
                continue
            if rid in loopback:
                inputs = speculation.node_batch.per_request_input_tensors[rid]
                for input_name, _ in speculation.consumed_edges:
                    inputs[input_name] = []
                threaded_continuing.add(rid)
                continue
            if ran is not None:
                if rid in ran:
                    inputs = speculation.node_batch.per_request_input_tensors[rid]
                    for input_name, _ in speculation.consumed_edges:
                        inputs[input_name] = []
                    threaded_continuing.add(rid)
                else:
                    dropped.add(rid)
                continue
            rid_outputs = outputs_N.get(rid, {})
            ok = True
            for input_name, _ in speculation.consumed_edges:
                tensors = rid_outputs.get(input_name, [])
                if not tensors:
                    ok = False
                    break
                speculation.node_batch.per_request_input_tensors[rid][input_name] \
                    = list(tensors)
            if ok:
                threaded_continuing.add(rid)
            else:
                dropped.add(rid)

        if dropped:
            logger.warning(
                "Speculation: dropped rids %s (no loop-back output from N)",
                sorted(dropped),
            )
            # Its final chunks go back below; it must not flush or report done.
            speculation.node_batch.drop_rids(dropped)
            for r in dropped:
                speculation.node_batch.per_request_input_tensors.pop(r, None)
                speculation.scheduled_batch.discard_rid(r)
                speculation.node_batch.per_request_input_metadata.pop(r, None)
        speculation.continuing_rids = threaded_continuing
        speculation.dropped = dropped

    def follow(self, pending: PendingBatch) -> Speculation | None:
        head = self.lockstep.follow_head()
        if head is None or not head.speculative:
            return None
        if head.spec_from_seq != pending.tp_seq:
            # From some other step: the serial path gets it in FIFO order.
            return None
        key = self.scheduler._key
        if head.node_name != pending.node_name or \
                key(head.node_name, head.graph_walk) != key(pending.node_name, pending.graph_walk):
            # Leaders only speculate same-node loop-backs; leave it serial.
            logger.warning(
                "Worker %s: speculative head %s/%s (seq %d) does not match its "
                "parent batch %s/%s; leaving it for the serial path",
                self.worker_id, head.node_name, head.graph_walk, head.spec_seq,
                pending.node_name, pending.graph_walk,
            )
            return None

        # Same gate as the serial path: a step must not be built against a page
        # state that is still mid-replay. Every poll retries the owed moves, so
        # returning here makes progress where reading the half-applied state as
        # "not ready" never could.
        if not self.lockstep.settle_delta(
            # Of the steps before the head, only N may still index a page of the head's hit
            caught_up=pending.node_batch.commit_done.is_set(),
        ):
            return None

        batch_N = pending.batch
        # The leader names its rids by wire string; these are this rank's handles.
        head_rids = self.lockstep.tp_rids(head)
        # before anything is popped: lists that do not match the rows raise
        chunk_ranges = tp_chunk_ranges(head, head_rids)
        # rows whose input read failed here: built without, as on the serial path
        unread = self.lockstep.unread_rows(head, head_rids)
        head_walk_idx = dict(zip(head_rids, head.walk_idx, strict=True)) if head.walks else None
        in_batch = batch_N.request_to_worker_graph
        # rows read here: prepped (continuing) or popped (fresh); every row,
        # unread ones included, runs the leader's continuation of N
        in_n = [r for r in head_rids if r in in_batch]
        # N's rows mid-prefill loop back on nothing of N's: as on the leader,
        # their next chunk is cut from the input held since their first. N's
        # chunks came from the leader, so every rank splits these out alike.
        held = batch_N.incomplete_node_rids
        mid_prefill = {r for r in in_n if r in held} if held else set()
        continuing = [r for r in in_n if r not in unread and r not in mid_prefill]
        fresh = [r for r in head_rids if r not in in_batch and r not in unread]
        if not in_n:
            return None
        spec_walk = head.walks[head_walk_idx[in_n[0]]] if head.walks else head.graph_walk
        # The leader already chose the target; this just reports its loop
        # context. speculate_node's eligibility filter cannot run here -- it
        # requires the node be a parallel LEADER node.
        spec_target_info = self._graph_runtime.get_spec_target(
            batch_N.node_name, head.node_name, spec_walk, in_n[0],
        )
        if spec_target_info is None:
            return None

        # Polling the StreamBuffers stays on this side: they hold tensors.
        polled: list[tuple[int, StreamingEdge]] = []
        per_rid_counts: list[int] = []
        for r in continuing:
            edges = self.streams.poll_for(r, head.node_name)
            polled.extend((r, e) for e in edges)
            per_rid_counts.append(len(edges))

        # Fresh rids are popped BEFORE the continuing prep, so the only undo
        # a failure needs is push_back_node. The reverse order leaves a
        # successful all-or-nothing prep to unwind, with no API for it.
        popped = self.lockstep.pop_ready_rids(
            head.node_name, head.graph_walk, fresh,
            walks=head.walks or None,
            walk_idx=[head_walk_idx[r] for r in fresh] if head.walks else None,
        )

        def _return_all_polled() -> None:
            for r, se in polled:
                self.streams.return_edge(r, se)

        if popped is None:
            # ``pop_ready_rids`` scans with ``allow_reload=False``: an offloaded
            # rid reads not-ready and waits for a replayed delta. The delta is
            # settled above, so a rid still off-device is a divergence, not a lag.
            _return_all_polled()
            self.lockstep.warn_unchecked_read_failure(head, fresh)
            return None

        # output_signals are handled by SpeculationBuilder.assemble
        fresh_wg, fresh_edges = popped.wg_ids, popped.edge_specs

        # The leader's list is the composition every rank runs: all-or-nothing,
        # no local finished-rid skipping, no room_for_continuing cap.
        prep = self._graph_runtime.prep_follow_spec_rids(SpeculationPrepInput(
            spec_node_name=head.node_name,
            curr_node_name=batch_N.node_name,
            graph_walk=spec_walk,
            rids=continuing,
            room_for_continuing=None,
            streaming_edges=self.streams.edge_specs(polled),
            streaming_edges_per_rid=per_rid_counts,
        ))
        if prep is None:
            # prep rolled its own ingests back; hand the fresh rids their ready
            # slots back too, so the serial path can still pick them up.
            self._graph_runtime.push_back_node(
                head.node_name, list(fresh_wg), list(fresh_wg.values()),
            )
            _return_all_polled()
            return None

        consumed = set(prep.consumed_streaming_edge_idxs)
        consumed_streaming_edges: dict[int, list[StreamingEdge]] = {}
        for i, (r, se) in enumerate(polled):
            if i in consumed:
                self.streams.mark_ingested(r, se)
                consumed_streaming_edges.setdefault(r, []).append(se)
            else:
                self.streams.return_edge(r, se)

        # Both seeded with their rid lists, so the `in prep_inputs` test below
        # still separates prepped rids from fresh ones -- a ready rid with no
        # prepped edge gets an empty mapping, as a zero-length slice used to.
        prep_inputs = self.streams.input_tensors(
            prep.input_edges, head.node_name, prep.ready_rids,
        ).by_rid
        fresh_inputs, fresh_final_edges, _ = self.streams.input_tensors(
            fresh_edges, head.node_name, list(fresh_wg),
        )
        prep_wg = dict(zip(prep.ready_rids, prep.wg_ids, strict=True))

        new_request_to_worker_graph: dict[int, int] = {}
        per_request_inputs: dict[int, NameToTensorList] = {}
        for rid in head_rids:  # wire order == the leader's step order, split rows ordered
            if rid in prep_inputs:
                new_request_to_worker_graph[rid] = prep_wg[rid]
                per_request_inputs[rid] = prep_inputs[rid]
            elif rid in unread:
                per_request_inputs[rid] = {}
            elif rid in mid_prefill:
                new_request_to_worker_graph[rid] = in_batch[rid]
                per_request_inputs[rid] = {}
            else:
                new_request_to_worker_graph[rid] = fresh_wg[rid]
                per_request_inputs[rid] = fresh_inputs[rid]
        request_walks = {r: head.walks[w] for r, w in head_walk_idx.items()} if head.walks else None
        if unread:
            # In with no inputs, as on the serial path: the step has rank 0's
            # rows until its read check takes these out. An unread row not in
            # N is no continuing row, so voiding the speculation
            # (``StepPipeline._clear_speculation``) pushes it back though it was never popped.
            # Harmless only while every later head naming it is read-checked,
            # so ``unread_rows`` keeps it out of the pop: rank 0 checks each
            # step taking the input until one runs (``Lockstep.consume_reads``),
            # and that step's check takes the row out on every rank.
            new_request_to_worker_graph, _ = self.lockstep.with_unread_rows(
                head, head_rids, unread, new_request_to_worker_graph, request_walks or {},
            )

        # Committed: the serial path must not see the head any more.
        self.lockstep.pop_follow_head()
        if fresh:
            self.scheduler.note_follow_pop(key(head.node_name, head.graph_walk))
        logger.debug(
            "Follow-speculating: %s %s (seq %d from %d)",
            head.node_name, head_rids, head.spec_seq, head.spec_from_seq,
        )
        speculation = self._spec_builder.assemble(
            pending, spec_target_info,
            new_request_to_worker_graph, per_request_inputs,
            consumed_streaming_edges,
            [r for r in in_n if r not in mid_prefill] if mid_prefill else in_n,
            fresh_final_edges,
            is_same_node=True, spec_id=prep.spec_id, tp_seq=head.spec_seq,
            request_walks=request_walks,
            chunk_ranges=chunk_ranges,
            incomplete_node_rids=unfinished_chunk_rids(chunk_ranges),
        )
        # N's rows, as on the leader: N's settle readies them, not a cleared speculation
        speculation.continuing_rids.update(mid_prefill)
        if chunk_ranges:
            # a fresh row's first chunk caps its Loop, as the leader's merge did
            self.lockstep.cap_chunk_loops(speculation.scheduled_batch)
        if head.read_check:
            speculation.node_batch.read_check_rids = tuple(head_rids)
        return speculation

    @staticmethod
    def step_voids_head(pending: PendingBatch) -> str | None:
        """Why N's verdict voids a head built on it, or ``None``. Both fields are
        final once N's result is taken (``StepPipeline._step_result``); the leader clears on exactly these two."""
        if pending.node_batch.admit_error is not None:
            return "admit_error"
        if pending.node_batch.failed_requests:
            return "failed rids"
        return None

    def build(
        self, pending: PendingBatch | None,
    ) -> tuple[Speculation | None, tuple[str, str] | None]:
        """Speculate the step after ``pending`` (N+1 from N), or a
        yield-away batch when fairness says so. Returns the speculation
        and the (node, walk) to exclude on the non-speculative path."""
        speculation = None
        yield_away_from_target = None
        batch = None
        # Build nothing on a refusal; the caller's admit fence is what
        # makes it knowable here. Ordinary speculation would be dropped anyway,
        # but a yield-away batch survives ``StepPipeline._clear_speculation`` and is
        # already broadcast — so the eviction below would land after its
        # head went out, and the follower would admit it a delta behind.
        # Skipping keeps that eviction in the gap between steps.
        #
        # The marker below stays outside this: a follower settles N on
        # exactly one of {head, marker}, and it moves no pages.
        admit_refused = (
            pending is not None
            and pending.node_batch.admit_error is not None
        )

        if pending is not None and self.can_speculate(pending.batch):
            # Fairness check (peek-based, replaces the old iter-
            # counter cap): only break the spec chain when there's
            # another (node, walk) actually ready to schedule on
            # this worker. On single-walk workers (Orpheus LLM,
            # Orpheus SNAC) this returns False and we always speculate.
            # Keyed on the walk the chain speculates into, not the step's:
            # after a mixed step that is the combined walk, which no row has,
            # so the chain's own parked rows would count as other work.
            target = self._choose_spec_target(pending.batch)
            chain_key = (
                (target.info.node_name, target.info.graph_walk) if target is not None
                else (pending.node_name, pending.graph_walk)
            )
            # A prompt chain yields from its first step: speculating its next
            # chunk must not keep decode rows waiting a second prompt step. A
            # walk with no token budget runs its prompts whole and keeps
            # main's order.
            other_ready = self._peek and self.scheduler.chain_yields(
                chain_key, self._consecutive,
                chain_key in self.chunk_loops.speculated and self.scheduler.chunks(*chain_key),
            )
            must_yield_away = self._consecutive >= self._max_consecutive or other_ready
            if target is not None and not must_yield_away and not admit_refused:
                if self.enable_nvtx:
                    range_push("worker.speculate", synchronize=False)
                _t0 = _time.perf_counter() if self.timer.period else 0.0
                speculation = self._try_speculate_next(pending, target)
                if self.timer.period:
                    self.timer.record("speculate", _time.perf_counter() - _t0)
                if self.enable_nvtx:
                    range_pop(synchronize=False)
                if speculation is not None and not speculation.is_yield_away:
                    # Broadcast the head now, during forward N, so
                    # followers build it during theirs (-1: not parallel).
                    speculation.tp_seq = self.lockstep.broadcast(
                        speculation.node_batch,
                        speculative=True, spec_from_seq=pending.tp_seq,
                    )
            self.lockstep.maybe_broadcast_nospec(pending, speculation)
            # ``yield_away_from_target`` stays None when the admit
            # refused, so the non-speculative path below schedules
            # without the fairness exclusion — it should be free to pick
            # up the batch ``_handle_admit_failure`` just pushed back.
            if speculation is None and not admit_refused:
                if (
                    not must_yield_away and self._peek and self._consecutive < 1
                    and self.scheduler.chunks(pending.node_name, pending.graph_walk)
                ):
                    # The chain could not continue, so there is no chain to
                    # keep going: other ready work goes first, as it would
                    # after the chain's next step. Without this, the step after
                    # a prompt step whose prompts all finished (or whose rows
                    # mid-prefill are not speculated, `speculate_mid_prefill`)
                    # starts another prompt while decode rows wait out both.
                    # Only after a step under a token budget: elsewhere a
                    # prompt runs whole, and main's order stands.
                    must_yield_away = self.scheduler.has_ready_excluding(chain_key)
                yield_away_from_target = chain_key if must_yield_away else None
                with self.timer.span("worker.schedule_yield_away"):
                    batch = self.scheduler.next_batch(exclude_target=yield_away_from_target)
                if batch is not None:
                    speculation = self._yield_away_speculation(batch)
            if speculation is not None and speculation.is_yield_away:
                # A leader stamps the seq it sends; a follower's
                # batch keeps the one it came off the FIFO with.
                ya_seq = self.lockstep.broadcast(speculation.node_batch)
                speculation.tp_seq = ya_seq if ya_seq >= 0 else speculation.scheduled_batch.tp_seq

        return speculation, yield_away_from_target

    def _yield_away_speculation(
        self, batch: ScheduledBatch, gave_way: bool = False,
    ) -> Speculation:
        """A step independent of the one in flight, built now so it overlaps
        it; broadcast by the caller once N's marker is out."""
        node_batch = build_executing_batch(self._services, batch)
        logger.debug("Yield away: %s %s", batch.node_name, node_batch.request_ids)
        return Speculation(
            scheduled_batch=batch,
            node_batch=node_batch,
            consumed_edges=set(),
            continuing_rids=set(),  # n/a
            partition=self.request_state.get_partition_for_node(batch.node_name),
            is_new_iter=False,
            is_same_node=False,
            is_yield_away=True,
            gave_way=gave_way,
        )
