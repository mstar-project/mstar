from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

from mstar.communication.tensors import TensorStore
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.profile.format import GraphTiming, RxInfo, TxInfo
from mstar.utils.containers import ParallelList


class SpeculationOutput(NamedTuple):
    node_name: str
    graph_walk: str
    # Loop context for the TARGET. Not used for filtering -- prep_spec_rids
    # owns that -- but the speculative batch carries it, and the postprocess
    # that follows needs it to match its pending loop stops. The selection
    # step already computes both, so reporting them avoids a second
    # ingest_for_speculation (which mutates speculative slot state).
    is_new_loop_iter: bool = False
    loop_name: str | None = None
    # The TARGET's output edge names, same as PopRidsOutput carries for a
    # normally-scheduled batch. A speculated batch never goes through
    # pop_rids, so without these the caller has to ask again on every forward
    # pass -- and decode, the hot path, is almost entirely speculated.
    output_signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class PendingLoopStop:
    """A loop stop produced by this iteration's check_stop.

    Lives for exactly one iteration: the routing pass that follows uses it to
    drop the outputs of rids that "overstayed" their stop, then clears it.
    """
    rid: int
    graph_walk: str
    loop_name: str


class EdgeSpec(NamedTuple):
    signal: str
    next_node: str
    uuids: list[int]

    # only valid for streaming edges
    is_final_streaming_chunk: bool = False


class PopRidsOutput(NamedTuple):
    wg_ids: ParallelList[int, int]
    # flat, rid-major over wg_ids.keys(); input_edges_per_rid[i] edges belong
    # to rid wg_ids.keys()[i] -- same layout as SpeculationPrepOutput, so
    # _build_executing_batch can walk either the same way.
    input_edges: list[EdgeSpec]
    input_edges_per_rid: list[int]
    # The node's output edge names, structural and identical for every rid in
    # the batch. Reported by the POP rather than asked for separately: it is
    # needed at completion, and deriving it from the tensors the model returned
    # would take whatever the model happened to emit, including names no edge
    # carries. Trailing with a default so a caller that does not need them can
    # leave it off.
    output_signals: tuple[str, ...] = ()


class SpeculationPrepInput(NamedTuple):
    spec_node_name: str
    curr_node_name: str
    graph_walk: str
    rids: list[int]
    room_for_continuing: int
    streaming_edges: list[EdgeSpec]
    streaming_edges_per_rid: list[int]


class SpeculationPrepOutput(NamedTuple):
    consumed_streaming_edge_idxs: list[int]
    ready_rids: list[int]
    # parallel to ready_rids: the worker graph each is speculating in
    wg_ids: list[int]
    input_edges: list[EdgeSpec]
    input_edges_per_rid: list[int]


class ReadyNodeSpec(NamedTuple):
    node_name: str
    graph_walk: str
    rids: list[int]


class FreedTensors(NamedTuple):
    """Tensors a runtime dereferenced to zero and dropped from the bookkeeper,
    for the caller to finish tearing down.

    ``registered`` is each uuid's ``mem_registered`` flag, which the forget
    took with it and the transport still needs: it is what decides whether the
    memory has to be unregistered. Hand the pair straight to
    ``TensorCommunicationManager.cleanup_collectable``.
    """
    uuids: list[int]
    registered: list[bool]

    @classmethod
    def none(cls) -> "FreedTensors":
        return cls([], [])


class RouteInput(NamedTuple):
    partition: str
    graph_walk: str
    node_name: str
    output_signals: list[str]
    # rid -> the worker graph it is running this node in
    wg_ids: ParallelList[int, int]
    # tensor uuids for every (rid, output signal), flat and rid-major over
    # wg_ids.keys(); num_tensors[i * len(output_signals) + s] of them belong to
    # rid i's signal s. The runtime looks the metadata up in the store.
    tensors: list[int]
    num_tensors: list[int]


class RouteOutput(NamedTuple):
    completion_id: int

    # NOTE: An outgoing edge can carry a tensor from an earlier batch via
    # accumulated outputs and loop outputs, so register and local streaming
    # have to be by UUID instead of indexing the UUID list passed in.
    register_uuids: list[int]
    register_rids: list[int]

    # Count these into the conductor's per-signal new-token totals. Indices
    # into RouteInput.tensors deliberately to avoid double-counting.
    new_token_output_idxs: list[int]

    # Push these into the request's stream buffer for their edge. Keyed by
    # edge name -> (rid, uuid) per tensor.
    local_streaming_by_signal: dict[str, ParallelList[int, int]]

    # The rids this completion will actually build a frame for -- one bound
    # for a peer worker (INPUT_SIGNALS) or the conductor (WORKER_GRAPHS_DONE).
    # Only those need `per_request_info`, and preparing it is not free for a
    # runtime that has to encode it: the object is mutated in place every
    # pass, so it cannot be cached. On a single-worker deployment inside a
    # loop neither frame goes out on most passes.
    #
    # ``None`` means "not computed, pass it for everyone" -- the Python
    # runtime hands the live object over untouched, so it has nothing to save.
    rids_needing_request_info: frozenset[int] | None = None
    # Consumed inputs the completion itself dropped to zero, for the caller to
    # tear down -- see ``cleanup_consumed_inputs``. Empty in the normal order,
    # where that call has already taken them.
    freed_inputs: FreedTensors = FreedTensors.none()



# One request's profiling for its WORKER_GRAPHS_DONE: rx_info, tx_info and
# graph_timings. All three are populated only under enable_prof.
Profiling = tuple[list[RxInfo], list[TxInfo], dict[tuple[str, str], GraphTiming]]


class SendInput(NamedTuple):
    completion_id: int
    # Opaque to the runtime: it forwards each on the frames it builds.
    per_request_info: ParallelList[int, CurrentForwardPassInfo]
    new_token_counts: ParallelList[int, dict[str, int]]

    # WORKER_GRAPHS_DONE fields the runtime cannot derive, so they come in here.
    # rid -> {edge name -> tokens consumed}; from StreamBuffer._consumed
    stream_tokens_consumed: ParallelList[int, dict[str, int]] | None = None
    # rid -> (rx_info, tx_info, graph_timings). All three are populated only
    # under enable_prof, so this is None in production.
    profiling: ParallelList[int, Profiling] | None = None


class GraphRuntime(ABC):
    """Owns a worker's graph state: its worker graphs' per-request queues,
    the routing and sharding derived from them, loop state, and the request
    id <-> handle table. The worker drives it through this contract and keeps
    only what cannot live behind it -- forward-pass info (a wire object it
    forwards opaquely) and stream buffers (which hold tensors).
    """

    # --------- Bookkeeping ----------
    @abstractmethod
    def set_node_metadata(
        self, parallel_nodes: set[str],
        parallel_leader_nodes: set[str],
        tp_async_nodes: set[str]
    ):
        pass

    @abstractmethod
    def add_request(
        self, request_id: str,
        partition: str,
        graph_walk: str,
        partition_worker_graph_ids: list[int],
        worker_graph_to_workers: ParallelList[int, list[str]]
    ) -> int:
        """
        Returns the integer handle for this request.
        """
        pass

    @abstractmethod
    def remove_request(
        self, rid: int
    ):
        """``rid`` is the handle ``add_request`` returned, not the uuid.

        Handles are RECYCLED, so anything keyed by one must be purged here or on
        the same event. A leaked string rid was harmless -- the string never
        recurred -- but a leaked handle silently attaches to whichever request
        gets that handle next, which reads as one request inheriting another's
        state. Known handle-keyed maps: MicroScheduler.{failed_rids,
        admit_errors, held_until, backlog, pending_tp_follow_count} (purged by
        clear_rid), Worker.{_last_active, _pending_removes}, the runtime's
        pending loop stops, and the tensor manager's per-request maps.
        """
        pass

    @abstractmethod
    def get_sharding_config(self, rid: int) -> ShardingConfig | None:
        pass

    # --------- rid <-> handle, at the process boundary ----------
    #
    # Handles are worker-internal. Messages carry the string, so the only
    # translations are: de-intern at each send site, intern at each receive
    # handler. In worker.py that is 12 sends and 9 receive handlers.

    @abstractmethod
    def get_rid_string(self, handle: int) -> str:
        """For a message about to leave this process."""
        pass

    @abstractmethod
    def get_rid_handle(self, rid: str) -> int | None:
        """For a message that just arrived.

        None when the rid is unknown -- a message can legitimately arrive for a
        request this rank already removed (the micro-scheduler already treats
        that as "not ready"), so this must not raise on the race.
        """
        pass

    @abstractmethod
    def set_walk(self, rid: int, partition: str, walk: str):
        pass

    @abstractmethod
    def set_speculatively_scheduled(
        self, node: str, wg_id: int, rids: list[int],
        speculatively_scheduled: bool
    ):
        pass

    @abstractmethod
    def is_speculatively_scheduled(
        self, node: str, wg_id: int, rid: int,
    ) -> bool:
        """Whether this rid's node is marked speculatively scheduled.

        The flag must SURVIVE node completion: its rids are still in flight
        for the speculative N+1 step, so the node must stay out of the ready
        set until the speculation resolves.
        """
        pass

    @abstractmethod
    def get_dynamic_loop_iters(
        self, request_ids: list[int],
        partition: str,
    ) -> ParallelList[int, dict[str,int]]:
        pass

    @abstractmethod
    def is_async_schedulable(self, node_name: str, graph_walk: str) -> bool:
        """Whether this node opts into async scheduling. Structural, so it
        takes no rid."""
        pass

    @abstractmethod
    def get_output_signals(self, node_name: str, graph_walk: str) -> list[str]:
        """The node's output signal names. Structural: the edge objects are
        per-request copies, but their names are not."""
        pass

    @abstractmethod
    def reset_outputs(
        self, node_name: str, rids: list[int], wg_ids: list[int],
    ):
        """Drop stale output tensor_info before a pass writes new ones."""
        pass

    @abstractmethod
    def cleanup_consumed_inputs(
        self, node_name: str, rids: list[int], wg_ids: list[int],
    ) -> FreedTensors:
        """Release the input tensors the just-executed node consumed.

        Returns the ones that became collectable and so still need the
        transport-side teardown -- shm files, arena slots, memory
        unregistration -- which lives on the tensor manager and not here. A
        runtime that holds the manager itself may do that in place and return
        ``FreedTensors.none()``; the Rust one has only the bookkeeper, so its
        freed tensors would otherwise sit until the request is torn down.
        """
        pass

    @abstractmethod
    def mark_stream_partition_done(self, rid: int, partition: str):
        """The consuming pass saw the final streaming chunk. Reported on the
        WORKER_GRAPHS_DONE that follows, unless the node was speculative."""
        pass

    @abstractmethod
    def get_consumed_edges(
        self, source_node: str, dest_node: str, graph_walk: str,
    ) -> set[tuple[str, str]]:
        """The (signal, dest) pairs ``source_node`` emits into ``dest_node``.

        Structural -- edge names and destinations do not vary by request -- so
        it takes no rid. A speculative batch uses it to know which of the
        in-flight batch's outputs it will consume.
        """
        pass

    @abstractmethod
    def get_worker_graph_id_for_node(
        self, node: str, graph_walk: str,
    ) -> int:
        """The owning worker graph. Keyed on the walk too: the same node can
        belong to different worker graphs in different walks (prefill vs
        decode), which is what walk_node_to_worker_graph_id indexes today."""
        pass

    # --------- Inputs ----------
    @abstractmethod
    def ingest_inputs_batch(
        self,
        # rid -> the signal arriving for it
        signals: ParallelList[int, EdgeSpec],
        can_buffer: bool=True,
        is_streaming: bool=False,
    ) -> list[int]:
        """
        Returns a list of signal indices that remain uningested
        (used in streaming for re-storing uningested edges).

        For streaming, this function must gate on whether all non-streaming
        inputs have already been ingested. The gate is re-evaluated per signal,
        since ingesting one can make another node eligible.

        EdgeSpec rather than a signal-only type because
        ``is_final_streaming_chunk`` has to survive the ingest: the pass that
        CONSUMES the chunk is the one that reports partition_done, so the flag
        lives in the node's input slot until then.

        """
        pass

    # --------- Scheduling ----------
    @abstractmethod
    def pop_rids(
        self, node_name: str,
        graph_walk: str,
        request_ids: list[int],
        check_ready: bool=False,
    ) -> PopRidsOutput | None:
        """
        Returns rids and worker graph ids for the batch. If check_ready is set,
        then this function checks if the rids are ready and either pops all or.
        nothing (returning None if it is nothing). Otherwise, it is assumed
        that the rids are already known to be ready.

        For check_ready, engine-level ready-ness is assumed to be a prerequisite.
        """
        pass

    @abstractmethod
    def has_ready_excluding(
        self, exclude_rids: set[int],
        exclude_target: tuple[str, str] | None=None,
    ) -> bool:
        """
        Graph-level check; does not include schedule-level backlog
        calculation; e.g., this may return False even when the backlog exists,
        so the backlog must be checked first.

        exclude_rids includes failed_rids, pending_removes, and held_until.
        """
        pass

    @abstractmethod
    def get_ready_nodes(
        self, exclude_rids: set[int],
        target: tuple[str, str] | None=None,
        exclude_target: tuple[str, str] | None=None,
    ) -> list[ReadyNodeSpec]:
        """
        Graph-level ready check. The output list must be filtered for engine-
        level ready-ness separately.

        exclude_rids includes failed_rids, pending_removes, and held_until.
        """
        pass

    @abstractmethod
    def push_back_node(
        self, node_name: str,
        rids: list[int],
        wg_ids: list[int]
    ):
        pass

    # --------- Speculation ----------
    @abstractmethod
    def speculate_node(
        self, node_name: str,
        graph_walk: str,
        sample_rid: int,
    ) -> list[SpeculationOutput]:
        """
        Returns a list of nodes that are ready for speculation, checking
        against whether the node is async enabled (known internally), as
        well as whether the node is TP async compatible.
        """
        pass

    @abstractmethod
    def prep_spec_rids(
        self, input: SpeculationPrepInput
    ) -> SpeculationPrepOutput:
        """
        Ingest streaming edges; rollback if node is not ready.
        This also checks pending loop stops and loops that are on their final
        iter, automatically filtering out those rids. It is assumed that rids
        are pre-filtered for removes.
        """
        pass

    @abstractmethod
    def get_spec_target(
        self, curr_node_name: str, spec_node_name: str,
        graph_walk: str, sample_rid: int,
    ) -> SpeculationOutput | None:
        """Loop context for a spec target chosen elsewhere.

        ``speculate_node`` applies the eligibility filter, which a follower
        cannot: that filter requires the node be in ``parallel_leader_nodes``,
        and a follower is by definition not the leader. The leader already
        decided; this just reports what the target is.
        """
        pass

    @abstractmethod
    def prep_follow_spec_rids(
        self, input: SpeculationPrepInput
    ) -> SpeculationPrepOutput | None:
        """The TP-follower counterpart of ``prep_spec_rids``.

        A follower runs the leader's composition exactly: rank 0 committed to
        it and sits on the collective until every follower joins, so this is
        ALL-OR-NOTHING (None rolls the whole set back) and applies neither the
        loop-completion filter nor ``room_for_continuing`` -- both are local
        decisions the leader already made for everyone.
        """
        pass

    # --------- Postprocess ----------
    @abstractmethod
    def stop_loops_batched(
        self, partition: str,
        graph_walk: str,
        last_node_run: str,
        loop_names: ParallelList[int, list[str]]
    ):
        """
        (1) Stop loops in the graph
        (2) Updates pending_loop_stops list
        (3) Send loop done messages to peer workers

        Rids whose current walk does not contain a named loop are filtered out
        here (that is a model bug, logged and dropped), so callers pass whatever
        check_stop produced.
        """
        pass

    @abstractmethod
    def apply_peer_loop_stops(
        self, rid: int, partition: str,
        loop_stop_times: dict[str, NestedLoopIndices],
    ):
        """A peer's STOP_LOOPS landing here.

        Stops only the loops whose incoming stop is NEWER than what this rank
        has (``label_context_gt``), and does NOT fan out again -- the loop that
        originated the stop already told everyone.
        """
        pass

    @abstractmethod
    def has_pending_loop_stop(
        self, rid: int, graph_walk: str, loop_name: str,
    ) -> bool:
        pass

    @abstractmethod
    def pending_loop_stop_rids(
        self, graph_walk: str, loop_name: str,
    ) -> set[int]:
        pass

    @abstractmethod
    def clear_pending_loop_stops(self):
        """Pending stops are good for one iteration only."""
        pass

    @abstractmethod
    def complete_and_route_batch(
        self, input: RouteInput,
        tensor_store: TensorStore
    ) -> RouteOutput:
        """
        (1) Mark node complete, do _cleanup_consumed_inputs
        (2) Process node outputs
        (3) Set persist and update ref counts on the tensor store

        Stores the outputs that need to be sent as internal state, to be
        accessed when send_batch is called. This includes the partition,
        graph walk, rids involved, nested loop indices, output routing, etc.
        These are keyed on completion_id.

        The nested loop indices are the node's loop context BEFORE this
        completion: marking the node complete advances the loop counters, so
        they are snapshotted here, first, and the send reports that snapshot
        (on RESULT_TENSORS and as WORKER_GRAPHS_DONE's output loop indices).
        """
        pass

    @abstractmethod
    def send_outputs(
        self,
        input: SendInput,
    ):
        """
        (1) Send outputs to other workers
        (2) Buffer persist signals (the buffered signals will be internal to
        this graph runtime)
        (3) Buffer new token counts (also internal to graph runtime)
        (4) Output signals -> api server
        (5) Remote streaming tensors
        (6) WG done messages to the conductor
        """
        pass
