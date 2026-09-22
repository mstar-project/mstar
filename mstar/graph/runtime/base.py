
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

from mstar.communication.tensors import TensorStore
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.utils.containers import ParallelList

#
# NOTE:
# (1) rids will be interned once at ingestion, then even in the Python code
# will be referred to by their integer handle. When being sent to other workers,
# then the handle will be dereferenced to the actual uuid string.
#
# (2) TensorStore stays Python and keeps dict[uuid, Tensor]. A separate
# bookkeeper -- metadata plus ref_cnt / persist / mem_registered -- is what
# gets a Rust backend, so Rust never sees a tensor and never has to drop a
# Python object. Because the bookkeeper holds the metadata, the routing call
# only needs uuid handles.
#
# (3) per-partition info will also probably eventually be ported into the
# GraphRuntime so that we don't have to do a hop to Python for everything
# involving the per-request info. For a first step, it can just be passed
# in as a msgpack-encoded blob. Some parts will be immediately ported over,
# like the pending persist signals.
# msgpack is self-describing, so "opaque" only means opaque to Python's type
# system -- Rust can still read the handful of keys it needs out of the blob
# (graph_walk, partition_name, resource_publish_info, fwd_index) without
# owning the type.
#
# (4) pending loops stops and stop loop idxs will also be owned by Rust, with
# an API to query them in the speculative path (if needed)
#
# (5) WORKER_GRAPHS_DONE is built and sent by Rust. Splitting it out would be
# the worst of both: Python would need persist_signals as
# dict[str, list[TensorPointerInfo]], output_loop_indices and
# output_signal_names handed back per request, which is the per-rid object
# construction the batching exists to avoid. The fields Rust cannot derive are
# small and come in on SendInput:
#   - rx_info / tx_info / graph_timings are populated only under enable_prof
#     (all three call sites are inside `if self.enable_prof:`), so in
#     production they are empty -- hence one optional `profiling` field.
#   - stream_tokens_consumed comes from StreamBuffer._consumed.
#   - partition_done is derived from the pass's final_stream_rids.
# The worker->conductor edge already speaks typed msgpack
# (mstar.communication.wire), as do all the others, so there is no codec
# prerequisite left here.
#
# (7) Request handles are recycled, which string rids effectively were not.
# Anything keyed by a handle must be purged when the request is removed, or a
# stale entry silently attaches to whichever request next gets that handle.
# The ones that exist today: MicroScheduler.{failed_rids, admit_errors,
# held_until, backlog, pending_tp_follow_count} (via clear_rid),
# Worker._last_active / _pending_removes / _pending_loop_stops, and the tensor
# manager's per-request maps. clear_rid covers the scheduler set; the rest are
# worth auditing as part of the refactor.
#
# (6) Rust sends directly: RawZmqCommunicator::send takes &self with its peer
# table behind a Mutex, so the runtime can hold an Arc to the same instance
# the Python communicator wraps. The receive side should decode and ingest in
# Rust too, or peer edges become Python objects again on arrival.
#


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


class RouteInput(NamedTuple):
    partition: str
    graph_walk: str
    node_name: str
    output_signals: list[str]
    # rid -> the worker graph it is running this node in
    wg_ids: ParallelList[int, int]
    # tensor uuids for every (rid, output signal), flat and rid-major over
    # wg_ids.keys(); num_tensors[i * len(output_signals) + s] of them belong to
    # rid i's signal s. Rust looks the metadata up in the bookkeeper.
    tensors: list[int]
    num_tensors: list[int]

    # TODO: function to build this object from a node batch


class RouteOutput(NamedTuple):
    completion_id: int
    # parallel: tensor at register_tensor_idxs[i] belongs to register_rids[i].
    # Indices are into RouteInput.tensors. Rust owns mem_registered, so this is
    # already deduped by uuid and excludes anything already staged.
    register_tensor_idxs: list[int]
    register_rids: list[int]
    new_token_output_idxs: list[int]
    local_streaming_tensor_idxs: list[int]
    # The rids this completion will actually build a frame for -- one bound
    # for a peer worker (INPUT_SIGNALS) or the conductor (WORKER_GRAPHS_DONE).
    # Only those need `per_request_info`, and preparing it is not free: the
    # object is mutated in place every pass, so it cannot be cached and has to
    # be re-encoded. On a single-worker deployment inside a loop neither frame
    # goes out on most passes, and the whole encode is wasted.
    #
    # ``None`` means "not computed, prepare it for everyone", so a runtime
    # that does not work this out keeps behaving as before -- Python's does
    # not, and does not need to: it hands the live object over untouched.
    rids_needing_request_info: frozenset[int] | None = None


class SendInput(NamedTuple):
    completion_id: int
    # Opaque to the runtime for now: a Rust backend reads the few keys it needs
    # (graph_walk, partition_name, resource_publish_info, fwd_index) out of the
    # msgpack without owning the type.
    per_request_info: ParallelList[int, CurrentForwardPassInfo]
    new_token_counts: ParallelList[int, dict[str, int]]
    # Snapshotted BEFORE the completion that produced this batch, because
    # mark_node_complete and stop_loops both advance loop state -- the runtime
    # cannot re-derive the pre-completion value at send time.
    nested_loop_indices: ParallelList[int, NestedLoopIndices]

    # WORKER_GRAPHS_DONE fields the runtime cannot derive, so they come in here.
    # rid -> {edge name -> tokens consumed}; from StreamBuffer._consumed
    stream_tokens_consumed: ParallelList[int, dict[str, int]] | None = None
    # rid -> msgpack(rx_info, tx_info, graph_timings). All three are populated
    # only under enable_prof, so this is None in production.
    profiling: ParallelList[int, bytes] | None = None

    # only for SHM arena tensor management
    # list[(tensor_idx, segment, offset)]
    shm_locs: list[tuple[int, int, int]] | None = None


class GraphRuntime(ABC):
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
        clear_rid), Worker.{_last_active, _pending_removes,
        _pending_loop_stops}, and the tensor manager's per-request maps.
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
    ):
        """Release the input tensors the just-executed node consumed."""
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

        A Rust backend should decode peer frames and ingest them here directly,
        or every peer edge becomes a Python object again on arrival.
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
        (5) Remote streaing tensors
        (6) WG done messages to the conductor

        A Rust backend sends these itself: RawZmqCommunicator::send takes &self
        with its peer table behind a Mutex, so it can hold an Arc to the same
        instance the Python communicator wraps -- no hop back to Python.
        """
        pass


#
# The output path is then four crossings per forward pass, each O(1) in batch
# size:
#
#   uuids = tensor_store.store_batch(...)        # Rust mints handles + records
#   out   = runtime.complete_and_route_batch(..) # route, refcount, persist
#   locs  = <stage arena in Python>              # .contiguous() / host.copy_()
#   runtime.send_outputs(.., shm_locs=locs)      # peers, api server, conductor
#
# Step 3 stays Python because it is torch; the reservation inside it is already
# Rust (SegmentedShmArena). With Rust owning mem_registered it also decides
# WHAT needs registering, so the dedup-by-uuid and skip-if-registered logic
# moves with the store.
#
