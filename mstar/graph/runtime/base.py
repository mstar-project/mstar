from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import NamedTuple

from mstar.communication.tensors import NameToTensorList, TensorStore
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge
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


class EdgeTuple(NamedTuple):
    """One edge of a ``ColumnarEdgeSpecs``, reassembled."""
    rid: int
    signal: str
    uuids: list[int]
    is_final_streaming_chunk: bool


class InputTensors(NamedTuple):
    """What a batch build needs off a block, in one walk of the columns."""
    by_rid: dict[int, NameToTensorList]
    # The rids whose edge carried a stream's final chunk. The CONSUMING pass
    # reports the partition done, which is why this rides with the inputs.
    final_stream_rids: set[int]


@dataclass
class ColumnarEdgeSpecs:
    """A batch of edges as columns rather than one object per edge; minimizes
    allocation of Strings on the Rsut end, as well as building of Python objects
    and Python <> Rust marshalling.
    """

    # len = the number of DISTINCT signal names in the batch, which is at most
    # the destination node's input count -- one to three in practice.
    signal_names: list[str]

    # len = the total number of tensors, flat over the edges in order
    uuids: list[int]

    # len = the number of edges
    tensors_per_edge: list[int]
    signal_name_idxs: list[int]
    # Per EDGE, not per rid: it replaces a run-length column, so the
    # "flat, rid-major" invariant stops being something every caller honours.
    rids: list[int]
    is_final_streaming_chunk: list[bool]

    # Only the INGEST path carries a destination. A pop or a prep reports
    # edges for a node the caller just named, so both leave these None.
    # len = the number of distinct destinations / the number of edges
    next_nodes: list[str] | None = None
    next_node_idxs: list[int] | None = None

    # signal name -> its index, for the dedup. A CACHE, not part of the value
    # -- hence out of repr and eq -- and re-derived on demand, because most
    # blocks are built without one (Rust fills the columns itself, and
    # select_rids copies the names across).
    _signal_idx_of: dict[str, int] = field(
        default_factory=dict, repr=False, compare=False,
    )
    _next_node_idx_of: dict[str, int] = field(
        default_factory=dict, repr=False, compare=False,
    )

    @classmethod
    def empty(cls) -> "ColumnarEdgeSpecs":
        return cls([], [], [], [], [], [])

    def add(
        self, rid: int, signal: str, uuids, is_final_streaming_chunk: bool,
        next_node: str | None = None,
    ) -> None:
        """Append one edge. For a producer on the Python side; Rust fills the
        same columns natively.

        ``next_node`` only for the ingest side -- a pop or a prep reports edges
        for a node the caller already named.
        """
        if next_node is not None:
            if self.next_nodes is None:
                self.next_nodes, self.next_node_idxs = [], []
            self.next_node_idxs.append(self._next_node_idx(next_node))
        self.signal_name_idxs.append(self._signal_idx(signal))
        self.rids.append(rid)
        self.is_final_streaming_chunk.append(is_final_streaming_chunk)
        before = len(self.uuids)
        self.uuids.extend(uuids)
        self.tensors_per_edge.append(len(self.uuids) - before)

    def __len__(self) -> int:
        """The number of EDGES. Not the number of rids -- one rid contributes
        one edge per ready input."""
        return len(self.rids)

    def add_edge(self, rid: int, edge: GraphEdge) -> None:
        """Append a real ``GraphEdge``, destination and all.

        The ingest side's producer: its edges arrived from elsewhere rather
        than being read out of a runtime's own state, and unlike a pop they do
        NOT all share a destination -- one INPUT_SIGNALS can carry edges for
        several nodes -- so the destination is carried.
        """
        self.add(
            rid, edge.name, [info.uuid for info in edge.tensor_info],
            edge._final_stream_chunk, next_node=edge.next_node,
        )

    def to_edges(
        self,
        tensor_store: TensorStore,
        is_streaming: bool,
        next_node: str | None = None,
    ) -> list[GraphEdge]:
        """Rebuild real ``GraphEdge`` objects, for the Python runtime's ingest.

        ``next_node`` overrides the column, for a caller whose whole batch
        shares one destination.
        """
        edges = []
        uuid_start = 0
        for i in range(len(self.rids)):
            uuid_end = uuid_start + self.tensors_per_edge[i]
            edges.append(
                GraphEdge(
                    name=self.signal_names[self.signal_name_idxs[i]],
                    next_node=next_node if next_node is not None
                        else self.next_nodes[self.next_node_idxs[i]],
                    tensor_info=[
                        tensor_store.get_info(self.uuids[j])
                        for j in range(uuid_start, uuid_end)
                    ],
                    is_streaming=is_streaming,
                    _final_stream_chunk=self.is_final_streaming_chunk[i],
                )
            )
            uuid_start = uuid_end
        return edges

    def to_input_tensors(
        self, get_tensor, rids: list[int] | None = None,
    ) -> InputTensors:
        """One walk for both things a batch build needs: each rid's inputs as
        ``{signal: [tensor]}``, and the rids whose edge carried a stream's
        final chunk.

        ``rids`` seeds the result so a rid with no ready edges still gets an
        empty mapping -- what slicing a zero-length run used to give. Left
        None, only the rids that actually have edges appear.
        """
        by_rid: dict[int, NameToTensorList] = (
            {} if rids is None else {rid: {} for rid in rids}
        )
        final_rids: set[int] = set()
        names = self.signal_names
        uuids = self.uuids
        at = 0
        for i, rid in enumerate(self.rids):
            end = at + self.tensors_per_edge[i]
            slot = by_rid.get(rid)
            if slot is None:
                slot = by_rid[rid] = {}
            slot[names[self.signal_name_idxs[i]]] = [
                get_tensor(uuids[j]) for j in range(at, end)
            ]
            if self.is_final_streaming_chunk[i]:
                final_rids.add(rid)
            at = end
        return InputTensors(by_rid, final_rids)

    def edge_tuples(self) -> list[EdgeTuple]:
        """One ``EdgeTuple`` per edge.

        For tests and debugging; production walks the columns. Comparing two
        blocks field by field would be sensitive to ``signal_names`` ORDER,
        which is an implementation detail -- the Python runtime discovers names
        in ``ready_inputs`` order and Rust in the node's input order -- so a
        parity check wants this instead.
        """
        out = []
        at = 0
        for i, rid in enumerate(self.rids):
            end = at + self.tensors_per_edge[i]
            out.append(EdgeTuple(
                rid=rid,
                signal=self.signal_names[self.signal_name_idxs[i]],
                uuids=self.uuids[at:end],
                is_final_streaming_chunk=self.is_final_streaming_chunk[i],
            ))
            at = end
        return out

    def select_rids(self, keep) -> "ColumnarEdgeSpecs":
        """The edges belonging to ``keep``, as a new block.

        Off the hot path: a batch that fits under its cap is passed through
        whole, and a rid is only dropped on a failure or a removal.
        ``signal_names`` is carried over as-is -- an entry no surviving edge
        points at is harmless.
        """
        out = ColumnarEdgeSpecs(
            signal_names=list(self.signal_names),
            uuids=[], tensors_per_edge=[], signal_name_idxs=[], rids=[],
            is_final_streaming_chunk=[],
            next_nodes=None if self.next_nodes is None
                else list(self.next_nodes),
            next_node_idxs=None if self.next_node_idxs is None else [],
        )
        at = 0
        for i, rid in enumerate(self.rids):
            end = at + self.tensors_per_edge[i]
            if rid in keep:
                out.rids.append(rid)
                out.signal_name_idxs.append(self.signal_name_idxs[i])
                out.tensors_per_edge.append(self.tensors_per_edge[i])
                out.is_final_streaming_chunk.append(
                    self.is_final_streaming_chunk[i]
                )
                out.uuids.extend(self.uuids[at:end])
                if out.next_node_idxs is not None:
                    out.next_node_idxs.append(self.next_node_idxs[i])
            at = end
        return out

    def extend(self, other: "ColumnarEdgeSpecs") -> None:
        """Fold another block in, in place.
        ``other``'s signal names are remapped as a correctness guard.
        """
        remap = [self._signal_idx(name) for name in other.signal_names]
        self.signal_name_idxs.extend(remap[i] for i in other.signal_name_idxs)
        self.rids.extend(other.rids)
        self.tensors_per_edge.extend(other.tensors_per_edge)
        self.is_final_streaming_chunk.extend(other.is_final_streaming_chunk)
        self.uuids.extend(other.uuids)
        if other.next_node_idxs is not None:
            if self.next_nodes is None:
                self.next_nodes, self.next_node_idxs = [], []
            node_remap = [
                self._next_node_idx(n) for n in (other.next_nodes or ())
            ]
            self.next_node_idxs.extend(
                node_remap[i] for i in other.next_node_idxs
            )

    def _signal_idx(self, name: str) -> int:
        """``name``'s index, appending it if it is new.

        The cache is re-derived whenever it does not cover ``signal_names``.
        """
        if len(self._signal_idx_of) != len(self.signal_names):
            self._signal_idx_of = {
                name: i for i, name in enumerate(self.signal_names)
            }
        idx = self._signal_idx_of.get(name)
        if idx is None:
            idx = self._signal_idx_of[name] = len(self.signal_names)
            self.signal_names.append(name)
        return idx

    def _next_node_idx(self, name: str) -> int:
        """``name``'s index among the destinations; see ``_signal_idx``, which
        this mirrors including the cache being re-derived on demand."""
        if len(self._next_node_idx_of) != len(self.next_nodes):
            self._next_node_idx_of = {
                name: i for i, name in enumerate(self.next_nodes)
            }
        idx = self._next_node_idx_of.get(name)
        if idx is None:
            idx = self._next_node_idx_of[name] = len(self.next_nodes)
            self.next_nodes.append(name)
        return idx


class EdgeSpec(NamedTuple):
    """One arriving signal, on the INGEST side only.

    The pop and prep directions report ``ColumnarEdgeSpecs`` instead -- there
    the producer is Rust and the consumer never wants a per-edge object, so an
    object per edge was pure overhead. Ingest still starts from real
    ``GraphEdge`` objects, so it keeps this until that path is converted too.
    """
    signal: str
    next_node: str
    uuids: list[int]

    # only valid for streaming edges
    is_final_streaming_chunk: bool = False


class PopRidsOutput(NamedTuple):
    wg_ids: ParallelList[int, int]
    # Columns, and each edge carries its own rid -- so there is no run-length
    # column to keep parallel and no rid-major invariant for callers to honour.
    # Same shape as SpeculationPrepOutput, so _build_executing_batch walks
    # either the same way.
    input_edges: ColumnarEdgeSpecs
    # The node's output edge names, structural and identical for every rid in
    # the batch. Reported by the POP rather than asked for separately: it is
    # needed at completion, and deriving it from the tensors the model returned
    # would take whatever the model happened to emit, including names no edge
    # carries. Trailing with a default so a caller that does not need them can
    # leave it off.
    output_signals: tuple[str, ...] = ()


class SpeculationPrepInput(NamedTuple):
    # TODO: streaming_edges wants to be a ColumnarEdgeSpecs like everything
    # else -- `next_node` is redundant here (the poll filters on it, so it is
    # always spec_node_name) and per-edge rids would retire
    # streaming_edges_per_rid. Held back because
    # SpeculationPrepOutput.consumed_streaming_edge_idxs index into this flat
    # list, and the rollback in _prep_one_spec_rid works in the same index
    # space; converting that needs care, and it is worth ~38% of a path that
    # only runs on streaming-consumer workers.
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
    # As PopRidsOutput: columns, with the rid on each edge. A ready rid with no
    # edges simply has none here, so callers that want an entry for it pass
    # ready_rids to ``to_input_tensors``.
    input_edges: ColumnarEdgeSpecs


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
        # One column set; each edge carries the rid it arrived for and its own
        # destination, since one message can hold edges for several nodes.
        signals: ColumnarEdgeSpecs,
        can_buffer: bool=True,
        is_streaming: bool=False,
    ) -> list[int]:
        """
        Returns a list of signal indices that remain uningested
        (used in streaming for re-storing uningested edges).

        For streaming, this function must gate on whether all non-streaming
        inputs have already been ingested. The gate is re-evaluated per signal,
        since ingesting one can make another node eligible.

        ``is_final_streaming_chunk`` is a column rather than being dropped
        because it has to survive the ingest: the pass that CONSUMES the chunk
        is the one that reports partition_done, so the flag lives in the node's
        input slot until then.

        Returned indices are into the block's edge columns, which is the order
        the caller built them in -- that is how a streaming caller maps a
        refusal back to the chunk it has to hand to its StreamBuffer.
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
        Returns rids and worker graph ids for the batch, plus the ready inputs
        as columns. If check_ready is set,
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
