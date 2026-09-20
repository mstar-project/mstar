
from abc import ABC, abstractmethod
from typing import NamedTuple

from mstar.communication.tensors import TensorStore
from mstar.conductor.request_info import CurrentForwardPassInfo

#
# NOTE:
# (1) rids will be interned once at ingestion, then even in the Python code
# will be referred to by their integer handle. When being sent to other workers,
# then the handle will be dereferenced to the actual uuid string.
#
# (2) The TensorStore class will have the tracking of persist / refcounts / etc
# in rust to prevent more hops to Python in complete_and_route_batch.
# Rust owns the whole record -- metadata (dims/dtype/stride/nbytes/address/
# source_*/shm_*) plus ref_cnt / persist / mem_registered. The torch.Tensor
# itself stays a Python object that Rust holds opaquely (Py<PyAny>) and hands
# back for get_tensor. Because Rust has the record, TensorSpec collapses to
# just the uuid handle, and store_batch mints those handles instead of
# str(uuid4()) -- so (1)'s interning costs nothing on the output path.
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
# (6) Rust sends directly: RawZmqCommunicator::send takes &self with its peer
# table behind a Mutex, so the runtime can hold an Arc to the same instance
# the Python communicator wraps. The receive side should decode and ingest in
# Rust too, or peer edges become Python objects again on arrival.
#


class SpeculationOutput(NamedTuple):
    node_name: str
    graph_walk: str


class EdgeSpec(NamedTuple):
    signal: str
    uuids: list[int]

    # only valid for streaming edges
    is_final_streaming_chunk: bool = False


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
    rid_to_wg: dict[int, int]
    rids: list[int]
    # uuid handles minted by TensorStore.store_batch; Rust looks up the rest
    tensors: list[int]
    # (rid, signal idx) -> number of tensors
    num_tensors_per_rid_signal: dict[(int, int), int]

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


class SendInput(NamedTuple):
    completion_id: int
    per_request_info: dict[int, CurrentForwardPassInfo]
    new_token_counts: dict[int, dict[str, int]]

    # for WORKER_GRAPHS_DONE; see note (5)
    # rid -> {edge name -> tokens consumed}
    stream_tokens_consumed: dict[int, dict[str, int]] | None = None
    # rids whose partition finished on this pass (from final_stream_rids)
    partition_done_rids: list[int] | None = None
    # rid -> msgpack(rx_info, tx_info, graph_timings); None unless enable_prof
    profiling: dict[int, bytes] | None = None

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

    # TODO: add request, remove request

    @abstractmethod
    def set_speculatively_scheduled(
        self, node: str, wgid: int, rids: list[int],
        speculatively_scheduled: bool
    ):
        pass

    # --------- Scheduling ----------
    @abstractmethod
    def pop_rids(
        self, node_name: str,
        graph_walk: str,
        request_ids: list[int],
        check_ready: bool=False,
    ) -> list[int]:
        """
        Returns worker graph ids for the batch. If check_ready is set, then
        this function checks if the rids are ready and either pops all or none.
        Otherwise, it is assumed that the rids are already known to be ready.
        """
        pass

    @abstractmethod
    def has_ready_excluding(
        self, exclude_rids: set[str],
        exclude_target: tuple[str, str] | None = None,
    ) -> bool:
        """
        Graph-level check; does not include schedule-level backlog
        calculation.
        """
        pass

    @abstractmethod
    def get_ready_nodes(
        self, exclude_rids: list[int],
        target: tuple[str, str] | None = None,
        exclude_target: tuple[str, str] | None = None,
    ) -> list[ReadyNodeSpec]:
        """
        Graph-level ready check. The output list must be filtered for engine-
        level ready-ness separately.
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

    # --------- Postprocess ----------
    @abstractmethod
    def stop_loops_batched(
        self, partition: str, graph_walk: str,
        rid_to_loop_names: dict[int, list[str]]
    ):
        """
        (1) Stop loops in the graph
        (2) Updates pending_loop_stops list
        (3) Send loop done messages to peer workers
        """
        pass

    @abstractmethod
    def complete_and_route_batch(
        self, input: RouteInput,
        tensor_store: TensorStore
    ) -> RouteOutput:
        """
        (1) Mark node complete
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
