import logging
from copy import deepcopy
from dataclasses import dataclass, field

from mstar.communication.tensors import TensorCommunicationManager
from mstar.conductor.request_info import CurrentForwardPassInfo, PerLabelSeqInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    NameAndDest,
    NodeAndGraphWalk,
    NodeCompletionOutput,
    TensorPointerInfo,
)
from mstar.graph.graph_io import WorkerGraphIO, format_graph_edge_list
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.graph.special_destinations import EMIT_TO_CLIENT, SPECIAL_DESTINATIONS
from mstar.model.base import WorkerGraph
from mstar.streaming.stream_buffer import StreamBuffer

logger = logging.getLogger(__name__)


@dataclass
class NodeOutputRouting:
    routed_to_this_worker_graph: list[GraphEdge]
    is_first_tp_rank: bool
    persist: list[GraphEdge] # outputs that are going back to the conductor
    to_workers: dict[str, list[GraphEdge]] # worker id to signals
    emit_to_client: list[GraphEdge] = field(default_factory=list)
    new_token_outputs: list[GraphEdge] = field(default_factory=list)
    completed_worker_graph_ids: list[str] = field(default_factory=list)
    streaming_to_workers: dict[str, list[GraphEdge]] = field(default_factory=dict)  # streaming edges to other workers
    streaming_local: list[GraphEdge] = field(default_factory=list)  # streaming edges staying on this worker


@dataclass
class WorkerGraphQueues:
    """
    For a single worker graph, keeps track of which nodes are waiting on which
    inputs for each request, and which nodes are ready to run per request.
    """
    worker_graph_id: str
    graph_walks: set[str] # e.g., this worker graph is active during decode and image_gen
                          # but not the prefill graph walk
    worker_graph: WorkerGraph
    per_request_queues: dict[str, WorkerGraphIO]
    tensor_manager: TensorCommunicationManager

    def __post_init__(self):
        self.nodes = set(self.worker_graph.section.get_nodes().keys())
        self.loops = set(self.worker_graph.section.get_loops().keys())

    def process_new_inputs(
        self, request_id: str, inputs: list[GraphEdge],
        can_buffer: bool=True
    ) -> list[GraphEdge]:
        """Ingest inputs into this worker graph's per-request io.

        Returns the edges that were NOT routed here (because their next_node
        is not a node in this worker graph) so the caller can try the next
        worker graph. Works for both normal and streaming inputs — the per-
        node io routes by name and lets ``ReadySignals.is_ready_for_streaming``
        light up the streaming readiness set on its own.
        """
        assert request_id in self.per_request_queues, \
            f"Tried to process new inputs for unknown request ID {request_id}"
        queue = self.per_request_queues[request_id]
        not_ingested: list[GraphEdge] = []
        for inp in inputs:
            if not queue.ingest_input(inp, can_buffer):
                not_ingested.append(inp)
        return not_ingested

    def process_new_streaming_inputs(
        self, request_id: str, inputs: list[GraphEdge],
        can_buffer: bool=True
    ) -> list[GraphEdge]:
        assert request_id in self.per_request_queues, \
            f"Tried to process new inputs for unknown request ID {request_id}"
        queue = self.per_request_queues[request_id]
        not_ingested: list[GraphEdge] = []
        for inp in inputs:
            if (inp.next_node not in queue.ready_for_streaming) or (not queue.ingest_input(inp, can_buffer)):
                not_ingested.append(inp)
        return not_ingested

    def is_done(self, request_id) -> bool:
        assert request_id in self.per_request_queues, \
            f"Tried to check queue done state for unknown request ID {request_id}"
        queue = self.per_request_queues[request_id]
        return queue.wg_state_registry.is_done

    def add_request(self, request_id: str):
        """
        Initialize queues for a new request
        """
        section_copy = deepcopy(self.worker_graph.section)
        queue = WorkerGraphIO(section_copy, wg_id=self.worker_graph_id)
        queue.register_communication_info(
            self.tensor_manager, request_id
        )
        self.per_request_queues[request_id] = queue

    def remove_request(self, request_id: str):
        """
        Delete queues for a completed/removed request (saw EOS)
        """
        self.per_request_queues.pop(request_id, None)

    def get_ready_node_names(self) -> dict[str, set[str]]:
        """
        Returns mapping of request id to ready node names for that request
        """
        return {
            request_id: q.ready_node_names \
                for (request_id, q) in self.per_request_queues.items()
        }

    def get_ready_for_streaming(self, request_id: str):
        assert request_id in self.per_request_queues, \
            f"Tried to check ready for streaming for unknown request ID {request_id}"
        return self.per_request_queues[request_id].ready_for_streaming

    def pop_ready_nodes(
        self, request_id: str, node_names: list[str]
    ) -> list[GraphNode]:
        """
        Remove the given node names from the ready queue for the request and
        return the corresponding GraphNode objects.
        """
        nodes = []
        if request_id in self.per_request_queues:
            q = self.per_request_queues[request_id]
            for name in node_names:
                q.ready_node_names.discard(name)
                nodes.append(q.nodes[name])
        return nodes

    def push_back_node(
        self, request_id: str, node: GraphNode
    ) -> None:
        """Push a previously popped node back onto the ready queue (e.g., after OOM hold)."""
        if request_id in self.per_request_queues:
            self.per_request_queues[request_id].ready_node_names.add(node.name)

    def reset(self, request_id):
        """
        At the end of a worker graph, reset the queues for a request so it can
        be used for the next full model forward pass.
        """
        self.per_request_queues[request_id].clear()

    def stop_loops(
        self, request_id: str, loop_names: set[str]
    ) -> set[NameAndDest]:
        """Register a finish signal for each named loop and return the union
        of their ``_loop_back_inputs`` so the caller can drop those (name, dest)
        edges from the current iter's output routing.
        """
        assert request_id in self.per_request_queues, \
            f"Tried to stop loops for unknown request ID {request_id}"
        queue = self.per_request_queues[request_id]
        loop_back_signals: set[NameAndDest] = set()
        for name in loop_names:
            if name not in queue.loops:
                continue
            queue.register_loop_finish_signal(name)
            loop_back_signals.update(queue.loops[name]._loop_back_inputs)
        return loop_back_signals

    def mark_node_complete(
        self, request_id: str, node_name: str
    ) -> NodeCompletionOutput:
        """Complete a node in this worker graph's per-request io and return
        the registry's NodeCompletionOutput (output_edges + filtered_signals)."""
        assert request_id in self.per_request_queues, \
            f"Tried to complete node {node_name!r} for unknown request ID {request_id}"
        return self.per_request_queues[request_id].mark_node_complete(node_name)

    def get_dynamic_loop_iters(self, request_id: str) -> dict[str, int]:
        assert request_id in self.per_request_queues, \
            f"Tried to get dynamic loop iters for unknown request ID {request_id}"
        queue = self.per_request_queues[request_id]
        return queue.get_loop_indices()


@dataclass
class PerPartitionInfo:
    current_fwd_info: CurrentForwardPassInfo
    # graph_walk_worker_graph_ids = worker graphs for current graph walk
    graph_walk_worker_graph_ids: list[str] = field(default_factory=list) # for this worker
    stream_partition_done: bool = False  # set True when last chunk pops with is_final


@dataclass
class PerRequestInfo:
    """
    Information about a request that the worker needs to keep track of:
    - node_to_worker: for all nodes. This is, e.g., how we say that if
        an output goes to (LLM, decode graph walk), what worker that points to.
    - worker_graph_ids: mainly redundant information / syntactic sugar. This is
        the list of worker graph IDs that are on this worker and used by this request
        (across all possible graph walks)
    - current_graph_walk: which computation path we’re currently on, e.g., prefill,
        decode, image_gen, etc.
    - graph_walk_worker_graph_ids: worker graph IDs used in the current graph walk (e.g., if there
        is a prefill LLM worker graph and decode LLM worker graph and we are in decode,
        this list only includes the decode worker graph)
    - pending_persist_signals: buffered persist signals awaiting flush on
        WORKER_GRAPHS_DONE
    - partition_fwd_infos: per-partition forward info for the colocated case
        where multiple partitions run on the same worker
    - tensors: TBD
    """
    node_to_workers: dict[NodeAndGraphWalk, list[str]]  # for all nodes
    dyn_loop_to_workers: dict[NodeAndGraphWalk, list[str]]
    worker_graph_ids: list[str] # for this worker
    sharding_config: ShardingConfig

    pending_persist_signals: list[GraphEdge] = field(default_factory=list)
    pending_new_tokens: dict[str, list[int]] = field(default_factory=dict)
    stream_buffers: dict[str, StreamBuffer] = field(default_factory=dict)  # edge_name -> StreamBuffer
    current_output_chunks: list[str] = field(default_factory=list)
    output_loop_indices: dict[str, NestedLoopIndices] = field(default_factory=dict)

    per_partition_info: dict[str, PerPartitionInfo] = field(default_factory=dict)


@dataclass
class WorkerGraphsManager:
    """
    Manages the worker graphs that this worker is responsible for, and the queues
    for each graph and request. Also keeps track of which nodes belong
    to which worker graphs, and which worker graphs belong to which graph walks, for
    routing external outputs to the correct worker.
    """
    queues: dict[str, WorkerGraphQueues] # worker graph id to queues
    per_request_info: dict[str, PerRequestInfo] # request id to info
    base_sharding_config: ShardingConfig
    worker_id: str

    # The following two are for routing purposes:
    all_worker_graph_ids_to_graph_walks: dict[str, set[str]] # for worker graphs on different workers too
    all_worker_graph_ids_to_nodes: dict[str, set[str]] # for worker graphs on different workers too
    all_worker_graph_ids_to_dyn_loops: dict[str, set[str]]

    # Maps node_name -> partition_name. Populated from the model's partitions
    # and graph walk definitions. Used to look up which partition a node belongs
    # to in the colocated case.
    node_to_partition: dict[str, str] = field(default_factory=dict)

    # Inverted index: (graph_walk, node_name) -> worker_graph_id.
    # Built in __post_init__ from all_worker_graph_ids_to_graph_walks +
    # all_worker_graph_ids_to_nodes. Lets get_worker_graph_id_for_node skip
    # the linear scan over the request's worker_graph_ids.
    walk_node_to_worker_graph_id: dict[tuple[str, str], str] = field(default_factory=dict)

    def __post_init__(self):
        for wg_id, walks in self.all_worker_graph_ids_to_graph_walks.items():
            for walk in walks:
                for node in self.all_worker_graph_ids_to_nodes.get(wg_id, set()):
                    # Multiple worker graphs may share a (walk, node) only when
                    # walks are co-partitioned; the last write wins. This index
                    # is only consulted with (walk, node) pairs that the
                    # request's per_request_info already includes, so the
                    # ambiguity is moot for routing.
                    self.walk_node_to_worker_graph_id[(walk, node)] = wg_id

    def update_request_info(
        self, request_id: str,
        partition_name,
        current_fwd_info: CurrentForwardPassInfo | None=None,
        per_label_seq_info: PerLabelSeqInfo | None=None,
    ):
        req_info = self.per_request_info[request_id]
        part_info = req_info.per_partition_info[partition_name]

        if current_fwd_info is not None:
            graph_walk = current_fwd_info.graph_walk
            if self.get_graph_walk(request_id, partition_name) != graph_walk:
                part_info.graph_walk_worker_graph_ids = [
                    graph_id for graph_id in self.per_request_info[request_id].worker_graph_ids \
                        if graph_walk in self.all_worker_graph_ids_to_graph_walks[graph_id]
                ]
            part_info.current_fwd_info = current_fwd_info

        if per_label_seq_info is not None:
            fwd_info = self.get_fwd_info(request_id, partition_name)
            fwd_info.per_label_seq_info.update(per_label_seq_info)

    def get_graph_walk(self, request_id: str, partition_name: str):
        return self.get_fwd_info(request_id, partition_name).graph_walk

    def get_seq_info(self, request_id: str, partition_name: str):
        return self.get_fwd_info(request_id, partition_name).per_label_seq_info

    def get_fwd_number(self, request_id: str, partition_name: str):
        return self.get_fwd_info(request_id, partition_name).fwd_index

    def has_partition(self,  request_id: str, partition_name: str):
        return partition_name in self.per_request_info[request_id].per_partition_info

    def get_fwd_info(self, request_id: str, partition_name: str):
        part_info = self.per_request_info[request_id].per_partition_info[partition_name]
        return part_info.current_fwd_info

    def get_partition_for_node(self, node_name: str) -> str | None:
        """Look up which partition a node belongs to."""
        return self.node_to_partition.get(node_name)

    def process_new_inputs(
        self,
        request_id: str,
        inputs: list[GraphEdge],
        can_buffer: bool=True
    ) -> list[GraphEdge]:
        """Route arriving inputs to the per-request io of every active worker
        graph for this request.

        Returns the leftover edges that no worker graph claimed (their
        ``next_node`` lives on a different worker). Caller uses these for
        cross-worker routing.
        """
        for part_info in self.per_request_info[request_id].per_partition_info.values():
            worker_graph_ids = part_info.graph_walk_worker_graph_ids
            for worker_graph_id in worker_graph_ids:
                inputs = self.queues[worker_graph_id].process_new_inputs(
                    request_id, inputs, can_buffer=can_buffer
                )
        return inputs

    def process_new_streaming_inputs(
        self,
        request_id: str,
        inputs: list[GraphEdge],
        can_buffer: bool=True
    ) -> list[GraphEdge]:
        for part_info in self.per_request_info[request_id].per_partition_info.values():
            worker_graph_ids = part_info.graph_walk_worker_graph_ids
            for worker_graph_id in worker_graph_ids:
                inputs = self.queues[worker_graph_id].process_new_streaming_inputs(
                    request_id, inputs, can_buffer=can_buffer
                )
        return inputs

    def get_worker_graph_id_for_node(
        self, request_id: str, node_name: str
    ) -> str:
        partition = self.get_partition_for_node(node_name)
        graph_walk = self.get_graph_walk(request_id, partition)
        wg_id = self.walk_node_to_worker_graph_id.get((graph_walk, node_name))
        if wg_id is None:
            raise RuntimeError(
                f"Could not find worker graph for node {node_name!r}, "
                f"request {request_id!r}, graph_walk {graph_walk!r}"
            )
        return wg_id

    def mark_node_complete(
        self, request_id: str, worker_graph_id: str, node_name: str,
    ) -> NodeCompletionOutput:
        """Complete a node in the given worker graph's per-request io.

        Returns the registry's ``NodeCompletionOutput`` carrying
        ``output_edges`` (entity static outputs + any loop terminal outputs)
        and ``filtered_signals`` (loop-back (name, dest) pairs to drop).
        """
        return self.queues[worker_graph_id].mark_node_complete(request_id, node_name)

    def register_output_loop_indices(
        self, request_id: str,
        loop_indices: NestedLoopIndices,
        output_name: str
    ):
        self.per_request_info[request_id].output_loop_indices[output_name] = loop_indices

    def get_output_loop_indices(self, request_id: str):
        return self.per_request_info[request_id].output_loop_indices

    def process_node_outputs(
        self, request_id: str,
        node_name: str,
        outputs: list[GraphEdge],
        graph_walk: str,
    ) -> NodeOutputRouting:
        """After a node has finished, route its outputs.

        Updates ready/waiting state in worker graphs on this worker, and
        builds the cross-worker routing map for edges destined elsewhere.
        """
        # (0) separate streaming edges — they bypass the queue system
        streaming_edges = [edge for edge in outputs if edge.is_streaming]
        non_streaming_outputs = [edge for edge in outputs if not edge.is_streaming]

        # (1) find persist (to-conductor) and new-token-output edges
        to_conductor = [edge for edge in non_streaming_outputs if edge.persist]
        new_token_outputs = [edge for edge in non_streaming_outputs if edge.conductor_new_token]

        sharding_config = self.per_request_info[request_id].sharding_config
        group = sharding_config.get_sharding_group(node_name, graph_walk)
        # No group → singleton/non-TP; treat as rank 0.
        is_first_tp_rank = group is None or group._tp_rank == 0

        # (2) route each output edge to its destination worker graph via the
        # inverted index. Compute the per-rank fanout first; ingest *this
        # worker's* sliced edge into the local wg (so the local consumer
        # sees the right tensor_info); fan the rest out to cross-worker
        # routing. Edges that don't map to any local wg fall through to
        # external for the same cross-worker pass.
        routed_to_this_worker: list[GraphEdge] = []
        external_outputs: list[GraphEdge] = []
        to_workers: dict[str, list[GraphEdge]] = {}
        for edge in non_streaming_outputs:
            wg_id = self.walk_node_to_worker_graph_id.get((graph_walk, edge.next_node))
            if wg_id is not None and wg_id in self.queues:
                fanout = sharding_config.fanout_graph_edges(
                    edge, source_node=node_name,
                    source_graph_walk=graph_walk,
                    dest_graph_walk=graph_walk,
                )
                this_worker_edge = fanout.pop(self.worker_id, None)
                if this_worker_edge is not None:
                    leftover = self.queues[wg_id].process_new_inputs(
                        request_id, [this_worker_edge], can_buffer=True,
                    )
                    if leftover:
                        # local wg declined (e.g., no per-request io yet);
                        # route to self via the cross-worker path
                        to_workers.setdefault(self.worker_id, []).extend(leftover)
                    else:
                        routed_to_this_worker.append(this_worker_edge)
                for (wkr, wkr_edge) in fanout.items():
                    to_workers.setdefault(wkr, []).append(wkr_edge)
            else:
                external_outputs.append(edge)

        # Sweep all worker graphs the request is registered with for THIS walk
        # to see which became done. A wg can become done without having
        # ingested any edge in this call — e.g. when the just-completed node's
        # outputs all target EMPTY_DESTINATION / EMIT_TO_CLIENT / a streaming
        # partition (Orpheus prefill, BAGEL vae_decoder, Code2Wav).
        completed_worker_graph_ids: list[str] = []
        for wg_id in self.per_request_info[request_id].worker_graph_ids:
            if graph_walk not in self.all_worker_graph_ids_to_graph_walks[wg_id]:
                continue
            queue = self.queues[wg_id]
            if queue.is_done(request_id):
                completed_worker_graph_ids.append(wg_id)
                queue.reset(request_id)

        # (3) get mapping of worker to external outputs
        # Skip edges whose next_node is a special destination (e.g.,
        # EMIT_TO_CLIENT is a virtual destination, not a real node on any worker).
        # Note: persist edges may ALSO route to a worker
        # (e.g., concat_text outputs text_emb -> LLM with persist=True),
        # so we do NOT filter on persist here.
        emit_to_client: list[GraphEdge] = []
        for edge in external_outputs:
            node_graph_walk = NodeAndGraphWalk(
                node=edge.next_node, graph_walk=graph_walk
            )
            # Compute the per-worker fanout once so it is available in both
            # the SPECIAL_DESTINATIONS branch (emit_to_client.extend) and
            # the cross-worker dispatch loop below. Computing it inside only
            # one branch leaves ``fanout`` unbound when control reaches the
            # other — a latent crash in any multi-worker config whose
            # external edges target a known node on a remote worker
            # (e.g. BAGEL CFG-parallel's cross-LLM edges).
            fanout = sharding_config.fanout_graph_edges(
                edge, source_node=node_name,
                source_graph_walk=graph_walk,
                dest_graph_walk=graph_walk,
            )
            if node_graph_walk not in self.per_request_info[request_id].node_to_workers:
                if edge.next_node in SPECIAL_DESTINATIONS or edge.persist:
                    if edge.next_node == EMIT_TO_CLIENT:
                        emit_to_client.extend(fanout.values())
                    continue  # e.g., emit_to_client — already captured in to_conductor
                raise ValueError(
                    f"Output edge targets unknown node/graph walk: {node_graph_walk}. "
                    f"Check graph construction."
                )
            for (wkr, wkr_edge) in fanout.items():
                to_workers.setdefault(wkr, []).append(wkr_edge)

        # (4) route streaming edges — find destination workers for streaming outputs
        streaming_to_workers: dict[str, list[GraphEdge]] = {}
        streaming_local: list[GraphEdge] = []
        my_node_names = set()
        for gid in self.per_request_info[request_id].worker_graph_ids:
            my_node_names.update(self.all_worker_graph_ids_to_nodes.get(gid, []))

        for edge in streaming_edges:
            fanout = sharding_config.fanout_graph_edges(
                edge, source_node=node_name,
                source_graph_walk=graph_walk,
                dest_graph_walk=None
            )
            this_worker_edge = fanout.pop(self.worker_id, None)
            if this_worker_edge:
                streaming_local.append(this_worker_edge)
            for (wkr, wkr_edge) in fanout.items():
                streaming_to_workers.setdefault(wkr, []).append(wkr_edge)

        logger.debug(
            ("Finished processing outputs from rid %s. \n"
             "Routed to this worker: %s; sent to others: %s; persist signals: %s; streaming: %d"),
            request_id, format_graph_edge_list(routed_to_this_worker),
            format_graph_edge_list(external_outputs), format_graph_edge_list(to_conductor),
            len(streaming_edges),
        )
        if completed_worker_graph_ids:
            logger.debug("Completed %d worker graphs", len(completed_worker_graph_ids))

        return NodeOutputRouting(
            routed_to_this_worker_graph=routed_to_this_worker,
            persist=to_conductor,
            to_workers=to_workers,
            emit_to_client=emit_to_client,
            new_token_outputs=new_token_outputs,
            completed_worker_graph_ids=completed_worker_graph_ids,
            streaming_to_workers=streaming_to_workers,
            streaming_local=streaming_local,
            is_first_tp_rank=is_first_tp_rank
        )

    def stop_loops(
        self, request_id: str,
        partition: str,
        loop_names: set[str],
        req_info: CurrentForwardPassInfo | None = None,
        last_node_run: str | None = None
    ) -> set[NameAndDest]:
        """Register a finish signal for each named loop across this partition's
        worker graphs and return the union of their loop-back ``(name, dest)``
        pairs so the caller can drop them from output routing on the iter
        that triggered the stop.

        If ``req_info`` and ``last_node_run`` are provided AND the last-run
        node lives on a given worker graph, this also snapshots the current
        nested-loop iteration indices into ``req_info.loop_stop_times[name]``
        as a ``NestedLoopIndices`` (used by the conductor's stop-ordering to
        suppress duplicate stop messages).
        """
        part_info = self.per_request_info[request_id].per_partition_info[partition]
        worker_graph_ids = part_info.graph_walk_worker_graph_ids
        stopped_loop_back_signals: set[NameAndDest] = set()
        # In disaggregated mode the same Loop name can exist on multiple worker
        # graphs (each with its own _finish_signal), so this still has to fan out.
        for worker_graph_id in worker_graph_ids:
            stopped_loop_back_signals |= self.queues[worker_graph_id].stop_loops(
                request_id, loop_names,
            )

        # loop_stop_times is a single observation per loop, so we only need
        # the worker graph that owns the last-run node. Direct index lookup
        # instead of iterating.
        if req_info is not None and last_node_run is not None:
            graph_walk = self.get_graph_walk(request_id, partition)
            owner_wg_id = self.walk_node_to_worker_graph_id.get(
                (graph_walk, last_node_run)
            )
            if owner_wg_id is not None and owner_wg_id in self.queues:
                wgio = self.queues[owner_wg_id].per_request_queues.get(request_id)
                if wgio is not None:
                    for name in loop_names & wgio.loops.keys():
                        req_info.loop_stop_times[name] = wgio.get_nested_loop_idxs(
                            target_loop_name=name,
                        )
        return stopped_loop_back_signals

    def get_nested_loop_idxs_for_node(
        self, request_id: str, partition: str, node_name: str
    ) -> NestedLoopIndices:
        graph_walk = self.get_graph_walk(request_id, partition)
        wgid = self.walk_node_to_worker_graph_id[ (graph_walk, node_name)]
        wgio = self.queues[wgid].per_request_queues.get(request_id)
        return wgio.get_nested_loop_idxs_for_node(node_name)

    def get_dynamic_loop_iters(
        self, request_id: str,
        partition: str,
    ) -> dict[str, int]:
        part_info = self.per_request_info[request_id].per_partition_info[partition]
        worker_graph_ids = part_info.graph_walk_worker_graph_ids

        iter_counts: dict[str, int] = {}
        for worker_graph_id in worker_graph_ids:
            iter_counts.update(
                self.queues[worker_graph_id].get_dynamic_loop_iters(request_id)
            )
        return iter_counts

    def add_request(
        self, request_id: str,
        partition_worker_graph_ids: list[str], # for this worker's worker graphs
        worker_graph_to_workers: dict[str, list[str]], # for other / all worker graphs
        current_fwd_info: CurrentForwardPassInfo,
    ):
        """
        Set up queues and info for a new request. This includes adding the request
        to the relevant worker graph queues, and updating the mapping of which worker
        is responsible for which nodes for this request (for output routing).
        """

        current_graph_walk = current_fwd_info.graph_walk
        my_worker_graph_ids = [gid for gid in partition_worker_graph_ids if gid in self.queues]
        partition_name = current_fwd_info.partition_name

        for graph_id in partition_worker_graph_ids:
            if graph_id in self.queues:
                self.queues[graph_id].add_request(request_id)


        if request_id not in self.per_request_info:
            # Note: conductor.py passes the same worker_graph_to_worker dict
            # on every NewRequest for a given request(i.e., for every partition).
            # So the below logic only needs to be done once.
            node_to_workers = {}
            dyn_loop_to_workers = {}
            for worker_graph_id, worker_ids in worker_graph_to_workers.items():
                if worker_graph_id not in self.all_worker_graph_ids_to_graph_walks:
                    continue
                for graph_walk in self.all_worker_graph_ids_to_graph_walks[worker_graph_id]:
                    node_to_workers.update({
                        NodeAndGraphWalk(
                            node=name,
                            graph_walk=graph_walk
                        ): worker_ids for name in self.all_worker_graph_ids_to_nodes[worker_graph_id]
                    })

                    for loop_name in self.all_worker_graph_ids_to_dyn_loops[worker_graph_id]:
                        dyn_loop_to_workers.setdefault(NodeAndGraphWalk(
                            node=loop_name,
                            graph_walk=graph_walk
                        ), []).extend(worker_ids)


            sharding_config = self.base_sharding_config.clone_empty()
            sharding_config.setup(node_to_workers)
            self.per_request_info[request_id] = PerRequestInfo(
                node_to_workers=node_to_workers,
                dyn_loop_to_workers=dyn_loop_to_workers,
                worker_graph_ids=my_worker_graph_ids,
                sharding_config=sharding_config,
                per_partition_info={
                    partition_name: PerPartitionInfo(
                        graph_walk_worker_graph_ids=[
                            graph_id for graph_id in my_worker_graph_ids
                            if current_graph_walk in self.all_worker_graph_ids_to_graph_walks[graph_id]
                        ],
                        current_fwd_info=current_fwd_info,
                    )
                }
            )
        else:
            # Just do partition-specific work: updating worker_graph_ids, instantiating PerPartitionInfo
            req_info = self.per_request_info[request_id]
            req_info.worker_graph_ids += my_worker_graph_ids
            req_info.per_partition_info[partition_name] = PerPartitionInfo(
                graph_walk_worker_graph_ids=[
                    graph_id for graph_id in my_worker_graph_ids
                    if current_graph_walk in self.all_worker_graph_ids_to_graph_walks[graph_id]
                ],
                current_fwd_info=current_fwd_info
            )

    def remove_request(self, request_id: str):
        if request_id in self.per_request_info:
            for queue_id in self.per_request_info[request_id].worker_graph_ids:
                self.queues[queue_id].remove_request(request_id)
            del self.per_request_info[request_id]

    def get_dyn_loop_workers(self, request_id: str, partition_name: str, loop_name: str):
        return self.per_request_info[request_id].dyn_loop_to_workers[NodeAndGraphWalk(
            node=loop_name, graph_walk=self.get_graph_walk(request_id, partition_name)
        )]

    def buffer_persist_signals(
            self, request_id: str,
            signals: list[GraphEdge]
        ):
        """Extend the pending persist signals for a request."""
        self.per_request_info[request_id].pending_persist_signals.extend(signals)

    def buffer_new_tokens(
        self, request_id: str,
        new_tokens: dict[str, list[int]]
    ):
        """Update the pending new tokens for a request."""
        for name, tokens in new_tokens.items():
            if name not in self.per_request_info[request_id].pending_new_tokens:
                self.per_request_info[request_id].pending_new_tokens[name] = []
            self.per_request_info[request_id].pending_new_tokens[name].extend(tokens)

    def buffer_output_signals(self, request_id: str, out_signals: list[GraphEdge]):
        self.per_request_info[request_id].current_output_chunks += [
            signal.name for signal in out_signals
        ]

    def flush_persist_signals(self, request_id: str) -> dict[str, list[TensorPointerInfo]]:
        """Pop and return all buffered persist signals for a request.

        Converts from internal list[GraphEdge] to the dict format
        expected by the conductor (name -> list[TensorPointerInfo]).
        """
        info = self.per_request_info[request_id]
        signals = info.pending_persist_signals
        info.pending_persist_signals = []
        result: dict[str, list[TensorPointerInfo]] = {}
        for edge in signals:
            result[edge.name] = edge.tensor_info
        return result

    def flush_new_tokens(self, request_id: str) -> dict[str, list[int]]:
        """Pop and return all buffered new tokens for a request."""
        info = self.per_request_info[request_id]
        new_tokens = info.pending_new_tokens
        info.pending_new_tokens = {}
        return new_tokens

    def flush_output_signals(self, request_id: str) -> list[str]:
        info = self.per_request_info[request_id]
        out_chunks = list(info.current_output_chunks)  # copy before clearing
        info.current_output_chunks.clear()
        return out_chunks
