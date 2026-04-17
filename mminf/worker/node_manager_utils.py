import logging
from copy import deepcopy
from dataclasses import dataclass, field

from mminf.communication.tensors import TensorCommunicationManager
from mminf.conductor.request_info import CurrentForwardPassInfo, PerLabelSeqInfo
from mminf.graph.base import (
    DynamicLoop,
    FilteredEdges,
    GraphEdge,
    GraphNode,
    GraphSection,
    Loop,
    Parallel,
    Sequential,
    TensorPointerInfo,
)
from mminf.graph.loop_index import build_loop_index_tree, update_loop_index_tree
from mminf.graph.request_queues import (
    PerRequestNodeQueues,
    ProcessedInputs,
    format_graph_edge_list,
)
from mminf.graph.special_destinations import EMIT_TO_CLIENT, SPECIAL_DESTINATIONS
from mminf.model.base import WorkerGraph
from mminf.streaming.stream_buffer import StreamBuffer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NodeAndGraphWalk:
    """
    Tuple of node name and graph walk, e.g., (LLM, decode) or (flow, image_gen)
    """
    node: str
    graph_walk: str


@dataclass
class NodeOutputRouting:
    routed_to_this_worker_graph: list[GraphEdge]
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
    per_request_queues: dict[str, PerRequestNodeQueues] # request_id -> queue
    tensor_manager: TensorCommunicationManager

    def process_new_inputs(self, request_id: str, inputs: list[GraphEdge]) -> ProcessedInputs:
        """
        Add new inputs for a request, and update waiting/ready nodes accordingly.
        Returns any signals that should be sent to other worker graphs.
        """
        return self.per_request_queues[request_id].process_new_inputs(inputs)

    def process_new_streaming_inputs(self, request_id: str, inputs: list[GraphEdge]) -> ProcessedInputs:
        """
        Add new inputs for a request, and update waiting/ready nodes accordingly.
        Returns any signals that should be sent to other worker graphs.
        """
        return self.per_request_queues[request_id].process_streaming_input(inputs)

    def is_done(self, request_id) -> bool:
        q = self.per_request_queues[request_id]
        return q.waiting is None and len(q.ready) == 0 and len(q.waiting_for_stream) == 0

    def add_request(self, request_id: str):
        """
        Initialize queues for a new request
        """
        section_copy = deepcopy(self.worker_graph.section)
        section_copy.register_communication_info(
            self.tensor_manager, request_id
        )
        self.per_request_queues[request_id] = PerRequestNodeQueues(
            waiting=section_copy,
            full_section=section_copy,
            worker_graph_id=self.worker_graph_id
        )

    def remove_request(self, request_id: str):
        """
        Delete queues for a completed/removed request (saw EOS)
        """
        if request_id in self.per_request_queues:
            del self.per_request_queues[request_id]

    def get_ready_node_names(self) -> dict[str, str]:
        """
        Returns mapping of request id to ready node names for that request
        """
        return {
            request_id: [s.name for s in q.ready] \
                for (request_id, q) in self.per_request_queues.items()
        }

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
            pop_idxs = set(
                [i for i, node in enumerate(q.ready) if node.name in set(node_names)]
            )
            nodes = [q.ready[i] for i in pop_idxs]
            q.ready = [node for i, node in enumerate(q.ready) if i not in pop_idxs]
        return nodes

    def push_back_node(
        self, request_id: str, node: GraphNode
    ) -> None:
        """Push a previously popped node back onto the ready queue (e.g., after OOM hold)."""
        if request_id in self.per_request_queues:
            self.per_request_queues[request_id].ready.append(node)

    def reset(self, request_id):
        """
        At the end of a worker graph, reset the queues for a request so it can
        be used for the next full model forward pass.
        """
        self.per_request_queues[request_id].reset()

    def stop_loops(self, request_id: str, loop_names: set[str]):
        def _stop_loops(section: GraphSection):
            if isinstance(section, Sequential) or isinstance(section, Parallel):
                for sec in section.sections:
                    _stop_loops(sec)
                return
            if isinstance(section, DynamicLoop):
                if section.name in loop_names:
                    section.register_finished()
            if isinstance(section, Loop):
                # including dynamic loops
                _stop_loops(section._curr_iter_section)
        _stop_loops(self.per_request_queues[request_id].waiting)

    def get_dynamic_loop_iters(self, request_id: str) -> dict[str, int]:
        iter_dict = {}
        def _get_iters(section: GraphSection):
            if isinstance(section, Sequential) or isinstance(section, Parallel):
                for sec in section.sections:
                    _get_iters(sec)
                return
            if isinstance(section, DynamicLoop):
                iter_dict[section.name] = section.curr_iter
            if isinstance(section, Loop):
                # including dynamic loops
                _get_iters(section._curr_iter_section)
        _get_iters(self.per_request_queues[request_id].waiting)
        return iter_dict


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
    node_to_worker: dict[NodeAndGraphWalk, str]  # for all nodes
    dyn_loop_to_workers: dict[NodeAndGraphWalk, list[str]]
    worker_graph_ids: list[str] # for this worker

    pending_persist_signals: list[GraphEdge] = field(default_factory=list)
    pending_new_tokens: dict[str, list[int]] = field(default_factory=dict)
    stream_buffers: dict[str, StreamBuffer] = field(default_factory=dict)  # edge_name -> StreamBuffer
    current_output_chunks: list[str] = field(default_factory=list)

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

    # The following two are for routing purposes:
    all_worker_graph_ids_to_graph_walks: dict[str, set[str]] # for worker graphs on different workers too
    all_worker_graph_ids_to_nodes: dict[str, set[str]] # for worker graphs on different workers too
    all_worker_graph_ids_to_dyn_loops: dict[str, set[str]]

    # Maps node_name -> partition_name. Populated from the model's partitions
    # and graph walk definitions. Used to look up which partition a node belongs
    # to in the colocated case.
    node_to_partition: dict[str, str] = field(default_factory=dict)

    def update_request_info(
        self, request_id: str,
        partition_name,
        current_fwd_info: CurrentForwardPassInfo | None=None,
        per_label_seq_info: PerLabelSeqInfo | None=None,
    ):
        req_info = self.per_request_info[request_id]
        part_info = req_info.per_partition_info[partition_name]

        if current_fwd_info is not None:
            partition_name = partition_name or getattr(current_fwd_info, 'partition_name', 'default')
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
    ):
        """
        Updates queues with new inputs for a request.

        Routes inputs to the worker graphs of the currently-active walks
        for each partition this request participates in.
        """
        for part_info in self.per_request_info[request_id].per_partition_info.values():
            worker_graph_ids = part_info.graph_walk_worker_graph_ids
            for worker_graph_id in worker_graph_ids:
                inputs = self.queues[worker_graph_id].process_new_inputs(request_id, inputs).for_other_worker_graphs
        return inputs

    def process_new_streaming_inputs(
        self,
        request_id: str,
        inputs: list[GraphEdge],
    ):
        """
        Updates queues with new inputs for a request.

        Routes inputs to the worker graphs of the currently-active walks
        for each partition this request participates in.
        """
        for part_info in self.per_request_info[request_id].per_partition_info.values():
            worker_graph_ids = part_info.graph_walk_worker_graph_ids
            for worker_graph_id in worker_graph_ids:
                inputs = self.queues[worker_graph_id].process_new_streaming_inputs(
                    request_id, inputs
                ).for_other_worker_graphs
        return inputs


    def get_worker_graph_id_for_node(
        self, request_id: str, node_name: str
    ) -> str:
        partition = self.get_partition_for_node(node_name)
        graph_walk = self.get_graph_walk(request_id, partition)
        for gid in self.per_request_info[request_id].worker_graph_ids:
            if (graph_walk in self.all_worker_graph_ids_to_graph_walks[gid]) and \
                        (node_name in self.all_worker_graph_ids_to_nodes[gid]):
                return gid
        raise RuntimeError(f"Could not find worker graph for node {node_name}, request {request_id}")


    def get_waiting_node(
        self, request_id: str, worker_graph_id: str
    ):
        return self.queues[worker_graph_id].per_request_queues[request_id].waiting


    def complete_loops(
        self, request_id: str, worker_graph_id: str,
        output_edges: list[GraphEdge], done_node: str
    ) -> FilteredEdges:
        queue = self.queues[worker_graph_id].per_request_queues[request_id]
        if queue.waiting is None:
            return FilteredEdges(
                kept=output_edges,
                filtered_out=[]
            )
        out = queue.waiting.complete_loops(done_node)
        queue.waiting = out.new_waiting

        filter_result = out.filter_out_loop_back(output_edges)
        filter_result.kept += out.outputs
        return filter_result


    def process_node_outputs(
        self, request_id: str,
        outputs: list[GraphEdge],
        graph_walk: str,
    ) -> NodeOutputRouting:
        """
        After a node has finished processing, use its outputs to update
        worker graph queues, and return any outputs that should be sent to other
        worker graphs or the conductor.

        I.e., it updates ready/waiting queues for worker graphs on this current
        worker, and directs external outputs to worker graphs on the appropriate
        (different) worker.
        """
        # (0) separate streaming edges — they bypass the queue system
        streaming_edges = [edge for edge in outputs if edge.is_streaming]
        non_streaming_outputs = [edge for edge in outputs if not edge.is_streaming]

        # (1) find back_to_conductor flags
        to_conductor = [edge for edge in non_streaming_outputs if edge.persist]
        new_token_outputs = [edge for edge in non_streaming_outputs if edge.conductor_new_token]

        # (2) process all internal-facing outputs
        worker_graph_ids = [
            gid
            for gid in self.per_request_info[request_id].worker_graph_ids
            if graph_walk in self.all_worker_graph_ids_to_graph_walks[gid]
        ]

        completed_worker_graph_ids = []
        routed_to_this_worker: list[GraphEdge] = [] # list of graph edges
        external_outputs: list[GraphEdge] = non_streaming_outputs
        for worker_graph_id in worker_graph_ids:
            queue = self.queues[worker_graph_id] # ready / waiting graph node queue
            # process_new_inputs consumes outputs that are used as
            # node inputs within `queue`
            processed_inputs = queue.process_new_inputs(request_id, external_outputs)

            # keep updating outputs to be the edges that have not yet been utilized
            external_outputs = processed_inputs.for_other_worker_graphs
            routed_to_this_worker += processed_inputs.routed_to_this_worker_graph
            if queue.is_done(request_id):
                completed_worker_graph_ids.append(worker_graph_id)
                queue.reset(request_id)
            # all outputs left over at this point are external outputs (to nodes
            # in different workers)

        # (3) get mapping of worker to external outputs
        # Skip edges whose next_node is a special destination (e.g.,
        # EMIT_TO_CLIENT is a virtual destination, not a real node on any worker).
        # Note: back_to_conductor edges may ALSO route to a worker
        # (e.g., concat_text outputs text_emb -> LLM with back_to_conductor=True),
        # so we do NOT filter on back_to_conductor here.
        to_workers: dict[str, list[GraphEdge]] = {}
        emit_to_client: list[GraphEdge] = []
        for edge in external_outputs:
            node_graph_walk = NodeAndGraphWalk(
                node=edge.next_node, graph_walk=graph_walk
            )
            if node_graph_walk not in self.per_request_info[request_id].node_to_worker:
                if edge.next_node in SPECIAL_DESTINATIONS or edge.persist:
                    if edge.next_node == EMIT_TO_CLIENT:
                        emit_to_client.append(edge)
                    continue  # e.g., emit_to_client — already captured in to_conductor
                raise ValueError(
                    f"Output edge targets unknown node/graph walk: {node_graph_walk}. "
                    f"Check graph construction."
                )
            worker_id = self.per_request_info[request_id].node_to_worker[node_graph_walk]
            if worker_id not in to_workers:
                to_workers[worker_id] = []
            to_workers[worker_id].append(edge)

        # (4) route streaming edges — find destination workers for streaming outputs
        streaming_to_workers: dict[str, list[GraphEdge]] = {}
        streaming_local: list[GraphEdge] = []
        my_node_names = set()
        for gid in self.per_request_info[request_id].worker_graph_ids:
            my_node_names.update(self.all_worker_graph_ids_to_nodes.get(gid, []))

        for edge in streaming_edges:
            if edge.next_node in my_node_names:
                # Destination node is on this worker — store locally
                streaming_local.append(edge)
            else:
                # Find the worker that has this node (check all graph walks)
                dest_worker = None
                for wg_id, worker_id in self.per_request_info[request_id].node_to_worker.items():
                    if wg_id.node == edge.next_node:
                        dest_worker = worker_id
                        break
                if dest_worker is not None:
                    streaming_to_workers.setdefault(dest_worker, []).append(edge)
                else:
                    logger.warning(
                        "Streaming edge to %s has no known destination worker", edge.next_node,
                    )

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
        )

    def stop_loops(
        self, request_id: str,
        partition: str,
        loop_names: set[str],
        req_info: CurrentForwardPassInfo | None = None,
        last_node_run: str | None = None
    ):
        """
        Stops dynamic loops based on stop signals. If req_info is also passed in,
        then this also updates req_info.loop_stop_times with the urrent loop indices.
        """
        part_info = self.per_request_info[request_id].per_partition_info[partition]
        worker_graph_ids = part_info.graph_walk_worker_graph_ids
        for worker_graph_id in worker_graph_ids:
            self.queues[worker_graph_id].stop_loops(request_id, loop_names)
            waiting = self.queues[worker_graph_id].per_request_queues[request_id].waiting

            if req_info is not None and (
                last_node_run in self.all_worker_graph_ids_to_nodes[worker_graph_id]
            ):
                for name in loop_names:
                    if name in req_info.loop_stop_times:
                        update_loop_index_tree(
                            index_tree=req_info.loop_stop_times[name],
                            graph=waiting,
                            fwd_idx=req_info.fwd_index
                        )
                    else:
                        req_info.loop_stop_times[name] = build_loop_index_tree(
                            graph=waiting,
                            fwd_idx=req_info.fwd_index
                        )

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
        worker_graph_to_worker: dict[str, str], # for other / all worker graphs
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
            node_to_worker = {}
            dyn_loop_to_workers = {}
            for worker_graph_id, worker_id in worker_graph_to_worker.items():
                if worker_graph_id not in self.all_worker_graph_ids_to_graph_walks:
                    continue
                for graph_walk in self.all_worker_graph_ids_to_graph_walks[worker_graph_id]:
                    node_to_worker.update({
                        NodeAndGraphWalk(
                            node=name,
                            graph_walk=graph_walk
                        ): worker_id for name in self.all_worker_graph_ids_to_nodes[worker_graph_id]
                    })

                    for loop_name in self.all_worker_graph_ids_to_dyn_loops[worker_graph_id]:
                        dyn_loop_to_workers.setdefault(NodeAndGraphWalk(
                            node=loop_name,
                            graph_walk=graph_walk
                        ), []).append(worker_id)

            self.per_request_info[request_id] = PerRequestInfo(
                node_to_worker=node_to_worker,
                dyn_loop_to_workers=dyn_loop_to_workers,
                worker_graph_ids=my_worker_graph_ids,
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
