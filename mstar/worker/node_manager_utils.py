import logging
from copy import deepcopy
from dataclasses import dataclass, field

from mstar.communication.tensors import TensorCommunicationManager
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.engine.resources import PublishedInfo
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    NameAndDest,
    NodeAndGraphWalk,
    NodeCompletionOutput,
)
from mstar.graph.graph_io import WorkerGraphIO
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
    completed_worker_graph_ids: list[int] = field(default_factory=list)
    streaming_to_workers: dict[str, list[GraphEdge]] = field(default_factory=dict)  # streaming edges to other workers
    streaming_local: list[GraphEdge] = field(default_factory=list)  # streaming edges staying on this worker


@dataclass
class WorkerGraphQueues:
    """
    For a single worker graph, keeps track of which nodes are waiting on which
    inputs for each request, and which nodes are ready to run per request.
    """
    worker_graph_id: int
    graph_walks: set[str] # e.g., this worker graph is active during decode and image_gen
                          # but not the prefill graph walk
    worker_graph: WorkerGraph
    per_request_queues: dict[int, WorkerGraphIO]
    tensor_manager: TensorCommunicationManager

    def __post_init__(self):
        self.nodes = set(self.worker_graph.section.get_nodes().keys())
        self.loops = set(self.worker_graph.section.get_loops().keys())

    def process_new_inputs(
        self, rid: int, inputs: list[GraphEdge],
        can_buffer: bool=True
    ) -> list[GraphEdge]:
        """Ingest inputs into this worker graph's per-request io.

        Returns the edges that were NOT routed here (because their next_node
        is not a node in this worker graph) so the caller can try the next
        worker graph. Works for both normal and streaming inputs — the per-
        node io routes by name and lets ``ReadySignals.is_ready_for_streaming``
        light up the streaming readiness set on its own.
        """
        assert rid in self.per_request_queues, \
            f"Tried to process new inputs for unknown request ID {rid}"
        queue = self.per_request_queues[rid]
        not_ingested: list[GraphEdge] = []
        for inp in inputs:
            if not queue.ingest_input(inp, can_buffer):
                not_ingested.append(inp)
        return not_ingested

    def is_done(self, rid) -> bool:
        assert rid in self.per_request_queues, \
            f"Tried to check queue done state for unknown request ID {rid}"
        queue = self.per_request_queues[rid]
        return queue.wg_state_registry.is_done

    def add_request(self, rid: int):
        """
        Initialize queues for a new request
        """
        section_copy = deepcopy(self.worker_graph.section)
        queue = WorkerGraphIO(section_copy, wg_id=self.worker_graph_id)
        queue.register_communication_info(
            self.tensor_manager, rid
        )
        self.per_request_queues[rid] = queue

    def remove_request(self, rid: int):
        """
        Delete queues for a completed/removed request (saw EOS)
        """
        self.per_request_queues.pop(rid, None)

    def get_ready_node_names(self) -> dict[int, set[str]]:
        """
        Returns mapping of request id to ready node names for that request
        """
        return {
            rid: q.ready_node_names \
                for (rid, q) in self.per_request_queues.items()
        }

    def get_ready_for_streaming(self, rid: int):
        assert rid in self.per_request_queues, \
            f"Tried to check ready for streaming for unknown request ID {rid}"
        return self.per_request_queues[rid].ready_for_streaming

    def pop_ready_nodes(
        self, rid: int, node_names: list[str]
    ) -> list[GraphNode]:
        """
        Remove the given node names from the ready queue for the request and
        return the corresponding GraphNode objects.
        """
        nodes = []
        if rid in self.per_request_queues:
            q = self.per_request_queues[rid]
            for name in node_names:
                q.ready_node_names.discard(name)
                nodes.append(q.nodes[name])
        return nodes

    def reset(self, rid):
        """
        At the end of a worker graph, reset the queues for a request so it can
        be used for the next full model forward pass.
        """
        self.per_request_queues[rid].clear()

    def stop_loops(
        self, rid: int, loop_names: set[str]
    ) -> set[NameAndDest]:
        """Register a finish signal for each named loop and return the union
        of their ``_loop_back_inputs`` so the caller can drop those (name, dest)
        edges from the current iter's output routing.
        """
        assert rid in self.per_request_queues, \
            f"Tried to stop loops for unknown request ID {rid}"
        queue = self.per_request_queues[rid]
        loop_back_signals: set[NameAndDest] = set()
        for name in loop_names:
            if name not in queue.loops:
                continue
            queue.register_loop_finish_signal(name)
            loop_back_signals.update(queue.loops[name]._loop_back_inputs)
        return loop_back_signals

    def mark_node_complete(
        self, rid: int, node_name: str
    ) -> NodeCompletionOutput:
        """Complete a node in this worker graph's per-request io and return
        the registry's NodeCompletionOutput (output_edges + filtered_signals)."""
        assert rid in self.per_request_queues, \
            f"Tried to complete node {node_name!r} for unknown request ID {rid}"
        return self.per_request_queues[rid].mark_node_complete(node_name)

    def get_dynamic_loop_iters(self, rid: int) -> dict[str, int]:
        assert rid in self.per_request_queues, \
            f"Tried to get dynamic loop iters for unknown request ID {rid}"
        queue = self.per_request_queues[rid]
        return queue.get_loop_indices()


@dataclass
class PerPartitionInfo:
    current_fwd_info: CurrentForwardPassInfo
    # graph_walk_worker_graph_ids = worker graphs for current graph walk
    graph_walk_worker_graph_ids: list[int] = field(default_factory=list) # for this worker
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
    - partition_fwd_infos: per-partition forward info for the colocated case
        where multiple partitions run on the same worker
    - tensors: TBD
    """
    node_to_workers: dict[NodeAndGraphWalk, list[str]]  # for all nodes
    dyn_loop_to_workers: dict[NodeAndGraphWalk, list[str]]
    worker_graph_ids: list[int] # for this worker
    sharding_config: ShardingConfig

    stream_buffers: dict[str, StreamBuffer] = field(default_factory=dict)  # edge_name -> StreamBuffer

    per_partition_info: dict[str, PerPartitionInfo] = field(default_factory=dict)


@dataclass
class WorkerGraphsManager:
    """
    Manages the worker graphs that this worker is responsible for, and the queues
    for each graph and request. Also keeps track of which nodes belong
    to which worker graphs, and which worker graphs belong to which graph walks, for
    routing external outputs to the correct worker.
    """
    queues: dict[int, WorkerGraphQueues] # worker graph id to queues
    per_request_info: dict[int, PerRequestInfo] # rid handle to info
    base_sharding_config: ShardingConfig
    worker_id: str

    # The following two are for routing purposes:
    all_worker_graph_ids_to_graph_walks: dict[int, set[str]] # for worker graphs on different workers too
    all_worker_graph_ids_to_nodes: dict[int, set[str]] # for worker graphs on different workers too
    all_worker_graph_ids_to_dyn_loops: dict[int, set[str]]

    # Maps node_name -> partition_name. Populated from the model's partitions
    # and graph walk definitions. Used to look up which partition a node belongs
    # to in the colocated case.
    node_to_partition: dict[str, str] = field(default_factory=dict)

    # (graph_walk, node_name) -> worker_graph_id, built once at init.
    # PythonGraphRuntime has the same index; this copy stays until the three
    # readers below (process_node_outputs routing, get_nested_loop_idxs_for_node)
    # move to the runtime. Both are derived from the same immutable inputs, so
    # they cannot diverge.
    walk_node_to_worker_graph_id: dict[tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self):
        for wg_id, walks in self.all_worker_graph_ids_to_graph_walks.items():
            for walk in walks:
                for node in self.all_worker_graph_ids_to_nodes.get(wg_id, set()):
                    self.walk_node_to_worker_graph_id[(walk, node)] = wg_id

    def update_request_info(
        self, rid: int,
        partition_name,
        current_fwd_info: CurrentForwardPassInfo | None=None,
        resource_publish_info: dict[str, PublishedInfo] | None=None,
    ):
        req_info = self.per_request_info[rid]
        part_info = req_info.per_partition_info[partition_name]

        if current_fwd_info is not None:
            graph_walk = current_fwd_info.graph_walk
            if self.get_graph_walk(rid, partition_name) != graph_walk:
                part_info.graph_walk_worker_graph_ids = [
                    graph_id for graph_id in self.per_request_info[rid].worker_graph_ids \
                        if graph_walk in self.all_worker_graph_ids_to_graph_walks[graph_id]
                ]
            part_info.current_fwd_info = current_fwd_info

        if resource_publish_info is not None:
            fwd_info = self.get_fwd_info(rid, partition_name)
            fwd_info.update_publish_info(resource_publish_info)

    def get_graph_walk(self, rid: int, partition_name: str):
        return self.get_fwd_info(rid, partition_name).graph_walk

    def get_publish_info(self, rid: int, partition_name: str):
        return self.get_fwd_info(rid, partition_name).resource_publish_info

    def get_fwd_number(self, rid: int, partition_name: str):
        return self.get_fwd_info(rid, partition_name).fwd_index

    def has_partition(self,  rid: int, partition_name: str):
        return partition_name in self.per_request_info[rid].per_partition_info

    def get_fwd_info(self, rid: int, partition_name: str):
        part_info = self.per_request_info[rid].per_partition_info[partition_name]
        return part_info.current_fwd_info

    def get_partition_for_node(self, node_name: str) -> str | None:
        """Look up which partition a node belongs to."""
        return self.node_to_partition.get(node_name)

    def add_request(
        self, rid: int,
        partition_worker_graph_ids: list[int], # for this worker's worker graphs
        worker_graph_to_workers: dict[int, list[str]], # for other / all worker graphs
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

        # The per-request queue lifecycle belongs to PythonGraphRuntime, which
        # runs first on this path and shares this exact queues dict.

        if rid not in self.per_request_info:
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
            self.per_request_info[rid] = PerRequestInfo(
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
            req_info = self.per_request_info[rid]
            req_info.worker_graph_ids += my_worker_graph_ids
            req_info.per_partition_info[partition_name] = PerPartitionInfo(
                graph_walk_worker_graph_ids=[
                    graph_id for graph_id in my_worker_graph_ids
                    if current_graph_walk in self.all_worker_graph_ids_to_graph_walks[graph_id]
                ],
                current_fwd_info=current_fwd_info
            )

    def remove_request(self, rid: int):
        # Queue teardown belongs to PythonGraphRuntime.remove_request.
        self.per_request_info.pop(rid, None)

