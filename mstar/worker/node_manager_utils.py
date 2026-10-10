import logging
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field

from mstar.communication.tensors import TensorCommunicationManager
from mstar.conductor.request_info import CurrentForwardPassInfo, merge_publish_info
from mstar.engine.resources import PublishedInfo
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    NameAndDest,
    NodeCompletionOutput,
)
from mstar.graph.graph_io import WorkerGraphIO
from mstar.model.base import WorkerGraph
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.utils.ipc_format import InputSignals

logger = logging.getLogger(__name__)


#: Renders a rid handle as its wire id. Installed by whoever owns the rid
#: table -- the graph runtime; ``None`` where there is none (tests). Anything
#: that puts a rid in TEXT goes through this: handles are worker-internal and
#: RECYCLED, so a raw one in a message that leaves the process -- a
#: FAIL_REQUESTS body becomes the client's HTTP error detail -- names whatever
#: request holds that handle by the time someone reads it.
RidStr = Callable[[int], str]


def _rid_name(rid_str: RidStr | None, handle: int) -> str:
    """Never raises. Every caller is reporting a rid that could not be found,
    and the rid table drops a handle's name on release -- so the lookup that
    renders it is exactly as likely to fail as the one that got here."""
    if rid_str is None:
        return f"<unresolved rid handle {handle}>"
    try:
        return rid_str(handle)
    except (KeyError, IndexError):
        return f"<released rid handle {handle}>"


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
    rid_str: RidStr | None = None

    def __post_init__(self):
        self.nodes = set(self.worker_graph.section.get_nodes().keys())
        self.loops = set(self.worker_graph.section.get_loops().keys())
        # Copied per request by unpickling, several times cheaper than deepcopy.
        self._section_blob = pickle.dumps(
            self.worker_graph.section, protocol=pickle.HIGHEST_PROTOCOL
        )

    def process_new_inputs(
        self, request_id: int, inputs: list[GraphEdge],
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
            f"Tried to process new inputs for unknown request ID {_rid_name(self.rid_str, request_id)}"
        queue = self.per_request_queues[request_id]
        not_ingested: list[GraphEdge] = []
        for inp in inputs:
            if not queue.ingest_input(inp, can_buffer):
                not_ingested.append(inp)
        return not_ingested

    def is_done(self, request_id) -> bool:
        assert request_id in self.per_request_queues, \
            f"Tried to check queue done state for unknown request ID {_rid_name(self.rid_str, request_id)}"
        queue = self.per_request_queues[request_id]
        return queue.wg_state_registry.is_done

    def add_request(self, request_id: int):
        """
        Initialize queues for a new request
        """
        section_copy = pickle.loads(self._section_blob)
        queue = WorkerGraphIO(section_copy, wg_id=self.worker_graph_id)
        queue.register_communication_info(
            self.tensor_manager, request_id
        )
        self.per_request_queues[request_id] = queue

    def remove_request(self, request_id: int):
        """
        Delete queues for a completed/removed request (saw EOS)
        """
        self.per_request_queues.pop(request_id, None)

    def get_ready_node_names(self) -> dict[int, set[str]]:
        """
        Returns mapping of request id to ready node names for that request
        """
        return {
            request_id: q.ready_node_names \
                for (request_id, q) in self.per_request_queues.items()
        }

    def pop_ready_nodes(
        self, request_id: int, node_names: list[str]
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

    def reset(self, request_id):
        """
        At the end of a worker graph, reset the queues for a request so it can
        be used for the next full model forward pass.
        """
        self.per_request_queues[request_id].clear()

    def stop_loops(
        self, request_id: int, loop_names: set[str]
    ) -> set[NameAndDest]:
        """Register a finish signal for each named loop and return the union
        of their ``_loop_back_inputs`` so the caller can drop those (name, dest)
        edges from the current iter's output routing.
        """
        assert request_id in self.per_request_queues, \
            f"Tried to stop loops for unknown request ID {_rid_name(self.rid_str, request_id)}"
        queue = self.per_request_queues[request_id]
        loop_back_signals: set[NameAndDest] = set()
        for name in loop_names:
            if name not in queue.loops:
                continue
            queue.register_loop_finish_signal(name)
            loop_back_signals.update(queue.loops[name]._loop_back_inputs)
        return loop_back_signals

    def mark_node_complete(
        self, request_id: int, node_name: str
    ) -> NodeCompletionOutput:
        """Complete a node in this worker graph's per-request io and return
        the registry's NodeCompletionOutput (output_edges + filtered_signals)."""
        assert request_id in self.per_request_queues, \
            f"Tried to complete node {node_name!r} for unknown request ID {_rid_name(self.rid_str, request_id)}"
        return self.per_request_queues[request_id].mark_node_complete(node_name)

    def get_dynamic_loop_iters(self, request_id: int) -> dict[str, int]:
        assert request_id in self.per_request_queues, \
            f"Tried to get dynamic loop iters for unknown request ID {_rid_name(self.rid_str, request_id)}"
        queue = self.per_request_queues[request_id]
        return queue.get_loop_indices()


@dataclass
class PerPartitionInfo:
    current_fwd_info: CurrentForwardPassInfo
    # Publication produced by this worker since its last completion message.
    # Kept separate from current_fwd_info, which also contains peer state.
    pending_resource_publish_info: dict[str, PublishedInfo] = field(
        default_factory=dict
    )


@dataclass
class PerRequestInfo:
    """What the worker still owns for a request.

    Everything graph-shaped -- node/loop routing, the sharding config, the
    worker graph ids, the live queues -- belongs to PythonGraphRuntime. What
    is left is the forward-pass info (a wire object) and the stream buffers
    (which hold real tensors), neither of which can live behind the runtime's
    contract.
    """
    # edge_name -> StreamBuffer
    stream_buffers: dict[str, StreamBuffer] = field(default_factory=dict)
    # consumer node -> edge_name -> StreamBuffer, the same buffers indexed by consumer
    stream_buffers_by_consumer: dict[str, dict[str, StreamBuffer]] = field(default_factory=dict)
    # edges whose final chunk a built step consumed; a returned chunk leaves again
    ended_streams: set[str] = field(default_factory=set)
    # conductor inputs for a walk the stream has not moved this partition into
    parked_inputs: list[InputSignals] = field(default_factory=list)
    per_partition_info: dict[str, PerPartitionInfo] = field(default_factory=dict)


@dataclass
class RequestStateManager:
    """Per-request forward-pass info and stream buffers.

    Was WorkerGraphsManager, which owned the worker graphs too; those moved to
    PythonGraphRuntime one method at a time. What remains is request state the
    runtime deliberately does not take: CurrentForwardPassInfo is a wire object
    it treats as opaque, and a StreamBuffer holds tensors.
    """
    per_request_info: dict[int, PerRequestInfo] = field(default_factory=dict)

    # node_name -> partition_name, from the model's partitions and graph walk
    # definitions. Used to find a node's partition in the colocated case.
    node_to_partition: dict[str, str] = field(default_factory=dict)

    def add_request(self, request_id: int, current_fwd_info: CurrentForwardPassInfo):
        """Register a partition for a request, adding the request if new.

        The conductor sends one NewRequest per partition, so this is called
        once per partition for a given request_id.
        """
        info = self.per_request_info.setdefault(request_id, PerRequestInfo())
        info.per_partition_info[current_fwd_info.partition_name] = (
            PerPartitionInfo(current_fwd_info=current_fwd_info)
        )

    def remove_request(self, request_id: int):
        self.per_request_info.pop(request_id, None)

    def update_request_info(
        self, request_id: int,
        partition_name: str,
        current_fwd_info: CurrentForwardPassInfo | None = None,
        resource_publish_info: dict[str, PublishedInfo] | None = None,
    ):
        info = self.per_request_info.get(request_id)
        if info is None:
            return
        part_info = info.per_partition_info.get(partition_name)
        if part_info is None:
            return
        if current_fwd_info is not None:
            part_info.current_fwd_info = current_fwd_info
        if resource_publish_info is not None:
            part_info.current_fwd_info.update_publish_info(resource_publish_info)

    def buffer_publish_info(
        self,
        request_id: int,
        partition_name: str,
        published: dict[str, PublishedInfo],
    ) -> None:
        pending = self.per_request_info[request_id].per_partition_info[
            partition_name
        ].pending_resource_publish_info
        # finalize_batch also merges these objects into current_fwd_info.
        # Detach the pending delta so a later peer update to that aggregate
        # cannot leak another rank into this worker's completion message.
        merge_publish_info(
            pending, {name: info.clone() for name, info in published.items()}
        )

    def flush_publish_info(
        self, request_id: int, partition_name: str,
    ) -> dict[str, PublishedInfo]:
        part_info = self.per_request_info[request_id].per_partition_info[
            partition_name
        ]
        published = part_info.pending_resource_publish_info
        part_info.pending_resource_publish_info = {}
        return published

    def get_fwd_info(self, request_id: int, partition_name: str):
        return self.per_request_info[request_id].per_partition_info[
            partition_name
        ].current_fwd_info

    def get_pending_publish_info(
        self, request_id: int, partition_name: str,
    ) -> dict[str, PublishedInfo]:
        return self.per_request_info[request_id].per_partition_info[
            partition_name
        ].pending_resource_publish_info

    def get_partition_for_node(self, node_name: str) -> str | None:
        return self.node_to_partition.get(node_name)
