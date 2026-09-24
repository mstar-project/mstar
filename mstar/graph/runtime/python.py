import logging
from dataclasses import dataclass, field

from mstar.api_server.request_types import APIServerMessage, ResultTensors
from mstar.communication import wire
from mstar.communication.communicator import BaseCommunicator
from mstar.communication.tensors import TensorCommunicationManager, TensorStore
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import (
    GraphEdge,
    GraphNode,
    NameAndDest,
    NodeAndGraphWalk,
    TensorPointerInfo,
)
from mstar.graph.graph_io import format_graph_edge_list
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.graph.runtime import sharding
from mstar.graph.runtime.base import (
    EdgeSpec,
    FreedTensors,
    GraphRuntime,
    ParallelList,
    PendingLoopStop,
    PopRidsOutput,
    ReadyNodeSpec,
    RouteInput,
    RouteOutput,
    SendInput,
    SpeculationOutput,
    SpeculationPrepInput,
    SpeculationPrepOutput,
)
from mstar.graph.special_destinations import (
    EMIT_TO_CLIENT,
    SPECIAL_DESTINATIONS,
)
from mstar.model.base import WorkerGraph
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    InputSignals,
    StopLoops,
    WorkerGraphsDone,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.worker.node_manager_utils import (
    NodeCompletionOutput,
    NodeOutputRouting,
    WorkerGraphQueues,
)

logger = logging.getLogger(__name__)


@dataclass
class GraphRuntimePartitionInfo:
    graph_walk: str
    graph_walk_worker_graph_ids: list[int] = field(default_factory=list) # for this worker
    stream_partition_done: bool = False  # set True when last chunk pops with is_final

@dataclass
class _SpecRidPrep:
    """One rid prepped for speculation, with enough to undo it."""
    rid: int
    node: GraphNode
    consumed_idxs: list[int]
    input_edges: list[EdgeSpec]
    into_signals: list[tuple[int, str]]
    into_next_iter: list[tuple[int, str]]


@dataclass
class CompletionState:
    """What a completed batch needs to hand to the send that follows it."""
    partition: str
    graph_walk: str
    node_name: str
    routing: dict[int, NodeOutputRouting]


@dataclass
class GraphRuntimeRequestInfo:
    partition_info: dict[str, GraphRuntimePartitionInfo]
    worker_graph_ids: list[int]
    node_to_workers: dict[NodeAndGraphWalk, list[str]]
    dyn_loop_to_workers: dict[NodeAndGraphWalk, list[str]]
    sharding_config: ShardingConfig
    # Per-loop stop indices. Worker-only, so it lives here rather than riding
    # on CurrentForwardPassInfo across the wire.
    loop_stop_times: dict[str, NestedLoopIndices] = field(default_factory=dict)
    # Buffered between send_outputs calls and flushed onto WORKER_GRAPHS_DONE,
    # so a persist signal cannot race the message that announces it.
    pending_persist_signals: list[GraphEdge] = field(default_factory=list)
    pending_new_token_counts: dict[str, int] = field(default_factory=dict)
    current_output_chunks: list[str] = field(default_factory=list)
    output_loop_indices: dict[str, NestedLoopIndices] = field(
        default_factory=dict
    )


class PythonGraphRuntime(GraphRuntime):
    """The reference implementation, and the thing a Rust backend is diffed
    against. Everything below is today's behavior behind the new contract; no
    new semantics belong here.
    """

    def __init__(
        self,
        # TODO: remove none default
        my_worker_id: str=None,
        my_worker_graphs: list[WorkerGraph]=None,
        all_wg_ids_to_graph_walks: dict[int, set[str]]=None,
        all_wg_ids_to_dyn_loops: dict[int, set[str]]=None,
        all_wg_ids_to_nodes: dict[int, set[str]]=None,
        node_to_partition: dict[str, str]=None,
        sharding_config: ShardingConfig=None,
        tensor_manager: TensorCommunicationManager=None,
        communicator: BaseCommunicator=None,
    ):
        self._my_worker_id = my_worker_id
        self._communicator = communicator
        # Descriptors for rebuilding an edge from its uuids. complete_and_route
        # takes the store as an argument; ingest has no such parameter, so it
        # reads the one the queues were built against.
        self._tensor_store = (
            None if tensor_manager is None else tensor_manager.tensor_store
        )

        # rid interning
        self._rids: list[str | None] = []
        self._rid_to_handle: dict[str, int] = {}
        self._available_handles: list[int] = []

        # TP metadata for speculative scheduling
        self._parallel_nodes: set[str] = set()
        self._parallel_leader_nodes: set[str] = set()
        self._tp_async_nodes: set[str] = set()

        # per-request info needed by the graph runtime
        self._request_info: dict[int, GraphRuntimeRequestInfo] = {}

        # Loop stops from this iteration's check_stop; cleared every iteration.
        self._pending_loop_stops: set[PendingLoopStop] = set()

        # Routing parked between complete_and_route_batch and send_outputs.
        self._completions: dict[int, CompletionState] = {}
        self._completion_counter = 0

        if my_worker_graphs is None:
            return # TODO: remove once rest is written

        # worker graph info
        self._queues = {
            worker_graph.worker_graph_id: WorkerGraphQueues(
                worker_graph_id=worker_graph.worker_graph_id,
                graph_walks=worker_graph.graph_walks,
                worker_graph=worker_graph,
                per_request_queues={},
                tensor_manager=tensor_manager
            )
            for worker_graph in my_worker_graphs
        }
        self._all_wg_ids_to_graph_walks = all_wg_ids_to_graph_walks
        self._all_wg_ids_to_dyn_loops = all_wg_ids_to_dyn_loops
        self._all_wg_ids_to_nodes = all_wg_ids_to_nodes

        # (graph_walk, node) -> worker graph. Saves a linear scan over the
        # request's worker graphs on every routing decision. Two worker graphs
        # can share a (walk, node) only when their walks are co-partitioned, so
        # last-write-wins is unambiguous for the pairs this is queried with.
        self._walk_node_to_wg_id: dict[tuple[str, str], int] = {}
        for wg_id, walks in all_wg_ids_to_graph_walks.items():
            for walk in walks:
                for node in all_wg_ids_to_nodes.get(wg_id, set()):
                    self._walk_node_to_wg_id[(walk, node)] = wg_id
        self._node_to_partition = node_to_partition
        self._sharding_config = sharding_config


    # --------- Bookkeeping ----------

    def set_node_metadata(
        self, parallel_nodes: set[str],
        parallel_leader_nodes: set[str],
        tp_async_nodes: set[str]
    ):
        self._parallel_nodes = parallel_nodes
        self._parallel_leader_nodes = parallel_leader_nodes
        self._tp_async_nodes = tp_async_nodes

    def add_request(
        self, request_id: str,
        partition: str,
        graph_walk: str,
        partition_worker_graph_ids: list[int],
        worker_graph_to_workers: ParallelList[int, list[str]]
    ) -> int:
        # The conductor sends one NewRequest PER PARTITION, so this runs
        # several times for one request. Minting a fresh handle each time
        # would orphan the previous one's queues -- a deepcopy of the whole
        # graph section per extra partition, never freed, because
        # remove_request only ever sees the last handle.
        handle = self._rid_to_handle.get(request_id)
        if handle is None:
            if not self._available_handles:
                handle = len(self._rids)
                self._rids.append(request_id)
            else:
                handle = self._available_handles.pop()
                self._rids[handle] = request_id
            self._rid_to_handle[request_id] = handle

        if self._my_worker_id is None:
            return handle # TODO: remove this once rest is updated

        my_worker_graph_ids = [gid for gid in partition_worker_graph_ids if gid in self._queues]
        if handle not in self._request_info:
            # Note: conductor.py passes the same worker_graph_to_worker dict
            # on every NewRequest for a given request(i.e., for every partition).
            # So the below logic only needs to be done once.
            node_to_workers = sharding.node_to_workers(
                worker_graph_to_workers,
                self._all_wg_ids_to_graph_walks,
                self._all_wg_ids_to_nodes,
            )
            dyn_loop_to_workers = {}
            for worker_graph_id, worker_ids in worker_graph_to_workers:
                if worker_graph_id not in self._all_wg_ids_to_graph_walks:
                    continue
                for wg_graph_walk in self._all_wg_ids_to_graph_walks[worker_graph_id]:
                    for loop_name in self._all_wg_ids_to_dyn_loops[worker_graph_id]:
                        dyn_loop_to_workers.setdefault(NodeAndGraphWalk(
                            node=loop_name,
                            graph_walk=wg_graph_walk
                        ), []).extend(worker_ids)
            sharding_config = self._sharding_config.clone_empty()
            sharding_config.setup(node_to_workers)
            self._request_info[handle] = GraphRuntimeRequestInfo(
                partition_info={},
                worker_graph_ids=[],
                node_to_workers=node_to_workers,
                dyn_loop_to_workers=dyn_loop_to_workers,
                sharding_config=sharding_config
            )

        for graph_id in partition_worker_graph_ids:
            if graph_id in self._queues:
                self._queues[graph_id].add_request(handle)

        self._request_info[handle].partition_info[partition] = GraphRuntimePartitionInfo(
            graph_walk=graph_walk,
            graph_walk_worker_graph_ids=[
                graph_id for graph_id in my_worker_graph_ids
                if graph_walk in self._all_wg_ids_to_graph_walks[graph_id]
            ]
        )
        self._request_info[handle].worker_graph_ids += my_worker_graph_ids

        return handle

    def remove_request(
        self, rid: int
    ):
        request_id = self._rids[rid]
        if request_id is None:
            return  # already removed; remove is idempotent by design
        # Routing parked by complete_and_route_batch whose send never ran --
        # an exception between the two abandons it. Handles are recycled, so a
        # stale entry would make the next request to get this integer send
        # another request's outputs.
        for cid in [
            cid for cid, c in self._completions.items()
            if rid in c.routing and len(c.routing) == 1
        ]:
            del self._completions[cid]
        for c in self._completions.values():
            c.routing.pop(rid, None)

        info = self._request_info.pop(rid, None)
        if info is not None:
            for wg_id in info.worker_graph_ids:
                self._queues[wg_id].remove_request(rid)
        del self._rid_to_handle[request_id]
        self._rids[rid] = None
        # Recycling means a stale handle held anywhere else now points at a
        # DIFFERENT request; see GraphRuntime.remove_request for what has to be
        # purged alongside this.
        self._available_handles.append(rid)

    def get_rid_string(self, handle: int) -> str:
        return self._rids[handle]

    def get_rid_handle(self, rid: str) -> int | None:
        return self._rid_to_handle.get(rid)

    def set_walk(self, rid: int, partition: str, walk: str):
        request_info = self._request_info.get(rid)
        if request_info is None:
            return
        part_info = request_info.partition_info.get(partition)
        if part_info is None or part_info.graph_walk == walk:
            return
        part_info.graph_walk = walk
        # The walk selects which of the request's worker graphs are live, so it
        # has to be re-derived here; leaving it stale would route this pass's
        # inputs into the previous walk's graphs.
        part_info.graph_walk_worker_graph_ids = [
            wg_id for wg_id in request_info.worker_graph_ids
            if walk in self._all_wg_ids_to_graph_walks[wg_id]
        ]

    def set_speculatively_scheduled(
        self, node: str, wg_id: int, rids: list[int],
        speculatively_scheduled: bool
    ):
        queues = self._queues[wg_id].per_request_queues
        for rid in rids:
            if rid not in queues:
                continue
            queues[rid].get_node(node)._speculatively_scheduled = speculatively_scheduled

    def is_speculatively_scheduled(
        self, node: str, wg_id: int, rid: int,
    ) -> bool:
        wgio = self._queues[wg_id].per_request_queues.get(rid)
        return wgio is not None and wgio.get_node(node)._speculatively_scheduled

    def get_dynamic_loop_iters(
        self, request_ids: list[int],
        partition: str,
    ) -> ParallelList[int, dict[str, int]]:
        values = []
        for rid in request_ids:
            iter_counts: dict[str, int] = {}
            part_info = self._request_info[rid].partition_info[partition]
            for wg_id in part_info.graph_walk_worker_graph_ids:
                iter_counts.update(self._queues[wg_id].get_dynamic_loop_iters(rid))
            values.append(iter_counts)
        return ParallelList(list(request_ids), values)

    def get_walk(self, rid: int, partition: str) -> str:
        return self._request_info[rid].partition_info[partition].graph_walk

    def check_dyn_loop(self, rid: int, partition: str, loop_name: str) -> bool:
        """Whether this request's current walk actually contains ``loop_name``.

        A stop for a loop the walk does not have is a model bug, not a
        protocol one, so it is logged and dropped rather than raised.
        """
        ngw = NodeAndGraphWalk(
            node=loop_name, graph_walk=self.get_walk(rid, partition),
        )
        if ngw not in self._request_info[rid].dyn_loop_to_workers:
            logger.error(
                "Tried to stop loop %s from graph walk %s, which does not "
                "include this loop! Ignoring this signal. This indicates a "
                "potential logical bug in the model.",
                loop_name, ngw.graph_walk,
            )
            return False
        return True

    def get_dyn_loop_workers(
        self, rid: int, partition: str, loop_name: str,
    ) -> list[str]:
        ngw = NodeAndGraphWalk(
            node=loop_name, graph_walk=self.get_walk(rid, partition),
        )
        return self._request_info[rid].dyn_loop_to_workers[ngw]

    def get_sharding_config(self, rid: int) -> ShardingConfig | None:
        """None for a rid this rank does not know: callers on the teardown and
        TP-fanout paths can legitimately race a removal."""
        info = self._request_info.get(rid)
        return None if info is None else info.sharding_config

    def _section_node(self, node_name: str, graph_walk: str):
        wg_id = self.get_worker_graph_id_for_node(node_name, graph_walk)
        return self._queues[wg_id].worker_graph.section.get_nodes().get(node_name)

    def is_async_schedulable(self, node_name: str, graph_walk: str) -> bool:
        node = self._section_node(node_name, graph_walk)
        return node is not None and node.enable_async_scheduling

    def get_output_signals(self, node_name: str, graph_walk: str) -> list[str]:
        node = self._section_node(node_name, graph_walk)
        if node is None:
            return []
        return sorted({edge.name for edge in node.outputs})

    def reset_outputs(
        self, node_name: str, rids: list[int], wg_ids: list[int],
    ):
        for rid, wg_id in zip(rids, wg_ids, strict=True):
            wgio = self._queues[wg_id].per_request_queues.get(rid)
            if wgio is not None:
                wgio.get_node(node_name).reset_outputs()

    def cleanup_consumed_inputs(
        self, node_name: str, rids: list[int], wg_ids: list[int],
    ) -> FreedTensors:
        for rid, wg_id in zip(rids, wg_ids, strict=True):
            wgio = self._queues[wg_id].per_request_queues.get(rid)
            if wgio is not None:
                wgio.get_node(node_name).ready_signals.clear()
                wgio.ready_node_names.discard(node_name)
        # ``clear`` dereferences through the tensor manager this runtime was
        # built with, which runs the teardown as it goes. Nothing is left for
        # the caller.
        return FreedTensors.none()

    def mark_stream_partition_done(self, rid: int, partition: str):
        info = self._request_info.get(rid)
        if info is not None and partition in info.partition_info:
            info.partition_info[partition].stream_partition_done = True

    def get_consumed_edges(
        self, source_node: str, dest_node: str, graph_walk: str,
    ) -> set[tuple[str, str]]:
        wg_id = self.get_worker_graph_id_for_node(source_node, graph_walk)
        section = self._queues[wg_id].worker_graph.section
        nodes = section.get_nodes()
        node = nodes.get(source_node)
        if node is None:
            return set()
        return {
            (edge.name, edge.next_node) for edge in node.outputs
            if edge.next_node == dest_node
        }

    def get_worker_graph_id_for_node(
        self, node: str, graph_walk: str,
    ) -> int:
        wg_id = self._walk_node_to_wg_id.get((graph_walk, node))
        if wg_id is None:
            raise RuntimeError(
                f"Could not find worker graph for node {node!r}, "
                f"graph_walk {graph_walk!r}"
            )
        return wg_id

    # --------- Inputs ----------

    def _edge_from_spec(
        self, spec: EdgeSpec, is_streaming: bool,
    ) -> GraphEdge:
        return GraphEdge(
            name=spec.signal,
            next_node=spec.next_node,
            tensor_info=[
                self._tensor_store.get_info(uuid) for uuid in spec.uuids
            ],
            is_streaming=is_streaming,
            _final_stream_chunk=spec.is_final_streaming_chunk,
        )

    def ingest_inputs_batch(
        self,
        signals: ParallelList[int, EdgeSpec],
        can_buffer: bool = True,
        is_streaming: bool = False,
    ) -> list[int]:
        uningested: list[int] = []
        for i, (rid, spec) in enumerate(signals):
            info = self._request_info.get(rid)
            if info is None:
                uningested.append(i)  # never admitted here, or already removed
                continue
            edge = self._edge_from_spec(spec, is_streaming)
            # The streaming gate is re-evaluated per signal on purpose:
            # ingesting one can be what makes the next node eligible.
            if not self._ingest_one(rid, info, edge, can_buffer, is_streaming):
                uningested.append(i)
        return uningested

    def _ingest_one(
        self, rid: int, info: GraphRuntimeRequestInfo, edge: GraphEdge,
        can_buffer: bool, is_streaming: bool,
    ) -> bool:
        """Offer the edge to each live worker graph until one claims it.

        A node can refuse an edge it owns (name mismatch, or both ready slots
        already full), which is why this is a claim loop and not a lookup.
        """
        for part_info in info.partition_info.values():
            for wg_id in part_info.graph_walk_worker_graph_ids:
                wgio = self._queues[wg_id].per_request_queues.get(rid)
                if wgio is None:
                    continue
                if is_streaming and edge.next_node not in wgio.ready_for_streaming:
                    continue
                if wgio.ingest_input(edge, can_buffer):
                    return True
        return False

    # --------- Scheduling ----------

    def pop_rids(
        self, node_name: str,
        graph_walk: str,
        request_ids: list[int],
        check_ready: bool = False,
    ) -> PopRidsOutput | None:
        wg_id = self.get_worker_graph_id_for_node(node_name, graph_walk)
        queue = self._queues.get(wg_id)
        if queue is None:
            return None
        if check_ready:
            # All or nothing: verified for every rid before anything is popped,
            # so a partially ready set is retried intact later.
            for rid in request_ids:
                wgio = queue.per_request_queues.get(rid)
                if wgio is None or node_name not in wgio.ready_node_names:
                    return None  # unknown rid (removed here) counts as not ready

        rids: list[int] = []
        wg_ids: list[int] = []
        input_edges: list[EdgeSpec] = []
        input_edges_per_rid: list[int] = []
        for rid in request_ids:
            popped = queue.pop_ready_nodes(rid, [node_name])
            if not popped:
                continue
            assert len(popped) == 1
            node = popped[0]
            ready = node.ready_signals.ready_inputs
            rids.append(rid)
            wg_ids.append(wg_id)
            input_edges_per_rid.append(len(ready))
            for signal, edge in ready.items():
                input_edges.append(EdgeSpec(
                    signal=signal,
                    next_node=edge.next_node,
                    uuids=[info.uuid for info in edge.tensor_info],
                    is_final_streaming_chunk=edge._final_stream_chunk,
                ))
        return PopRidsOutput(
            wg_ids=ParallelList(rids, wg_ids),
            input_edges=input_edges,
            input_edges_per_rid=input_edges_per_rid,
            output_signals=self.get_output_signals(node_name, graph_walk),
        )

    def _scan_ready(
        self, exclude_rids: set[int],
        target: tuple[str, str] | None,
        exclude_target: tuple[str, str] | None,
    ):
        """Every (node, walk, rid) whose graph inputs are satisfied.

        Graph level only -- engine readiness (is the KV cache read in?) is the
        caller's to apply, because it can fail a request, which is a scheduling
        decision rather than a graph one.
        """
        target_node, target_walk = target if target is not None else (None, None)
        for queue in self._queues.values():
            for rid, node_names in queue.get_ready_node_names().items():
                if rid in exclude_rids or rid not in self._request_info:
                    continue
                for node_name in node_names:
                    if target_node is not None and node_name != target_node:
                        continue
                    partition = self._node_to_partition.get(node_name)
                    if partition is None:
                        continue
                    walk = self.get_walk(rid, partition)
                    if target_walk is not None and walk != target_walk:
                        continue
                    if exclude_target is not None \
                            and (node_name, walk) == exclude_target:
                        continue
                    yield node_name, walk, rid

    def has_ready_excluding(
        self, exclude_rids: set[int],
        exclude_target: tuple[str, str] | None = None,
    ) -> bool:
        # Stops at the first match instead of building the full list. Graph
        # readiness is a necessary condition for schedulability, so a False
        # here lets the caller skip its engine-level pass entirely.
        for _ in self._scan_ready(exclude_rids, None, exclude_target):
            return True
        return False

    def get_ready_nodes(
        self, exclude_rids: set[int],
        target: tuple[str, str] | None = None,
        exclude_target: tuple[str, str] | None = None,
    ) -> list[ReadyNodeSpec]:
        grouped: dict[tuple[str, str], list[int]] = {}
        for node_name, walk, rid in self._scan_ready(
            exclude_rids, target, exclude_target,
        ):
            grouped.setdefault((node_name, walk), []).append(rid)
        return [
            ReadyNodeSpec(node_name, walk, rids)
            for (node_name, walk), rids in grouped.items()
        ]

    def push_back_node(
        self, node_name: str,
        rids: list[int],
        wg_ids: list[int]
    ):
        """Return a popped node to the ready set, e.g. after an OOM hold."""
        for rid, wg_id in zip(rids, wg_ids, strict=True):
            queue = self._queues.get(wg_id)
            if queue is None:
                continue
            wgio = queue.per_request_queues.get(rid)
            if wgio is not None:
                wgio.ready_node_names.add(node_name)

    # --------- Speculation ----------

    def speculate_node(
        self, node_name: str,
        graph_walk: str,
        sample_rid: int,
    ) -> list[SpeculationOutput]:
        wg_id = self.get_worker_graph_id_for_node(node_name, graph_walk)
        wgio = self._queues[wg_id].per_request_queues.get(sample_rid)
        if wgio is None:
            return []
        node = wgio.nodes[node_name]
        if not node.outputs:
            return []  # nothing to feed a spec target

        ready = wgio.ingest_for_speculation(node.outputs, node_name)
        wgio.clear_speculative_inputs()

        out: list[SpeculationOutput] = []
        for info in ready:
            target = wgio.nodes[info.node_name]
            if not target.enable_async_scheduling:
                # The destination opts out of async scheduling; mirrors the
                # source-side check. Without it a structurally ineligible
                # destination could be picked and then dropped per rid.
                continue
            if info.node_name in self._parallel_nodes:
                # A parallel node is a target only under TP async, from the
                # leader, as a same-node loop-back: for a transition INTO one
                # a follower has no in-flight batch to rebuild a head from.
                if not (
                    info.node_name in self._tp_async_nodes
                    and info.node_name in self._parallel_leader_nodes
                    and info.node_name == node_name
                ):
                    continue
            out.append(SpeculationOutput(
                node_name=info.node_name, graph_walk=graph_walk,
                is_new_loop_iter=info.is_new_loop_iter,
                loop_name=info.loop_name,
                output_signals=tuple(
                    self.get_output_signals(info.node_name, graph_walk)
                ),
            ))
        return out

    def _spec_target_info(
        self, rid: int, wgio, curr_node_name: str, spec_node_name: str,
    ):
        """The spec target's loop context, as ingest_for_speculation reports it.

        Taken from ONE rid and applied to the whole batch, matching today's
        sample-based behaviour: the loop a node belongs to and whether the
        target is a loop-back are structural, so they do not vary by request.
        """
        node = wgio.nodes[curr_node_name]
        ready = wgio.ingest_for_speculation(node.outputs, curr_node_name)
        wgio.clear_speculative_inputs()
        for info in ready:
            if info.node_name == spec_node_name:
                return info
        return None

    @staticmethod
    def _streaming_edges_by_rid(
        input: SpeculationPrepInput,
    ) -> list[tuple[int, list[tuple[int, EdgeSpec]]]]:
        """Slice the flat streaming edges per rid, keeping each one's index in
        the flat list so the caller can be told what was consumed."""
        per_rid: list[tuple[int, list[tuple[int, EdgeSpec]]]] = []
        cursor = 0
        for i, rid in enumerate(input.rids):
            count = input.streaming_edges_per_rid[i]
            per_rid.append((rid, [
                (cursor + k, input.streaming_edges[cursor + k])
                for k in range(count)
            ]))
            cursor += count
        return per_rid

    def get_spec_target(
        self, curr_node_name: str, spec_node_name: str,
        graph_walk: str, sample_rid: int,
    ) -> SpeculationOutput | None:
        wg_id = self.get_worker_graph_id_for_node(curr_node_name, graph_walk)
        wgio = self._queues[wg_id].per_request_queues.get(sample_rid)
        if wgio is None or not wgio.nodes[curr_node_name].outputs:
            return None
        info = self._spec_target_info(
            sample_rid, wgio, curr_node_name, spec_node_name,
        )
        if info is None:
            return None
        return SpeculationOutput(
            node_name=info.node_name,
            graph_walk=graph_walk,
            is_new_loop_iter=info.is_new_loop_iter,
            loop_name=info.loop_name,
            output_signals=tuple(
                self.get_output_signals(info.node_name, graph_walk)
            ),
        )

    def prep_follow_spec_rids(
        self, input: SpeculationPrepInput
    ) -> SpeculationPrepOutput | None:
        wg_id = self.get_worker_graph_id_for_node(
            input.spec_node_name, input.graph_walk
        )
        queue = self._queues[wg_id]
        # A follower always speculates the same node it is running.
        same_node = True

        prepped: list[_SpecRidPrep] = []
        for rid, indexed_edges in self._streaming_edges_by_rid(input):
            wgio = queue.per_request_queues.get(rid)
            result = None if wgio is None else self._prep_one_spec_rid(
                rid, wgio, indexed_edges, input.curr_node_name,
                input.spec_node_name, same_node,
            )
            if result is None:
                # All or nothing: undo everything prepped so far, or this rank
                # joins the collective with a batch the leader never sent.
                for done in prepped:
                    self._undo_spec_ingest(
                        done.node, done.into_signals, done.into_next_iter,
                    )
                return None
            prepped.append(result)

        return SpeculationPrepOutput(
            consumed_streaming_edge_idxs=[
                idx for p in prepped for idx in p.consumed_idxs
            ],
            ready_rids=[p.rid for p in prepped],
            wg_ids=[wg_id] * len(prepped),
            input_edges=[e for p in prepped for e in p.input_edges],
            input_edges_per_rid=[len(p.input_edges) for p in prepped],
        )

    def prep_spec_rids(
        self, input: SpeculationPrepInput
    ) -> SpeculationPrepOutput:
        wg_id = self.get_worker_graph_id_for_node(
            input.spec_node_name, input.graph_walk
        )
        queue = self._queues[wg_id]
        same_node = input.spec_node_name == input.curr_node_name

        per_rid_edges = self._streaming_edges_by_rid(input)

        sample_wgio = queue.per_request_queues.get(input.rids[0]) \
            if input.rids else None
        spec_info = None if sample_wgio is None else self._spec_target_info(
            input.rids[0], sample_wgio, input.curr_node_name,
            input.spec_node_name,
        )

        consumed_idxs: list[int] = []
        ready_rids: list[int] = []
        wg_ids: list[int] = []
        input_edges: list[EdgeSpec] = []
        input_edges_per_rid: list[int] = []

        for rid, indexed_edges in per_rid_edges:
            wgio = queue.per_request_queues.get(rid)
            if wgio is None:
                continue
            if spec_info is not None and not self._can_continue_loop(
                rid, wgio, spec_info, input.graph_walk
            ):
                continue  # loop already finished; no further work to speculate
            if (
                input.room_for_continuing is not None
                and len(ready_rids) >= input.room_for_continuing
            ):
                # Room is spoken for by the backlog. Skipped BEFORE any
                # streaming ingest, so there is nothing to roll back.
                continue

            prepped = self._prep_one_spec_rid(
                rid, wgio, indexed_edges, input.curr_node_name,
                input.spec_node_name, same_node,
            )
            if prepped is None:
                continue
            consumed_idxs.extend(prepped.consumed_idxs)
            ready_rids.append(rid)
            wg_ids.append(wg_id)
            input_edges.extend(prepped.input_edges)
            input_edges_per_rid.append(len(prepped.input_edges))

        return SpeculationPrepOutput(
            consumed_streaming_edge_idxs=consumed_idxs,
            ready_rids=ready_rids,
            wg_ids=wg_ids,
            input_edges=input_edges,
            input_edges_per_rid=input_edges_per_rid,
        )

    def _can_continue_loop(
        self, rid: int, wgio, spec_info, graph_walk: str,
    ) -> bool:
        """False once this rid's loop has ended: a stop is already pending for
        it, or the next iteration would be past the last one."""
        if not spec_info.is_new_loop_iter:
            return True
        if self.has_pending_loop_stop(rid, graph_walk, spec_info.loop_name):
            return False
        loop = wgio.loops.get(spec_info.loop_name)
        if loop is not None and (
            loop.curr_iter + 1 >= loop.max_iters or loop._finish_signal
        ):
            return False
        return True

    @staticmethod
    def _undo_spec_ingest(
        node, into_signals: list[tuple[int, str]],
        into_next_iter: list[tuple[int, str]],
    ):
        """Pull ingested streaming chunks back out of the node's ready slots.

        The two slots are removed from separately, which is why the ingest
        tracks which one each chunk landed in. Registry state was never
        touched -- _speculatively_scheduled was held True across the ingest --
        so the caller only has to return the chunks to their StreamBuffers.
        """
        for _idx, name in into_signals:
            node.ready_signals.remove(name)
        for _idx, name in into_next_iter:
            node.ready_next_iter.remove(name)

    def _prep_one_spec_rid(
        self, rid: int, wgio, indexed_edges: list[tuple[int, EdgeSpec]],
        curr_node_name: str, spec_node_name: str, same_node: bool,
    ) -> "_SpecRidPrep | None":
        """Ingest this rid's stream chunks, check readiness, gather inputs.

        Returns the prep (including what it ingested, so a caller doing
        all-or-nothing can undo it), or None after rolling its own back.
        """
        node = wgio.nodes[spec_node_name]
        # Held True across the ingest so a streaming input cannot re-add the
        # node to the ready queue underneath us.
        node._speculatively_scheduled = True

        # ingest_input reports success without saying WHICH slot it used, so
        # the slot is inferred by peeking before the call. The rollback below
        # needs that: the two slots are removed from separately.
        # (index, signal name): the index is what the caller needs back, the
        # name is what the rollback removes by.
        into_signals: list[tuple[int, str]] = []
        into_next_iter: list[tuple[int, str]] = []
        for idx, spec in indexed_edges:
            edge = self._edge_from_spec(spec, is_streaming=True)
            already_ready = edge.name in node.ready_signals.ready_names
            if node.ingest_input(edge, can_buffer=same_node):
                target = into_next_iter if already_ready else into_signals
                target.append((idx, edge.name))

        wgio.ingest_for_speculation(
            wgio.nodes[curr_node_name].outputs, curr_node_name
        )
        fully_ready = node.is_ready_for_speculation(
            check_next_iter=same_node, allow_streaming=False,
        )
        wgio.clear_speculative_inputs()
        node._speculatively_scheduled = False  # reset in case the rid is dropped

        if not fully_ready:
            self._undo_spec_ingest(node, into_signals, into_next_iter)
            return None

        slots = node.ready_next_iter if same_node else node.ready_signals
        return _SpecRidPrep(
            rid=rid,
            node=node,
            consumed_idxs=[idx for idx, _name in into_next_iter + into_signals],
            input_edges=[
                EdgeSpec(
                    signal=name,
                    next_node=edge.next_node,
                    uuids=[info.uuid for info in edge.tensor_info],
                    is_final_streaming_chunk=edge._final_stream_chunk,
                ) for name, edge in slots.ready_inputs.items()
            ],
            into_signals=into_signals,
            into_next_iter=into_next_iter,
        )

    def _process_node_outputs(
        self, rid: int,
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

        sharding_config = self.get_sharding_config(rid)
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
            wg_id = self._walk_node_to_wg_id.get((graph_walk, edge.next_node))
            if wg_id is not None and wg_id in self._queues:
                fanout = sharding_config.fanout_graph_edges(
                    edge, source_node=node_name,
                    source_graph_walk=graph_walk,
                    dest_graph_walk=graph_walk,
                )
                this_worker_edge = fanout.pop(self._my_worker_id, None)
                if this_worker_edge is not None:
                    leftover = self._queues[wg_id].process_new_inputs(
                        rid, [this_worker_edge], can_buffer=True,
                    )
                    if leftover:
                        # local wg declined (e.g., no per-request io yet);
                        # route to self via the cross-worker path
                        to_workers.setdefault(self._my_worker_id, []).extend(leftover)
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
        completed_worker_graph_ids: list[int] = []
        for wg_id in self._request_info[rid].worker_graph_ids:
            if graph_walk not in self._all_wg_ids_to_graph_walks[wg_id]:
                continue
            queue = self._queues[wg_id]
            if queue.is_done(rid):
                completed_worker_graph_ids.append(wg_id)
                queue.reset(rid)

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
            if node_graph_walk not in self._request_info[rid].node_to_workers:
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
        for gid in self._request_info[rid].worker_graph_ids:
            my_node_names.update(self._all_wg_ids_to_nodes.get(gid, []))

        for edge in streaming_edges:
            fanout = sharding_config.fanout_graph_edges(
                edge, source_node=node_name,
                source_graph_walk=graph_walk,
                dest_graph_walk=None
            )
            this_worker_edge = fanout.pop(self._my_worker_id, None)
            if this_worker_edge:
                streaming_local.append(this_worker_edge)
            for (wkr, wkr_edge) in fanout.items():
                streaming_to_workers.setdefault(wkr, []).append(wkr_edge)

        logger.debug(
            ("Finished processing outputs from rid %s. \n"
             "Routed to this worker: %s; sent to others: %s; persist signals: %s; streaming: %d"),
            rid, format_graph_edge_list(routed_to_this_worker),
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


    def _mark_node_complete(
        self, rid: int, wg_id: int, node_name: str,
    ) -> NodeCompletionOutput:
        """Returns ``output_edges`` (static outputs plus any loop terminal
        outputs) and ``filtered_signals`` (loop-back (name, dest) pairs the
        caller must drop from routing)."""
        return self._queues[wg_id].mark_node_complete(rid, node_name)

    def get_nested_loop_idxs_for_node(
        self, rid: int, partition: str, node_name: str
    ) -> NestedLoopIndices:
        graph_walk = self.get_walk(rid, partition)
        wgid = self._walk_node_to_wg_id[(graph_walk, node_name)]
        wgio = self._queues[wgid].per_request_queues.get(rid)
        return wgio.get_nested_loop_idxs_for_node(node_name)


    # --------- Postprocess ----------

    def stop_loops_batched(
        self, partition: str,
        graph_walk: str,
        last_node_run: str,
        loop_names: ParallelList[int, list[str]]
    ):
        for rid, names in loop_names:
            wanted = {
                name for name in names
                if self.check_dyn_loop(rid, partition, name)
            }
            if not wanted:
                continue
            self._stop_loops_for_rid(rid, partition, wanted, last_node_run)
            self._pending_loop_stops.update(
                PendingLoopStop(rid, graph_walk, name) for name in wanted
            )
            self._fan_out_loop_stops(rid, partition, wanted)

    def _stop_loops_for_rid(
        self, rid: int, partition: str, loop_names: set[str],
        last_node_run: str | None,
    ) -> set[NameAndDest]:
        """Register the finish signal on every worker graph carrying a named
        loop, and return the union of their loop-back (name, dest) pairs so the
        caller drops those from the triggering iteration's output routing.

        In disaggregated mode one loop name can exist on several worker graphs,
        each with its own finish signal, so this still fans out locally.
        """
        part_info = self._request_info[rid].partition_info[partition]
        stopped: set[NameAndDest] = set()
        for wg_id in part_info.graph_walk_worker_graph_ids:
            stopped |= self._queues[wg_id].stop_loops(rid, loop_names)

        # A stop time is one observation per loop, so only the worker graph
        # owning the last-run node needs to be asked.
        if last_node_run is not None:
            owner = self._walk_node_to_wg_id.get(
                (part_info.graph_walk, last_node_run)
            )
            wgio = (
                None if owner is None or owner not in self._queues
                else self._queues[owner].per_request_queues.get(rid)
            )
            if wgio is not None:
                times = self._request_info[rid].loop_stop_times
                for name in loop_names & wgio.loops.keys():
                    times[name] = wgio.get_nested_loop_idxs(
                        target_loop_name=name,
                    )
        return stopped

    def _fan_out_loop_stops(
        self, rid: int, partition: str, loop_names: set[str],
    ):
        """Tell the peers sharing each loop that it is done."""
        if self._communicator is None:
            return
        per_worker: dict[str, set[str]] = {}
        for loop_name in loop_names:
            for worker in self.get_dyn_loop_workers(rid, partition, loop_name):
                per_worker.setdefault(worker, set()).add(loop_name)
        for worker, names in per_worker.items():
            if worker == self._my_worker_id:
                continue
            self._communicator.send(
                entity_id=worker,
                msg=WorkerMessage(
                    message_type=WorkerMessageType.STOP_LOOPS,
                    body=StopLoops(
                        request_id=self.get_rid_string(rid),
                        loop_names=names,
                        loop_stop_times=self._loop_stop_times(rid),
                        partition_name=partition,
                    ),
                ),
            )

    def apply_peer_loop_stops(
        self, rid: int, partition: str,
        loop_stop_times: dict[str, NestedLoopIndices],
    ):
        request_info = self._request_info.get(rid)
        if request_info is None or partition not in request_info.partition_info:
            return
        mine = request_info.loop_stop_times
        newer: set[str] = set()
        for name, stop_time in loop_stop_times.items():
            if name not in mine or stop_time.label_context_gt(mine[name], name):
                newer.add(name)
            mine[name] = stop_time
        if newer:
            # No last_node_run and no fan-out: the originating rank already
            # took the snapshot and told everyone.
            self._stop_loops_for_rid(rid, partition, newer, None)

    def _loop_stop_times(self, rid: int) -> dict[str, NestedLoopIndices]:
        """Only ever read to build the STOP_LOOPS this runtime sends, so it
        stays off the contract."""
        return self._request_info[rid].loop_stop_times

    def has_pending_loop_stop(
        self, rid: int, graph_walk: str, loop_name: str,
    ) -> bool:
        return PendingLoopStop(rid, graph_walk, loop_name) \
            in self._pending_loop_stops

    def pending_loop_stop_rids(
        self, graph_walk: str, loop_name: str,
    ) -> set[int]:
        return {
            stop.rid for stop in self._pending_loop_stops
            if stop.loop_name == loop_name and stop.graph_walk == graph_walk
        }

    def clear_pending_loop_stops(self):
        self._pending_loop_stops.clear()

    def complete_and_route_batch(
        self, input: RouteInput,
        tensor_store: TensorStore
    ) -> RouteOutput:
        rids, wg_ids = input.wg_ids.keys, input.wg_ids.values
        n_signals = len(input.output_signals)
        uuid_to_idx = {uuid: i for i, uuid in enumerate(input.tensors)}
        # Drop the previous pass's outputs here rather than making the caller
        # remember: this is the only thing that writes them, so the reset
        # belongs on the same round trip.
        self.reset_outputs(input.node_name, list(rids), list(wg_ids))

        routing_per_rid: dict[int, NodeOutputRouting] = {}
        register_idxs: list[int] = []
        register_rids: list[int] = []
        new_token_idxs: list[int] = []
        local_streaming_idxs: list[int] = []
        staged: set[int] = set()

        cursor = 0
        for i, (rid, wg_id) in enumerate(zip(rids, wg_ids, strict=True)):
            # Slice this rid's uuids back out of the flat, rid-major layout.
            per_signal: dict[str, list[int]] = {}
            for sig_i, signal in enumerate(input.output_signals):
                count = input.num_tensors[i * n_signals + sig_i]
                per_signal[signal] = input.tensors[cursor:cursor + count]
                cursor += count

            node = self._queues[wg_id].per_request_queues[rid].get_node(
                input.node_name
            )
            # Descriptors come back from the store, which is what lets the
            # caller hand us uuids instead of TensorPointerInfo objects.
            owned: set[int] = set()
            for edge in node.outputs:
                uuids = per_signal.get(edge.name)
                if not uuids:
                    continue
                edge.tensor_info = [tensor_store.get_info(u) for u in uuids]
                owned.update(uuids)

            completion = self._mark_node_complete(rid, wg_id, input.node_name)
            routing = self._process_node_outputs(
                rid, node_name=input.node_name,
                outputs=[edge.clone() for edge in completion.output_edges],
                graph_walk=input.graph_walk,
            )
            routing_per_rid[rid] = routing

            if owned:
                for edge in routing.persist:
                    for info in edge.tensor_info:
                        tensor_store.set_metadata(info.uuid, persist=True)
                # persist is deliberately absent: those tensors are held alive
                # by the persist marker, and counting them here would
                # double-count a signal whose destination is EMPTY_DESTINATION
                # (the conductor computes that reference when it unpersists).
                routed_edges = (
                    routing.routed_to_this_worker_graph
                    + routing.emit_to_client
                    + routing.streaming_local
                    + sum(routing.to_workers.values(), start=[])
                    + sum(routing.streaming_to_workers.values(), start=[])
                )
                self._set_output_ref_counts(tensor_store, owned, routed_edges)

            # What the caller has to stage for remote reads. Deduped by uuid
            # and skipping anything already registered, so a re-emitted edge
            # does not stage twice.
            for edge in (
                routing.persist + routing.emit_to_client
                + sum(routing.to_workers.values(), start=[])
                + sum(routing.streaming_to_workers.values(), start=[])
            ):
                for info in edge.tensor_info:
                    idx = uuid_to_idx.get(info.uuid)
                    if idx is None or info.uuid in staged:
                        continue
                    staged.add(info.uuid)
                    register_idxs.append(idx)
                    register_rids.append(rid)

            for edge in routing.new_token_outputs:
                for info in edge.tensor_info:
                    idx = uuid_to_idx.get(info.uuid)
                    if idx is not None:
                        new_token_idxs.append(idx)
            for edge in routing.streaming_local:
                for info in edge.tensor_info:
                    idx = uuid_to_idx.get(info.uuid)
                    if idx is not None:
                        local_streaming_idxs.append(idx)

        self._completion_counter += 1
        completion_id = self._completion_counter
        self._completions[completion_id] = CompletionState(
            partition=input.partition,
            graph_walk=input.graph_walk,
            node_name=input.node_name,
            routing=routing_per_rid,
        )
        return RouteOutput(
            completion_id=completion_id,
            register_tensor_idxs=register_idxs,
            register_rids=register_rids,
            new_token_output_idxs=new_token_idxs,
            local_streaming_tensor_idxs=local_streaming_idxs,
        )

    @staticmethod
    def _set_output_ref_counts(
        tensor_store: TensorStore,
        owned_uuids: set[int],
        routed_edges: list[GraphEdge],
    ):
        """Adjust from the safety hold of 1 to the real fanout."""
        actual = dict.fromkeys(owned_uuids, 0)
        for edge in routed_edges:
            for info in edge.tensor_info:
                if info.uuid in actual:
                    actual[info.uuid] += 1
        for uuid, count in actual.items():
            delta = count - 1
            if delta > 0:
                tensor_store.increment_ref(uuid, n=delta)
            elif delta < 0:
                tensor_store.dereference(uuid, n=-delta)

    def _peek_completion(self, completion_id: int) -> "CompletionState":
        """For testing purposes only.
        """
        return self._completions[completion_id]

    def send_outputs(
        self,
        input: SendInput,
    ):
        completion = self._completions.pop(input.completion_id)
        partition = completion.partition
        fwd_infos = dict(iter(input.per_request_info))
        new_token_counts = dict(iter(input.new_token_counts))
        nested_idxs = dict(iter(input.nested_loop_indices))
        consumed = (
            {} if input.stream_tokens_consumed is None
            else dict(iter(input.stream_tokens_consumed))
        )
        profiling = (
            {} if input.profiling is None else dict(iter(input.profiling))
        )

        for rid, routing in completion.routing.items():
            fwd_info = fwd_infos.get(rid)
            info = self._request_info.get(rid)

            for worker_id, edges in routing.to_workers.items():
                self._send_input_signals(rid, worker_id, edges, fwd_info, partition)

            if routing.persist and info is not None:
                info.pending_persist_signals.extend(routing.persist)

            counts = new_token_counts.get(rid)
            if counts and info is not None:
                for name, count in counts.items():
                    info.pending_new_token_counts[name] = (
                        info.pending_new_token_counts.get(name, 0) + count
                    )

            if routing.emit_to_client and info is not None:
                info.current_output_chunks.extend(
                    edge.name for edge in routing.emit_to_client
                )
                for edge in routing.emit_to_client:
                    info.output_loop_indices[edge.name] = nested_idxs.get(rid)
                    self._communicator.send("api_server", APIServerMessage(
                        message_type="result_tensors",
                        body=ResultTensors(
                            request_id=self.get_rid_string(rid),
                            modality=edge.output_modality,
                            graph_edge=edge,
                            loop_indices=nested_idxs.get(rid),
                            metadata={},
                        ),
                    ))

            # streaming_local is NOT here: it feeds a StreamBuffer, which holds
            # real tensors and so stays on the Python side. RouteOutput's
            # local_streaming_tensor_idxs is how the caller finds those.
            for worker_id, edges in routing.streaming_to_workers.items():
                self._send_input_signals(rid, worker_id, edges, fwd_info, partition)

            if routing.completed_worker_graph_ids and info is not None:
                part = info.partition_info.get(partition)
                node = self._queues[
                    self.get_worker_graph_id_for_node(
                        completion.node_name, completion.graph_walk,
                    )
                ].per_request_queues.get(rid)
                speculative = node is not None and node.get_node(
                    completion.node_name
                )._speculatively_scheduled
                self._send_worker_graphs_done(
                    rid, routing, info, fwd_info, partition,
                    stream_tokens_consumed=consumed.get(rid, {}),
                    # A speculatively-scheduled node has not really finished
                    # the partition, so it must not report done.
                    partition_done=(
                        part is not None and part.stream_partition_done
                        and not speculative
                    ),
                    profiling=profiling.get(rid),
                )

    def _send_input_signals(
        self, rid: int, worker_id: str, edges: list[GraphEdge],
        fwd_info: CurrentForwardPassInfo | None, partition: str,
    ):
        self._communicator.send(worker_id, WorkerMessage(
            message_type=WorkerMessageType.INPUT_SIGNALS,
            body=InputSignals(
                request_id=self.get_rid_string(rid),
                inputs=edges,
                request_info=fwd_info,
                partition_name=partition,
            ),
        ))

    def _send_worker_graphs_done(
        self, rid: int, routing: NodeOutputRouting,
        info: GraphRuntimeRequestInfo,
        fwd_info: CurrentForwardPassInfo | None,
        partition: str,
        stream_tokens_consumed: dict[str, int],
        partition_done: bool,
        profiling: bytes | None,
    ):
        persist_signals: dict[str, list[TensorPointerInfo]] = {}
        for edge in info.pending_persist_signals:
            persist_signals[edge.name] = edge.tensor_info
        info.pending_persist_signals = []
        new_token_counts = info.pending_new_token_counts
        info.pending_new_token_counts = {}
        output_signal_names = list(info.current_output_chunks)
        info.current_output_chunks.clear()

        rx_info, tx_info, graph_timings = [], [], {}
        if profiling is not None:
            rx_info, tx_info, graph_timings = wire.decode(profiling)

        self._communicator.send("conductor", ConductorMessage(
            message_type=ConductorMessageType.WORKER_GRAPHS_DONE,
            body=WorkerGraphsDone(
                request_id=self.get_rid_string(rid),
                worker_graph_ids=routing.completed_worker_graph_ids,
                is_first_tp_rank=routing.is_first_tp_rank,
                persist_signals=persist_signals,
                new_token_counts=new_token_counts,
                output_signal_names=output_signal_names,
                resource_publish_info=(
                    {} if fwd_info is None else fwd_info.resource_publish_info
                ),
                partition_name=partition,
                partition_done=partition_done,
                stream_tokens_consumed=stream_tokens_consumed,
                output_loop_indices=info.output_loop_indices,
                graph_timings=graph_timings,
                rx_info=rx_info,
                tx_info=tx_info,
            ),
        ))
