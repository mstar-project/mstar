from dataclasses import dataclass, field

from mstar.communication.tensors import TensorCommunicationManager, TensorStore
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import NodeAndGraphWalk
from mstar.graph.runtime.base import (
    EdgeSpec,
    GraphRuntime,
    ParallelList,
    PopRidsOutput,
    ReadyNodeSpec,
    RouteInput,
    RouteOutput,
    SendInput,
    SpeculationOutput,
    SpeculationPrepInput,
    SpeculationPrepOutput,
)
from mstar.model.base import WorkerGraph
from mstar.worker.node_manager_utils import WorkerGraphQueues


@dataclass
class GraphRuntimePartitionInfo:
    graph_walk: str
    graph_walk_worker_graph_ids: list[int] = field(default_factory=list) # for this worker
    stream_partition_done: bool = False  # set True when last chunk pops with is_final

@dataclass
class GraphRuntimeRequestInfo:
    partition_info: dict[str, GraphRuntimePartitionInfo]
    worker_graph_ids: list[int]
    node_to_workers: dict[NodeAndGraphWalk, list[str]]
    dyn_loop_to_workers: dict[NodeAndGraphWalk, list[str]]
    sharding_config: ShardingConfig


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
    ):
        self._my_worker_id = my_worker_id

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


    @property
    def queues(self) -> dict[int, WorkerGraphQueues]:
        """Shared with WorkerGraphsManager while the port is in flight; the
        runtime owns the per-request lifecycle (add_request / remove_request)."""
        return self._queues

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
            node_to_workers = {}
            dyn_loop_to_workers = {}
            for worker_graph_id, worker_ids in worker_graph_to_workers:
                if worker_graph_id not in self._all_wg_ids_to_graph_walks:
                    continue
                for wg_graph_walk in self._all_wg_ids_to_graph_walks[worker_graph_id]:
                    node_to_workers.update({
                        NodeAndGraphWalk(
                            node=name,
                            graph_walk=wg_graph_walk
                        ): worker_ids for name in self._all_wg_ids_to_nodes[worker_graph_id]
                    })

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

    def ingest_inputs_batch(
        self,
        signals: ParallelList[int, EdgeSpec],
        can_buffer: bool = True,
        is_streaming: bool = False,
    ) -> list[int]:
        raise NotImplementedError

    # --------- Scheduling ----------

    def pop_rids(
        self, node_name: str,
        graph_walk: str,
        request_ids: list[int],
        check_ready: bool = False,
    ) -> PopRidsOutput | None:
        raise NotImplementedError

    def has_ready_excluding(
        self, exclude_rids: set[int],
        exclude_target: tuple[str, str] | None = None,
    ) -> bool:
        raise NotImplementedError

    def get_ready_nodes(
        self, exclude_rids: set[int],
        target: tuple[str, str] | None = None,
        exclude_target: tuple[str, str] | None = None,
    ) -> list[ReadyNodeSpec]:
        raise NotImplementedError

    def push_back_node(
        self, node_name: str,
        rids: list[int],
        wg_ids: list[int]
    ):
        raise NotImplementedError

    # --------- Speculation ----------

    def speculate_node(
        self, node_name: str,
        graph_walk: str,
        sample_rid: int,
    ) -> list[SpeculationOutput]:
        raise NotImplementedError

    def prep_spec_rids(
        self, input: SpeculationPrepInput
    ) -> SpeculationPrepOutput:
        raise NotImplementedError

    # --------- Postprocess ----------

    def stop_loops_batched(
        self, partition: str, graph_walk: str,
        loop_names: ParallelList[int, list[str]]
    ):
        raise NotImplementedError

    def complete_and_route_batch(
        self, input: RouteInput,
        tensor_store: TensorStore
    ) -> RouteOutput:
        raise NotImplementedError

    def send_outputs(
        self,
        input: SendInput,
    ):
        raise NotImplementedError
