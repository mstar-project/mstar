"""What the worker's components share: the long-lived objects (``WorkerServices``)
and the read-only facts about this worker's place in the deployment
(``WorkerLayout``). Neither holds a component's state."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch

from mstar.worker.chunk_loops import ChunkLoops
from mstar.worker.engine_manager import refuse_combined_walks_across_tp_groups

if TYPE_CHECKING:
    from mstar.communication.base import BaseCommunicator

    from mstar.communication.tensors import TensorCommunicationManager
    from mstar.distributed.base import ShardingConfig
    from mstar.distributed.communication import WorkerParallelGroups
    from mstar.graph.base import GraphSection
    from mstar.graph.runtime.base import GraphRuntime
    from mstar.model.base import Model, WorkerGraph
    from mstar.streaming.topology import Connection
    from mstar.worker.engine_manager import EngineManager
    from mstar.worker.node_manager_utils import RequestStateManager
    from mstar.worker.rebuild import Rebuilds
    from mstar.worker.stream_buffers import StreamBufferManager


@dataclass(frozen=True, slots=True)
class WorkerServices:
    """The objects ``Worker`` builds once and passes to each component."""
    worker_id: str
    device: torch.device
    # worker-graph queues and readiness, sharding, push-back, and the rid table
    graph_runtime: GraphRuntime
    # one engine per node: admit, plan, forward and commit, remove_request, and
    # the resident delta and page trails TP followers replay
    engine_manager: EngineManager
    # async reads, ACKs, in-flight read checks, forced cleanup on remove
    tensor_manager: TensorCommunicationManager
    request_state: RequestStateManager
    communicator: BaseCommunicator
    # TP, SP and joint process groups
    parallel_groups: WorkerParallelGroups
    # what this worker rebuilds a dropped request's KV from
    rebuilds: Rebuilds
    # the consumer side of streaming edges, and the producer side's walk hooks
    streams: StreamBufferManager


def _no_entries() -> Mapping:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class WorkerLayout:
    """What the deployment looks like from this worker: fixed at startup, and
    computable without building one."""
    node_names: frozenset[str] = frozenset()
    # the (node, walk) pairs this worker's graphs run
    my_walks: frozenset[tuple[str, str]] = frozenset()
    node_to_partition: Mapping[str, str] = field(default_factory=_no_entries)
    chunk_loops: ChunkLoops = field(default_factory=ChunkLoops)
    # nodes whose instance spans more than one rank, on any axis, and the nodes
    # this worker leads. Instance rank 0 is rank 0 on every axis: keying the
    # leader off one axis's rank alone would elect one per row of the others
    # (ranks 0 and 2 of a tp2*sp2 instance). Pure SP or CFG nodes count, as
    # their all-to-alls and gathers need the whole instance.
    parallel_nodes: frozenset[str] = frozenset()
    parallel_leader_nodes: frozenset[str] = frozenset()
    # Every parallel node on a worker is in one instance (``validate_deployment``
    # checks it), so the worker leads all of them or follows all of them.
    is_tp_follower: bool = False
    # (node, real walk) -> combined walk; scheduling keys use the combined one
    combined_walk_of: Mapping[tuple[str, str], str] = field(default_factory=_no_entries)
    # (node, walk) keys on the side lane; known once ``EngineManager.build`` has
    # validated them, so ``Worker`` fills this in after building the engines
    side_keys: frozenset[tuple[str, str]] = frozenset()
    # wg id -> its nodes, for every worker graph of the deployment
    all_wg_nodes: Mapping[int, set[str]] = field(default_factory=_no_entries)
    # (graph_walk, node) -> innermost enclosing loop
    innermost_loop_by_node: Mapping[tuple[str, str], str] = field(default_factory=_no_entries)
    # the connections streaming into this worker, and their edges
    consumer_connections: tuple[Connection, ...] = ()
    streaming_edge_names: frozenset[str] = frozenset()

    @property
    def followed_nodes(self) -> frozenset[str]:
        """The nodes rank 0 of their instance decides for: what runs on them,
        and what gets evicted from them."""
        return self.parallel_nodes - self.parallel_leader_nodes

    @classmethod
    def derive(
        cls, model: Model | None, my_worker_graphs: list[WorkerGraph],
        all_wg_nodes: Mapping[int, set[str]], sharding_config: ShardingConfig,
        parallel_groups: WorkerParallelGroups, model_config: dict | None,
    ) -> WorkerLayout:
        """Pure, save that it refuses combined walks spanning TP groups."""
        node_names: set[str] = set()
        my_walks: set[tuple[str, str]] = set()
        for wg in my_worker_graphs:
            wg_nodes = wg.section.get_nodes()
            node_names.update(wg_nodes)
            my_walks.update((node, walk) for node in wg_nodes for walk in wg.graph_walks)
        parallel = frozenset(
            node for node in node_names if parallel_groups.get_instance_world_size_for_node(node) > 1
        )
        leaders = frozenset(
            node for node in node_names if parallel_groups.get_instance_rank_for_node(node) == 0
        )
        chunk_loops = ChunkLoops.from_graphs(
            my_worker_graphs, (model_config or {}).get("speculate_mid_prefill"),
            model.chunking_policy_of() if model is not None else {},
        )
        node_to_partition: dict[str, str] = {}
        innermost: Mapping[tuple[str, str], str] = {}
        consumers: list[Connection] = []
        if model is not None:
            refuse_combined_walks_across_tp_groups(model, sharding_config, my_walks)
            walks = model.get_graph_walk_graphs()
            for pdef in model.get_partitions():
                for walk_name in pdef.graph_walks:
                    section = walks.get(walk_name)
                    if section:
                        for node_name in section.get_nodes():
                            node_to_partition[node_name] = pdef.name
            innermost = innermost_loops(walks)
            topology = model.get_partition_topology()
            if topology:
                consumers = [
                    conn for conn in topology.connections
                    if not node_names.isdisjoint(partition_nodes(conn.to_partition, model))
                ]
        return cls(
            node_names=frozenset(node_names),
            my_walks=frozenset(my_walks),
            node_to_partition=MappingProxyType(node_to_partition),
            chunk_loops=chunk_loops,
            parallel_nodes=parallel,
            parallel_leader_nodes=leaders,
            is_tp_follower=bool(parallel - leaders),
            combined_walk_of=MappingProxyType(model.combined_walk_of() if model is not None else {}),
            all_wg_nodes=all_wg_nodes,
            innermost_loop_by_node=MappingProxyType(innermost),
            consumer_connections=tuple(consumers),
            streaming_edge_names=frozenset(conn.edge_name for conn in consumers),
        )


def innermost_loops(walks: Mapping[str, GraphSection]) -> dict[tuple[str, str], str]:
    """Map each ``(graph_walk, node)`` to its innermost enclosing loop: the
    candidate with no other candidate strictly inside it. Structural, so it is
    derived once rather than asked of the graph runtime per step."""
    cache: dict[tuple[str, str], str] = {}
    for walk_name, section in walks.items():
        loops = section.get_loops()
        for node_name in section.get_nodes():
            enclosing = {
                name: loop for name, loop in loops.items()
                if node_name in loop.get_nodes()
            }
            for name, loop in enclosing.items():
                if not set(loop.section.get_loops()) & enclosing.keys():
                    cache[(walk_name, node_name)] = name
                    break
    return cache


def partition_nodes(partition_name: str, model: Model) -> list[str]:
    """The node names that belong to a partition.

    Recurses into each walk so nodes nested in a ``Loop`` / ``Sequential``
    / ``Parallel`` are included. Taking the section's own ``name`` would
    return the Loop's name (e.g. ``"talker_decode_loop"``) instead of the
    consuming node's, so its ``StreamBuffer`` would never be created and
    routing the streamed edge would fail with a ``KeyError``.
    """
    walks = model.get_graph_walk_graphs()
    for pdef in model.get_partitions():
        if pdef.name == partition_name:
            nodes: set[str] = set()
            for walk_name in pdef.graph_walks:
                section = walks.get(walk_name)
                if section is not None:
                    nodes.update(section.get_nodes().keys())
            return list(nodes)
    return []
