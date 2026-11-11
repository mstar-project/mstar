"""What the worker's components share: the long-lived objects (``WorkerServices``)
and the read-only facts about this worker's place in the deployment
(``WorkerLayout``). Neither holds a component's state."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch

from mstar.worker.chunk_loops import ChunkLoops
from mstar.worker.engine_manager import refuse_combined_walks_across_tp_groups
from mstar.worker.scheduling_policy import WalkPriority, resolve_walk_priorities

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
    # per-axis (TP, SP, CFG) and lockstep-instance process groups
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
    # this worker's (node, walk) pairs -> the entry deciding their priority (the
    # serving config's ``scheduling.walk_priorities``, else ``Model.get_walk_priorities``)
    walk_priorities: Mapping[tuple[str, str], WalkPriority] = field(default_factory=_no_entries)
    # the serving config's ``scheduling``: its policy ("priority", or "round_robin"
    # whatever the priorities), and the ``max_wait`` of a key no entry bounds
    scheduling_policy: str = "priority"
    default_max_wait: int | None = None
    # (node, walk) keys on the side lane; known once ``EngineManager.build`` has
    # validated them, so ``Worker`` fills this in after building the engines
    side_keys: frozenset[tuple[str, str]] = frozenset()
    # (node, walk) -> its streamed signals that go as tensors there
    # (``InlineStreams.tensor_streams``), likewise filled in after the engines
    tensor_streams: Mapping[tuple[str, str], frozenset[str]] = field(default_factory=_no_entries)
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
        priorities: dict[tuple[str, str], WalkPriority] = {}
        scheduling = SchedulingSettings.from_config((model_config or {}).get("scheduling"))
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
            priorities = {
                pair: entry for pair, entry in walk_priorities_of(model, walks, scheduling.entries).items()
                if pair in my_walks
            }
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
            walk_priorities=MappingProxyType(priorities),
            scheduling_policy=scheduling.policy,
            default_max_wait=scheduling.max_wait,
            all_wg_nodes=all_wg_nodes,
            innermost_loop_by_node=MappingProxyType(innermost),
            consumer_connections=tuple(consumers),
            streaming_edge_names=frozenset(conn.edge_name for conn in consumers),
        )


@dataclass(frozen=True)
class SchedulingSettings:
    """The serving config's ``scheduling`` block, checked."""

    policy: str = "priority"
    max_wait: int | None = None
    # the deployment's own priorities, which replace the model's; None keeps the model's
    entries: tuple[WalkPriority, ...] | None = None

    POLICIES = ("priority", "round_robin")
    KEYS = frozenset({"policy", "max_wait", "walk_priorities"})
    ENTRY_KEYS = frozenset(WalkPriority._fields)

    @classmethod
    def from_config(cls, block: Any) -> SchedulingSettings:
        if block is None:
            return cls()
        if not isinstance(block, Mapping) or block.keys() - cls.KEYS:
            raise ValueError(f"scheduling must be a mapping of {sorted(cls.KEYS)}, got {block!r}")
        policy = block.get("policy", "priority")
        if policy not in cls.POLICIES:
            raise ValueError(f"scheduling.policy must be one of {cls.POLICIES}, got {policy!r}")
        max_wait = block.get("max_wait")
        if max_wait is not None and (not isinstance(max_wait, int) or isinstance(max_wait, bool) or max_wait < 1):
            raise ValueError(f"scheduling.max_wait must be a positive int, got {max_wait!r}")
        entries = block.get("walk_priorities")
        if entries is not None:
            if not isinstance(entries, list) or not all(
                isinstance(e, Mapping) and not e.keys() - cls.ENTRY_KEYS for e in entries
            ):
                raise ValueError(
                    "scheduling.walk_priorities must be a list of mappings of "
                    f"{sorted(cls.ENTRY_KEYS)}, got {entries!r}"
                )
            entries = tuple(WalkPriority(**e) for e in entries)
        return cls(policy=policy, max_wait=max_wait, entries=entries)


def walk_priorities_of(
    model: Model, walks: Mapping[str, GraphSection], entries: Sequence[WalkPriority] | None = None,
) -> dict[tuple[str, str], WalkPriority]:
    """``entries`` (the serving config's), else ``Model.get_walk_priorities``, resolved
    to the (node, walk) pairs of the model's walks they reach. Refuses an entry that names
    no node or walk the model runs: a typo there would silently leave its target at
    round robin."""
    owner = "scheduling.walk_priorities" if entries is not None else f"{type(model).__name__}.get_walk_priorities"
    return resolve_walk_priorities(
        model.get_walk_priorities() if entries is None else entries,
        {walk: section.get_nodes() for walk, section in walks.items()},
        owner,
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
