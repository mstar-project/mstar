import math
from collections import defaultdict
from dataclasses import dataclass

from mstar.graph.base import GraphEdge, NodeAndGraphWalk


@dataclass
class ShardingGroup:
    nodes: set[str]
    tp_size: int
    graph_walks: set[str] | None = None  # None = all graph walks
    _tp_rank: int | None = None  # set by the conductor

    def __post_init__(self):
        self.nodes = set(self.nodes)
        if self.graph_walks is not None:
            self.graph_walks = set(self.graph_walks)

    def register_workers(self, workers: list[str], my_tp_rank: int | None = None):
        assert len(workers) == self.tp_size, (
            f"register_workers got {len(workers)} workers but tp_size={self.tp_size}"
        )
        self._workers = list(workers)
        self._workers_set = set(workers)
        self._worker_to_tp_rank = {
            w: i for i, w in enumerate(workers)
        }
        if my_tp_rank is not None:
            self._tp_rank = my_tp_rank

    def clone_empty(self):
        return ShardingGroup(
            nodes=self.nodes,
            tp_size=self.tp_size,
            graph_walks=self.graph_walks,
            _tp_rank=self._tp_rank
        )

    def key_str(self):
        key = "///".join(sorted(self.nodes))
        if self.graph_walks is not None:
            key += "|" + "///".join(sorted(self.graph_walks))
        return key

@dataclass
class ShardDestination:
    worker: str
    full_tensor: bool
    tp_rank: int
    start_idxs: list[int] | None = None
    end_idxs: list[int] | None = None


@dataclass
class ShardingConfig:
    groups: list[ShardingGroup]
    tp_enabled_nodes: set[str]
    shard_dim: dict[str, int | None]  # signal name to shard dim (None/absent for replicated)

    def clone_empty(self):
        return ShardingConfig(
            groups=[group.clone_empty() for group in self.groups],
            tp_enabled_nodes=self.tp_enabled_nodes,
            shard_dim=self.shard_dim
        )

    def get_sharding_group(
        self, node: str, graph_walk: str,
    ):
        return self.group_mapping.get(NodeAndGraphWalk(node, graph_walk))

    def __post_init__(self):
        self.group_mapping: dict[NodeAndGraphWalk, ShardingGroup] = {}
        self.node_to_worker: dict[NodeAndGraphWalk, list[str]] = {}
        self.tp_enabled_nodes = set(self.tp_enabled_nodes)
        self._setup_done = False

    def assert_stream_consumer_compatibility(self, streaming_consumers: set[str]):
        """
        Asserts that stream consumers do not have graph-walk specific worker
        configuration
        """
        for node in streaming_consumers:
            if NodeAndGraphWalk(node, None) not in self.group_mapping:
                raise RuntimeError((
                    f"Node {node} is a streaming consumer but has a custom graph walk -> worker "
                    "configuration. For streaming consumers, it is disallowed to to map different "
                    "graph walks to different workers."
                ))

    def setup(
        self, node_to_workers: dict[NodeAndGraphWalk, list[str]]
    ):
        # Note: this function can be called multiple times, e.g., for different
        # partitions, and each will build on this object's existing group_mapping
        # and node_to_worker dictionaries. Now, however, it only needs to be
        # called once.
        self._setup_done = True
        self.node_to_worker.update(node_to_workers)
        graph_walks = {x.graph_walk for x in node_to_workers.keys()}

        for group in self.groups:
            add_none_flag = group.graph_walks is None
            gws = group.graph_walks if group.graph_walks is not None else graph_walks
            for node in group.nodes:
                for gw in gws:
                    key = NodeAndGraphWalk(node, gw)
                    if key in node_to_workers:
                        self.group_mapping[key] = group
                        group.register_workers(node_to_workers[key])
                    if add_none_flag:
                        # For streaming edges, we don't know the graph walk a priori,
                        # so we add this entry to the group_mapping to be able to look up
                        # streaming receivers. NOTE: we enforce that, for streaming
                        # consumers, no custom graph walk configuration is specified for TP,
                        # so all streaming consumers will have add_none_flag be True.
                        none_key = NodeAndGraphWalk(node, None)
                        existing = self.group_mapping.get(none_key)
                        assert existing is None or existing is group, (
                            f"Two groups both claim node {node!r} with graph_walks=None — "
                            f"streaming consumers must have exactly one ShardingGroup."
                        )
                        self.group_mapping[none_key] = group

        node_and_workers_to_gws = {}
        num_walk_combos: dict[str, int] = defaultdict(lambda: 0)
        for node_gw, workers in node_to_workers.items():
            (node, gw) = node_gw.node, node_gw.graph_walk
            if node_gw in self.group_mapping:
                continue
            workers = tuple(workers)
            if (node, workers) in node_and_workers_to_gws:
                node_and_workers_to_gws[(node, workers)].append(gw)
            else:
                num_walk_combos[node] += 1
                node_and_workers_to_gws[(node, workers)] = [gw]

        for (node, workers), gws in node_and_workers_to_gws.items():
            singleton = ShardingGroup(
                nodes=[node], tp_size=1, graph_walks=gws,
            )

            singleton.register_workers(
                list(workers), my_tp_rank=0
            )
            for gw in gws:
                key = NodeAndGraphWalk(node, gw)
                self.group_mapping[key] = singleton
            if num_walk_combos[node] == 1:
                self.group_mapping[NodeAndGraphWalk(node, None)] = singleton

    def compute_fanout(
        self, signal: str, source_graph_walk: str,
        source_node: str, dest_node: str,
        shard_dim_sizes: list[int], # for TensorPointerInfo list
        dest_graph_walk: str | None,
        source_tp_rank: int | None = None,
    ) -> list[ShardDestination]:
        if not self._setup_done:
            raise RuntimeError("Must call setup before compute_fanout")

        source = NodeAndGraphWalk(source_node, source_graph_walk)
        dest = NodeAndGraphWalk(dest_node, dest_graph_walk)
        source_group = self.group_mapping.get(source)
        dest_group = self.group_mapping.get(dest)
        shard_dim = self.shard_dim.get(signal)

        if source_group is None:
            # special source, like API server: one rank
            source_tp_rank = 0
            source_tp_size = 1
            source_worker = None
            source_worker_set = set()
        else:
            if source_tp_rank is None:
                assert source_group._tp_rank is not None, (
                    f"source group for {source} has no _tp_rank set; "
                    "call register_workers with my_tp_rank, or pass source_tp_rank explicitly"
                )
                source_tp_rank = source_group._tp_rank
            source_worker = source_group._workers[source_tp_rank]
            source_worker_set = source_group._workers_set
            source_tp_size = source_group.tp_size

        if dest_group is None:
            dest_worker_set = {""}
            dest_worker_to_tp_rank = {"": 0}
            dest_tp_size = 1
            dest_tp_rank = 0
        else:
            dest_worker_set = dest_group._workers_set
            dest_tp_size = dest_group.tp_size
            dest_tp_rank = dest_group._tp_rank
            dest_worker_to_tp_rank = dest_group._worker_to_tp_rank

        fanout = []
        if shard_dim is None:  # replicated
            if source_worker in dest_worker_set:
                fanout.append(ShardDestination(
                    worker=source_worker, full_tensor=True,
                    tp_rank=dest_worker_to_tp_rank.get(source_worker, 0)
                ))
            if source_tp_rank == 0:
                # find all workers that do not already have the tensor
                workers_needing_tensor = dest_worker_set - source_worker_set
                for worker in workers_needing_tensor:
                    fanout.append(ShardDestination(
                        worker=worker, full_tensor=True,
                        tp_rank=dest_worker_to_tp_rank.get(worker, 0)
                    ))
        else:  # sharded
            # like in vLLM, we will enforce that the TP size divides the shard dim size
            total_sizes = [s * source_tp_size for s in shard_dim_sizes]
            assert all([s % dest_tp_size == 0 for s in total_sizes]), (
                f"some total shard dim size from {total_sizes} not divisible by "
                f"dest tp_size {dest_tp_size}"
            )
            dest_shard_sizes = [s // dest_tp_size for s in total_sizes]

            source_shard_starts = [source_tp_rank * s for s in shard_dim_sizes]
            source_shard_ends = [(source_tp_rank + 1) * s for s in shard_dim_sizes]

            for dest_tp_rank in range(dest_tp_size):
                dest_shard_starts = [dest_tp_rank * s for s in dest_shard_sizes]
                dest_shard_ends = [(dest_tp_rank + 1) * s for s in dest_shard_sizes]

                if source_shard_ends[0] <= dest_shard_starts[0]:
                    break  # beyond the point of overlap, can stop
                if source_shard_starts[0] >= dest_shard_ends[0]:
                    continue  # not reached the point of overlap yet, keep looking

                worker = dest_worker_to_tp_rank[dest_tp_rank]
                fanout.append(ShardDestination(
                    worker=worker,
                    full_tensor=False,
                    start_idxs=[
                        max(src_s, dest_s)
                        for (src_s, dest_s) in zip(source_shard_starts, dest_shard_starts, strict=False)
                    ],
                    end_idxs=[
                        min(src_e, dest_e)
                        for (src_e, dest_e) in zip(source_shard_ends, dest_shard_ends, strict=False)
                    ],
                    tp_rank=dest_worker_to_tp_rank.get(worker, 0)
                ))

        return fanout

    def fanout_graph_edges(
        self, graph_edge: GraphEdge,
        source_node: str,
        source_graph_walk: str,
        dest_graph_walk: str | None,
        source_tp_rank: int | None = None,
    ) -> dict[str, GraphEdge]: # dest worker to graph edge
        # canonical form: leading shard dim
        shard_dim_sizes = [
            info.dims[0] for info in graph_edge.tensor_info
        ]
        fanout = self.compute_fanout(
            signal=graph_edge.name, source_graph_walk=source_graph_walk,
            source_node=source_node, dest_node=graph_edge.next_node,
            shard_dim_sizes=shard_dim_sizes,
            dest_graph_walk=dest_graph_walk,
            source_tp_rank=source_tp_rank
        )

        source = NodeAndGraphWalk(source_node, source_graph_walk)
        source_group = self.group_mapping.get(source)
        if source_group is None:
            source_tp_size = 1
        else:
            source_tp_size = source_group.tp_size

        result = {}
        for item in fanout:
            new_edge = graph_edge.clone()
            new_edge.tensor_info = []
            new_edge._shard_dim = self.shard_dim.get(new_edge.name)
            new_edge._total_fanin = self.compute_fanin(
                signal=new_edge.name, source_tp_size=source_tp_size,
                dest_node=new_edge.next_node,
                dest_graph_walk=dest_graph_walk,
                dest_tp_rank=item.tp_rank
            )
            if item.full_tensor:
                new_edge.tensor_info = graph_edge.tensor_info
            else:
                for i, info in enumerate(graph_edge.tensor_info):
                    # compute new dims
                    start_idx = item.start_idxs[i]
                    end_idx = item.end_idxs[i]
                    new_dims = list(info.dims)

                    # canonical form has leading shard dim
                    new_dims[0] = end_idx - start_idx

                    # compute offset and new nbytes
                    element_size = info.nbytes // math.prod(info.dims)
                    row_nbytes = element_size * math.prod(info.dims[1:])
                    offset = start_idx * row_nbytes
                    new_nbytes = (end_idx - start_idx) * row_nbytes

                    new_info = info.clone()
                    new_info.dims = tuple(new_dims)
                    new_info.nbytes = new_nbytes
                    new_info.offset = offset

                    new_edge.tensor_info.append(new_info)
            result[item.worker] = new_edge
        return result

    def compute_fanin(
        self, signal: str, source_tp_size: int,
        dest_node: str, dest_graph_walk: str,
        dest_tp_rank: int
    ) -> int:  # number of source workers contributing to this rank's tensor
        if self.shard_dim.get(signal) is None:  # replicated
            return 1
        dest_group = self.group_mapping.get(NodeAndGraphWalk(dest_node, dest_graph_walk))
        if dest_group is None:
            # special destination like API server
            dest_tp_size = 1
        else:
            dest_tp_size = dest_group.tp_size
        # Scaled integer coords (total = source_tp_size * dest_tp_size).
        dest_lo = dest_tp_rank * source_tp_size
        dest_hi = dest_lo + source_tp_size
        return sum(
            1 for r in range(source_tp_size)
            if r * dest_tp_size < dest_hi and (r + 1) * dest_tp_size > dest_lo
        )
