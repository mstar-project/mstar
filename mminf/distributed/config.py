from dataclasses import dataclass

import numpy as np

from mminf.graph.base import GraphEdge


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
        if my_tp_rank is not None:
            self._tp_rank = my_tp_rank


@dataclass
class ShardDestination:
    worker: str
    full_tensor: bool
    start_idxs: list[int] | None = None
    end_idx: list[int] | None = None


NodeAndGraphWalk = tuple[str, str]


@dataclass
class ShardingConfig:
    groups: list[ShardingGroup]
    shard_dim: dict[str, int | None]  # signal name to shard dim (None/absent for replicated)

    def setup(
        self, node_to_worker: dict[NodeAndGraphWalk, list[str]]
    ):
        self.node_to_worker = node_to_worker
        graph_walks = {gw for (_node, gw) in node_to_worker.keys()}
        self.group_mapping: dict[NodeAndGraphWalk, ShardingGroup] = {}
        for group in self.groups:
            gws = group.graph_walks if group.graph_walks is not None else graph_walks
            for node in group.nodes:
                for gw in gws:
                    self.group_mapping[(node, gw)] = group

        for (node, gw) in node_to_worker:
            if (node, gw) in self.group_mapping:
                continue
            singleton = ShardingGroup(
                nodes=[node], tp_size=1, graph_walks=[gw],
            )
            singleton.register_workers(
                list(node_to_worker[(node, gw)][0:1]), my_tp_rank=0
            )
            self.group_mapping[(node, gw)] = singleton

    def compute_fanout(
        self, signal: str, source_graph_walk: str,
        source_node: str, dest_node: str,
        shard_dim_sizes: list[int], # for TensorPointerInfo list
        dest_graph_walk: str | None = None,
        source_tp_rank: int | None = None,
    ) -> list[ShardDestination]:
        if not hasattr(self, "group_mapping"):
            raise RuntimeError("Must call setup before compute_fanout")

        if dest_graph_walk is None:
            dest_graph_walk = source_graph_walk
        source = (source_node, source_graph_walk)
        dest = (dest_node, dest_graph_walk)
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
            dest_worker_set = set()
            dest_tp_size = 1
            dest_tp_rank = 0
        else:
            dest_worker_set = dest_group._workers_set
            dest_tp_size = dest_group.tp_size
            dest_tp_rank = dest_group._tp_rank

        fanout = []
        if shard_dim is None:  # replicated
            if source_worker in dest_worker_set:
                fanout.append(ShardDestination(
                    worker=source_worker, full_tensor=True
                ))
            if source_tp_rank == 0:
                # find all workers that do not already have the tensor
                workers_needing_tensor = dest_worker_set - source_worker_set
                for worker in workers_needing_tensor:
                    fanout.append(ShardDestination(
                        worker=worker, full_tensor=True
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

                fanout.append(ShardDestination(
                    worker=dest_group._workers[dest_tp_rank] if dest_group else "api_server",
                    full_tensor=False,
                    start_idx=[
                        max(src_s, dest_s) \
                            for (src_s, dest_s) in zip(source_shard_starts, dest_shard_starts)
                    ],
                    end_idx=[
                        min(src_e, dest_e) \
                            for (src_e, dest_e) in zip(source_shard_ends, dest_shard_ends)
                    ]
                ))

        return fanout
    
    def fanout_graph_edges(
        self, graph_edge: GraphEdge,
        source_node: str,
        source_graph_walk: str,
        dest_graph_walk: str | None = None,
        source_tp_rank: int | None = None,
    ) -> dict[str, GraphEdge]: # dest worker to graph edge
        if dest_graph_walk is None:
            dest_graph_walk = source_graph_walk
        
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

        result = {}
        for item in fanout:
            new_edge = graph_edge.clone()
            new_edge.tensor_info = []
            if item.full_tensor:
                new_edge.tensor_info = graph_edge.tensor_info
            else:
                for i, info in enumerate(graph_edge.tensor_info):
                    # compute new dims
                    start_idx = item.start_idxs[i]
                    end_idx = item.end_idx[i]
                    new_dims = list(info.dims)

                    # canonical form has leading shard dim
                    new_dims[0] = end_idx - start_idx
                    
                    # compute offset and new nbytes
                    element_size = info.nbytes // np.prod(info.dims)
                    offset = start_idx * element_size * np.prod(info.dims[1:])
                    new_nbytes = (end_idx - start_idx) * element_size * np.prod(info.dims[1:])

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
    ) -> int:  # number of source workers contributing to this rank's tensor
        if self.shard_dim.get(signal) is None:  # replicated
            return 1
        dest_group = self.group_mapping.get((dest_node, dest_graph_walk))
        if dest_group is None:
            # special destination like API server
            dest_tp_size, dest_tp_rank = 1, 0
        else:
            dest_tp_size, dest_tp_rank = dest_group.tp_size, dest_group._tp_rank
        # Scaled integer coords (total = source_tp_size * dest_tp_size).
        dest_lo = dest_tp_rank * source_tp_size
        dest_hi = dest_lo + source_tp_size
        return sum(
            1 for r in range(source_tp_size)
            if r * dest_tp_size < dest_hi and (r + 1) * dest_tp_size > dest_lo
        )


    def compute_all_fanouts(
        self, signal: str, source_graph_walk: str,
        source_node: str, dest_node: str,
        shard_dim_size: int,
        dest_graph_walk: str | None = None,
    ) -> dict[str, list[ShardDestination]]:  # source worker -> shard destinations
        source = (source_node, source_graph_walk)
        source_group = self.group_mapping.get(source)
        tp_size = source_group.tp_size if source_group is not None else 1
        fanouts = {}
        for tp_rank in range(tp_size):
            worker = source_group._workers[tp_rank] if source_group else None
            fanouts[worker] = self.compute_fanout(
                signal=signal, source_graph_walk=source_graph_walk,
                source_node=source_node, dest_node=dest_node,
                shard_dim_size=shard_dim_size, dest_graph_walk=dest_graph_walk,
                source_tp_rank=tp_rank
            )
        return fanouts
