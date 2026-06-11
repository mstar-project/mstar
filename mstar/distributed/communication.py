from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist


class TPCommGroup:
    def __init__(
        self,
        my_global_rank: int,
        my_group_rank: int,
        group_members: list[int]
    ):
        self.global_rank = my_global_rank
        self.rank = my_group_rank
        self.group_members = group_members
        self.world_size = len(group_members)
        self.device_group = None
        self.initialized = False

    @classmethod
    def trivial(cls) -> "TPCommGroup":
        """A degenerate single-rank group. All collectives are no-ops;
        ``init_process_group`` does nothing. Useful as the default for
        non-TP runs so the same code path works everywhere."""
        return cls(my_global_rank=0, my_group_rank=0, group_members=[0])

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if dim < 0:
            # Convert negative dim to positive
            dim += input_.dim()
        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        # Allocate output tensor
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        # All-gather
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        # Reshape
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor
    
    def barrier(self):
        if self.world_size == 1:
            return
        dist.barrier(group=self.device_group)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        dist.all_reduce(input_, group=self.device_group)
        return input_
    
    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output_tensor = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        # Perform reduce-scatter operation
        dist.reduce_scatter_tensor(
            output_tensor, input_tensor, group=self.device_group
        )

        # Reshape before returning
        return output_tensor.movedim(0, dim).contiguous()
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from source rank to all ranks."""
        if self.world_size == 1:
            return tensor
        dist.broadcast(tensor, self.group_members[src], self.device_group)
        return tensor


@dataclass
class WorkerTPGroups:
    num_workers: int
    global_rank: int
    # True iff any worker in the run uses TP. Set by GlobalTPConfig from
    # the global worker-graph view so all ranks agree.
    any_tp: bool = False
    # Every distinct TP rank tuple in the run, sorted for stable iteration
    # order across workers. Set by GlobalTPConfig. ``init_dist`` calls
    # ``dist.new_group`` once per entry on every rank — including ranks
    # that aren't members of the group — because PyTorch assigns an
    # auto-incrementing tag inside ``new_group`` that all ranks must agree
    # on; asymmetric call counts deadlock the participating ranks.
    world_tp_groups: list[tuple[int, ...]] = field(default_factory=list)
    node_to_group: dict[str, TPCommGroup] = field(default_factory=dict)

    def add(self, node: str, comm_group: TPCommGroup):
        # disallow colocation of multiple comm groups on the same node
        if node in self.node_to_group and self.node_to_group[node].group_members != comm_group.group_members:
            raise RuntimeError(
                f"Node {node} already has a comm group assigned for worker {self.global_rank}"
            )
        if node not in self.node_to_group:
            self.node_to_group[node] = comm_group
    
    def init_dist(
        self, init_method="tcp://127.0.0.1:29500",
    ):
        """Initialize the NCCL world group and per-node TP subgroups.

        Every worker calls ``dist.init_process_group`` when *any* worker
        in the run participates in TP (``self.any_tp``) — otherwise ranks
        with no local TP would skip the call and the TP-participating
        ranks would hang waiting for them.

        Subgroup creation: PyTorch's ``dist.new_group`` is collective on
        the global world. It assigns an auto-incrementing tag inside the
        call that every rank must agree on; if non-member ranks skip the
        call, the tag counter drifts and member ranks deadlock. We
        therefore call ``new_group`` once per distinct TP rank tuple on
        every rank — members keep the returned handle, non-members
        discard it.
        """
        torch.cuda.set_device(self.global_rank)
        if not self.any_tp:
            return

        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=self.num_workers,
            rank=self.global_rank,
        )

        rank_tuple_to_pg: dict[tuple[int, ...], "dist.ProcessGroup"] = {}
        for rank_tuple in self.world_tp_groups:
            rank_tuple_to_pg[rank_tuple] = dist.new_group(ranks=list(rank_tuple))

        seen: set[int] = set()
        for comm_group in self.node_to_group.values():
            if id(comm_group) in seen:
                continue
            seen.add(id(comm_group))
            if comm_group.world_size == 1:
                comm_group.initialized = True
                continue
            comm_group.device_group = rank_tuple_to_pg[tuple(comm_group.group_members)]
            comm_group.initialized = True
    
    def get_tp_config_for_node(self, node: str) -> TPCommGroup:
        if node not in self.node_to_group:
            self.node_to_group[node] = TPCommGroup.trivial()
        return self.node_to_group[node]

    def barrier_all(self) -> None:
        """Global barrier across every worker process in the run.

        No-op when ``any_tp`` is False (no NCCL world was initialized in
        ``init_dist``). Otherwise calls ``dist.barrier()`` on the default
        global process group, syncing both TP-participating and non-TP
        workers. Used at phase boundaries that require all ranks to be
        ready — e.g. between CUDA-graph warmup and the worker's main
        loop, so a TP leader can't send a ``ScheduleTPNode`` to a
        follower that's still inside ``engine.warmup``.
        """
        if not self.any_tp:
            return
        dist.barrier()


class GlobalTPConfig:
    def __init__(
        # leaving type annotation as Any due to circular import
        self, worker_graphs: dict[str, Any],
        worker_ids: list[str]
    ):
        self.num_workers = len(worker_ids)
        any_tp = any(wg.tp_size > 1 for wg in worker_graphs.values())
        world_tp_groups: list[tuple[int, ...]] = sorted({
            tuple(rank_group)
            for wg in worker_graphs.values()
            for rank_group in wg._tp_ranks
            if len(rank_group) > 1
        })
        self.per_worker_config: dict[str, WorkerTPGroups] = {
            wid: WorkerTPGroups(
                global_rank=i, num_workers=self.num_workers,
                any_tp=any_tp,
                world_tp_groups=world_tp_groups,
            ) for i, wid in enumerate(worker_ids)
        }

        # (global rank, (group ranks...)) -> comm group
        self.comm_groups: dict[tuple[int, tuple], TPCommGroup] = {}
        for wg in worker_graphs.values():
            for rank_group in wg._tp_ranks:
                rank_group_tuple = tuple(rank_group)
                for i, rank in enumerate(rank_group):
                    key = (rank, rank_group_tuple)
                    if key not in self.comm_groups:
                        self.comm_groups[key] = TPCommGroup(
                            my_global_rank=rank,
                            my_group_rank=i,
                            group_members=rank_group
                        )
                    for node in wg.section.get_nodes().keys():
                        self.per_worker_config[worker_ids[rank]].add(
                            node,  self.comm_groups[key]
                        )
        
