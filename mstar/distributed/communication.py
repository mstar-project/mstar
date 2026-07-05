from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist


class CommGroup:
    """A communication group over one axis of the worker device mesh.

    Serves both parallelism axes: tensor parallelism (all-reduce /
    all-gather of row-parallel projections) and Ulysses sequence
    parallelism (all-to-all around attention). ``rank`` / ``world_size``
    are relative to this group; ``global_rank`` is the worker's rank in
    the world.
    """

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
    def trivial(cls) -> "CommGroup":
        """A degenerate single-rank group. All collectives are no-ops;
        ``init_process_group`` does nothing. Useful as the default for
        non-parallel runs so the same code path works everywhere."""
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

    def all_to_all(
        self,
        input_: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        """Redistribute ``input_`` across the group: split it into
        ``world_size`` pieces along ``scatter_dim`` (piece i goes to rank i)
        and concatenate the pieces received from every rank along
        ``gather_dim``.

        This is the primitive behind Ulysses sequence parallelism: with
        ``scatter_dim`` = heads and ``gather_dim`` = sequence it converts a
        sequence-sharded ``[seq/P, heads, dim]`` tensor into a head-sharded
        ``[seq, heads/P, dim]`` one (and the reverse with the dims swapped).

        ``scatter_sizes`` / ``gather_sizes`` give per-rank extents for an
        uneven split / gather (e.g. a sequence length not divisible by the
        group size); when ``None`` the respective dimension is split evenly
        (and ``scatter_dim`` must then be divisible by ``world_size``).
        """
        if self.world_size == 1:
            return input_
        world_size = self.world_size
        if scatter_sizes is None:
            assert input_.size(scatter_dim) % world_size == 0, (
                f"all_to_all: scatter_dim {scatter_dim} size "
                f"{input_.size(scatter_dim)} not divisible by world_size "
                f"{world_size}; pass scatter_sizes for an uneven split"
            )
            chunk = input_.size(scatter_dim) // world_size
            scatter_sizes = [chunk] * world_size
        send = [
            t.contiguous()
            for t in torch.split(input_, scatter_sizes, dim=scatter_dim)
        ]
        if gather_sizes is None:
            gather_sizes = [send[self.rank].size(gather_dim)] * world_size
        base_shape = list(send[self.rank].shape)
        recv = []
        for r in range(world_size):
            shape = list(base_shape)
            shape[gather_dim] = gather_sizes[r]
            recv.append(
                torch.empty(shape, dtype=input_.dtype, device=input_.device)
            )
        dist.all_to_all(recv, send, group=self.device_group)
        return torch.cat(recv, dim=gather_dim).contiguous()


@dataclass
class WorkerParallelGroups:
    """Per-worker view of the parallelism comm groups in the run.

    Tensor parallelism and sequence parallelism are orthogonal axes of the
    same device mesh, so a node may carry one comm group of each. The
    combined TP x SP block is the node's lockstep *instance* — the unit the
    conductor routes a request to — exposed via
    ``get_instance_rank_for_node`` / ``get_instance_world_size_for_node``.
    """

    num_workers: int
    global_rank: int
    # True iff any worker in the run uses TP or SP. Set by
    # GlobalParallelConfig from the global worker-graph view so all ranks
    # agree.
    any_parallelism: bool = False
    # Every distinct TP / SP rank tuple in the run — the union of both mesh
    # axes, sorted for stable iteration order across workers. Set by
    # GlobalParallelConfig. ``init_dist`` calls ``dist.new_group`` once per
    # entry on every rank — including ranks that aren't members of the
    # group — because PyTorch assigns an auto-incrementing tag inside
    # ``new_group`` that all ranks must agree on; asymmetric call counts
    # deadlock the participating ranks.
    world_parallel_groups: list[tuple[int, ...]] = field(default_factory=list)
    # Per-node comm groups, one per mesh axis (the SP group all-to-alls
    # around attention; the TP group all-reduces the row-parallel
    # projections).
    node_to_tp_group: dict[str, CommGroup] = field(default_factory=dict)
    node_to_sp_group: dict[str, CommGroup] = field(default_factory=dict)

    def add(self, node: str, comm_group: CommGroup):
        # disallow colocation of multiple comm groups on the same node
        if node in self.node_to_tp_group and self.node_to_tp_group[node].group_members != comm_group.group_members:
            raise RuntimeError(
                f"Node {node} already has a comm group assigned for worker {self.global_rank}"
            )
        if node not in self.node_to_tp_group:
            self.node_to_tp_group[node] = comm_group

    def add_sp(self, node: str, comm_group: CommGroup):
        # SP analogue of ``add`` — the sequence-parallel comm group for a node.
        if node in self.node_to_sp_group and self.node_to_sp_group[node].group_members != comm_group.group_members:
            raise RuntimeError(
                f"Node {node} already has an SP comm group assigned for worker {self.global_rank}"
            )
        if node not in self.node_to_sp_group:
            self.node_to_sp_group[node] = comm_group

    def init_dist(
        self, init_method="tcp://127.0.0.1:29500",
    ):
        """Initialize the NCCL world group and per-node parallel subgroups.

        Every worker calls ``dist.init_process_group`` when *any* worker
        in the run participates in TP or SP (``self.any_parallelism``) —
        otherwise ranks with no local parallelism would skip the call and
        the participating ranks would hang waiting for them.

        Subgroup creation: PyTorch's ``dist.new_group`` is collective on
        the global world. It assigns an auto-incrementing tag inside the
        call that every rank must agree on; if non-member ranks skip the
        call, the tag counter drifts and member ranks deadlock. We
        therefore call ``new_group`` once per distinct rank tuple on
        every rank — members keep the returned handle, non-members
        discard it.
        """
        torch.cuda.set_device(self.global_rank)
        if not self.any_parallelism:
            return

        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=self.num_workers,
            rank=self.global_rank,
        )

        # One subgroup per distinct rank tuple across BOTH mesh axes —
        # ``world_parallel_groups`` is the sorted union, so every rank
        # iterates it identically; ``new_group`` is collective and
        # tag-ordered (see the field comment). A tuple shared by a TP and
        # an SP group (degenerate meshes) maps to one subgroup.
        rank_tuple_to_pg: dict[tuple[int, ...], "dist.ProcessGroup"] = {}
        for rank_tuple in self.world_parallel_groups:
            rank_tuple_to_pg[rank_tuple] = dist.new_group(ranks=list(rank_tuple))

        seen: set[int] = set()
        for comm_group in (
            list(self.node_to_tp_group.values()) + list(self.node_to_sp_group.values())
        ):
            if id(comm_group) in seen:
                continue
            seen.add(id(comm_group))
            if comm_group.world_size == 1:
                comm_group.initialized = True
                continue
            comm_group.device_group = rank_tuple_to_pg[tuple(comm_group.group_members)]
            comm_group.initialized = True

    def get_tp_config_for_node(self, node: str) -> CommGroup:
        if node not in self.node_to_tp_group:
            self.node_to_tp_group[node] = CommGroup.trivial()
        return self.node_to_tp_group[node]

    def get_sp_config_for_node(self, node: str) -> CommGroup:
        if node not in self.node_to_sp_group:
            self.node_to_sp_group[node] = CommGroup.trivial()
        return self.node_to_sp_group[node]

    def get_instance_rank_for_node(self, node: str) -> int:
        """This worker's rank within ``node``'s lockstep instance — the
        TP x SP block, row-major [sp][tp] to match
        ``WorkerGraph._instance_ranks``. 0 iff this worker leads the
        instance (it is rank 0 in both its TP and SP comm groups)."""
        tp = self.get_tp_config_for_node(node)
        sp = self.get_sp_config_for_node(node)
        return sp.rank * tp.world_size + tp.rank

    def get_instance_world_size_for_node(self, node: str) -> int:
        """Total ranks in ``node``'s lockstep instance (tp_size * sp_size)."""
        return (
            self.get_tp_config_for_node(node).world_size
            * self.get_sp_config_for_node(node).world_size
        )

    def barrier_all(self) -> None:
        """Global barrier across every worker process in the run.

        No-op when ``any_parallelism`` is False (no NCCL world was
        initialized in ``init_dist``). Otherwise calls ``dist.barrier()``
        on the default global process group, syncing participating and
        non-participating workers alike. Used where every rank must be
        ready before proceeding — e.g. between CUDA-graph warmup and the
        worker's main loop, so an instance leader can't send a
        ``ScheduleTPNode`` to a follower that's still inside
        ``engine.warmup``.
        """
        if not self.any_parallelism:
            return
        dist.barrier()


class GlobalParallelConfig:
    def __init__(
        # leaving type annotation as Any due to circular import
        self, worker_graphs: dict[str, Any],
        worker_ids: list[str]
    ):
        self.num_workers = len(worker_ids)
        any_parallelism = any(
            wg.tp_size > 1 or wg.sp_size > 1 for wg in worker_graphs.values()
        )
        world_tp_groups: list[tuple[int, ...]] = sorted({
            tuple(rank_group)
            for wg in worker_graphs.values()
            for rank_group in wg._tp_ranks
            if len(rank_group) > 1
        })
        world_sp_groups: list[tuple[int, ...]] = sorted({
            tuple(rank_group)
            for wg in worker_graphs.values()
            for rank_group in wg._sp_ranks
            if len(rank_group) > 1
        })
        # The union of both axes, sorted — the per-rank ``new_group``
        # schedule (see the ``world_parallel_groups`` field comment).
        world_parallel_groups: list[tuple[int, ...]] = sorted(
            set(world_tp_groups) | set(world_sp_groups)
        )
        self.per_worker_config: dict[str, WorkerParallelGroups] = {
            wid: WorkerParallelGroups(
                global_rank=i, num_workers=self.num_workers,
                any_parallelism=any_parallelism,
                world_parallel_groups=world_parallel_groups,
            ) for i, wid in enumerate(worker_ids)
        }

        # (global rank, (group ranks...)) -> comm group, for each mesh axis.
        self.comm_groups: dict[tuple[int, tuple], CommGroup] = {}
        self.sp_comm_groups: dict[tuple[int, tuple], CommGroup] = {}
        for wg in worker_graphs.values():
            for rank_group in wg._tp_ranks:
                rank_group_tuple = tuple(rank_group)
                for i, rank in enumerate(rank_group):
                    key = (rank, rank_group_tuple)
                    if key not in self.comm_groups:
                        self.comm_groups[key] = CommGroup(
                            my_global_rank=rank,
                            my_group_rank=i,
                            group_members=rank_group
                        )
                    for node in wg.section.get_nodes().keys():
                        self.per_worker_config[worker_ids[rank]].add(
                            node,  self.comm_groups[key]
                        )
            for rank_group in wg._sp_ranks:
                rank_group_tuple = tuple(rank_group)
                for i, rank in enumerate(rank_group):
                    key = (rank, rank_group_tuple)
                    if key not in self.sp_comm_groups:
                        self.sp_comm_groups[key] = CommGroup(
                            my_global_rank=rank,
                            my_group_rank=i,
                            group_members=rank_group
                        )
                    for node in wg.section.get_nodes().keys():
                        self.per_worker_config[worker_ids[rank]].add_sp(
                            node, self.sp_comm_groups[key]
                        )

