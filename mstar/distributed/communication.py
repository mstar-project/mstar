import socket
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any
from urllib.parse import urlparse

import torch
import torch.distributed as dist
from torch.distributed.constants import default_pg_nccl_timeout, default_pg_timeout
from torch.distributed.distributed_c10d import _set_pg_timeout

from mstar.distributed.all_reduce import AllReduce
from mstar.distributed.base import PARALLEL_AXES
from mstar.distributed.utils import resolve_dist_timeout


def default_group_timeout(backend: str) -> timedelta:
    """The timeout ``init_process_group`` gives ``backend``'s group when none is passed.

    With ``init_method`` it also sets the store to it; with ``store=`` it
    leaves the store alone, so the caller passes this to both.
    """
    if dist.Backend(backend) == dist.Backend.NCCL and default_pg_nccl_timeout is not None:
        return default_pg_nccl_timeout
    return default_pg_timeout


def rendezvous_store(init_method: str, rank: int, world_size: int, timeout: timedelta) -> dist.TCPStore:
    """The ``tcp://host:port`` store every rank meets at, listening on ``host`` only.

    ``init_process_group(init_method="tcp://...")`` has rank 0 serve its
    TCPStore on every interface whatever the host, so rank 0 binds the
    listener itself and hands it over.
    """
    url = urlparse(init_method)
    if url.scheme != "tcp" or not url.hostname or not url.port:
        raise ValueError(f"init_method must be tcp://host:port; got {init_method!r}")
    if rank != 0:
        return dist.TCPStore(url.hostname, url.port, world_size, is_master=False, timeout=timeout)
    listener = socket.create_server((url.hostname, url.port), backlog=max(world_size, 128))
    # the store owns the descriptor from here
    return dist.TCPStore(
        url.hostname, url.port, world_size, is_master=True, timeout=timeout,
        wait_for_workers=False, master_listen_fd=listener.detach(),
    )


@torch.compiler.disable
def _all_to_all_single(
    group: dist.ProcessGroup, send: torch.Tensor, recv: torch.Tensor,
    recv_splits: list[int] | None = None, send_splits: list[int] | None = None,
) -> torch.Tensor:
    dist.all_to_all_single(recv, send, recv_splits, send_splits, group=group)
    return recv


class CommGroup:
    """A communication group over one axis of the worker device mesh, or over
    a whole lockstep instance.

    An axis group runs the model's collectives: tensor parallelism
    (all-reduce / all-gather of row-parallel projections) or Ulysses sequence
    parallelism (all-to-all around attention). An instance group only agrees
    on the host (``barrier``, ``agree``), so it has a gloo group and
    no device group. ``rank`` / ``world_size``
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
        # host channel, for rank votes (``agree``), ``barrier`` and object
        # collectives (workspace rendezvous) that should not go through NCCL:
        # a gloo twin of ``device_group``, or on XPU the mixed group's gloo
        # side; a lockstep instance's only group
        self.cpu_group = None
        self.initialized = False
        # see ``init_allreduce_fusion``
        self._ar_fusion_ws: int | None = None
        self._ar_fusion_max_tokens = 0
        # NCCL until ``init_all_reduce`` sets up the deployment's backend
        self._all_reduce = AllReduce(self)

    def init_all_reduce(self, mode: str | None = None, max_kb: int | None = None) -> None:
        """Set up the all-reduce backend the deployment config's ``tp_allreduce``
        (``mode``) and ``tp_allreduce_max_kb`` (``max_kb``) ask for (see
        ``AllReduce.new``). Collective across the group — every member calls
        this from ``init_dist`` at the same point."""
        self._all_reduce = AllReduce.new(self, mode, max_kb)

    def _device_pg(self) -> "dist.ProcessGroup":
        """The group a device collective runs on. torch reads ``group=None`` as
        the whole world, so a group without one refuses instead."""
        if self.device_group is None:
            raise ValueError(
                f"device collective on ranks {self.group_members}: a host-only (lockstep instance) "
                "group, or one init_dist has not set up"
            )
        return self.device_group

    def agree(self, values: list[int], op: dist.ReduceOp = dist.ReduceOp.MIN) -> list[int]:
        """Reduce each of ``values`` over the group's ranks with ``op``: ``MIN`` ANDs
        flags, ``MAX`` takes the largest. Collective: every member calls it at the
        same point. Over ``cpu_group``, so the answer costs no device sync."""
        if self.world_size == 1 or not values:
            return list(values)
        if self.cpu_group is None:
            raise ValueError(
                f"agree over ranks {self.group_members} before init_dist set up their cpu_group"
            )
        t = torch.tensor(values, dtype=torch.int64)
        dist.all_reduce(t, op=op, group=self.cpu_group)
        return t.tolist()

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
        dist.all_gather_into_tensor(output_tensor, input_, group=self._device_pg())
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
        """On the host: waits on the other ranks, never on a device."""
        if self.world_size == 1:
            return
        dist.barrier(group=self.cpu_group)

    def all_reduce_buffer(
        self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Where to write the partial that the next ``all_reduce`` sums. On
        the symmetric-memory path this is the symm buffer, so ``all_reduce``
        skips its copy in and returns the sum in a new tensor: use its return
        value, and all-reduce nothing else on this group in between."""
        return self._all_reduce.buffer(shape, dtype, device)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return self._all_reduce.all_reduce(input_)

    def init_allreduce_fusion(
        self, hidden: int, dtype: torch.dtype, max_tokens: int = 2048,
    ) -> bool:
        """Set up ``allreduce_add_rmsnorm``'s fused kernel for ``[<=max_tokens,
        hidden]`` inputs; larger inputs keep the NCCL path. Collective: every
        member calls it at the same point. Returns whether it is available."""
        if self.world_size == 1 or self._ar_fusion_ws is not None:
            return self._ar_fusion_ws is not None
        from mstar.distributed.ar_fusion import create_workspace

        self._ar_fusion_ws = create_workspace(
            self, max_tokens, hidden, dtype,
        )
        self._ar_fusion_max_tokens = max_tokens if self._ar_fusion_ws is not None else 0
        return self._ar_fusion_ws is not None

    def allreduce_add_rmsnorm(
        self, x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        eps: float, weight_bias: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``r = all_reduce(x) + residual``; returns ``(rms_norm(r), r)``,
        scaling by ``weight + weight_bias``. ``x`` is a row-parallel
        projection's partial sum (its linear skipped ``reduce_results``)."""
        if (self._ar_fusion_ws is not None
                and x.shape[0] <= self._ar_fusion_max_tokens
                and x.is_contiguous() and residual.is_contiguous()):
            from mstar.distributed.ar_fusion import allreduce_add_rmsnorm

            return allreduce_add_rmsnorm(
                x, residual, weight, eps, weight_bias, self._ar_fusion_ws,
            )
        r = self.all_reduce(x) + residual
        v = r.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
        return (v * (weight.float() + weight_bias)).to(r.dtype), r

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
            output_tensor, input_tensor, group=self._device_pg()
        )

        # Reshape before returning
        return output_tensor.movedim(0, dim).contiguous()

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from source rank to all ranks."""
        if self.world_size == 1:
            return tensor
        dist.broadcast(tensor, self.group_members[src], self._device_pg())
        return tensor

    def all_to_all(
        self,
        input_: torch.Tensor | Sequence[torch.Tensor],
        scatter_dim: int,
        gather_dim: int,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
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

        ``input_`` may be several tensors that differ only in the extent of
        ``scatter_dim``, split evenly (q, k and v with their own head counts):
        they go in one exchange and come back as a tuple. Each exchange is one
        ``all_to_all_single`` over a buffer packed in a single copy, which
        replays from a CUDA graph. The collective is hidden from torch.compile,
        so a compiled caller breaks around this call; tracing it as a
        functional collective instead gave an illegal memory access in
        Cosmos3's batched capture.
        """
        tensors = (input_,) if isinstance(input_, torch.Tensor) else tuple(input_)
        if self.world_size == 1:
            return input_ if isinstance(input_, torch.Tensor) else tensors
        world_size = self.world_size
        # positive dims: unflatten below adds a dim, which would shift a negative one
        scatter_dim %= tensors[0].dim()
        gather_dim %= tensors[0].dim()
        if scatter_sizes is not None and len(set(scatter_sizes)) > 1:
            assert len(tensors) == 1, "all_to_all: several tensors need an even scatter"
            out = self._all_to_all_uneven_scatter(tensors[0], scatter_dim, gather_dim, scatter_sizes, gather_sizes)
            return out if isinstance(input_, torch.Tensor) else (out,)
        # per tensor [.., N, ..] -> [P, .., N/P, ..]; rank r's block of the send buffer holds
        # its piece of each tensor one after another, so each tensor unpacks contiguous
        blocks = [x.unflatten(scatter_dim, (world_size, -1)).movedim(scatter_dim, 0) for x in tensors]
        numels = [blk[0].numel() for blk in blocks]
        send = blocks[0].new_empty(world_size, sum(numels))
        for blk, piece in zip(blocks, send.split(numels, dim=1), strict=True):
            piece.view(blk.shape).copy_(blk)
        if gather_sizes is None or len(set(gather_sizes)) == 1:
            # the block received from rank r has this rank's own block's layout
            recv = _all_to_all_single(self._device_pg(), send, torch.empty_like(send))
            outs = [
                piece.view(blk.shape).movedim(0, gather_dim).flatten(gather_dim, gather_dim + 1)
                for blk, piece in zip(blocks, recv.split(numels, dim=1), strict=True)
            ]
        else:
            # rank r's block: each tensor's piece with the gathered dim at rank r's extent
            shapes = [
                [blk.shape[1:gather_dim + 1] + (n,) + blk.shape[gather_dim + 2:] for blk in blocks]
                for n in gather_sizes
            ]
            rank_numels = [[int(torch.Size(shape).numel()) for shape in row] for row in shapes]
            recv = _all_to_all_single(
                self._device_pg(), send.flatten(), send.new_empty(sum(map(sum, rank_numels))),
                [sum(row) for row in rank_numels], [sum(numels)] * world_size,
            )
            per_rank = [
                [piece.view(shape) for piece, shape in zip(block.split(row), shape_row, strict=True)]
                for block, row, shape_row in zip(recv.split([sum(row) for row in rank_numels]), rank_numels, shapes,
                                                 strict=True)
            ]
            outs = [torch.cat([pieces[i] for pieces in per_rank], dim=gather_dim) for i in range(len(tensors))]
        return outs[0] if isinstance(input_, torch.Tensor) else tuple(outs)

    def _all_to_all_uneven_scatter(
        self, input_: torch.Tensor, scatter_dim: int, gather_dim: int,
        scatter_sizes: list[int], gather_sizes: list[int] | None,
    ) -> torch.Tensor:
        chunks = input_.split(scatter_sizes, dim=scatter_dim)
        send = torch.cat([chunk.contiguous().flatten() for chunk in chunks])
        mine = list(chunks[self.rank].shape)
        if gather_sizes is None:
            gather_sizes = [mine[gather_dim]] * self.world_size
        shapes = [mine[:gather_dim] + [n] + mine[gather_dim + 1:] for n in gather_sizes]
        numels = [int(torch.Size(shape).numel()) for shape in shapes]
        recv = _all_to_all_single(
            self._device_pg(), send, input_.new_empty(sum(numels)),
            numels, [chunk.numel() for chunk in chunks],
        )
        pieces = [
            piece.view(shape)
            for piece, shape in zip(recv.split(numels), shapes, strict=True)
        ]
        return torch.cat(pieces, dim=gather_dim)


@dataclass
class JointGroups:
    """A node's comm groups on this worker: one per mesh axis, for the model's
    own collectives, and ``instance``, the whole lockstep instance (its ranks
    in ``WorkerGraph._instance_ranks`` order, which the conductor's instance
    rank follows too). Anything every rank of the instance must agree on — a
    barrier, a vote, a symmetric-state check — is one host-side collective on
    ``instance``."""

    tp_group: CommGroup
    sp_group: CommGroup
    instance: CommGroup
    cfg_group: CommGroup = field(default_factory=CommGroup.trivial)

    @property
    def rank(self):
        """This worker's rank in the node's lockstep instance; 0 leads it."""
        return self.instance.rank

    @property
    def world_size(self):
        """Ranks in the node's lockstep instance."""
        return self.instance.world_size

    def broadcast(self, tensor: torch.Tensor) -> torch.Tensor:
        """Broadcast from instance rank 0 to the whole instance: down each axis
        in turn, from the member at coordinate 0, which instance rank 0 is on
        every axis. Each step spreads its value along one more axis, so any
        order works."""
        for axis in reversed(PARALLEL_AXES):
            tensor = getattr(self, f"{axis}_group").broadcast(tensor)
        return tensor

    def agree(self, values: list[int], op: dist.ReduceOp = dist.ReduceOp.MIN) -> list[int]:
        """``CommGroup.agree`` over the whole lockstep instance: one host
        all-reduce on ``instance``."""
        return self.instance.agree(values, op)


@dataclass
class WorkerParallelGroups:
    """Per-worker view of the parallelism comm groups in the run.

    The parallel axes (``PARALLEL_AXES``) are orthogonal axes of one device
    mesh, so a node may carry one comm group of each. The whole block is the
    node's lockstep *instance* — the unit the conductor routes a request to —
    exposed via ``get_instance_rank_for_node`` /
    ``get_instance_world_size_for_node``.
    """

    num_workers: int
    global_rank: int
    # True iff any worker in the run has a parallel axis. Set by
    # GlobalParallelConfig from the global worker-graph view so all ranks
    # agree.
    any_parallelism: bool = False
    # Every distinct axis rank tuple in the run — the union over the mesh
    # axes, sorted for stable iteration order across workers. Set by
    # GlobalParallelConfig. ``init_dist`` calls ``dist.new_group`` once per
    # entry on every rank — including ranks that aren't members of the
    # group — because PyTorch assigns an auto-incrementing tag inside
    # ``new_group`` that all ranks must agree on; asymmetric call counts
    # deadlock the participating ranks.
    world_parallel_groups: list[tuple[int, ...]] = field(default_factory=list)
    # Per-node comm groups by mesh axis (``PARALLEL_AXES``): the SP group
    # all-to-alls around attention; the TP group all-reduces the row-parallel
    # projections.
    axis_groups: dict[str, dict[str, CommGroup]] = field(
        default_factory=lambda: {axis: {} for axis in PARALLEL_AXES}
    )
    # Per-node lockstep instance groups (``JointGroups.instance``), gloo (on
    # XPU the world's mixed group, like the XPU axis groups);
    # a single-axis instance shares that axis group's.
    node_to_instance_comm: dict[str, CommGroup] = field(default_factory=dict)
    # Global topology metadata. Unlike the comm-group maps above, these maps
    # describe every replica, including ones this worker does not host. A
    # shape is the per-axis sizes, in ``PARALLEL_AXES`` order.
    node_to_parallel_shapes: dict[
        str, frozenset[tuple[int, ...]]
    ] = field(default_factory=dict)
    node_to_instance_groups: dict[
        str, frozenset[tuple[int, ...]]
    ] = field(default_factory=dict)
    _device: torch.device | None = field(default=None, init=False, repr=False)

    node_to_joint_group: dict[str, JointGroups] = field(default_factory=dict)
    # Process-group timeout in seconds for load and warmup, from the config's
    # ``dist_timeout_s``. None = torch default.
    dist_timeout_s: float | None = None
    # Small-message all-reduce path from the deployment config's
    # ``tp_allreduce``. None = NCCL.
    tp_allreduce: str | None = None
    # Largest message it takes, from ``tp_allreduce_max_kb``; None = 512.
    tp_allreduce_max_kb: int | None = None

    def add(self, node: str, comm_group: CommGroup, axis: str):
        # disallow colocation of multiple comm groups on the same node and axis
        per_node = self.axis_groups[axis]
        if node in per_node and per_node[node].group_members != comm_group.group_members:
            raise RuntimeError(
                f"Node {node} already has a {axis} comm group assigned for worker {self.global_rank}"
            )
        per_node.setdefault(node, comm_group)

    def init_dist(
        self, init_method="tcp://127.0.0.1:29500",
        device: torch.device | None = None,
    ):
        """Initialize the NCCL world group and per-node parallel subgroups.

        Every worker calls ``dist.init_process_group`` when *any* worker
        in the run has a parallel axis (``self.any_parallelism``) —
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
        device = device or torch.device("cuda", self.global_rank)
        self._device = device
        if device.type != "cpu":
            torch.accelerator.set_device_index(device)
        backend = dist.get_default_backend_for_device(device.type)
        if not self.any_parallelism:
            return
        if device.type == "xpu":
            backend = f"cpu:gloo,xpu:{backend}"

        timeout_kwargs = resolve_dist_timeout(self.dist_timeout_s)
        # a configured dist_timeout_s covers the rendezvous store too
        timeout = timeout_kwargs.get("timeout", default_group_timeout(backend))
        dist.init_process_group(
            backend=backend,
            store=rendezvous_store(init_method, self.global_rank, self.num_workers, timeout),
            world_size=self.num_workers,
            rank=self.global_rank,
            device_id=device,
            timeout=timeout,
        )

        # Gloo groups, members only: one per rank tuple of a mesh axis, and one
        # per multi-axis lockstep instance, which agrees only on the host and
        # so gets no NCCL communicator (a single-axis instance shares its
        # axis's). With the default group bound to a device, a non-member's
        # call would join an ncclCommSplit that members never make for a gloo
        # group. torch names a members-only group after its ranks and how many
        # groups this process holds, so every rank makes them before the device
        # groups (registered only on their members, they would skew the count),
        # in one global order, and a rank outside a tuple makes a group of just
        # itself: the count, and so the name the members rendezvous under, is
        # then equal on every member. On XPU an axis's device group doubles as
        # its cpu group, and an instance gets the world's mixed group, made on
        # every rank.
        axis_tuples = set(self.world_parallel_groups)
        instance_tuples = {
            block for blocks in self.node_to_instance_groups.values() for block in blocks if len(block) > 1
        }
        rank_tuple_to_cpu_pg: dict[tuple[int, ...], "dist.ProcessGroup"] = {}
        for rank_tuple in sorted(axis_tuples | instance_tuples):
            member = self.global_rank in rank_tuple
            if device.type == "xpu":
                if rank_tuple in axis_tuples:
                    continue  # the device group doubles as the cpu group below
                cpu_pg = dist.new_group(ranks=list(rank_tuple), **timeout_kwargs)
            else:
                cpu_pg = dist.new_group(
                    ranks=list(rank_tuple) if member else [self.global_rank],
                    backend="gloo", use_local_synchronization=True, **timeout_kwargs,
                )
            if member:
                rank_tuple_to_cpu_pg[rank_tuple] = cpu_pg

        # One device subgroup per distinct rank tuple over every mesh axis —
        # ``world_parallel_groups`` is the sorted union, so every rank
        # iterates it identically; ``new_group`` is collective and
        # tag-ordered (see the field comment). A tuple shared by two axes'
        # groups (degenerate meshes) maps to one subgroup.
        rank_tuple_to_pg: dict[tuple[int, ...], "dist.ProcessGroup"] = {}
        for rank_tuple in self.world_parallel_groups:
            rank_tuple_to_pg[rank_tuple] = dist.new_group(
                ranks=list(rank_tuple), **timeout_kwargs
            )
            if device.type == "xpu":
                rank_tuple_to_cpu_pg[rank_tuple] = rank_tuple_to_pg[rank_tuple]

        seen: set[int] = set()
        for comm_group in (group for per_node in self.axis_groups.values() for group in per_node.values()):
            if id(comm_group) in seen:
                continue
            seen.add(id(comm_group))
            if comm_group.world_size == 1:
                comm_group.initialized = True
                continue
            comm_group.device_group = rank_tuple_to_pg[tuple(comm_group.group_members)]
            comm_group.cpu_group = rank_tuple_to_cpu_pg[tuple(comm_group.group_members)]
            comm_group.initialized = True
            # Collective rendezvous within the group. Members visit shared
            # groups in the same relative order: GlobalParallelConfig fills
            # every worker's dicts in one global iteration order.
            comm_group.init_all_reduce(self.tp_allreduce, self.tp_allreduce_max_kb)

        for node, comm_group in self.node_to_instance_comm.items():
            if comm_group.world_size > 1:
                cpu_pg = rank_tuple_to_cpu_pg.get(tuple(comm_group.group_members))
                if cpu_pg is None:
                    raise ValueError(
                        f"node {node!r}: instance ranks {comm_group.group_members} are not "
                        f"among its instance groups "
                        f"{sorted(self.node_to_instance_groups.get(node, ()))}"
                    )
                comm_group.cpu_group = cpu_pg
            comm_group.initialized = True

    def _all_groups(self) -> list[CommGroup]:
        """Every axis group, then every instance group."""
        return [
            *(group for per_node in self.axis_groups.values() for group in per_node.values()),
            *self.node_to_instance_comm.values(),
        ]

    def get_joint_group_for_node(self, node: str) -> JointGroups:
        if node not in self.node_to_joint_group:
            axes = {
                f"{axis}_group": self.axis_groups[axis].setdefault(node, CommGroup.trivial())
                for axis in PARALLEL_AXES
            }
            instance = self.node_to_instance_comm.get(node)
            if instance is None:
                if any(group.world_size > 1 for group in axes.values()):
                    # every rank would lead it, and every vote on it would pass
                    raise ValueError(f"node {node!r} has a parallel group but no lockstep instance group")
                instance = CommGroup.trivial()
            self.node_to_joint_group[node] = JointGroups(**axes, instance=instance)
        return self.node_to_joint_group[node]

    def get_instance_rank_for_node(self, node: str) -> int:
        return self.get_joint_group_for_node(node).rank

    def get_instance_world_size_for_node(self, node: str) -> int:
        return self.get_joint_group_for_node(node).world_size

    def all_in_same_group(self, nodes: list[str]) -> bool:
        """Whether every node shares one group on every mesh axis.

        Per axis, and only where someone parallelizes: unregistered reads
        as the single-rank group, so a TP-only run would otherwise be rejected
        for "differing" on SP. By membership, not identity — the lazy getters
        mint a fresh group per node (and would cache one for a remote node).
        """
        for per_node in self.axis_groups.values():
            members = [
                tuple(group.group_members) if group is not None else (0,)
                for group in (per_node.get(node) for node in nodes)
            ]
            if all(len(m) == 1 for m in members):
                continue
            if len(set(members)) != 1:
                return False
        return True

    def all_have_compatible_parallel_shape(self, nodes: set[str]) -> bool:
        """Whether every replica uses the same size on every mesh axis."""
        shapes: set[tuple[int, ...]] = set()
        for node in nodes:
            global_shapes = self.node_to_parallel_shapes.get(node)
            if global_shapes:
                shapes.update(global_shapes)
                continue
            shapes.add(tuple(
                group.world_size if (group := self.axis_groups[axis].get(node)) is not None else 1
                for axis in PARALLEL_AXES
            ))
        return len(shapes) <= 1

    def resource_needs_remote_transfer(
        self, nodes: set[str], local_nodes: set[str],
    ) -> bool:
        """Whether a logical resource spans more than one worker instance."""
        instance_groups: set[tuple[int, ...]] = set()
        topology_known = True
        for node in nodes:
            groups = self.node_to_instance_groups.get(node)
            if not groups:
                topology_known = False
                break
            instance_groups.update(groups)
        if topology_known:
            return len(instance_groups) > 1
        # Preserve sensible behavior for hand-built/test configs that lack the
        # global maps: another named consumer necessarily needs a transfer.
        return bool(nodes - local_nodes)

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

    def restore_default_timeout(self) -> None:
        """Give every group this rank holds torch's default timeout back."""
        # the long timeout is for load and warmup; in serving a wedged rank
        # must abort the others, not leave them waiting for hours
        if not self.any_parallelism or not resolve_dist_timeout(self.dist_timeout_s):
            return
        groups = [dist.group.WORLD]
        for comm_group in self._all_groups():
            groups += [comm_group.device_group, comm_group.cpu_group]
        for pg in {id(g): g for g in groups if g is not None}.values():
            nccl = dist.get_backend(pg) == dist.Backend.NCCL
            _set_pg_timeout(default_pg_nccl_timeout if nccl else default_pg_timeout, pg)


class GlobalParallelConfig:
    def __init__(
        # leaving type annotation as Any due to circular import
        self, worker_graphs: dict[str, Any],
        worker_ids: list[str],
        dist_timeout_s: float | None = None,
        tp_allreduce: str | None = None,
        tp_allreduce_max_kb: int | None = None,
    ):
        self.num_workers = len(worker_ids)
        any_parallelism = any(wg.instance_size > 1 for wg in worker_graphs.values())
        node_to_parallel_shapes: dict[str, set[tuple[int, ...]]] = {}
        node_to_instance_groups: dict[str, set[tuple[int, ...]]] = {}
        for wg in worker_graphs.values():
            shape = tuple(wg.axis_sizes[axis] for axis in PARALLEL_AXES)
            for node in wg.section.get_nodes():
                node_to_parallel_shapes.setdefault(node, set()).add(shape)
                node_to_instance_groups.setdefault(node, set()).update(
                    tuple(group) for group in wg.replica_ranks
                )
        frozen_parallel_shapes = {
            node: frozenset(shapes)
            for node, shapes in node_to_parallel_shapes.items()
        }
        frozen_instance_groups = {
            node: frozenset(groups)
            for node, groups in node_to_instance_groups.items()
        }
        # The union over every axis, sorted — the per-rank ``new_group``
        # schedule (see the ``world_parallel_groups`` field comment).
        world_parallel_groups: list[tuple[int, ...]] = sorted({
            tuple(rank_group)
            for wg in worker_graphs.values()
            for rank_groups in wg._axis_ranks.values()
            for rank_group in rank_groups
            if len(rank_group) > 1
        })
        self.per_worker_config: dict[str, WorkerParallelGroups] = {
            wid: WorkerParallelGroups(
                global_rank=i, num_workers=self.num_workers,
                any_parallelism=any_parallelism,
                world_parallel_groups=world_parallel_groups,
                dist_timeout_s=dist_timeout_s,
                tp_allreduce=tp_allreduce,
                tp_allreduce_max_kb=tp_allreduce_max_kb,
                node_to_parallel_shapes=frozen_parallel_shapes,
                node_to_instance_groups=frozen_instance_groups,
            ) for i, wid in enumerate(worker_ids)
        }

        # (axis, global rank, (group ranks...)) -> comm group; axis
        # "instance" for the whole instance
        comm_groups: dict[tuple[str, int, tuple], CommGroup] = {}
        for wg in worker_graphs.values():
            for axis, rank_groups in wg._axis_ranks.items():
                for rank_group in rank_groups:
                    for i, rank in enumerate(rank_group):
                        key = (axis, rank, tuple(rank_group))
                        if key not in comm_groups:
                            comm_groups[key] = CommGroup(
                                my_global_rank=rank, my_group_rank=i, group_members=rank_group,
                            )
                        for node in wg.section.get_nodes():
                            self.per_worker_config[worker_ids[rank]].add(node, comm_groups[key], axis)
            for block in wg._instance_ranks:
                for i, rank in enumerate(block):
                    key = ("instance", rank, tuple(block))
                    if key not in comm_groups:
                        comm_groups[key] = CommGroup(my_global_rank=rank, my_group_rank=i, group_members=block)
                    for node in wg.section.get_nodes():
                        self.per_worker_config[worker_ids[rank]].node_to_instance_comm.setdefault(
                            node, comm_groups[key],
                        )
