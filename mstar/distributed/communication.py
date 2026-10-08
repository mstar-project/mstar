import logging
import math
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

DIST_TIMEOUT_ENV = "MSTAR_DIST_TIMEOUT_S"

# Small-message all-reduce backend: "nccl" (default) or "symm_oneshot" /
# "symm_multimem", which route messages up to MSTAR_TP_SYMM_AR_MAX_KB through
# torch symmetric memory (NVLink one-shot / NVLS multicast). Reduction order
# differs from NCCL's, so bf16 can differ at the last bit — opt-in, from the
# deployment config's ``tp_allreduce`` or this env var, which overrides it.
TP_ALLREDUCE_ENV = "MSTAR_TP_ALLREDUCE"
TP_SYMM_AR_MAX_KB_ENV = "MSTAR_TP_SYMM_AR_MAX_KB"
TP_SYMM_AR_MODES = ("symm_oneshot", "symm_multimem")
# The symm_mem ops check the message byte size is aligned (at least
# max(4, element_size)) and vectorize at 16; anything else raises inside
# the op, so only 16-byte-multiple messages are routed there.
TP_SYMM_AR_ALIGN_BYTES = 16
# The symm_mem kernels dispatch only these.
TP_SYMM_AR_DTYPES = (torch.bfloat16, torch.float32)
# symm_multimem reduces messages up to this size in one kernel that writes the
# caller's tensor (in 16-byte stores, so that tensor must start 16-byte
# aligned). Every rank then reads the whole message through the switch, so
# larger ones take the split kernel and a copy out.
MULTIMEM_ONESHOT_MAX_BYTES = 32 * 1024


class _SymmAllReduce:
    """One persistent symmetric-memory buffer per group; all-reduce small
    messages through it (copy in, reduce, result back in the input — in-place
    semantics, except for an input that is the buffer itself)."""

    def __init__(self, device_group, device: torch.device, max_bytes: int, mode: str):
        import torch.distributed._symmetric_memory as symm

        self.mode = mode
        self.max_bytes = max_bytes
        self.group_name = device_group.group_name
        symm.enable_symm_mem_for_group(self.group_name)
        # Raw byte buffer; viewed per dtype at call time.
        self._buf = symm.empty(max_bytes, dtype=torch.uint8, device=device)
        handle = symm.rendezvous(self._buf, self.group_name)
        # Without NVLS the multimem kernel raises at its first call; fail
        # here instead, where the caller falls back to NCCL.
        if mode == "symm_multimem" and not handle.multicast_ptr:
            raise RuntimeError("no NVLink multicast for this group")

    def buffer(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return self._buf[: math.prod(shape) * dtype.itemsize].view(dtype).view(shape)

    def all_reduce_(self, input_: torch.Tensor) -> torch.Tensor:
        n = input_.numel()
        flat = input_.view(-1)
        view = self._buf[: n * input_.element_size()].view(input_.dtype)
        # data_ptr() breaks a torch.compile graph at every call, so a compiled
        # forward reads neither: it never gets the buffer (all_reduce_buffer)
        # and takes the split kernel.
        eager = not torch.compiler.is_compiling()
        if eager and flat.data_ptr() == view.data_ptr():
            # The producer wrote into the buffer, and the next call reuses it,
            # so the sum goes to a new tensor.
            input_ = torch.empty_like(input_)
            flat = input_.view(-1)
        else:
            view.copy_(flat)
        if (
            self.mode == "symm_multimem"
            and view.nbytes <= MULTIMEM_ONESHOT_MAX_BYTES
            and eager
            and flat.data_ptr() % 16 == 0
        ):
            torch.ops.symm_mem.multimem_one_shot_all_reduce_out(view, "sum", self.group_name, flat)
        elif self.mode == "symm_multimem":
            torch.ops.symm_mem.multimem_all_reduce_(view, "sum", self.group_name)
            flat.copy_(view)
        else:
            out = torch.ops.symm_mem.one_shot_all_reduce(view, "sum", self.group_name)
            flat.copy_(out)
        return input_


def resolve_dist_timeout(dist_timeout_s: float | None = None) -> dict[str, timedelta]:
    """Build the ``timeout`` kwarg for ``init_process_group`` / ``new_group``.

    ``MSTAR_DIST_TIMEOUT_S`` overrides the config's ``dist_timeout_s``; with
    neither set, PyTorch's default applies (``{}``). Large checkpoint loads
    (hundreds of GB at TP8) exceed that default, so deployments opt in.
    """
    raw = os.environ.get(DIST_TIMEOUT_ENV, "").strip()
    if raw:
        try:
            dist_timeout_s = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{DIST_TIMEOUT_ENV} must be a number of seconds, got {raw!r}"
            ) from exc
    if dist_timeout_s is None:
        return {}
    if dist_timeout_s <= 0:
        raise ValueError(f"Distributed timeout must be positive, got {dist_timeout_s}")
    return {"timeout": timedelta(seconds=float(dist_timeout_s))}


def _all_ranks_agree(flag: bool, cpu_group) -> bool:
    """AND ``flag`` across the ranks of ``cpu_group``."""
    t = torch.tensor([int(flag)])
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=cpu_group)
    return bool(t.item())


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
        # gloo twin of ``device_group``, for object collectives (workspace
        # rendezvous) that should not go through NCCL
        self.cpu_group = None
        self.initialized = False
        # see ``init_allreduce_fusion``
        self._ar_fusion_ws: int | None = None
        self._ar_fusion_max_tokens = 0
        self._symm_ar: _SymmAllReduce | None = None

    def _maybe_init_symm_allreduce(
        self, config_mode: str | None = None, config_max_kb: int | None = None,
    ) -> None:
        """Build the symmetric-memory all-reduce path if the deployment
        config's ``tp_allreduce`` (``config_mode``) or MSTAR_TP_ALLREDUCE,
        which overrides it, asks for it; ``tp_allreduce_max_kb`` (or
        MSTAR_TP_SYMM_AR_MAX_KB) sizes it. Collective across the group
        (rendezvous) — every member calls this from ``init_dist`` at the same
        point. Falls back to NCCL."""
        mode = (os.environ.get(TP_ALLREDUCE_ENV) or config_mode or "nccl").strip().lower()
        if mode not in TP_SYMM_AR_MODES:
            if mode != "nccl":
                logger.warning("unknown all-reduce mode %r; using NCCL", mode)
            return
        if self.world_size == 1 or self.device_group is None or not torch.cuda.is_available():
            return
        env_kb = os.environ.get(TP_SYMM_AR_MAX_KB_ENV)
        max_kb = int(env_kb if env_kb else 512 if config_max_kb is None else config_max_kb)
        if max_kb <= 0:
            return
        try:
            self._symm_ar = _SymmAllReduce(
                self.device_group, torch.device("cuda", torch.cuda.current_device()),
                max_kb * 1024, mode,
            )
        except Exception as exc:  # noqa: BLE001 — fall back to NCCL, loudly
            logger.warning(
                "all-reduce %s requested but symmetric memory could not be set up (%r); using NCCL",
                mode, exc,
            )
            self._symm_ar = None
        # every rank or none: one rank on NCCL while its peers wait in the symmetric
        # kernel for its signal deadlocks the first all-reduce
        if not _all_ranks_agree(self._symm_ar is not None, self.cpu_group):
            if self._symm_ar is not None:
                logger.warning("all-reduce %s: another rank could not set it up; using NCCL", mode)
            self._symm_ar = None
            return
        if self.rank == 0:
            logger.info(
                "all-reduce %s for messages up to %d KiB, ranks %s",
                mode, max_kb, self.group_members,
            )

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

    def _symm_fits(self, nbytes: int, dtype: torch.dtype) -> bool:
        symm = self._symm_ar
        return (
            symm is not None
            and dtype in TP_SYMM_AR_DTYPES
            and 0 < nbytes <= symm.max_bytes
            and nbytes % TP_SYMM_AR_ALIGN_BYTES == 0
        )

    def all_reduce_buffer(
        self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """Where to write the partial that the next ``all_reduce`` sums. On
        the symmetric-memory path this is the symm buffer, so ``all_reduce``
        skips its copy in and returns the sum in a new tensor: use its return
        value, and all-reduce nothing else on this group in between."""
        if not torch.compiler.is_compiling() and self._symm_fits(
            math.prod(shape) * dtype.itemsize, dtype
        ):
            return self._symm_ar.buffer(shape, dtype)
        return torch.empty(shape, dtype=dtype, device=device)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        # Small contiguous CUDA messages go through symmetric memory when
        # enabled; everything else (prefill-sized, strided, host, other
        # dtypes, or a byte size the op would reject as misaligned) stays on
        # NCCL.
        nbytes = input_.numel() * input_.element_size()
        if (
            self._symm_fits(nbytes, input_.dtype)
            and input_.is_cuda
            and input_.is_contiguous()
        ):
            return self._symm_ar.all_reduce_(input_)
        dist.all_reduce(input_, group=self.device_group)
        return input_

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
            self.rank, self.world_size, self.cpu_group, max_tokens, hidden, dtype,
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
class JointGroups:
    tp_group: CommGroup
    sp_group: CommGroup

    @property
    def rank(self):
        """This worker's rank within ``node``'s lockstep instance — the
        TP x SP block, row-major [sp][tp] to match
        ``WorkerGraph._instance_ranks``. 0 iff this worker leads the
        instance (it is rank 0 in both its TP and SP comm groups)."""
        return self.sp_group.rank * self.tp_group.world_size + self.tp_group.rank

    @property
    def world_size(self):
        """Total ranks in ``node``'s lockstep instance (tp_size * sp_size)."""
        return  self.tp_group.world_size * self.sp_group.world_size

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast from joint rank ``src`` to the whole TP x SP block.

        ``src`` is indexed like ``rank`` (row-major [sp][tp]); each subgroup
        takes its own coordinate out of it, since ``CommGroup.broadcast``
        indexes its own members. TP first: that fills src's SP row across
        every TP column, which the SP pass then broadcasts down.
        """
        if not 0 <= src < self.world_size:
            raise ValueError(
                f"broadcast src {src} outside the joint group "
                f"(world size {self.world_size})"
            )
        sp_src, tp_src = divmod(src, self.tp_group.world_size)
        tensor = self.tp_group.broadcast(tensor, tp_src)
        return self.sp_group.broadcast(tensor, sp_src)



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
    # Global topology metadata. Unlike the comm-group maps above, these maps
    # describe every replica, including ones this worker does not host.
    node_to_parallel_shapes: dict[
        str, frozenset[tuple[int, int]]
    ] = field(default_factory=dict)
    node_to_instance_groups: dict[
        str, frozenset[tuple[int, ...]]
    ] = field(default_factory=dict)
    _device: torch.device | None = field(default=None, init=False, repr=False)

    node_to_joint_group: dict[str, JointGroups] = field(default_factory=dict)
    # Process-group timeout in seconds from the deployment config's
    # ``dist_timeout_s``; ``MSTAR_DIST_TIMEOUT_S`` overrides. None = torch default.
    dist_timeout_s: float | None = None
    # Small-message all-reduce path from the deployment config's
    # ``tp_allreduce``; MSTAR_TP_ALLREDUCE overrides. None = NCCL.
    tp_allreduce: str | None = None
    # Largest message it takes, from ``tp_allreduce_max_kb``; None = 512.
    tp_allreduce_max_kb: int | None = None

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
        device: torch.device | None = None,
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
        device = device or torch.device("cuda", self.global_rank)
        self._device = device
        if device.type != "cpu":
            torch.accelerator.set_device_index(device)
        backend = dist.get_default_backend_for_device(device.type)
        if not self.any_parallelism:
            return

        timeout_kwargs = resolve_dist_timeout(self.dist_timeout_s)
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=self.num_workers,
            rank=self.global_rank,
            device_id=device,
            **timeout_kwargs,
        )

        # One subgroup per distinct rank tuple across BOTH mesh axes —
        # ``world_parallel_groups`` is the sorted union, so every rank
        # iterates it identically; ``new_group`` is collective and
        # tag-ordered (see the field comment). A tuple shared by a TP and
        # an SP group (degenerate meshes) maps to one subgroup.
        rank_tuple_to_pg: dict[tuple[int, ...], "dist.ProcessGroup"] = {}
        rank_tuple_to_cpu_pg: dict[tuple[int, ...], "dist.ProcessGroup"] = {}
        for rank_tuple in self.world_parallel_groups:
            rank_tuple_to_pg[rank_tuple] = dist.new_group(
                ranks=list(rank_tuple), **timeout_kwargs
            )
            # Members only: with the default group bound to a device, a
            # non-member's subgroup creation joins an ncclCommSplit that
            # members never call for a gloo group, and blocks forever.
            rank_tuple_to_cpu_pg[rank_tuple] = dist.new_group(
                ranks=list(rank_tuple), backend="gloo",
                use_local_synchronization=True, **timeout_kwargs,
            )

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
            comm_group.cpu_group = rank_tuple_to_cpu_pg[tuple(comm_group.group_members)]
            comm_group.initialized = True
            # Collective rendezvous within the group. Members visit shared
            # groups in the same relative order: GlobalParallelConfig fills
            # every worker's dicts in one global iteration order.
            comm_group._maybe_init_symm_allreduce(self.tp_allreduce, self.tp_allreduce_max_kb)

    def get_tp_config_for_node(self, node: str) -> CommGroup:
        if node not in self.node_to_tp_group:
            self.node_to_tp_group[node] = CommGroup.trivial()
        return self.node_to_tp_group[node]

    def get_sp_config_for_node(self, node: str) -> CommGroup:
        if node not in self.node_to_sp_group:
            self.node_to_sp_group[node] = CommGroup.trivial()
        return self.node_to_sp_group[node]

    def get_joint_group_for_node(self, node: str) -> JointGroups:
        if node not in self.node_to_joint_group:
            self.node_to_joint_group[node] = JointGroups(
                tp_group=self.get_tp_config_for_node(node),
                sp_group=self.get_sp_config_for_node(node)
            )
        return self.node_to_joint_group[node]

    def get_instance_rank_for_node(self, node: str) -> int:
        return self.get_joint_group_for_node(node).rank

    def get_instance_world_size_for_node(self, node: str) -> int:
        return self.get_joint_group_for_node(node).world_size

    def all_in_same_group(self, nodes: list[str]) -> bool:
        """Whether every node shares one (tp, sp) group.

        Per dimension, and only where someone parallelizes: unregistered reads
        as the single-rank group, so a TP-only run would otherwise be rejected
        for "differing" on SP. By membership, not identity — the lazy getters
        mint a fresh group per node (and would cache one for a remote node).
        """
        for per_node in (self.node_to_tp_group, self.node_to_sp_group):
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
        """Whether every replica uses the same tensor/sequence parallel size."""
        shapes: set[tuple[int, int]] = set()
        for node in nodes:
            global_shapes = self.node_to_parallel_shapes.get(node)
            if global_shapes:
                shapes.update(global_shapes)
                continue
            tp = self.node_to_tp_group.get(node)
            sp = self.node_to_sp_group.get(node)
            shapes.add((
                tp.world_size if tp is not None else 1,
                sp.world_size if sp is not None else 1,
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
        any_parallelism = any(
            wg.tp_size > 1 or wg.sp_size > 1 for wg in worker_graphs.values()
        )
        node_to_parallel_shapes: dict[str, set[tuple[int, int]]] = {}
        node_to_instance_groups: dict[str, set[tuple[int, ...]]] = {}
        for wg in worker_graphs.values():
            shape = (wg._tp_comm_size, wg.sp_size)
            instance_groups = wg._instance_ranks or [
                [rank] for rank in wg.ranks
            ]
            for node in wg.section.get_nodes():
                node_to_parallel_shapes.setdefault(node, set()).add(shape)
                node_to_instance_groups.setdefault(node, set()).update(
                    tuple(group) for group in instance_groups
                )
        frozen_parallel_shapes = {
            node: frozenset(shapes)
            for node, shapes in node_to_parallel_shapes.items()
        }
        frozen_instance_groups = {
            node: frozenset(groups)
            for node, groups in node_to_instance_groups.items()
        }
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
                dist_timeout_s=dist_timeout_s,
                tp_allreduce=tp_allreduce,
                tp_allreduce_max_kb=tp_allreduce_max_kb,
                node_to_parallel_shapes=frozen_parallel_shapes,
                node_to_instance_groups=frozen_instance_groups,
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
