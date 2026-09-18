import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


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
        # Fast all-reduce state; see ``_fast_all_reduce_buffer``. One
        # symmetric-memory buffer per dtype, created on first use.
        self._fast_ar_buffers: dict[torch.dtype, torch.Tensor] = {}
        self._fast_ar_kind: dict[torch.dtype, str] = {}
        self._fast_ar_group_name: str | None = None

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

    # Small all-reduces dominate a TP decode step: a 28-layer model issues
    # 2 per layer plus one for the vocab-parallel embedding, all of them
    # ``[bs, hidden]`` — 6 KiB at bs=1, hidden 3072. NCCL is the wrong tool
    # at that size. Measured on 2xH100 (NV18), inside a CUDA graph, 57
    # back-to-back all-reduces of 6 KiB:
    #
    #   dist.all_reduce                1.050 ms   (18.4 us each)
    #   symm_mem one-shot              0.490 ms   ( 8.6 us each)
    #
    # A single isolated ``dist.all_reduce`` is only 9.8 us; the other 8.6 us
    # is the per-call fork/join between the compute stream and
    # ProcessGroupNCCL's internal stream, which nothing can hide when the
    # collectives serialize the way a transformer's do. The one-shot kernel
    # runs on the calling stream, so it never pays it.
    #
    # ``MSTAR_FAST_ALLREDUCE``: ``1`` (default) uses the one-shot path where
    # it applies, ``0`` forces NCCL everywhere.
    # ``MSTAR_FAST_ALLREDUCE_MAX_KIB``: size cutoff, default 8 MiB. The two
    # kernels cross over against NCCL at different sizes, so the effective
    # cap is per kind (``_FAST_AR_KIND_MAX``) and this env var is an overall
    # ceiling on top. Measured on 2xH100/NV18, us per all-reduce, chained
    # and graph-replayed:
    #
    #             8 KiB    128 KiB    8 MiB
    #   NCCL      18.48      20.71    58.72
    #   one_shot   8.69       9.87    64.53   <- loses to NCCL by 8 MiB
    #   multimem   7.76       8.33    74.88   <- loses harder; NVLS is a
    #                                            small-message win only
    #
    # Re-measure with ``test/scratch/collective_shootout.py`` on new
    # topology; multicast availability and crossover are fabric-dependent.
    _fast_ar_enabled = os.environ.get("MSTAR_FAST_ALLREDUCE", "1") != "0"
    _fast_ar_max_bytes = int(
        float(os.environ.get("MSTAR_FAST_ALLREDUCE_MAX_KIB", "2048")) * 1024
    )
    # Per-kernel ceiling, applied under the env cap above.
    _FAST_AR_KIND_MAX = {"multimem": 2 << 20, "one_shot": 2 << 20}
    # Pre-registered at init; any other dtype falls back to NCCL rather
    # than rendezvous'ing mid-run. float16 is listed on purpose even
    # though torch has no one-shot kernel for it today — the probe in
    # ``register_fast_allreduce_buffers`` finds that out and skips it, so
    # this list needs no edit when one lands.
    _FAST_AR_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

    def _fast_all_reduce_buffer(self, input_: torch.Tensor):
        """A symmetric-memory staging buffer shaped like ``input_``, or None.

        One flat buffer per dtype, registered up front by
        ``register_fast_allreduce_buffers`` and sliced to fit here, so no
        collective and no allocation happens on this path.

        The buffer is shared by every all-reduce on the group, which is safe
        only because they serialize: mstar submits all engine work from one
        GPU executor thread onto the default stream, so two steps are never
        staging through it at once. Two graphs replayed concurrently on
        separate streams would need a buffer each.

        Returns None (caller falls back to NCCL) when the fast path does not
        apply: symmetric memory unavailable, tensor too large, not
        contiguous, or of a dtype that was not pre-registered.
        """
        if not self._fast_ar_enabled or self.device_group is None:
            return None
        if input_.device.type != "cuda" or not input_.is_contiguous():
            return None
        buf = self._fast_ar_buffers.get(input_.dtype)
        if buf is None:
            return None
        nbytes = input_.numel() * input_.element_size()
        kind_max = self._FAST_AR_KIND_MAX.get(
            self._fast_ar_kind.get(input_.dtype, "one_shot"), 0
        )
        if nbytes == 0 or nbytes > min(self._fast_ar_max_bytes, kind_max):
            return None
        return buf[: input_.numel()].view_as(input_)

    def register_fast_allreduce_buffers(self, device: torch.device) -> None:
        """Allocate and rendezvous the one-shot staging buffers.

        Called once from ``init_dist``, not lazily on first use, for two
        reasons. ``rendezvous`` is collective over the group, so doing it
        from inside a forward would make correctness depend on every rank
        meeting the same dtypes in the same order; and it cannot run under
        CUDA-graph capture at all, so a buffer that first appeared mid-run
        would be missing from exactly the graphs that need it.

        Best-effort and per dtype: torch has no one-shot kernel for every
        dtype (float16, today), and a dtype that cannot be registered must
        only lose itself to NCCL, not the ones that already succeeded.
        """
        if (
            not self._fast_ar_enabled
            or self.world_size == 1
            or self.device_group is None
            or device.type != "cuda"
        ):
            return
        try:
            import torch.distributed._symmetric_memory as symm_mem
        except Exception as exc:  # noqa: BLE001 - optional fast path
            logger.warning(
                "Fast all-reduce unavailable (%s: %s); using NCCL. Set "
                "MSTAR_FAST_ALLREDUCE=0 to silence.",
                type(exc).__name__, exc,
            )
            return

        group_name = self.device_group.group_name
        skipped: list[str] = []
        for dtype in self._FAST_AR_DTYPES:
            numel = self._fast_ar_max_bytes // dtype.itemsize
            try:
                buf = symm_mem.empty(numel, dtype=dtype, device=device)
                # Collective: every member must reach it for this dtype, in
                # this order. Failures below are local to the probe and
                # deterministic (a missing kernel), so the ranks agree.
                symm_mem.rendezvous(buf, group_name)
                # Prove a sliced view still resolves to the registered
                # allocation, and that a kernel exists for this dtype —
                # that is exactly how ``all_reduce`` will use it.
                probe = torch.zeros(8, dtype=dtype, device=device)
                # Prefer multimem (NVLink SHARP): measured faster than
                # one-shot at every size on 2xH100 -- 7.76 vs 8.62 us at
                # 8 KiB, 8.53 vs 9.90 at 128 KiB, 51.3 vs 64.6 at 8 MiB --
                # and unlike one-shot it still beats NCCL at 8 MiB, which
                # is what lets the cutoff go past 2 MiB. Not every fabric
                # exposes multicast, so fall back to one-shot.
                kind = "multimem"
                try:
                    buf[:8].copy_(probe)
                    torch.ops.symm_mem.multimem_one_shot_all_reduce_out(
                        buf[:8], "sum", group_name, probe,
                    )
                except Exception:  # noqa: BLE001 - no multicast support
                    kind = "one_shot"
                    torch.ops.symm_mem.one_shot_all_reduce_copy_out(
                        buf[:8], probe, "sum", group_name, probe,
                    )
            except Exception as exc:  # noqa: BLE001 - optional fast path
                skipped.append(f"{dtype} ({type(exc).__name__}: {exc})")
                continue
            self._fast_ar_kind[dtype] = kind
            self._fast_ar_buffers[dtype] = buf

        self._fast_ar_group_name = group_name
        if self._fast_ar_buffers:
            logger.info(
                "Fast all-reduce enabled for ranks %s: %s via %s, <= %d KiB",
                self.group_members,
                sorted(str(d) for d in self._fast_ar_buffers),
                sorted(set(self._fast_ar_kind.values())),
                self._fast_ar_max_bytes // 1024,
            )
        if skipped:
            logger.info(
                "Fast all-reduce falls back to NCCL for: %s", "; ".join(skipped),
            )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        buf = self._fast_all_reduce_buffer(input_)
        if buf is not None:
            # Both paths stage through the symmetric buffer and land the
            # result back in ``input_``, so the in-place contract holds.
            if self._fast_ar_kind.get(input_.dtype) == "multimem":
                # ``multimem_all_reduce_`` would leave the result in the
                # symmetric buffer and need a second copy back; the
                # ``_out`` form reduces straight into ``input_``, so this
                # stages exactly once, same as the one-shot path.
                buf.copy_(input_)
                torch.ops.symm_mem.multimem_one_shot_all_reduce_out(
                    buf, "sum", self._fast_ar_group_name, input_,
                )
            else:
                torch.ops.symm_mem.one_shot_all_reduce_copy_out(
                    buf, input_, "sum", self._fast_ar_group_name, input_,
                )
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
    _device: torch.device | None = field(default=None, init=False, repr=False)

    node_to_joint_group: dict[str, JointGroups] = field(default_factory=dict)

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

        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=self.num_workers,
            rank=self.global_rank,
            device_id=device,
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

        # Symmetric-memory staging buffers for the small-message all-reduce
        # fast path. Sorted by membership, not by the dict order above: this
        # rendezvous is collective over each subgroup, so every member has to
        # walk the groups it shares in the same sequence, and a rank that
        # belongs to several would not otherwise be guaranteed to.
        by_members = {
            tuple(g.group_members): g
            for g in (
                list(self.node_to_tp_group.values())
                + list(self.node_to_sp_group.values())
            )
            if g.world_size > 1
        }
        for members in sorted(by_members):
            by_members[members].register_fast_allreduce_buffers(device)

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
