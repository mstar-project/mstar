from dataclasses import dataclass, field
from typing import Any

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _symm_mode(mode: str) -> tuple[bool, bool]:
    """``MSTAR_SYMM_MEM_ALLREDUCE`` -> (use symmetric memory, prefer the multicast kernel).

    ``auto`` (the default), ``lamport`` and ``multimem`` take the NVLink-multicast path for
    what the Lamport kernel does not cover (see :func:`_lamport_mode`), falling back to the
    one-shot/two-shot kernels where NVLS is missing and to NCCL where the group spans nodes;
    ``1`` is the one-shot/two-shot path; ``0`` is NCCL only.
    """
    mode = mode.strip().lower()
    if mode in ("0", "off", "false", "nccl"):
        return False, False
    if mode == "1":
        return True, False
    return True, True  # auto / lamport / multimem


def _lamport_mode(mode: str) -> bool:
    """Whether ``MSTAR_SYMM_MEM_ALLREDUCE`` selects the Lamport one-shot tier
    (``mstar.distributed.lamport_allreduce``) for small 16-bit 2-D all-reduces and all-gathers:
    ``auto`` (the default), ``lamport`` and ``flashinfer`` do; ``multimem``, ``1`` and ``0`` do not."""
    return mode.strip().lower() in ("auto", "lamport", "flashinfer")


def _flashinfer_mode(mode: str) -> bool:
    """Whether the all-reduces of the Lamport tier run on flashinfer's TensorRT-LLM one-shot kernel
    when its comm module imports (``auto`` and ``flashinfer``; ``lamport`` keeps M*'s own kernel,
    which also serves the all-gathers and is the fallback without flashinfer)."""
    return mode.strip().lower() in ("auto", "flashinfer")


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
        if dim == 1 and input_.dim() == 2 and input_.is_cuda and self._symm_available():
            # small column shards of a decode batch: one Lamport launch instead of NCCL + reshapes
            channel = self._lamport_channel(input_, gather=True)
            if channel is not None:
                return channel.all_gather(input_)
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

    # --- small-message all-reduce over symmetric memory -------------------------------
    # A tensor-parallel decode step issues two small all-reduces per layer (a few KB to a
    # few hundred KB). Measured on an 8xH100 NVLink node at [rows, 7168] bf16: NCCL takes
    # 24-29 µs whatever the size; torch's symmetric-memory one-shot kernel (every rank reads
    # its peers over NVLink and reduces locally) 12-13 µs up to ~128 KB, and the two-shot
    # kernel (reduce-scatter + all-gather) 14-16 µs at 1 MB. Opt in with
    # On by default (MSTAR_SYMM_MEM_ALLREDUCE=auto: the NVLink-multicast kernel, in place on a ring of
    # buffers; =1 one-shot/two-shot; =0 NCCL). Messages up to
    # MSTAR_SYMM_MEM_ALLREDUCE_ONE_SHOT_MAX_BYTES (default 256 KiB) go one-shot, up to
    # MSTAR_SYMM_MEM_ALLREDUCE_MAX_BYTES (default 4 MiB) two-shot, larger ones stay on NCCL.
    _symm_enabled: bool | None = None
    _symm_multimem: bool = False
    _symm_one_shot_max_bytes: int = 256 * 1024
    _symm_max_bytes: int = 4 * 1024 * 1024
    # the Lamport one-shot kernel (mstar/distributed/lamport_allreduce.py): one launch, no barrier,
    # for 2-D bf16/fp16 messages of up to MSTAR_LAMPORT_ALLREDUCE_MAX_ROWS rows (decode batches); one
    # workspace per (dtype, width), built eagerly on first use (a collective, never under capture)
    _lamport_enabled: bool = False
    _lamport_flashinfer: bool = False
    _lamport_max_rows: int = 128

    def _symm_available(self) -> bool:
        if self._symm_enabled is None:
            import os

            mode = os.environ.get("MSTAR_SYMM_MEM_ALLREDUCE", "auto")
            enabled, self._symm_multimem = _symm_mode(mode)
            if enabled:
                try:
                    import torch.distributed._symmetric_memory  # noqa: F401

                    enabled = hasattr(torch.ops.symm_mem, "one_shot_all_reduce")
                except Exception:
                    enabled = False
                self._symm_max_bytes = int(os.environ.get("MSTAR_SYMM_MEM_ALLREDUCE_MAX_BYTES", self._symm_max_bytes))
                self._symm_one_shot_max_bytes = int(
                    os.environ.get("MSTAR_SYMM_MEM_ALLREDUCE_ONE_SHOT_MAX_BYTES", self._symm_one_shot_max_bytes))
                self._lamport_enabled = enabled and _lamport_mode(mode)
                self._lamport_flashinfer = self._lamport_enabled and _flashinfer_mode(mode)
                self._lamport_max_rows = int(os.environ.get("MSTAR_LAMPORT_ALLREDUCE_MAX_ROWS", self._lamport_max_rows))
            self._symm_enabled = enabled
            self._symm_bufs: dict = {}
            self._lamport_channels: dict = {}
        return self._symm_enabled

    # --- Lamport one-shot tier ----------------------------------------------------------
    def lamport_applies(self, shape, dtype: torch.dtype, device: torch.device) -> bool:
        """Whether a 2-D message of this shape would take the Lamport kernel (all-reduce, or
        all-gather along the last dim)."""
        if self.world_size == 1 or device.type != "cuda" or not self._symm_available() or not self._lamport_enabled:
            return False
        if len(shape) != 2 or dtype not in (torch.bfloat16, torch.float16):
            return False
        rows, width = int(shape[0]), int(shape[1])
        return 0 < rows <= self._lamport_max_rows and width > 0 and rows * width * 2 <= self._symm_max_bytes

    def _lamport_channel(self, x: torch.Tensor, gather: bool = False):
        """The workspace for ``x``'s (dtype, width), or None when the tier does not apply (or the
        channel is missing while a CUDA graph is being captured: the caller stays on its other path).
        All-reduces take flashinfer's kernel when selected and importable, all-gathers always ours."""
        if not self.lamport_applies(x.shape, x.dtype, x.device) or x.stride(1) != 1:
            return None
        kind = "gather" if gather or not self._lamport_flashinfer else "reduce"
        key = (kind, x.dtype, x.shape[1], x.device)
        channel = self._lamport_channels.get(key, False)
        if channel is False:
            if torch.cuda.is_current_stream_capturing():
                logger.warning("Lamport all-reduce channel %s first requested under CUDA-graph capture; "
                               "warm the shape up eagerly first. Falling back for this call.", key)
                return None
            from mstar.distributed.lamport_allreduce import FlashInferAllReduce, LamportAllReduce

            channel = None
            if kind == "reduce":
                try:
                    channel = FlashInferAllReduce(
                        self.device_group, self.rank, self.world_size, self._lamport_max_rows, x.shape[1], x.dtype,
                        x.device)
                    logger.info("all-reduce channel ready (flashinfer one-shot): %s x %d, up to %d rows, %d ranks",
                                x.dtype, x.shape[1], self._lamport_max_rows, self.world_size)
                except Exception as ex:  # flashinfer missing or its workspace failing: our kernel
                    logger.warning("flashinfer all-reduce unavailable (%s); using M*'s Lamport kernel", ex)
                    self._lamport_flashinfer = False
                    channel = None
            if channel is None:
                try:
                    channel = LamportAllReduce(
                        self.device_group.group_name, self.rank, self.world_size, self._lamport_max_rows,
                        x.shape[1], x.dtype, x.device)
                    logger.info("Lamport %s channel ready: %s x %d (%s), up to %d rows, %d ranks",
                                "all-gather" if gather else "all-reduce", x.dtype, x.shape[1], x.device,
                                self._lamport_max_rows, self.world_size)
                except (RuntimeError, ValueError) as ex:  # no symmetric memory across nodes, odd group sizes
                    logger.warning("Lamport all-reduce unavailable for this group (%s); using the other tiers", ex)
                    self._lamport_enabled = False
                    channel = None
            self._lamport_channels[key] = channel
        return channel or None

    # multimem (NVLink multicast, ``MSTAR_SYMM_MEM_ALLREDUCE=multimem``) reduces in place and hands
    # the symmetric buffer itself back, so each shape owns a ring of buffers: a result stays valid
    # until this many more all-reduces of the same shape have happened. Kimi K3 keeps at most one
    # result per shape live (attention output, then the FFN output), so 4 is ample.
    _SYMM_RING = 4

    def symm_applies(self, shape, dtype: torch.dtype, device: torch.device) -> bool:
        """Whether an all-reduce of this shape would take the symmetric-memory *buffer* path (the
        in-place multicast / one-shot kernels on a ring of buffers). Shapes the Lamport kernel
        takes are excluded: its input is any plain tensor, so producers need no buffer."""
        if self.world_size == 1 or device.type != "cuda" or not self._symm_available():
            return False
        if self.lamport_applies(shape, dtype, device):
            return False
        numel = 1
        for d in shape:
            numel *= int(d)
        return numel * torch.empty((), dtype=dtype).element_size() <= self._symm_max_bytes

    def symm_buffer(self, shape, dtype: torch.dtype, device: torch.device) -> torch.Tensor | None:
        """The next symmetric buffer for this shape, or None when the path does not apply. A
        producer writes its result straight into it (``torch.mm(..., out=buf)``, ``torch.add(...,
        out=buf)``) and hands it to :meth:`all_reduce_symm_buffer`: one copy launch less per
        all-reduce (279 per Kimi K3 decode step). Ring slots are reused, see ``_SYMM_RING``."""
        if not self.symm_applies(shape, dtype, device):
            return None
        try:
            return self._next_symm_buffer(tuple(int(d) for d in shape), dtype, device)
        except RuntimeError as ex:
            logger.warning("symmetric-memory all-reduce unavailable for this group (%s); using NCCL", ex)
            self._symm_enabled = False
            return None

    def _next_symm_buffer(self, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        import torch.distributed._symmetric_memory as symm_mem

        key = (shape, dtype, device)
        ring = self._symm_bufs.get(key)
        if ring is None:
            # first use of a shape: allocate the symmetric buffer(s) and rendezvous (a
            # collective; every rank sees the shapes in the same order). The engine's
            # warmup runs each capture shape eagerly first, so this never happens
            # inside a CUDA-graph capture.
            bufs = []
            for _ in range(self._SYMM_RING if self._symm_multimem else 1):
                buf = symm_mem.empty(*shape, dtype=dtype, device=device)
                symm_mem.rendezvous(buf, self.device_group.group_name)
                bufs.append(buf)
            ring = self._symm_bufs[key] = (bufs, [0])
        bufs, cursor = ring
        buf = bufs[cursor[0]]
        cursor[0] = (cursor[0] + 1) % len(bufs)
        return buf

    def _symm_all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        try:
            buf = self._next_symm_buffer(tuple(input_.shape), input_.dtype, input_.device)
        except RuntimeError as ex:  # e.g. ranks on several nodes: no symmetric memory, stay on NCCL
            logger.warning("symmetric-memory all-reduce unavailable for this group (%s); using NCCL", ex)
            self._symm_enabled = False
            dist.all_reduce(input_, group=self.device_group)
            return input_
        buf.copy_(input_)
        return self.all_reduce_symm_buffer(buf)

    def all_reduce_symm_buffer(self, buf: torch.Tensor) -> torch.Tensor:
        """Sum a buffer obtained from :meth:`symm_buffer` across the group; use the returned tensor."""
        name = self.device_group.group_name
        if self._symm_multimem:
            try:
                torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", name)
                return buf  # aliased ring slot, see _SYMM_RING
            except RuntimeError as ex:  # no NVLS support here: stay on the copying kernels
                logger.warning("symmetric-memory multimem all-reduce unavailable (%s); using one-shot/two-shot", ex)
                self._symm_multimem = False
        if buf.numel() * buf.element_size() <= self._symm_one_shot_max_bytes:
            return torch.ops.symm_mem.one_shot_all_reduce(buf, "sum", name)
        return torch.ops.symm_mem.two_shot_all_reduce_(buf, "sum", name)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """Sum ``input_`` across the group. Use the returned tensor: NCCL reduces in place,
        the symmetric-memory path returns a reduced tensor of its own."""
        if self.world_size == 1:
            return input_
        if input_.is_cuda and self._symm_available():
            channel = self._lamport_channel(input_)
            if channel is not None:
                return channel.all_reduce(input_)
            if input_.numel() * input_.element_size() <= self._symm_max_bytes:
                return self._symm_all_reduce(input_)
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
