"""All-reduce backends of a ``CommGroup``.

``AllReduce`` is NCCL. ``SymmMemAllReduce`` sends the small messages it can take
through torch symmetric memory (NVLink one-shot or NVLS multicast) and the rest
through NCCL. ``AllReduce.new`` picks one from the deployment config's
``tp_allreduce``.
"""

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from mstar.distributed.communication import CommGroup

logger = logging.getLogger(__name__)

# The ``tp_allreduce`` modes past "nccl", the default. Opt-in: their reduction
# order differs from NCCL's, so bf16 can differ at the last bit.
SYMM_MODES = ("symm_oneshot", "symm_multimem")
# The largest message, in KiB, the symmetric path takes without a ``tp_allreduce_max_kb``
DEFAULT_SYMM_MAX_KB = 512
# The symm_mem ops check the message byte size is aligned (at least
# max(4, element_size)) and vectorize at 16; anything else raises inside
# the op, so only 16-byte-multiple messages are routed there.
SYMM_ALIGN_BYTES = 16
# The symm_mem kernels dispatch only these.
SYMM_DTYPES = (torch.bfloat16, torch.float32)
# symm_multimem reduces messages up to this size in one kernel that writes the
# sum straight out, in 16-byte stores (so to a 16-byte-aligned tensor). Every
# rank then reads the whole message through the switch, so larger ones take
# the split kernel and a copy out.
MULTIMEM_ONESHOT_MAX_BYTES = 32 * 1024


def _all_ranks_agree(flag: bool, cpu_group) -> bool:
    """AND ``flag`` across the ranks of ``cpu_group``."""
    t = torch.tensor([int(flag)])
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=cpu_group)
    return bool(t.item())


class AllReduce:
    """NCCL for every message; the fallback of the backends below."""

    def __init__(self, device_group):
        self.device_group = device_group

    @classmethod
    def new(
        cls, group: "CommGroup", mode: str | None = None, max_kb: int | None = None,
    ) -> "AllReduce":
        """The backend the config's ``tp_allreduce`` (``mode``) names for ``group``,
        sized by ``tp_allreduce_max_kb`` (``max_kb``). Collective across the group
        (rendezvous): every member calls it at the same point. NCCL unless every
        rank could set symmetric memory up."""
        mode = (mode or "nccl").strip().lower()
        if mode not in SYMM_MODES:
            if mode != "nccl":
                logger.warning("unknown all-reduce mode %r; using NCCL", mode)
            return AllReduce(group.device_group)
        max_kb = DEFAULT_SYMM_MAX_KB if max_kb is None else int(max_kb)
        if (group.world_size == 1 or group.device_group is None or max_kb <= 0
                or not torch.cuda.is_available()):
            return AllReduce(group.device_group)
        backend = None
        try:
            backend = SymmMemAllReduce(
                group.device_group, torch.device("cuda", torch.cuda.current_device()),
                max_kb * 1024, mode,
            )
        except Exception as exc:  # noqa: BLE001 — fall back to NCCL, loudly
            logger.warning(
                "all-reduce %s requested but symmetric memory could not be set up (%r); using NCCL",
                mode, exc,
            )
        # every rank or none: one rank on NCCL while its peers wait in the symmetric
        # kernel for its signal deadlocks the first all-reduce
        if not _all_ranks_agree(backend is not None, group.cpu_group):
            if backend is not None:
                logger.warning("all-reduce %s: another rank could not set it up; using NCCL", mode)
            return AllReduce(group.device_group)
        if group.rank == 0:
            logger.info(
                "all-reduce %s for messages up to %d KiB, ranks %s",
                mode, max_kb, group.group_members,
            )
        return backend

    def buffer(
        self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device,
    ) -> torch.Tensor:
        """Where to write the partial that the next ``all_reduce`` sums."""
        return torch.empty(shape, dtype=dtype, device=device)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(input_, group=self.device_group)
        return input_


class SymmMemAllReduce(AllReduce):
    """One persistent symmetric-memory buffer per group; small messages are
    reduced through it (copy in, reduce, result back in the input — in-place
    semantics, except for an input that is the buffer itself)."""

    def __init__(self, device_group, device: torch.device, max_bytes: int, mode: str):
        import torch.distributed._symmetric_memory as symm

        super().__init__(device_group)
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

    def _fits(self, nbytes: int, dtype: torch.dtype) -> bool:
        return (
            dtype in SYMM_DTYPES
            and 0 < nbytes <= self.max_bytes
            and nbytes % SYMM_ALIGN_BYTES == 0
        )

    def buffer(
        self, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device,
    ) -> torch.Tensor:
        """The symm buffer itself where the message fits, so ``all_reduce``
        skips its copy in and returns the sum in a new tensor: use its return
        value, and all-reduce nothing else on this group in between."""
        nbytes = math.prod(shape) * dtype.itemsize
        if not torch.compiler.is_compiling() and self._fits(nbytes, dtype):
            return self._buf[:nbytes].view(dtype).view(shape)
        return super().buffer(shape, dtype, device)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # Small contiguous CUDA messages go through symmetric memory; everything
        # else (prefill-sized, strided, host, other dtypes, or a byte size the op
        # would reject as misaligned) stays on NCCL.
        if (
            self._fits(input_.numel() * input_.element_size(), input_.dtype)
            and input_.is_cuda
            and input_.is_contiguous()
        ):
            return self._reduce(input_)
        return super().all_reduce(input_)

    def _reduce(self, input_: torch.Tensor) -> torch.Tensor:
        n = input_.numel()
        flat = input_.view(-1)
        view = self._buf[: n * input_.element_size()].view(input_.dtype)
        # data_ptr() breaks a torch.compile graph at every call, so a compiled
        # forward reads neither: it never gets the buffer (``buffer``) and
        # takes the split kernel.
        eager = not torch.compiler.is_compiling()
        if eager and flat.data_ptr() == view.data_ptr():
            # The producer wrote into the buffer, and the next call reuses it,
            # so the sum goes to a new tensor.
            input_ = torch.empty_like(input_)
            flat = input_.view(-1)
        else:
            view.copy_(flat)
        if self.mode == "symm_multimem" and view.nbytes <= MULTIMEM_ONESHOT_MAX_BYTES and eager:
            # The kernel is the ranks' to agree on, so it can't follow this rank's
            # alignment: a tensor its 16-byte stores can't take gets the sum in an
            # aligned one and a copy.
            out = flat if flat.data_ptr() % 16 == 0 else torch.empty_like(flat)
            torch.ops.symm_mem.multimem_one_shot_all_reduce_out(view, "sum", self.group_name, out)
            if out is not flat:
                flat.copy_(out)
        elif self.mode == "symm_multimem":
            torch.ops.symm_mem.multimem_all_reduce_(view, "sum", self.group_name)
            flat.copy_(view)
        else:
            out = torch.ops.symm_mem.one_shot_all_reduce(view, "sum", self.group_name)
            flat.copy_(out)
        return input_
