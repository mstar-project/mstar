"""All-reduce backends of a ``CommGroup``, one per ``tp_allreduce`` mode.

``AllReduce`` is NCCL. ``SymmMemAllReduce`` sends the small messages it can take
through torch symmetric memory and the rest through NCCL; its subclasses are the
two kernels, NVLink one-shot and NVLS multicast. ``FlashInferAllReduce`` is
FlashInfer's one-kernel all-reduce. ``AllReduce.new`` picks one.
"""

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from mstar.distributed.communication import CommGroup

logger = logging.getLogger(__name__)

# The largest message, in KiB, the symmetric path takes without a ``tp_allreduce_max_kb``.
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


def flagged_on_any_rank(flags: Sequence[bool], groups: Iterable["CommGroup"]) -> list[bool]:
    """Per entry, OR ``flags`` across the ranks of each group in ``groups`` in
    turn: one gloo all-reduce (MAX) each, skipping a group of one rank. Over a
    TP group and then an SP group, an entry ends up set on every rank of the
    grid if any rank set it. Every rank passes the same number of entries, in
    the same order. CPU tensors: it waits on the other ranks, never on a device."""
    t = torch.tensor(list(flags), dtype=torch.int32)
    for group in groups:
        if group.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group.cpu_group)
    return [bool(f) for f in t.tolist()]


class AllReduce:
    """NCCL for every message, and the fallback of the backends below."""

    mode = "nccl"

    def __init__(self, device_group):
        self.device_group = device_group

    @classmethod
    def new(
        cls, group: "CommGroup", mode: str | None = None, max_kb: int | None = None,
    ) -> "AllReduce":
        """The backend the config's ``tp_allreduce`` (``mode``) names for ``group``,
        sized by ``tp_allreduce_max_kb`` (``max_kb``). Collective across the group
        (rendezvous): every member calls it at the same point. Refuses a mode it
        doesn't know, and stays on NCCL unless every rank could set symmetric
        memory up."""
        mode = (mode or "nccl").strip().lower()
        if mode == "nccl":
            return AllReduce(group.device_group)
        backend_cls = SYMM_BACKENDS.get(mode)
        if backend_cls is None:
            raise ValueError(f"tp_allreduce={mode!r} is not one of {['nccl', *SYMM_BACKENDS]}")
        max_kb = DEFAULT_SYMM_MAX_KB if max_kb is None else int(max_kb)
        if (group.world_size == 1 or group.device_group is None or max_kb <= 0
                or not torch.cuda.is_available()):
            return AllReduce(group.device_group)
        backend = None
        try:
            if backend_cls is FlashInferAllReduce:
                backend = FlashInferAllReduce(group, max_kb * 1024)
            else:
                backend = backend_cls(
                    group.device_group, torch.device("cuda", torch.cuda.current_device()),
                    max_kb * 1024,
                )
        except Exception as exc:  # noqa: BLE001 — fall back to NCCL, loudly
            logger.warning(
                "all-reduce %s requested but could not be set up (%r); using NCCL",
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


class SymmMemAllReduce(AllReduce, ABC):
    """One persistent symmetric-memory buffer per group; small messages are
    reduced through it (copy in, the subclass's kernel sums it across the group,
    result back in the input — in-place semantics, except for an input that is
    the buffer itself). The kernels' reduction order differs from NCCL's, so
    bf16 results can differ at the last bit."""

    def __init__(self, device_group, device: torch.device, max_bytes: int):
        import torch.distributed._symmetric_memory as symm

        super().__init__(device_group)
        self.max_bytes = max_bytes
        self.group_name = device_group.group_name
        symm.enable_symm_mem_for_group(self.group_name)
        # Raw byte buffer; viewed per dtype at call time.
        self._buf = symm.empty(max_bytes, dtype=torch.uint8, device=device)
        self._handle = symm.rendezvous(self._buf, self.group_name)

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
        # data_ptr() breaks a torch.compile graph at every call, so only eager code
        # checks pointers: a compiled forward never gets the buffer (``buffer``),
        # and multimem takes its split kernel there.
        eager = not torch.compiler.is_compiling()
        if eager and flat.data_ptr() == view.data_ptr():
            # The producer wrote into the buffer, and the next call reuses it,
            # so the sum goes to a new tensor.
            input_ = torch.empty_like(input_)
            flat = input_.view(-1)
        else:
            view.copy_(flat)
        self._sum_into(view, flat, eager)
        return input_

    @abstractmethod
    def _sum_into(self, buf: torch.Tensor, out: torch.Tensor, eager: bool) -> None:
        """Write the sum of every rank's ``buf`` (a view of the symm buffer) to ``out``."""


class OneShotAllReduce(SymmMemAllReduce):
    """Each rank reads every peer's buffer over NVLink and sums locally."""

    mode = "symm_oneshot"

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        # Inductor gives one_shot_all_reduce's input a persistent symmetric buffer
        # of its own, sized by the first call, so a forward compiled for dynamic
        # batch sizes fails at the next size; compiled code stays on NCCL.
        if torch.compiler.is_compiling():
            return AllReduce.all_reduce(self, input_)
        return super().all_reduce(input_)

    def _sum_into(self, buf: torch.Tensor, out: torch.Tensor, eager: bool) -> None:
        out.copy_(torch.ops.symm_mem.one_shot_all_reduce(buf, "sum", self.group_name))


class MultimemAllReduce(SymmMemAllReduce):
    """The NVSwitch sums the buffer through NVLS multicast."""

    mode = "symm_multimem"

    def __init__(self, device_group, device: torch.device, max_bytes: int):
        super().__init__(device_group, device, max_bytes)
        # Without NVLS the kernel raises at its first call; fail here instead,
        # where ``AllReduce.new`` falls back to NCCL.
        if not self._handle.multicast_ptr:
            raise RuntimeError("no NVLink multicast for this group")

    def _sum_into(self, buf: torch.Tensor, out: torch.Tensor, eager: bool) -> None:
        if eager and buf.nbytes <= MULTIMEM_ONESHOT_MAX_BYTES:
            # Every rank must run the same kernel, so the choice can't follow this
            # rank's alignment: an ``out`` the kernel's 16-byte stores can't take
            # gets the sum in an aligned tensor, then a copy.
            dst = out if out.data_ptr() % 16 == 0 else torch.empty_like(out)
            torch.ops.symm_mem.multimem_one_shot_all_reduce_out(buf, "sum", self.group_name, dst)
            if dst is not out:
                out.copy_(dst)
        else:
            torch.ops.symm_mem.multimem_all_reduce_(buf, "sum", self.group_name)
            out.copy_(buf)


class FlashInferAllReduce(AllReduce):
    """FlashInfer's one-kernel all-reduce for 2-D bf16 messages that fit its
    workspace, NCCL for the rest. The workspace is sized in bytes, so one serves
    any hidden size."""

    mode = "flashinfer"

    def __init__(self, group: "CommGroup", max_bytes: int):
        from mstar.distributed.ar_fusion import ARFusion
        from mstar.platforms import current_platform

        super().__init__(group.device_group)
        if not current_platform.is_cuda():
            raise RuntimeError("tp_allreduce='flashinfer' needs FlashInfer's comm module (NVIDIA GPUs)")
        # the workspace is sized in bytes: one bf16 column, max_bytes // 2 rows
        self._fusion = ARFusion.new(group, 1, torch.bfloat16, max_bytes // 2)
        if self._fusion is None:
            raise RuntimeError("no FlashInfer all-reduce workspace")

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        from mstar.distributed.ar_fusion import allreduce_oneshot

        if not torch.compiler.is_compiling() and self._fusion.accepts_allreduce(input_):
            return allreduce_oneshot(input_, self._fusion.ws_id)
        return super().all_reduce(input_)


# ``tp_allreduce`` -> its backend; "nccl" is ``AllReduce`` itself
SYMM_BACKENDS: dict[str, type[AllReduce]] = {
    backend.mode: backend for backend in (OneShotAllReduce, MultimemAllReduce, FlashInferAllReduce)
}
