"""The LoRA shrink and expand as custom ops, so ``torch.compile`` sees one opaque node each,
and ``lora_apply_``, the two as one op, which is what a linear target runs;
``lora_experts_apply_`` is an experts target's shrink and expands over combined ids.

``mstar::lora_expand_`` adds into ``y`` in place and says so (``mutates_args``):
functionalization then keeps every read of ``y`` on the right side of the add and never
drops it, whatever Inductor reorders around it.

The kernel registered here is the plain-torch reference (``backend.shrink_reference`` /
``expand_reference_`` / ``experts_reference_``), on any device, with no host sync and no data-dependent shape;
``triton_backend`` registers the CUDA kernels over it. Both read the same arguments with
the same meaning, so the fake (shape) implementations and ``torch.library.opcheck`` can
be checked on a CPU.

Rows of ``h`` the expand does not read (tokens on the zero adapter, columns past a
slot's rank) are unspecified; the reference writes zeros there.

``b`` must be finite everywhere, including ``b[slot, :, rank:]`` past each slot's rank: the
CUDA expand reads it up to the max rank against zeros in ``h``, so a NaN or inf there reaches
the output (the reference's one GEMM over every slot reads all of ``b``). The pool keeps those
columns zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mstar.engine.resources.lora.backend import (
    LoraBackend,
    expand_reference_,
    experts_reference_,
    gather_shrink_reference,
    shrink_reference,
)

if TYPE_CHECKING:
    from mstar.engine.resources.lora.pool import LoraAddressing


@torch.library.custom_op("mstar::lora_shrink", mutates_args=())
def lora_shrink(
    x: torch.Tensor, a: torch.Tensor, token_order: torch.Tensor,
    slot_indptr: torch.Tensor, token_slot: torch.Tensor, ranks: torch.Tensor, num_slices: int,
) -> torch.Tensor:
    """``h [T, slices * R]`` for ``x [T, in]`` against ``a [slots, slices * R, in]``."""
    return shrink_reference(x, a, token_slot, ranks, num_slices)


@lora_shrink.register_fake
def _lora_shrink_fake(x, a, token_order, slot_indptr, token_slot, ranks, num_slices):
    return x.new_empty(x.shape[0], a.shape[1])


@torch.library.custom_op("mstar::lora_expand_", mutates_args={"y"})
def lora_expand_(
    y: torch.Tensor, h: torch.Tensor, b: torch.Tensor, token_order: torch.Tensor,
    slot_indptr: torch.Tensor, token_slot: torch.Tensor, ranks: torch.Tensor, scalings: torch.Tensor,
    out_slices: list[int],
) -> None:
    """``y[:, slice] += scaling * h[:, slice ranks] @ b[slot, slice]^T`` in place. ``b`` must be
    finite past each slot's rank (see above)."""
    expand_reference_(y, h, b, token_slot, ranks, scalings, out_slices)


@lora_expand_.register_fake
def _lora_expand_fake(y, h, b, token_order, slot_indptr, token_slot, ranks, scalings, out_slices):
    return None


@torch.library.custom_op("mstar::lora_apply_", mutates_args={"y"})
def lora_apply_(
    y: torch.Tensor, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    token_order: torch.Tensor, slot_indptr: torch.Tensor, token_slot: torch.Tensor, ranks: torch.Tensor,
    scalings: torch.Tensor, out_slices: list[int],
) -> None:
    """``lora_expand_(y, lora_shrink(x, ...), ...)`` as one op, so a kernel may keep the
    shrink's output in whatever form suits its expand (split partials on CUDA). ``b`` must be
    finite past each slot's rank, as for ``lora_expand_``."""
    h = shrink_reference(x, a, token_slot, ranks, len(out_slices))
    expand_reference_(y, h, b, token_slot, ranks, scalings, out_slices)


@lora_apply_.register_fake
def _lora_apply_fake(y, x, a, b, token_order, slot_indptr, token_slot, ranks, scalings, out_slices):
    return None


@torch.library.custom_op("mstar::lora_gather_shrink", mutates_args=())
def lora_gather_shrink(
    ids: torch.Tensor, a: torch.Tensor, token_slot: torch.Tensor, ranks: torch.Tensor,
) -> torch.Tensor:
    """``h [T, slices * R]``: each token's column of its slot's ``a [slots, slices * R,
    vocab]``, an embedding target's shrink. A gather, so one implementation serves every
    device."""
    return gather_shrink_reference(ids, a, token_slot, ranks)


@lora_gather_shrink.register_fake
def _lora_gather_shrink_fake(ids, a, token_slot, ranks):
    return a.new_empty(ids.shape[0], a.shape[1])


@torch.library.custom_op("mstar::lora_experts_apply_", mutates_args={"y"})
def lora_experts_apply_(
    x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor, ids: torch.Tensor,
    row_weight: torch.Tensor, sorted_ids: torch.Tensor | None, expert_ids: torch.Tensor | None,
    num_post_padded: torch.Tensor | None, top_k: int, out_slices: list[int], block_m: int,
) -> None:
    """An experts target's shrink and per-slice expands (``LoraBackend.apply_experts_``).
    ``sorted_ids`` .. ``block_m`` are the rows' alignment (``experts_alignment``), which
    the CUDA kernel groups them by and this reference ignores."""
    experts_reference_(x, y, a, b, ids, row_weight, top_k, out_slices)


@lora_experts_apply_.register_fake
def _lora_experts_apply_fake(x, y, a, b, ids, row_weight, sorted_ids, expert_ids, num_post_padded, top_k,
                             out_slices, block_m):
    return None


def experts_config(ids: torch.Tensor, num_ids: int) -> dict:
    """The tiles of every LoRA GEMM over one layer's combined ids, which share one
    alignment and so one ``BLOCK_SIZE_M``: the base fused-MoE kernel's defaults, which pick
    by token and id counts alone."""
    from mstar.utils.fused_moe.kernels import get_default_config

    return get_default_config(M=ids.shape[0], E=num_ids, N=0, K=0, top_k=ids.shape[1])


def experts_alignment(ids: torch.Tensor, num_ids: int):
    """``ids`` grouped by id and padded to ``experts_config``'s ``BLOCK_SIZE_M`` (a
    ``MoeAlignment``), as the fused-MoE kernel reads them."""
    from mstar.utils.fused_moe import MoeAlignment
    from mstar.utils.fused_moe.align import moe_align_block_size

    block_m = experts_config(ids, num_ids)["BLOCK_SIZE_M"]
    return MoeAlignment(*moe_align_block_size(ids, block_m, num_ids), block_m)


@torch.compiler.assume_constant_result
def _grouped_on_cuda() -> bool:
    """Whether ``triton_backend`` registered ``lora_experts_apply_``'s CUDA kernel, the one
    reader of an alignment (not under ``backend: torch``). A constant to dynamo."""
    return torch._C._dispatch_has_kernel_for_dispatch_key("mstar::lora_experts_apply_", "CUDA")


class LoraOpsBackend(LoraBackend):
    """The custom ops: on CUDA the Triton kernels once ``triton_backend`` is imported,
    elsewhere the reference above, so a CPU test runs exactly the graph a GPU compiles."""

    def shrink(self, x, a, addressing: "LoraAddressing", ranks, num_slices=1):
        return lora_shrink(
            x, a, addressing.token_order, addressing.slot_indptr, addressing.token_slot, ranks, num_slices,
        )

    def gather_shrink(self, ids, a, addressing: "LoraAddressing", ranks):
        return lora_gather_shrink(ids, a, addressing.token_slot, ranks)

    def expand_fused(self, h, b, addressing: "LoraAddressing", ranks, scalings, y, out_slices):
        lora_expand_(
            y, h, b, addressing.token_order, addressing.slot_indptr, addressing.token_slot, ranks, scalings,
            list(out_slices),
        )

    def apply_linear_(self, x, y, a, b, addressing: "LoraAddressing", ranks, scalings, out_slices):
        lora_apply_(
            y, x, a, b, addressing.token_order, addressing.slot_indptr, addressing.token_slot, ranks,
            scalings, list(out_slices),
        )

    def align_experts(self, ids, num_ids):
        # the CUDA kernel groups rows by id; the reference reads them in order
        return experts_alignment(ids, num_ids) if ids.is_cuda and _grouped_on_cuda() else None

    def apply_experts_(self, x, y, a, b, ids, row_weight, top_k, out_slices, alignment=None):
        sorted_ids, expert_ids, num_post_padded, block_m = alignment or (None, None, None, 0)
        lora_experts_apply_(
            x, y, a, b, ids, row_weight, sorted_ids, expert_ids, num_post_padded, top_k, list(out_slices), block_m,
        )
