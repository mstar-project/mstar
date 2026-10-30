"""The LoRA shrink and expand as custom ops, so ``torch.compile`` sees one opaque node each,
and ``lora_apply_``, the two as one op, which is what a linear target runs.

``mstar::lora_expand_`` adds into ``y`` in place and says so (``mutates_args``):
functionalization then keeps every read of ``y`` on the right side of the add and never
drops it, whatever Inductor reorders around it.

The kernel registered here is the plain-torch reference (``backend.shrink_reference`` /
``expand_reference_``), on any device, with no host sync and no data-dependent shape;
``triton_backend`` registers the CUDA kernels over it. Both read the same arguments with
the same meaning, so the fake (shape) implementations and ``torch.library.opcheck`` can
be checked on a CPU.

Rows of ``h`` the expand does not read (tokens on the zero adapter, columns past a
slot's rank) are unspecified; the reference writes zeros there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mstar.engine.resources.lora.backend import (
    LoraBackend,
    expand_reference_,
    gather_shrink_reference,
    shrink_reference,
)

if TYPE_CHECKING:
    from mstar.engine.resources.lora.pool import LoraAddressing


@torch.library.custom_op("mstar::lora_shrink", mutates_args=())
def lora_shrink(
    x: torch.Tensor, a: torch.Tensor, token_order: torch.Tensor,
    slot_indptr: torch.Tensor, ranks: torch.Tensor, num_slices: int,
) -> torch.Tensor:
    """``h [T, slices * R]`` for ``x [T, in]`` against ``a [slots, slices * R, in]``."""
    return shrink_reference(x, a, token_order, slot_indptr, ranks, num_slices)


@lora_shrink.register_fake
def _lora_shrink_fake(x, a, token_order, slot_indptr, ranks, num_slices):
    return x.new_empty(x.shape[0], a.shape[1])


@torch.library.custom_op("mstar::lora_expand_", mutates_args={"y"})
def lora_expand_(
    y: torch.Tensor, h: torch.Tensor, b: torch.Tensor, token_order: torch.Tensor,
    slot_indptr: torch.Tensor, ranks: torch.Tensor, scalings: torch.Tensor,
    out_slices: list[int],
) -> None:
    """``y[:, slice] += scaling * h[:, slice ranks] @ b[slot, slice]^T`` in place."""
    expand_reference_(y, h, b, token_order, slot_indptr, ranks, scalings, out_slices)


@lora_expand_.register_fake
def _lora_expand_fake(y, h, b, token_order, slot_indptr, ranks, scalings, out_slices):
    return None


@torch.library.custom_op("mstar::lora_apply_", mutates_args={"y"})
def lora_apply_(
    y: torch.Tensor, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
    token_order: torch.Tensor, slot_indptr: torch.Tensor, ranks: torch.Tensor,
    scalings: torch.Tensor, out_slices: list[int],
) -> None:
    """``lora_expand_(y, lora_shrink(x, ...), ...)`` as one op, so a kernel may keep the
    shrink's output in whatever form suits its expand (split partials on CUDA)."""
    h = shrink_reference(x, a, token_order, slot_indptr, ranks, len(out_slices))
    expand_reference_(y, h, b, token_order, slot_indptr, ranks, scalings, out_slices)


@lora_apply_.register_fake
def _lora_apply_fake(y, x, a, b, token_order, slot_indptr, ranks, scalings, out_slices):
    return None


@torch.library.custom_op("mstar::lora_gather_shrink", mutates_args=())
def lora_gather_shrink(
    ids: torch.Tensor, a: torch.Tensor, token_order: torch.Tensor,
    slot_indptr: torch.Tensor, ranks: torch.Tensor,
) -> torch.Tensor:
    """``h [T, slices * R]``: each token's column of its slot's ``a [slots, slices * R,
    vocab]``, an embedding target's shrink. A gather, so one implementation serves every
    device."""
    return gather_shrink_reference(ids, a, token_order, slot_indptr, ranks)


@lora_gather_shrink.register_fake
def _lora_gather_shrink_fake(ids, a, token_order, slot_indptr, ranks):
    return a.new_empty(ids.shape[0], a.shape[1])


class LoraOpsBackend(LoraBackend):
    """The custom ops: on CUDA the Triton kernels once ``triton_backend`` is imported,
    elsewhere the reference above, so a CPU test runs exactly the graph a GPU compiles."""

    def shrink(self, x, a, addressing: "LoraAddressing", ranks, num_slices=1):
        return lora_shrink(x, a, addressing.token_order, addressing.slot_indptr, ranks, num_slices)

    def gather_shrink(self, ids, a, addressing: "LoraAddressing", ranks):
        return lora_gather_shrink(ids, a, addressing.token_order, addressing.slot_indptr, ranks)

    def expand_fused(self, h, b, addressing: "LoraAddressing", ranks, scalings, y, out_slices):
        lora_expand_(
            y, h, b, addressing.token_order, addressing.slot_indptr, ranks, scalings, list(out_slices),
        )

    def apply_linear_(self, x, y, a, b, addressing: "LoraAddressing", ranks, scalings, out_slices):
        lora_apply_(
            y, x, a, b, addressing.token_order, addressing.slot_indptr, ranks, scalings,
            list(out_slices),
        )
