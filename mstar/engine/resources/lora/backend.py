"""Kernels that add a batch's LoRA deltas: shrink ``h = x @ A^T``, then expand ``y += s * h @ B^T``.

Both read the step's ``LoraAddressing`` from the pool, never a model-side index, so the eager
and captured paths address slots the same way. Shapes, per layer of one target:

* ``x [T, in]``, ``y [T, out]`` (written in place), ``h [T, slices * R]``
* ``a [slots, slices * R, in]``, ``b [slots, out, R]``, ``ranks`` / ``scalings`` ``[slots]``

``out_slices`` are this rank's widths of a fused projection's outputs (``(q, k, v)``); slice
``i`` reads ``h[:, i*R:(i+1)*R]`` and writes its own columns of ``y``.

An embedding target's shrink is a gather instead (``gather_shrink``): its input is a token
id, ``in`` is the vocabulary, and ``x @ A^T`` of the id's one-hot is A's column for it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mstar.engine.resources.lora.pool import LoraAddressing


class LoraBackend(ABC):
    @abstractmethod
    def shrink(
        self, x: torch.Tensor, a: torch.Tensor, addressing: "LoraAddressing",
        ranks: torch.Tensor, num_slices: int = 1,
    ) -> torch.Tensor:
        """``h[t] = x[t] @ a[slot(t)]^T``. Rows of rank-0 slots, and columns past a slot's
        rank, may hold anything: the expand does not read them."""

    @abstractmethod
    def expand_fused(
        self, h: torch.Tensor, b: torch.Tensor, addressing: "LoraAddressing",
        ranks: torch.Tensor, scalings: torch.Tensor, y: torch.Tensor,
        out_slices: tuple[int, ...],
    ) -> None:
        """``y[t, slice i] += scalings[slot(t)] * h[t, slice i] @ b[slot(t), slice i]^T``."""

    def gather_shrink(
        self, ids: torch.Tensor, a: torch.Tensor, addressing: "LoraAddressing",
        ranks: torch.Tensor,
    ) -> torch.Tensor:
        """``h[t] = a[slot(t)][:, ids[t]]``, the shrink of a token's one-hot; an id outside
        ``[0, in)`` (e.g. an image position's ``-1``) contributes nothing."""
        return gather_shrink_reference(ids, a, addressing.token_order, addressing.slot_indptr, ranks)

    def apply_(
        self, x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
        addressing: "LoraAddressing", ranks: torch.Tensor, scalings: torch.Tensor,
        out_slices: tuple[int, ...], embedding: bool = False,
    ) -> torch.Tensor:
        """Shrink (a gather for an ``embedding`` target, whose ``x`` is token ids) then
        expand into ``y``; returns ``y``."""
        if x.shape[0] != addressing.token_order.shape[0]:
            raise ValueError(
                f"LoRA input has {x.shape[0]} rows but the step declared "
                f"{addressing.token_order.shape[0]}; the segments' spans must add up to "
                "the rows the linear sees"
            )
        if y.shape[-1] != sum(out_slices):
            raise ValueError(
                f"LoRA output has {y.shape[-1]} columns but the target's slices "
                f"{tuple(out_slices)} cover {sum(out_slices)}"
            )
        if embedding:
            h = self.gather_shrink(x, a, addressing, ranks)
            self.expand_fused(h, b, addressing, ranks, scalings, y, out_slices)
        else:
            self.apply_linear_(x, y, a, b, addressing, ranks, scalings, out_slices)
        return y

    def apply_linear_(
        self, x: torch.Tensor, y: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
        addressing: "LoraAddressing", ranks: torch.Tensor, scalings: torch.Tensor,
        out_slices: tuple[int, ...],
    ) -> None:
        """Shrink then expand into ``y``, for a linear target; a backend may fuse them."""
        h = self.shrink(x, a, addressing, ranks, len(out_slices))
        self.expand_fused(h, b, addressing, ranks, scalings, y, out_slices)


def _token_slots(token_order: torch.Tensor, slot_indptr: torch.Tensor, tokens: int):
    """``(slot of each token, whether it has one)`` from the slot-grouped layout.

    A token outside every slot's range (a capture bucket's tail) gets slot 0.
    """
    slots = slot_indptr.shape[0] - 1
    pos = torch.arange(tokens, device=token_order.device)
    # the slot owning sorted position p: how many slots end at or before it
    owner = torch.searchsorted(slot_indptr[1:].contiguous(), pos.to(slot_indptr.dtype), right=True)
    live = owner < slots
    # scatter into a spare last entry for the dead positions, so they write nothing real
    index = torch.where(live, token_order.long(), tokens)
    token_slot = torch.zeros(tokens + 1, dtype=torch.long, device=token_order.device)
    token_slot.scatter_(0, index, torch.where(live, owner, 0))
    has = torch.zeros(tokens + 1, dtype=torch.bool, device=token_order.device)
    has.scatter_(0, index, live)
    return token_slot[:tokens], has[:tokens]


def shrink_reference(
    x: torch.Tensor, a: torch.Tensor, token_order: torch.Tensor,
    slot_indptr: torch.Tensor, ranks: torch.Tensor, num_slices: int,
) -> torch.Tensor:
    """``h [T, slices * R]`` for ``x [T, in]`` against ``a [slots, slices * R, in]``, every
    token against every slot and then its own selected: no host sync and no data-dependent
    shape, so it compiles and captures, at ``slots`` times a dense LoRA's work. Zeros where
    the expand does not read (dead tokens, columns past a slot's rank)."""
    tokens = x.shape[0]
    slots, width, k = a.shape
    max_rank = width // num_slices
    token_slot, live = _token_slots(token_order, slot_indptr, tokens)
    h_all = (x @ a.reshape(slots * width, k).t()).view(tokens, slots, width)
    h = h_all.gather(1, token_slot.view(tokens, 1, 1).expand(tokens, 1, width)).squeeze(1)
    rank = ranks.long()[token_slot]
    col = torch.arange(width, device=x.device) % max_rank
    keep = live[:, None] & (col[None, :] < rank[:, None])
    # a select, not a multiply: a captured bucket's tail tokens may hold anything, inf included
    return torch.where(keep, h, torch.zeros((), dtype=h.dtype, device=h.device))


def expand_reference_(
    y: torch.Tensor, h: torch.Tensor, b: torch.Tensor, token_order: torch.Tensor,
    slot_indptr: torch.Tensor, ranks: torch.Tensor, scalings: torch.Tensor,
    out_slices: list[int],
) -> None:
    """``y[:, slice] += scaling * h[:, slice ranks] @ b[slot, slice]^T`` in place, as
    ``shrink_reference`` computes it."""
    tokens = y.shape[0]
    slots, _, max_rank = b.shape
    token_slot, live = _token_slots(token_order, slot_indptr, tokens)
    rank = ranks.long()[token_slot]
    live = live & (rank > 0)
    k = torch.arange(max_rank, device=y.device)
    # accumulate in at least fp32, as the kernels do
    acc = torch.promote_types(h.dtype, torch.float32)
    # [T, slots]: the token's scaling in its own slot's column, zero elsewhere (scatter
    # rather than one_hot, which checks its input's range on the host)
    scale = torch.where(live, scalings[token_slot].to(acc), 0.0)
    weight = torch.zeros(tokens, slots, dtype=acc, device=y.device)
    weight.scatter_(1, token_slot[:, None], scale[:, None])
    offset = 0
    for i, width in enumerate(out_slices):
        h_i = h[:, i * max_rank:(i + 1) * max_rank].to(acc)
        # a select, not a multiply: h past a slot's rank is unspecified
        h_i = torch.where(live[:, None] & (k[None, :] < rank[:, None]), h_i, 0.0)
        spread = (weight[:, :, None] * h_i[:, None, :]).reshape(tokens, slots * max_rank)
        b_i = b[:, offset:offset + width, :].permute(0, 2, 1).reshape(slots * max_rank, width)
        y[:, offset:offset + width] += (spread @ b_i.to(acc)).to(y.dtype)
        offset += width


def gather_shrink_reference(
    ids: torch.Tensor, a: torch.Tensor, token_order: torch.Tensor,
    slot_indptr: torch.Tensor, ranks: torch.Tensor,
) -> torch.Tensor:
    """Plain torch for ``gather_shrink``: fixed shapes and no host sync, so it compiles
    and captures on any device."""
    vocab = a.shape[2]
    ids = ids.long()
    slots, has = _token_slots(token_order, slot_indptr, ids.shape[0])
    live = has & (ids >= 0) & (ids < vocab) & (ranks[slots] > 0)
    h = a[slots, :, ids.clamp(0, vocab - 1)]  # [T, slices * R]
    return torch.where(live[:, None], h, torch.zeros((), dtype=h.dtype, device=h.device))

class TorchLoraBackend(LoraBackend):
    """The plain-torch reference (``shrink_reference`` / ``expand_reference_``), called
    directly: the custom ops in ``ops.py`` register the same functions as their default
    kernel, and Triton's over them on CUDA, so this is the one that never dispatches to a
    GPU kernel. Fine for a reference and for small pools."""

    def shrink(self, x, a, addressing, ranks, num_slices=1):
        return shrink_reference(x, a, addressing.token_order, addressing.slot_indptr, ranks, num_slices)

    def expand_fused(self, h, b, addressing, ranks, scalings, y, out_slices):
        expand_reference_(
            y, h, b, addressing.token_order, addressing.slot_indptr, ranks, scalings, list(out_slices),
        )


def make_backend(kind: str, device: torch.device) -> LoraBackend:
    if kind == "auto":
        kind = "triton" if device.type == "cuda" else "torch"
    if kind == "torch":
        return TorchLoraBackend()
    if kind == "triton":
        from mstar.engine.resources.lora.triton_backend import TritonLoraBackend

        return TritonLoraBackend()
    raise ValueError(f"unknown LoRA backend {kind!r}; expected auto, torch or triton")
