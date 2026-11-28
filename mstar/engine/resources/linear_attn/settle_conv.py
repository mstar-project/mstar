"""Settle a verify step's short-conv windows once the kept tokens are known.

The verify forward runs the conv with the window left as it was
(``causal_conv1d_update(..., update_state=False)``), so each row's live slot
still holds its pre-step window. A row that keeps ``kept`` of its ``block``
tokens ends with the last ``width`` entries of ``[window | block inputs]`` up
to its last kept token: entry ``j`` of the new window is entry ``kept + j`` of
that concatenation.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs; the CUDA path needs triton
    triton = None


if triton is not None:

    @triton.jit
    def _settle_conv_kernel(
        conv_ptr, slots_ptr, accepted_ptr, x_ptr,
        dim,
        stride_slots, stride_accepted,
        stride_conv_slot, stride_conv_dim, stride_conv_tok,
        stride_x_tok, stride_x_dim,
        null_slot_id,
        BLOCK: tl.constexpr,
        WIDTH: tl.constexpr,
        NP2_WIDTH: tl.constexpr,
        BLOCK_N: tl.constexpr,
        launch_pdl: tl.constexpr,
    ):
        if launch_pdl:
            tl.extra.cuda.gdc_wait()
        row = tl.program_id(0)
        slot = tl.load(slots_ptr + row * stride_slots).to(tl.int64)
        if slot == null_slot_id:
            if launch_pdl:
                tl.extra.cuda.gdc_launch_dependents()
            return
        # a row keeps its last token and the drafts it accepted
        kept = tl.minimum(tl.load(accepted_ptr + row * stride_accepted).to(tl.int32), BLOCK - 1) + 1
        feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        js = tl.arange(0, NP2_WIDTH)
        src = kept + js
        live = (js < WIDTH)[:, None] & (feats < dim)[None, :]
        from_window = (src < WIDTH)[:, None]
        base = conv_ptr + slot * stride_conv_slot + (feats * stride_conv_dim)[None, :]
        x_row = x_ptr + (row * BLOCK).to(tl.int64) * stride_x_tok + (feats * stride_x_dim)[None, :]
        old_at = base + (tl.minimum(src, WIDTH - 1) * stride_conv_tok)[:, None]
        new_at = x_row + (tl.maximum(src - WIDTH, 0) * stride_x_tok)[:, None]
        old = tl.load(old_at, mask=live & from_window, other=0.0)
        new = tl.load(new_at, mask=live & ~from_window, other=0.0)
        window = tl.where(from_window, old, new.to(old.dtype))
        # entry j reads entry kept + j of the same window: every read lands before any write
        tl.debug_barrier()
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(base + (js * stride_conv_tok)[:, None], window, mask=live)

def settle_conv(
    conv: torch.Tensor, slots: torch.Tensor, accepted: torch.Tensor, x: torch.Tensor, block: int,
    null_slot_id: int,
) -> None:
    """Each row's window in ``conv`` (``[slots, dim, width]``, one layer) after its
    kept tokens, in place. ``slots`` and ``accepted`` hold ``rows`` entries (a
    row on ``null_slot_id`` is skipped); ``x`` is the conv's input, ``[rows *
    block, dim]``. All three are read through their strides (``accepted`` is a
    column of the sampler's packed result). Capturable."""
    rows = slots.shape[0]
    _, dim, width = conv.shape
    if x.shape[0] != rows * block or x.shape[1] != dim:
        raise ValueError(f"settle_conv: x {tuple(x.shape)} for {rows} rows of {block} tokens and {dim} channels")
    if not conv.is_cuda:
        return settle_conv_reference(conv, slots, accepted, x, block, null_slot_id)
    from mstar.utils.causal_conv1d.kernels import _supports_pdl

    block_n = 256
    _settle_conv_kernel[(rows, triton.cdiv(dim, block_n))](
        conv, slots, accepted, x, dim,
        slots.stride(0), accepted.stride(0),
        conv.stride(0), conv.stride(1), conv.stride(2), x.stride(0), x.stride(1),
        null_slot_id, BLOCK=block, WIDTH=width, NP2_WIDTH=triton.next_power_of_2(width), BLOCK_N=block_n,
        launch_pdl=_supports_pdl(),
    )


def settle_conv_reference(
    conv: torch.Tensor, slots: torch.Tensor, accepted: torch.Tensor, x: torch.Tensor, block: int,
    null_slot_id: int,
) -> None:
    """`settle_conv` in torch: the reference, and the CPU path."""
    rows = slots.shape[0]
    _, dim, width = conv.shape
    real = slots != null_slot_id
    addr = slots.long()[real]
    if addr.numel() == 0:
        return
    # the kernel reads and writes each row's window in place: two live rows on one slot would race
    if addr.unique().numel() != addr.numel():
        raise AssertionError(f"settle_conv: live rows share a slot: {addr.tolist()}")
    kept = accepted[:rows].long().clamp(max=block - 1)[real] + 1
    starts = torch.arange(rows, device=slots.device)[real] * block
    window = conv.index_select(0, addr)
    src = kept[:, None] + torch.arange(width, device=kept.device)[None, :]
    from_window = src < width
    old = window.gather(2, src.clamp(max=width - 1)[:, None, :].expand(-1, dim, width))
    at = (starts[:, None] + src - width).clamp(min=0, max=x.shape[0] - 1)
    new = x[at.reshape(-1)].view(-1, width, dim).transpose(1, 2).to(conv.dtype)
    conv.index_copy_(0, addr, torch.where(from_window[:, None, :], old, new))
