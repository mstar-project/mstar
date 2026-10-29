"""Fused residual add + RMSNorm in Triton: ``mstar::fused_add_rms_norm``.

In place, like FlashInfer's kernel: ``residual += hidden_states``, then
``hidden_states = norm(residual) * weight`` (``* (1 + weight)`` for Gemma).
The sum stays in fp32 for the norm and each output is rounded once, to nearest
even, so the numerics match FlashInfer's.

A program normalizes ``rows_per_program`` whole rows as one tile.
``elems_per_thread`` sets the workgroup size from the tile, so one setting
carries across hidden sizes: the warps are the padded tile over (warp size x
elements per thread), between 1 and 16.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import triton
import triton.language as tl

KNOBS = ("elems_per_thread", "rows_per_program", "waves_per_eu")
DEFAULT_KNOBS: Mapping[str, int] = {"elems_per_thread": 16, "rows_per_program": 1, "waves_per_eu": 0}


@triton.jit
def _fused_add_rms_norm(
    h_ptr, r_ptr, w_ptr, rows, hidden, h_stride, r_stride, eps,
    GEMMA: tl.constexpr, BLOCK_N: tl.constexpr, ROWS: tl.constexpr,
):
    # one [ROWS, BLOCK_N] tile, so every row's loads are in flight before the
    # first store (a loop over rows would wait on each row's stores)
    cols = tl.arange(0, BLOCK_N)
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    in_row = cols < hidden
    mask = (row < rows)[:, None] & in_row[None, :]
    w = tl.load(w_ptr + cols, mask=in_row, other=0.0).to(tl.float32)
    if GEMMA:
        w = w + 1.0
    h = tl.load(h_ptr + row[:, None] * h_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row[:, None] * r_stride + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    total = h + r
    tl.store(r_ptr + row[:, None] * r_stride + cols[None, :], total.to(r_ptr.dtype.element_ty), mask=mask)
    rms_rcp = tl.rsqrt(tl.sum(total * total, axis=1) / hidden + eps)
    out = total * rms_rcp[:, None] * w[None, :]
    tl.store(h_ptr + row[:, None] * h_stride + cols[None, :], out.to(h_ptr.dtype.element_ty), mask=mask)


def _warp_size() -> int:
    return 64 if torch.version.hip else 32


def launch_options(hidden: int, knobs: Mapping[str, Any]) -> dict[str, int]:
    """The block width, rows per program and Triton launch options for rows of
    ``hidden`` columns at ``knobs``."""
    block, rows = triton.next_power_of_2(hidden), int(knobs["rows_per_program"])
    warps = block * rows // (_warp_size() * int(knobs["elems_per_thread"]))
    options = {"BLOCK_N": block, "ROWS": rows, "num_warps": max(1, min(16, warps))}
    if torch.version.hip:
        options["waves_per_eu"] = int(knobs.get("waves_per_eu", 0))
    return options


def fused_add_rms_norm(
    hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, gemma: bool,
    knobs: Mapping[str, Any] = DEFAULT_KNOBS,
) -> None:
    """``mstar::fused_add_rms_norm`` on 2-D tensors whose rows are contiguous."""
    rows, hidden = hidden_states.shape
    if rows == 0:
        return
    options = launch_options(hidden, knobs)
    grid = (triton.cdiv(rows, options["ROWS"]),)
    _fused_add_rms_norm[grid](
        hidden_states, residual, weight, rows, hidden, hidden_states.stride(0), residual.stride(0), eps,
        GEMMA=bool(gemma), **options,
    )
