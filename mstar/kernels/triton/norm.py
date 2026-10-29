"""Fused residual add + RMSNorm in Triton: ``mstar::fused_add_rms_norm``.

In place, like FlashInfer's kernel: ``residual += hidden_states``, then
``hidden_states = norm(residual) * weight`` (``* (1 + weight)`` for Gemma).
The sum stays in fp32 for the norm and each output is rounded once, to nearest
even, so the numerics match FlashInfer's.

A program normalizes ``rows_per_program`` whole rows. ``elems_per_thread``
sets the workgroup size from the row width, so one setting carries across
hidden sizes: the warps are the padded row over (warp size x elements per
thread), between 1 and 16.
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
    cols = tl.arange(0, BLOCK_N)
    in_row = cols < hidden
    w = tl.load(w_ptr + cols, mask=in_row, other=0.0).to(tl.float32)
    if GEMMA:
        w = w + 1.0
    for i in tl.static_range(ROWS):
        row = (tl.program_id(0) * ROWS + i).to(tl.int64)
        mask = in_row & (row < rows)
        h = tl.load(h_ptr + row * h_stride + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + row * r_stride + cols, mask=mask, other=0.0).to(tl.float32)
        total = h + r
        tl.store(r_ptr + row * r_stride + cols, total.to(r_ptr.dtype.element_ty), mask=mask)
        rms_rcp = tl.rsqrt(tl.sum(total * total, axis=0) / hidden + eps)
        tl.store(h_ptr + row * h_stride + cols, (total * rms_rcp * w).to(h_ptr.dtype.element_ty), mask=mask)


def _warp_size() -> int:
    return 64 if torch.version.hip else 32


def launch_options(hidden: int, knobs: Mapping[str, Any]) -> dict[str, int]:
    """The block width, rows per program and Triton launch options for rows of
    ``hidden`` columns at ``knobs``."""
    block = triton.next_power_of_2(hidden)
    warps = block // (_warp_size() * int(knobs["elems_per_thread"]))
    options = {"BLOCK_N": block, "ROWS": int(knobs["rows_per_program"]), "num_warps": max(1, min(16, warps))}
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
