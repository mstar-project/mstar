"""RMSNorm in Triton, a search candidate: fp32 math, one rounding per output.

One program normalizes ``ROWS`` rows of up to ``BLOCK_N`` columns.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm(
    x_ptr, w_ptr, out_ptr, rows, hidden, x_stride, out_stride, eps,
    BLOCK_N: tl.constexpr, ROWS: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    in_row = cols < hidden
    w = tl.load(w_ptr + cols, mask=in_row, other=0.0).to(tl.float32)
    for r in tl.static_range(ROWS):
        row = tl.program_id(0) * ROWS + r
        mask = in_row & (row < rows)
        x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)
        rms_rcp = tl.rsqrt(tl.sum(x * x, axis=0) / hidden + eps)
        tl.store(out_ptr + row * out_stride + cols, (x * rms_rcp * w).to(out_ptr.dtype.element_ty), mask=mask)


def make(num_warps: int, rows_per_program: int, waves_per_eu: int | None = None, record=None):
    """An ``mstar::rms_norm``-signature callable for 2-D input at these knobs.
    ``record``, if given, receives each launch's compiled kernel."""
    options = {"num_warps": num_warps}
    if waves_per_eu is not None:
        options["waves_per_eu"] = waves_per_eu

    def rms_norm(x, weight, eps, norm_dtype=None):
        rows, hidden = x.shape
        out = torch.empty_like(x)
        grid = (triton.cdiv(rows, rows_per_program),)
        compiled = _rms_norm[grid](
            x, weight, out, rows, hidden, x.stride(0), out.stride(0), eps,
            BLOCK_N=triton.next_power_of_2(hidden), ROWS=rows_per_program, **options,
        )
        if record is not None:
            record(compiled)
        return out

    return rms_norm
