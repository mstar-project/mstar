"""Z-Image's per-head q/k RMSNorm and rotary in one pass, for the compiled transformer.

Inductor splits ``ScaledRMSNorm`` + the rotary product into a variance reduction per q and k, an
apply pass and a layout copy (~42 ms of a ~800 ms 1024^2 request on an H200). This kernel reads
each 128-wide head row of q and k once from the fused qkv projection, and writes the rotated
rows contiguous ``[B, S, H, D]``, rounding to bf16 where the eager modules do: after the
normalization, after ``* weight``, and after the fp32 rotary product.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

_ROWS = 32

if triton is not None:

    @triton.jit
    def _qk_norm_rope_kernel(
        x_ptr, o_ptr, w_ptr, f_ptr, M, S, H, sb, ss, sh, eps, ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        rmask = (rows < M)[:, None]
        h = rows % H
        s = (rows // H) % S
        b = rows // (H * S)
        base = b.to(tl.int64) * sb + s.to(tl.int64) * ss + h.to(tl.int64) * sh
        cols = tl.arange(0, 128)
        x = tl.load(x_ptr + base[:, None] + cols[None, :], mask=rmask, other=0.0).to(tl.float32)
        rstd = tl.math.rsqrt(tl.sum(x * x, axis=1) / 128.0 + eps)
        xn = (x * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
        xn = (xn * tl.load(w_ptr + cols).to(tl.float32)[None, :]).to(tl.bfloat16).to(tl.float32)
        re, im = tl.split(tl.reshape(xn, [ROWS, 64, 2]))
        fidx = s.to(tl.int64)[:, None] * 128 + tl.arange(0, 64)[None, :] * 2
        cos = tl.load(f_ptr + fidx, mask=rmask, other=0.0)
        sin = tl.load(f_ptr + fidx + 1, mask=rmask, other=0.0)
        out = tl.reshape(tl.join(re * cos - im * sin, re * sin + im * cos), [ROWS, 128])
        tl.store(o_ptr + rows.to(tl.int64)[:, None] * 128 + cols[None, :], out.to(tl.bfloat16), mask=rmask)


def fits(q: torch.Tensor, k: torch.Tensor, weight: torch.Tensor, freqs: torch.Tensor) -> bool:
    return (
        triton is not None and q.is_cuda and q.dim() == 4 and q.shape[-1] == 128 and q.dtype == torch.bfloat16
        and q.shape == k.shape and q.stride(-1) == 1 and k.stride(-1) == 1 and weight.dtype == torch.bfloat16
        and freqs.dtype == torch.float32 and freqs.shape == (q.shape[1], 64, 2)
    )


@torch.library.custom_op("mstar::zimage_qk_norm_rope", mutates_args=())
def qk_norm_rope(
    q: torch.Tensor, k: torch.Tensor, wq: torch.Tensor, wk: torch.Tensor, freqs: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rope(norm_q(q)), rope(norm_k(k))`` for ``[B, S, H, 128]`` bf16 views and the
    ``[S, 64, 2]`` fp32 (cos, sin) tables; outputs contiguous."""
    bsz, seq, heads, _ = q.shape
    qo, ko = torch.empty(q.shape, dtype=q.dtype, device=q.device), torch.empty(q.shape, dtype=q.dtype, device=q.device)
    m = bsz * seq * heads
    freqs = freqs.contiguous()
    for x, out, w in ((q, qo, wq), (k, ko, wk)):
        _qk_norm_rope_kernel[(triton.cdiv(m, _ROWS),)](
            x, out, w, freqs, m, seq, heads, x.stride(0), x.stride(1), x.stride(2), float(eps), ROWS=_ROWS,
        )
    return qo, ko


@qk_norm_rope.register_fake
def _(q, k, wq, wk, freqs, eps):
    return torch.empty(q.shape, dtype=q.dtype, device=q.device), torch.empty(q.shape, dtype=q.dtype, device=q.device)
