"""Bit-exact replicas of torch's eager CUDA norm kernels that run at full bandwidth.

``compile_exact_ops: [norms]`` keeps a transformer's norms on torch's eager result so the
compiled forward matches eager bit for bit. For a per-head q/k RMSNorm (128 channels) torch's
``vectorized_layer_norm_kernel`` launches one 128-thread block per row, of which one warp does the
work; at 1024^2 that is ~110k blocks per call and 4 ms of klein-4B's ~60 ms denoise step. The
kernel here does the same arithmetic in the same order -- each lane sums the squares of its four
contiguous elements with fused multiply-adds, the 32 lane sums are combined in the warp-shuffle
tree order, then ``sum / N``, ``rsqrt(. + eps)`` and ``w * (rstd * x)`` in fp32 -- for 64 rows per
program, so its output is identical to torch's (pinned by test/modular/test_flux2_klein_compile_exact.py
on CUDA). That order is torch's private choice for NVIDIA's 32-lane warps, so the replica runs only
there, and only once it has reproduced the installed torch's result on the device.
"""

from __future__ import annotations

import functools
import logging

import torch
import torch.nn.functional as F

from mstar.platforms import current_platform

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

logger = logging.getLogger(__name__)

# torch's vectorized kernel: 4-element vectors, a 32-lane warp; one vector per lane at N = 128
_VEC, _LANES = 4, 32
EXACT_RMS_DIM = _VEC * _LANES
_ROWS = 64


if triton is not None:

    @triton.jit
    def _lane_tree(s, ROWS: tl.constexpr, WIDTH: tl.constexpr):
        # lane i += lane i + WIDTH/2, halving until one lane is left (WARP_SHFL_DOWN order)
        half: tl.constexpr = WIDTH // 2
        s = tl.reshape(s, [ROWS, 2, half])
        a, b = tl.split(tl.permute(s, [0, 2, 1]))
        return a + b

    @triton.jit
    def _exact_rms_norm_kernel(
        x_ptr, w_ptr, y_ptr, M, D1, D2, sx0, sx1, sx2, eps, ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        rmask = (rows < M)[:, None]
        i2 = rows % D2
        rest = rows // D2
        base = (rest // D1).to(tl.int64) * sx0 + (rest % D1).to(tl.int64) * sx1 + i2.to(tl.int64) * sx2
        cols = tl.arange(0, 128)
        x = tl.load(x_ptr + base[:, None] + cols[None, :], mask=rmask, other=0.0).to(tl.float32)
        # column 4 * lane + 2 * a + b -> the lane's four elements x_j, j = 2a + b
        xa0, xa1 = tl.split(tl.reshape(x, [ROWS, 32, 2, 2]))  # b = 0 / b = 1, [ROWS, 32, 2] over a
        x0, x2 = tl.split(xa0)
        x1, x3 = tl.split(xa1)
        s = x0 * x0
        s = tl.fma(x1, x1, s)
        s = tl.fma(x2, x2, s)
        s = tl.fma(x3, x3, s)
        s = _lane_tree(s, ROWS, 32)
        s = _lane_tree(s, ROWS, 16)
        s = _lane_tree(s, ROWS, 8)
        s = _lane_tree(s, ROWS, 4)
        s = _lane_tree(s, ROWS, 2)  # [ROWS, 1]
        rstd = tl.math.rsqrt(s / 128.0 + eps)
        w = tl.load(w_ptr + cols).to(tl.float32)[None, :]
        y = (w * (rstd * x)).to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + rows.to(tl.int64)[:, None] * 128 + cols[None, :], y, mask=rmask)


def _fits(x: torch.Tensor, weight: torch.Tensor | None) -> bool:
    return (
        triton is not None and x.is_cuda and current_platform.is_cuda()
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.shape[-1] == EXACT_RMS_DIM and x.stride(-1) == 1 and 1 <= x.dim() <= 4
        and weight is not None and weight.dtype == x.dtype and weight.is_contiguous()
        and _reproduces_torch(x.device, x.dtype)
    )


@functools.cache
def _reproduces_torch(device: torch.device, dtype: torch.dtype) -> bool:
    """Whether the replica gives torch's result on ``device``, checked once on random rows; if not,
    torch's kernel runs from then on. The first call is a warm-up forward, before any capture."""
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(3 * _ROWS, EXACT_RMS_DIM, generator=gen, device=device).to(dtype)
    w = torch.randn(EXACT_RMS_DIM, generator=gen, device=device).to(dtype)
    if torch.equal(_replica(x, w, 1e-6), F.rms_norm(x, (EXACT_RMS_DIM,), w, 1e-6)):
        return True
    logger.warning("exact_rms_norm: the replica differs from torch %s's RMSNorm on %s; using torch's kernel",
                   torch.__version__, device)
    return False


def _replica(x: torch.Tensor, weight: torch.Tensor, eps: float | None) -> torch.Tensor:
    if eps is None:  # torch's default: the epsilon of the fp32 math type
        eps = torch.finfo(torch.float32).eps
    lead = [1] * (4 - x.dim()) + list(x.shape[:-1])
    strides = [0] * (4 - x.dim()) + list(x.stride()[:-1])
    m = lead[0] * lead[1] * lead[2]
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if m:
        _exact_rms_norm_kernel[(triton.cdiv(m, _ROWS),)](
            x, weight, y, m, lead[1], lead[2], strides[0], strides[1], strides[2], float(eps), ROWS=_ROWS,
        )
    return y


@torch.library.custom_op("mstar::exact_rms_norm", mutates_args=())
def exact_rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float | None) -> torch.Tensor:
    """``F.rms_norm(x, (128,), weight, eps)`` with torch's eager CUDA result, bit for bit; other
    shapes and devices go to ``F.rms_norm`` itself. An op a compiled graph keeps whole: no graph
    break, and inductor never decomposes the norm. The output is contiguous."""
    if not _fits(x, weight):
        return F.rms_norm(x, (x.shape[-1],), weight, eps).contiguous()
    return _replica(x, weight, eps)


@exact_rms_norm.register_fake
def _(x, weight, eps):
    # torch's result type, contiguous like the replica's output
    return F.rms_norm(x, (x.shape[-1],), weight, eps).contiguous()
