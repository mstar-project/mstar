"""GroupNorm (+ SiLU) for a compiled VAE decode, with the reduction split across the image.

Inductor reduces each (batch, group) pair in one program, so a 1024^2 decode's GroupNorms run
32 programs over 4M elements each and take ~10 ms of a ~27 ms FLUX VAE decode on an H200. Here
every program reduces a tile of pixels for all groups (per-tile mean and M2), the tiles are
combined with Chan's formula, and a second pass normalizes, applies the affine and the optional
SiLU in one read and one write. Statistics are fp32 like the eager kernel's, but the result is
not eager's beyond the reduction order: normalize, affine and SiLU run in fp32 and round to the
output dtype once, where eager rounds the GroupNorm's output before its SiLU (and applies the
affine as one fused scale and shift per channel).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from mstar.model.components.diffusion.compile_utils import kept_exact

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None

if triton is not None:

    @triton.jit
    def _gn_stats_kernel(
        x_ptr, mean_ptr, m2_ptr, sb, sp, sc,
        C: tl.constexpr, G: tl.constexpr, BP: tl.constexpr,
    ):
        # pixel tiles on grid axis 0, which holds 2^31 - 1 programs; axis 1 holds 65,535
        blk = tl.program_id(0)
        b = tl.program_id(1)
        pix = blk * BP + tl.arange(0, BP)
        ch = tl.arange(0, C)
        x = tl.load(
            x_ptr + b.to(tl.int64) * sb + pix.to(tl.int64)[:, None] * sp + ch.to(tl.int64)[None, :] * sc
        ).to(tl.float32)
        x = tl.reshape(x, [BP, G, C // G])
        mean = tl.sum(tl.sum(x, axis=2), axis=0) / (BP * (C // G))  # [G]
        d = x - mean[None, :, None]
        m2 = tl.sum(tl.sum(d * d, axis=2), axis=0)
        out = (b * tl.num_programs(0) + blk) * G + tl.arange(0, G)
        tl.store(mean_ptr + out, mean)
        tl.store(m2_ptr + out, m2)

    @triton.jit
    def _gn_apply_kernel(
        x_ptr, y_ptr, w_ptr, bias_ptr, mean_ptr, rstd_ptr, sb, sp, sc,
        C: tl.constexpr, G: tl.constexpr, BP: tl.constexpr, SILU: tl.constexpr,
    ):
        b = tl.program_id(1)
        pix = tl.program_id(0) * BP + tl.arange(0, BP)
        ch = tl.arange(0, C)
        offs = b.to(tl.int64) * sb + pix.to(tl.int64)[:, None] * sp + ch.to(tl.int64)[None, :] * sc
        x = tl.load(x_ptr + offs).to(tl.float32)
        grp = ch // (C // G)
        mean = tl.load(mean_ptr + b * G + grp)
        rstd = tl.load(rstd_ptr + b * G + grp)
        y = (x - mean[None, :]) * rstd[None, :] * tl.load(w_ptr + ch).to(tl.float32)[None, :] \
            + tl.load(bias_ptr + ch).to(tl.float32)[None, :]
        if SILU:
            y = y / (1.0 + tl.exp(-y))
        tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty))


def _tile(c: int) -> int:
    return max(1, min(64, 8192 // c))


def _fits(x: torch.Tensor, weight, bias, num_groups: int) -> bool:
    if triton is None or not x.is_cuda or x.dim() != 4 or weight is None or bias is None:
        return False
    _, c, h, w = x.shape
    # Below 256^2 pixels inductor's own GroupNorm is as fast. The kernels write y with x's
    # strides, which y = empty_like(x) shares only for a dense x (NCHW or NHWC), and merge H and
    # W into one pixel index; every tile is full.
    dense = x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last)
    return (dense and (c & (c - 1)) == 0 and 32 <= c <= 1024 and c % num_groups == 0 and h * w >= 1 << 16
            and (h * w) % _tile(c) == 0 and x.stride(2) == w * x.stride(3))


@torch.library.custom_op("mstar::group_norm_silu", mutates_args=())
def group_norm_silu(
    x: torch.Tensor, weight: torch.Tensor | None, bias: torch.Tensor | None, num_groups: int, eps: float,
    silu: bool,
) -> torch.Tensor:
    """``silu(group_norm(x))`` (or the plain GroupNorm) of a 4-D tensor, layout preserved."""
    y = torch.empty_like(x)  # the layout the fake kernel promises
    if not _fits(x, weight, bias, num_groups):
        ref = F.group_norm(x, num_groups, weight, bias, eps)
        return y.copy_(F.silu(ref) if silu else ref)
    bsz, c, h, w = x.shape
    p = h * w
    bp = _tile(c)
    nblk = p // bp
    sb, sc, sp = x.stride(0), x.stride(1), x.stride(3)
    mean = torch.empty(bsz, nblk, num_groups, device=x.device, dtype=torch.float32)
    m2 = torch.empty_like(mean)
    _gn_stats_kernel[(nblk, bsz)](x, mean, m2, sb, sp, sc, C=c, G=num_groups, BP=bp, num_warps=8)
    # Chan's combine of the per-tile (mean, M2), all tiles holding bp * C / G elements
    gmean = mean.mean(1)
    var = (m2.sum(1) + (bp * (c // num_groups)) * (mean - gmean[:, None]).pow(2).sum(1)) / (p * (c // num_groups))
    rstd = torch.rsqrt(var + eps)
    _gn_apply_kernel[(nblk, bsz)](
        x, y, weight, bias, gmean.contiguous(), rstd.contiguous(), sb, sp, sc,
        C=c, G=num_groups, BP=bp, SILU=silu, num_warps=8,
    )
    return y


@group_norm_silu.register_fake
def _(x, weight, bias, num_groups, eps, silu):
    return torch.empty_like(x)


def norm_act(norm: torch.nn.GroupNorm, x: torch.Tensor, act: torch.nn.Module | None) -> torch.Tensor:
    """``act(norm(x))``: the split-reduction kernel inside a compiled CUDA forward, else the
    modules themselves (eager stays the reference's kernels, and so does a compiled module
    ``exclude_from_compile`` keeps exact)."""
    # the kernel fuses SiLU or nothing, in place of both modules
    fusable = (act is None or type(act) is torch.nn.SiLU) and not (kept_exact(norm) or kept_exact(act))
    if fusable and torch.compiler.is_compiling() and _fits(x, norm.weight, norm.bias, norm.num_groups):
        return group_norm_silu(x, norm.weight, norm.bias, norm.num_groups, norm.eps, act is not None)
    x = norm(x)
    return act(x) if act is not None else x
