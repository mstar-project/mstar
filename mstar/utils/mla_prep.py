"""Fused MLA pre-attention for the absorbed path: the latent RMSNorms, both RoPEs and the
latent's KV-cache write in two kernels, where the unfused path launches seven.

``q_norm`` normalizes the q_a slice of the fused q_a/kv_a projection into the contiguous q_c
that q_b reads. ``kv_write_q_rope`` runs after q_b: per token, one program normalizes the kv_a
slice, rotates k_pe and stores the ``[kv_c | k_pe]`` latent straight into its planned cache
slot, and another rotates every head's q_pe.

Both reproduce the unfused compiled path bit for bit. The norms sum their squares in the order
of FlashInfer's rmsnorm for rows of 256 to 3072 (lane i of 32 accumulates the 8-wide vectors
at ``256 j + 8 i``, then a butterfly across the lanes), and the RoPE rounds as Inductor's
contraction of it does, ``fma(x, cos, rotate(x) * sin)``. The eager unfused path rounds the
RoPE without the fma, so against it q_pe / k_pe differ by one bf16 ulp in ~1e-5 of values.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

_PDL: dict[torch.device, bool] = {}


def supports(q_lora_rank: int, kv_lora_rank: int, rope_dim: int) -> bool:
    """Widths whose norms the kernels sum in FlashInfer's order, and a power-of-two RoPE."""
    return (all(w % 256 == 0 and w <= 3072 for w in (q_lora_rank, kv_lora_rank))
            and rope_dim > 1 and rope_dim & (rope_dim - 1) == 0)


def _pdl(device: torch.device) -> bool:
    if device not in _PDL:
        _PDL[device] = torch.cuda.get_device_capability(device)[0] >= 9
    return _PDL[device]


@triton.jit
def _sum_squares(row, W: tl.constexpr):
    """fp32 sum of ``row[:W] ** 2`` in FlashInfer rmsnorm's order."""
    lane = tl.arange(0, 32)
    acc = tl.zeros([32], tl.float32)
    for j in tl.static_range(W // 256):
        for v in tl.static_range(8):
            x = tl.load(row + j * 256 + lane * 8 + v).to(tl.float32)
            acc += x * x
    for i in tl.static_range(5):  # the butterfly: lanes xor 1, 2, 4, 8, 16
        acc = tl.sum(tl.reshape(acc, [16 >> i, 2]), axis=1)
    return tl.sum(acc)


@triton.jit
def _rms_norm(row, w, eps, W: tl.constexpr, WP: tl.constexpr):
    """``row[:W] * rsqrt(mean(row ** 2) + eps) * w`` in fp32, over WP >= W lanes."""
    r = tl.rsqrt(_sum_squares(row, W) / W + eps)
    c = tl.arange(0, WP)
    x = tl.load(row + c, mask=c < W, other=0.0).to(tl.float32)
    return x * r * tl.load(w + c, mask=c < W, other=0.0).to(tl.float32)


@triton.jit
def _rope(x, d, cos, sin, mask):
    """Interleaved RoPE of ``x[d]`` in fp32: the pair (2i, 2i + 1) turns by its angle."""
    v = tl.load(x + d, mask=mask, other=0.0).to(tl.float32)
    pair = tl.load(x + (d ^ 1), mask=mask, other=0.0).to(tl.float32)
    return tl.fma(v, cos, tl.where(d % 2 == 0, -pair, pair) * sin)


@triton.jit
def _q_norm_kernel(x, stride_x, w, out, eps, Q: tl.constexpr, QP: tl.constexpr,
                   PDL: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    y = _rms_norm(x + t * stride_x, w, eps, Q, QP)
    c = tl.arange(0, QP)
    tl.store(out + t * Q + c, y.to(out.dtype.element_ty), mask=c < Q)


@triton.jit
def _kv_write_q_rope_kernel(
    x, stride_x, w, eps, cos, sin, stride_cs, cache, stride_cp, stride_ct, pages, offsets,
    n_write, q, stride_qt, stride_qh, q_pe,
    Q: tl.constexpr, L: tl.constexpr, LP: tl.constexpr, R: tl.constexpr, NOPE: tl.constexpr,
    H: tl.constexpr, HP: tl.constexpr, PDL: tl.constexpr,
):
    """Program (t, 0) stores token t's latent in its cache slot when t < n_write; (t, 1)
    rotates token t's q_pe for every head."""
    t = tl.program_id(0).to(tl.int64)
    if PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    d = tl.arange(0, R)
    c = tl.load(cos + t * stride_cs + d)
    s = tl.load(sin + t * stride_cs + d)
    if tl.program_id(1) == 0:
        if t < n_write:
            row = x + t * stride_x + Q
            kv = _rms_norm(row, w, eps, L, LP)
            k_pe = _rope(row + L, d, c, s, d < R)
            slot = cache + tl.load(pages + t) * stride_cp + tl.load(offsets + t) * stride_ct
            e = tl.arange(0, LP)
            tl.store(slot + e, kv.to(cache.dtype.element_ty), mask=e < L)
            tl.store(slot + L + d, k_pe.to(cache.dtype.element_ty))
    else:
        h = tl.arange(0, HP)[:, None]
        live = (h < H) & (d[None, :] < R)
        y = _rope(q + t * stride_qt + h * stride_qh + NOPE, d[None, :], c[None, :], s[None, :],
                  live)
        tl.store(q_pe + (t * H + h) * R + d[None, :], y.to(q_pe.dtype.element_ty), mask=live)


@torch.library.custom_op("glm52::mla_q_norm", mutates_args=())
def q_norm(fused: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``rmsnorm(fused[:, :Q])`` as a contiguous ``(T, Q)``, Q = ``weight``'s width: the q_c
    q_b reads, straight from the fused q_a/kv_a projection's rows."""
    T, Q = fused.shape[0], weight.shape[0]
    assert fused.stride(1) == 1 and fused.shape[1] >= Q
    out = fused.new_empty(T, Q)
    if T:
        pdl = _pdl(fused.device)
        # FlashInfer's rmsnorm takes the weight in the activation dtype; 4 warps for
        # decode-sized batches, 1 for prefill
        _q_norm_kernel[(T,)](
            fused, fused.stride(0), weight.to(fused.dtype), out, eps,
            Q=Q, QP=triton.next_power_of_2(Q), PDL=pdl, num_warps=4 if T <= 64 else 1,
            launch_pdl=pdl)
    return out


@q_norm.register_fake
def _(fused, weight, eps):
    return fused.new_empty(fused.shape[0], weight.shape[0])


def kv_write_q_rope(
    fused: torch.Tensor, weight: torch.Tensor, eps: float,
    cos: torch.Tensor, sin: torch.Tensor, q: torch.Tensor,
    cache: torch.Tensor, pages: torch.Tensor, offsets: torch.Tensor,
) -> torch.Tensor:
    """Store ``[rmsnorm(kv_a) | rope(k_pe)]`` for the first ``len(pages)`` tokens into the
    slots ``cache[pages[t], offsets[t]]``, and return the roped q_pe of every token and head.

    ``fused`` is ``(T, Q + L + R)`` rows of ``[q_a | kv_a | k_pe]``, L = ``weight``'s width;
    ``cos`` / ``sin`` the fp32 ``(T, 1, R)`` tables; ``q`` q_b's ``(T, H, nope + R)``;
    ``cache`` one layer's ``(pages, page_size, L + R)`` latents. Returns ``(T, H, R)``.
    """
    T, H = q.shape[0], q.shape[1]
    L, R = weight.shape[0], cos.shape[-1]
    n_write = pages.shape[0]
    assert fused.stride(1) == 1 and q.stride(2) == 1 and cache.stride(2) == 1
    assert cos.stride(-1) == 1 and cos.stride(0) == sin.stride(0) and cos.dtype == torch.float32
    assert cache.shape[-1] == L + R and cache.dtype == fused.dtype and n_write <= T
    q_pe = q.new_empty(T, H, R)
    if T:
        pdl = _pdl(fused.device)
        _kv_write_q_rope_kernel[(T, 2)](
            fused, fused.stride(0), weight.to(fused.dtype), eps, cos, sin, cos.stride(0),
            cache, cache.stride(0), cache.stride(1), pages, offsets, n_write,
            q, q.stride(0), q.stride(1), q_pe,
            Q=fused.shape[1] - L - R, L=L, LP=triton.next_power_of_2(L), R=R,
            NOPE=q.shape[2] - R, H=H, HP=triton.next_power_of_2(H), PDL=pdl, num_warps=1,
            launch_pdl=pdl)
    return q_pe
