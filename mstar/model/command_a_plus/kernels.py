"""Fused Triton kernels for the Command A+ decoder layer.

Each kernel replaces a chain of small PyTorch ops in the eager layer and
rounds to bf16 at the same points the eager chain does, so a fused layer and
an unfused layer differ only by the order of fp32 reductions.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_layernorm_kernel(
    residual_ptr, delta_ptr, weight_ptr, residual_out_ptr, normed_ptr,
    stride_r, stride_d, stride_ro, stride_n,
    H: tl.constexpr, BLOCK: tl.constexpr, HAS_DELTA: tl.constexpr, eps,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < H
    x = tl.load(residual_ptr + row * stride_r + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_DELTA:
        d = tl.load(delta_ptr + row * stride_d + offs, mask=mask, other=0.0).to(tl.float32)
        # The eager layer stores the residual sum in bf16 before normalizing it.
        h = (x + d).to(residual_out_ptr.dtype.element_ty)
        tl.store(residual_out_ptr + row * stride_ro + offs, h, mask=mask)
        x = h.to(tl.float32)
    mean = tl.sum(x, axis=0) / H
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / H
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = w * (centered * tl.rsqrt(var + eps))
    tl.store(normed_ptr + row * stride_n + offs, y.to(normed_ptr.dtype.element_ty), mask=mask)


def add_layernorm(
    residual: torch.Tensor, delta: torch.Tensor | None,
    weight: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``h = residual + delta`` (bf16), then bias-free LayerNorm of ``h``.

    Returns ``(h, layernorm(h))``. With ``delta=None`` returns
    ``(residual, layernorm(residual))``.
    """
    tokens, hidden = residual.shape
    normed = torch.empty_like(residual)
    residual_out = torch.empty_like(residual) if delta is not None else residual
    if tokens == 0:
        return residual_out, normed
    _add_layernorm_kernel[(tokens,)](
        residual, delta if delta is not None else residual, weight, residual_out, normed,
        residual.stride(0), (delta if delta is not None else residual).stride(0),
        residual_out.stride(0), normed.stride(0),
        H=hidden, BLOCK=triton.next_power_of_2(hidden), HAS_DELTA=delta is not None,
        eps=eps, num_warps=8,
    )
    return residual_out, normed


@triton.jit
def _sigmoid_topk_kernel(
    logits_ptr, weights_ptr, ids_ptr, stride_l,
    E: tl.constexpr, K: tl.constexpr, BLOCK_E: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_E)
    logits = tl.load(logits_ptr + row * stride_l + offs, mask=offs < E, other=-float("inf"))
    logits = logits.to(tl.float32)
    kk = tl.arange(0, K)
    top_vals = tl.zeros((K,), dtype=tl.float32)
    top_ids = tl.zeros((K,), dtype=tl.int32)
    for i in tl.static_range(K):
        best = tl.max(logits, axis=0)
        # lowest index among equal maxima
        idx = tl.min(tl.where(logits == best, offs, BLOCK_E), axis=0)
        top_vals = tl.where(kk == i, best, top_vals)
        top_ids = tl.where(kk == i, idx, top_ids)
        logits = tl.where(offs == idx, -float("inf"), logits)
    # Match the eager rounding: bf16 sigmoid, bf16 sum, bf16 quotient.
    s = tl.sigmoid(top_vals).to(weights_ptr.dtype.element_ty).to(tl.float32)
    total = tl.sum(s, axis=0).to(weights_ptr.dtype.element_ty).to(tl.float32)
    w = (s / total).to(weights_ptr.dtype.element_ty)
    tl.store(weights_ptr + row * K + kk, w)
    tl.store(ids_ptr + row * K + kk, top_ids)


def sigmoid_topk(logits: torch.Tensor, k: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k over router logits, sigmoid, renormalize. Returns ``(weights, int32 ids)``."""
    tokens, experts = logits.shape
    assert logits.stride(1) == 1
    weights = torch.empty((tokens, k), dtype=dtype, device=logits.device)
    ids = torch.empty((tokens, k), dtype=torch.int32, device=logits.device)
    if tokens:
        _sigmoid_topk_kernel[(tokens,)](
            logits, weights, ids, logits.stride(0),
            E=experts, K=k, BLOCK_E=triton.next_power_of_2(experts), num_warps=1,
        )
    return weights, ids


@triton.jit
def _route_align_kernel(
    logits_ptr, weights_ptr, ids_ptr, sorted_ptr, expert_ids_ptr, post_pad_ptr,
    T, stride_l,
    E: tl.constexpr, BLOCK_E: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr,
    BLOCK_T: tl.constexpr, PAD_LEN: tl.constexpr, BLOCK_PAD: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_T)
    experts = tl.arange(0, BLOCK_E)
    logits = tl.load(
        logits_ptr + rows[:, None] * stride_l + experts[None, :],
        mask=(rows[:, None] < T) & (experts[None, :] < E), other=-float("inf"),
    ).to(tl.float32)
    kk = tl.arange(0, K)
    top_vals = tl.zeros((BLOCK_T, K), dtype=tl.float32)
    top_ids = tl.zeros((BLOCK_T, K), dtype=tl.int32)
    for i in tl.static_range(K):
        best = tl.max(logits, axis=1)
        idx = tl.min(tl.where(logits == best[:, None], experts[None, :], BLOCK_E), axis=1)
        top_vals = tl.where(kk[None, :] == i, best[:, None], top_vals)
        top_ids = tl.where(kk[None, :] == i, idx[:, None], top_ids)
        logits = tl.where(experts[None, :] == idx[:, None], -float("inf"), logits)
    dtype = weights_ptr.dtype.element_ty
    s = tl.sigmoid(top_vals).to(dtype).to(tl.float32)
    total = tl.sum(s, axis=1).to(dtype).to(tl.float32)
    row_ok = rows[:, None] < T
    tl.store(weights_ptr + rows[:, None] * K + kk[None, :], (s / total[:, None]).to(dtype), mask=row_ok)
    tl.store(ids_ptr + rows[:, None] * K + kk[None, :], top_ids, mask=row_ok)

    # Block alignment, matching moe_align_block_size: each expert's slots are
    # padded to a multiple of BLOCK_M, experts laid out in index order.
    num_slots = T * K
    slot = tl.arange(0, BLOCK_T * K)
    slot_expert = tl.reshape(top_ids, (BLOCK_T * K,))
    valid = slot < num_slots
    onehot = (slot_expert[:, None] == experts[None, :]) & valid[:, None]
    counts = tl.sum(onehot.to(tl.int32), axis=0)
    padded = (counts + BLOCK_M - 1) // BLOCK_M * BLOCK_M
    offsets = tl.cumsum(padded, axis=0) - padded
    earlier = (slot_expert[:, None] == slot_expert[None, :]) & (slot[None, :] < slot[:, None]) & valid[None, :]
    rank = tl.sum(earlier.to(tl.int32), axis=1)
    dest = tl.sum(tl.where(onehot, offsets[None, :], 0), axis=1) + rank

    pad = tl.arange(0, BLOCK_PAD)
    tl.store(sorted_ptr + pad, tl.full((BLOCK_PAD,), 0, tl.int32) + num_slots, mask=pad < PAD_LEN)
    tl.debug_barrier()
    tl.store(sorted_ptr + dest, slot, mask=valid)
    tl.store(expert_ids_ptr + dest // BLOCK_M, slot_expert, mask=valid & (rank % BLOCK_M == 0))
    tl.store(post_pad_ptr, tl.sum(padded, axis=0))


ROUTE_ALIGN_MAX_SLOTS = 128


def route_align(
    logits: torch.Tensor, k: int, block_m: int, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``sigmoid_topk`` plus ``moe_align_block_size`` in one single-program kernel.

    Returns ``(weights, ids, (sorted_token_ids, expert_ids, num_tokens_post_padded))``.
    The alignment buffers are sized for at most ``min(E, slots)`` active
    experts, the bound ``invoke_fused_moe_kernel`` sizes its grid with.
    Meant for decode-sized batches (``tokens * k <= ROUTE_ALIGN_MAX_SLOTS``).
    """
    tokens, experts = logits.shape
    num_slots = tokens * k
    assert logits.stride(1) == 1 and 0 < num_slots <= ROUTE_ALIGN_MAX_SLOTS
    pad_len = triton.cdiv(num_slots + min(experts, num_slots) * (block_m - 1), block_m) * block_m
    device = logits.device
    weights = torch.empty((tokens, k), dtype=dtype, device=device)
    ids = torch.empty((tokens, k), dtype=torch.int32, device=device)
    sorted_ids = torch.empty((pad_len,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((pad_len // block_m,), dtype=torch.int32, device=device)
    post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    _route_align_kernel[(1,)](
        logits, weights, ids, sorted_ids, expert_ids, post_pad, tokens, logits.stride(0),
        E=experts, BLOCK_E=triton.next_power_of_2(experts), K=k, BLOCK_M=block_m,
        BLOCK_T=triton.next_power_of_2(tokens), PAD_LEN=pad_len,
        BLOCK_PAD=triton.next_power_of_2(pad_len), num_warps=4,
    )
    return weights, ids, (sorted_ids, expert_ids, post_pad)


@triton.jit
def _silu_mul_kernel(
    gate_up_ptr, out_ptr, stride_in, stride_out, I, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < I
    g = tl.load(gate_up_ptr + row * stride_in + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gate_up_ptr + row * stride_in + I + offs, mask=mask, other=0.0)
    dtype = out_ptr.dtype.element_ty
    a = (g * tl.sigmoid(g)).to(dtype)
    y = (a.to(tl.float32) * u.to(tl.float32)).to(dtype)
    tl.store(out_ptr + row * stride_out + offs, (y.to(tl.float32) * scale).to(dtype), mask=mask)


def silu_mul_into(gate_up: torch.Tensor, out: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """``out = scale * (silu(gate) * up)`` with ``gate_up = [gate | up]`` on the last dim.

    Rounds like ``F.silu(gate) * up`` in bf16; ``scale`` should be a power of
    two so it adds no rounding. Both tensors may be row-strided views.
    """
    tokens, two_i = gate_up.shape
    inter = two_i // 2
    assert out.shape == (tokens, inter) and gate_up.stride(1) == 1 and out.stride(1) == 1
    block = 1024
    if tokens:
        _silu_mul_kernel[(tokens, triton.cdiv(inter, block))](
            gate_up, out, gate_up.stride(0), out.stride(0), inter, scale, BLOCK=block, num_warps=4,
        )
    return out


@triton.jit
def _splitk_linear_kernel(
    x0_ptr, x1_ptr, w_ptr, out_ptr, M, N, K0, stride_x0, stride_x1, stride_wn, stride_os, stride_om,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, K_PER_SPLIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    k0 = pid_s * K_PER_SPLIT
    # Columns [0, K0) of the logical input come from x0, the rest from x1.
    if k0 < K0:
        x_rows = x0_ptr + offs_m[:, None] * stride_x0 + k0
    else:
        x_rows = x1_ptr + offs_m[:, None] * stride_x1 + (k0 - K0)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BK):
        a = tl.load(x_rows + kk + offs_k[None, :], mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * stride_wn + (k0 + kk + offs_k)[:, None],
                    mask=offs_n[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
    tl.store(
        out_ptr + pid_s * stride_os + offs_m[:, None] * stride_om + offs_n[None, :], acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


SPLITK_MAX_TOKENS = 16
_SPLITK = dict(BN=128, BK=128, SPLIT=8, num_warps=4, num_stages=3)


def splitk_supported(xs: tuple[torch.Tensor, ...], w: torch.Tensor) -> bool:
    k = sum(x.shape[1] for x in xs)
    k_per_split = k // _SPLITK["SPLIT"]
    return (
        len(xs) in (1, 2) and xs[0].is_cuda and 0 < xs[0].shape[0] <= SPLITK_MAX_TOKENS
        and all(x.shape[0] == xs[0].shape[0] and x.stride(1) == 1 for x in xs) and w.stride(1) == 1 and w.shape[1] == k
        and k % (_SPLITK["SPLIT"] * _SPLITK["BK"]) == 0 and xs[0].shape[1] % k_per_split == 0
    )


def splitk_linear(xs: tuple[torch.Tensor, ...], w: torch.Tensor) -> torch.Tensor:
    """fp32 split-K partials of ``cat(xs, -1) @ w.T`` with shape ``[SPLIT, M, N]``; sum over dim 0.

    cuBLAS picks a non-split kernel for these skinny decode shapes and leaves
    HBM bandwidth idle; splitting K fills the GPU. ``xs`` is one or two
    inputs read in place, so the caller needn't concatenate them. Callers
    fold the partial sum into their next elementwise kernel (see ``moe_combine``).
    """
    x0, x1 = xs[0], xs[-1]
    m, n = x0.shape[0], w.shape[0]
    k0 = x0.shape[1] if len(xs) == 2 else w.shape[1]
    split = _SPLITK["SPLIT"]
    out = torch.empty((split, m, n), dtype=torch.float32, device=x0.device)
    _splitk_linear_kernel[(triton.cdiv(n, _SPLITK["BN"]), split)](
        x0, x1, w, out, m, n, k0, x0.stride(0), x1.stride(0), w.stride(0), out.stride(0), out.stride(1),
        BM=max(16, triton.next_power_of_2(m)), BN=_SPLITK["BN"], BK=_SPLITK["BK"],
        K_PER_SPLIT=w.shape[1] // split, num_warps=_SPLITK["num_warps"], num_stages=_SPLITK["num_stages"],
    )
    return out


@triton.jit
def _moe_combine_kernel(
    base_ptr, cache_ptr, out_ptr, stride_bs, stride_b, stride_c0, stride_c1, stride_o, H, scale,
    TOPK: tl.constexpr, BASE_SPLITS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        acc += tl.load(cache_ptr + row * stride_c0 + k * stride_c1 + offs, mask=mask, other=0.0).to(tl.float32)
    base = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(BASE_SPLITS):
        base += tl.load(base_ptr + s * stride_bs + row * stride_b + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_o + offs, (base + scale * acc).to(out_ptr.dtype.element_ty), mask=mask)


def moe_combine(base: torch.Tensor, cache3: torch.Tensor, scale: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """``out = base + scale * cache3.sum(1)``, accumulated in fp32 and rounded once.

    ``base`` is ``[tokens, hidden]``, or ``[splits, tokens, hidden]`` split-K
    partials that are summed first.
    """
    tokens, topk, hidden = cache3.shape
    base3 = base if base.dim() == 3 else base.unsqueeze(0)
    assert base3.shape[1:] == (tokens, hidden) and cache3.stride(2) == 1 and base3.stride(2) == 1
    if out is None:
        out = torch.empty((tokens, hidden), dtype=cache3.dtype, device=cache3.device)
    block = 1024
    if tokens:
        _moe_combine_kernel[(tokens, triton.cdiv(hidden, block))](
            base3, cache3, out, base3.stride(0), base3.stride(1),
            cache3.stride(0), cache3.stride(1), out.stride(0),
            hidden, scale, TOPK=topk, BASE_SPLITS=base3.shape[0], BLOCK=block, num_warps=4,
        )
    return out
