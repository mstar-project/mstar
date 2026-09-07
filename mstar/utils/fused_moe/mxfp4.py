"""MXFP4 weight-only (W4A16) fused MoE in Triton, plus the SiTU-GLU activation.

Weights are stored exactly as the compressed-tensors ``mxfp4-pack-quantized`` checkpoint
lays them out (spec H), stacked over experts:

* ``w_packed [E, N, K // 2]`` uint8: two E2M1 codes per byte along ``K`` (low nibble =
  even k, high nibble = odd k),
* ``w_scale [E, N, K // 32]`` uint8: one E8M0 exponent per group of 32 input elements.

The grouped GEMM dequantizes each ``[BLOCK_N, 32]`` weight tile in registers
(``fp4(code) * 2 ** (scale - 127)``, exact in bf16) and feeds the tensor cores in bf16, so
its numerics equal the bf16 kernel on the dequantized weights. Token permutation and
padding reuse ``moe_align_block_size``; the activation kernel applies SiTU-GLU
(``beta * tanh(g / beta) * sigmoid(g) * lb * tanh(u / lb)``) between the two GEMMs.

This is the correctness-first Hopper path (weight-bandwidth-bound at small batch); the
W4A8 (FP8 tensor core) variant replaces the inner dot without changing the interface.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.utils.fused_moe.align import moe_align_block_size
from mstar.utils.fused_moe.kernels import get_default_config, moe_sum_reduce_triton

MXFP4_GROUP = 32


@triton.jit
def _e2m1_to_float(code):
    """4-bit E2M1 code (uint8 0..15) -> fp32 value in {0, .5, 1, 1.5, 2, 3, 4, 6} with sign."""
    sign = (code >> 3) & 1
    e = (code >> 1) & 3
    m = (code & 1).to(tl.float32)
    mag = tl.where(e == 0, 0.5 * m, (1.0 + 0.5 * m) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def fused_moe_mxfp4_kernel(
    a_ptr, b_packed_ptr, b_scale_ptr, c_ptr, topk_weights_ptr,
    sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,  # packed: bk indexes bytes (K/2)
    stride_se, stride_sn, stride_sk,  # scales: sk indexes groups (K/32)
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """One ``[BLOCK_SIZE_M, BLOCK_SIZE_N]`` output tile; ``K`` must be a multiple of 32
    and ``BLOCK_SIZE_K`` a multiple of 32."""
    tl.static_assert(BLOCK_SIZE_K % 32 == 0, "BLOCK_SIZE_K must cover whole MXFP4 groups")
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k32 = tl.arange(0, 32)
    offs_b16 = tl.arange(0, 16)
    a_ptrs = a_ptr + offs_token[:, None] // top_k * stride_am + offs_k32[None, :] * stride_ak
    # packed rows: [BLOCK_N, 16] bytes per 32-wide group
    b_ptrs = b_packed_ptr + off_experts * stride_be + offs_bn[:, None] * stride_bn + offs_b16[None, :] * stride_bk
    s_ptrs = b_scale_ptr + off_experts * stride_se + offs_bn * stride_sn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _g in range(0, K // 32):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)  # [BLOCK_M, 32]
        packed = tl.load(b_ptrs)  # [BLOCK_N, 16] uint8
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        codes = tl.reshape(tl.join(low, high), (BLOCK_SIZE_N, 32))  # k-interleaved
        vals = _e2m1_to_float(codes)
        scale = tl.exp2(tl.load(s_ptrs).to(tl.float32) - 127.0)  # [BLOCK_N]
        b = (vals * scale[:, None]).to(compute_type)  # [BLOCK_N, 32]
        accumulator += tl.dot(a, tl.trans(b))
        a_ptrs += 32 * stride_ak
        b_ptrs += 16 * stride_bk
        s_ptrs += stride_sk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_mxfp4_kernel(
    A: torch.Tensor, B_packed: torch.Tensor, B_scale: torch.Tensor, C: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor, num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool, top_k: int, config: dict, compute_type: tl.dtype,
) -> None:
    E, N, half_k = B_packed.shape
    K = half_k * 2
    assert K % MXFP4_GROUP == 0 and B_scale.shape == (E, N, K // MXFP4_GROUP), (B_packed.shape, B_scale.shape)
    assert A.shape[1] == K, (A.shape, K)

    def grid(META):
        return (triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    cfg = dict(config)
    cfg["BLOCK_SIZE_K"] = max(32, (cfg.get("BLOCK_SIZE_K", 32) // 32) * 32)
    fused_moe_mxfp4_kernel[grid](
        A, B_packed, B_scale, C, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, sorted_token_ids.shape[0], topk_ids.numel(),
        A.stride(0), A.stride(1),
        B_packed.stride(0), B_packed.stride(1), B_packed.stride(2),
        B_scale.stride(0), B_scale.stride(1), B_scale.stride(2),
        C.stride(-2), C.stride(-1),
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k, compute_type=compute_type, **cfg,
    )


@triton.jit
def _tanh_f32(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def situ_and_mul_kernel(
    gateup_ptr, out_ptr, hidden_size, beta, linear_beta,
    BLOCK_SIZE: tl.constexpr, HAS_LINEAR_BETA: tl.constexpr,
):
    """Row-wise SiTU-GLU over ``[rows, 2 * inter]`` -> ``[rows, inter]`` (fp32 math)."""
    half = hidden_size // 2
    pid = tl.program_id(0)
    gate_row = gateup_ptr + pid * hidden_size
    up_row = gate_row + half
    out_row = out_ptr + pid * half
    for start in tl.range(0, half, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < half
        g = tl.load(gate_row + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_row + offs, mask=mask, other=0.0).to(tl.float32)
        a = beta * _tanh_f32(g / beta) * tl.sigmoid(g)
        if HAS_LINEAR_BETA:
            u = linear_beta * _tanh_f32(u / linear_beta)
        tl.store(out_row + offs, (a * u).to(out_ptr.dtype.element_ty), mask=mask)


def situ_and_mul_triton(gateup: torch.Tensor, out: torch.Tensor, beta: float, linear_beta: float | None) -> None:
    assert gateup.is_contiguous() and out.is_contiguous()
    assert gateup.shape[0] == out.shape[0] and gateup.shape[1] == 2 * out.shape[1]
    situ_and_mul_kernel[(out.shape[0],)](
        gateup, out, gateup.shape[1], float(beta), float(linear_beta or 1.0),
        BLOCK_SIZE=512, HAS_LINEAR_BETA=linear_beta is not None,
    )


@torch.compiler.disable
def fused_experts_mxfp4(
    hidden_states: torch.Tensor,
    w13_packed: torch.Tensor, w13_scale: torch.Tensor,
    w2_packed: torch.Tensor, w2_scale: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    situ_beta: float = 4.0, situ_linear_beta: float | None = 25.0,
    reduce_results: bool = True,
) -> torch.Tensor:
    """MXFP4-weight MoE dispatch: ``hidden [T, K]`` bf16 -> ``[T, K]`` (or ``[T, top_k, K]``).

    ``w13_packed [E, 2*inter, K/2]`` (gate rows then up rows), ``w2_packed [E, K, inter/2]``.
    """
    assert hidden_states.dtype in (torch.bfloat16, torch.float16) and hidden_states.dim() == 2
    hidden_states = hidden_states.contiguous()
    num_tokens, hidden = hidden_states.shape
    E, two_inter, _ = w13_packed.shape
    inter = two_inter // 2
    assert w2_packed.shape == (E, hidden, inter // 2), (w2_packed.shape, (E, hidden, inter // 2))
    top_k = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(hidden_states.dtype).contiguous()
    config = get_default_config(M=num_tokens, E=E, N=two_inter, K=hidden, top_k=top_k)
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)
    m_topk = num_tokens * top_k
    cache1 = torch.empty((m_topk, two_inter), device=hidden_states.device, dtype=hidden_states.dtype)
    cache2 = torch.empty((m_topk, inter), device=hidden_states.device, dtype=hidden_states.dtype)
    cache3 = torch.empty((num_tokens, top_k, hidden), device=hidden_states.device, dtype=hidden_states.dtype)

    invoke_fused_moe_mxfp4_kernel(
        hidden_states, w13_packed, w13_scale, cache1, topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight=False, top_k=top_k, config=config, compute_type=compute_type,
    )
    situ_and_mul_triton(cache1, cache2, situ_beta, situ_linear_beta)
    invoke_fused_moe_mxfp4_kernel(
        cache2, w2_packed, w2_scale, cache3.view(m_topk, hidden), topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        mul_routed_weight=True, top_k=1, config=config, compute_type=compute_type,
    )
    if not reduce_results:
        return cache3
    output = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
    return output


@torch.compiler.disable
def fused_experts_bf16_situ(
    hidden_states: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    situ_beta: float = 4.0, situ_linear_beta: float | None = 25.0, reduce_results: bool = True,
) -> torch.Tensor:
    """The bf16 Triton grouped GEMM with the SiTU-GLU activation (dense-weight experts)."""
    from mstar.utils.fused_moe.kernels import invoke_fused_moe_kernel

    hidden_states = hidden_states.contiguous()
    num_tokens, hidden = hidden_states.shape
    E, two_inter, _ = w13.shape
    inter = two_inter // 2
    top_k = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(hidden_states.dtype).contiguous()
    config = get_default_config(M=num_tokens, E=E, N=two_inter, K=hidden, top_k=top_k)
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)
    m_topk = num_tokens * top_k
    cache1 = torch.empty((m_topk, two_inter), device=hidden_states.device, dtype=hidden_states.dtype)
    cache2 = torch.empty((m_topk, inter), device=hidden_states.device, dtype=hidden_states.dtype)
    cache3 = torch.empty((num_tokens, top_k, hidden), device=hidden_states.device, dtype=hidden_states.dtype)
    invoke_fused_moe_kernel(
        A=hidden_states, B=w13, C=cache1, topk_weights=topk_weights, topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids, expert_ids=expert_ids, num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False, top_k=top_k, config=config, compute_type=compute_type,
    )
    situ_and_mul_triton(cache1, cache2, situ_beta, situ_linear_beta)
    invoke_fused_moe_kernel(
        A=cache2, B=w2, C=cache3.view(m_topk, hidden), topk_weights=topk_weights, topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids, expert_ids=expert_ids, num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True, top_k=1, config=config, compute_type=compute_type,
    )
    if not reduce_results:
        return cache3
    output = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
    return output
