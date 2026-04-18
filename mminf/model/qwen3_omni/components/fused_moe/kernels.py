"""Triton kernels for fused MoE dispatch.

Ported from sglang's ``fused_moe_triton_kernels.py`` and
``fused_moe_triton_config.py`` (Apache-2.0).  Quantization branches
(fp8_w8a8, int8_w8a8, int8_w8a16, int4_w4a16) and the TMA / swap_ab /
fused-all-reduce / expert-parallel paths are stripped since mminf only
needs the bf16 single-node path.

The three kernels mirror sglang's layout:

* :func:`fused_moe_kernel` -- grouped GEMM; the same kernel is used for
  both the gate+up GEMM and the down GEMM.
* :func:`act_and_mul_kernel` -- per-slot SwiGLU activation on the
  ``(M*topk, 2*inter)`` intermediate.
* :func:`moe_sum_reduce_kernel` -- weight-free sum over the top-k
  dimension of the ``(M, topk, hidden)`` down-GEMM output.

The Python wrappers :func:`invoke_fused_moe_kernel`,
:func:`act_and_mul_triton`, :func:`moe_sum_reduce_triton` set up launch
grids and keep the Triton-specific boilerplate out of the runner.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Main grouped-GEMM kernel (used for both gate_up and down projections)
# ---------------------------------------------------------------------------


@triton.jit
def fused_moe_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Block sizes (compile-time)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """Compute one ``[BLOCK_SIZE_M, BLOCK_SIZE_N]`` tile of the MoE output.

    ``A`` holds the input rows (``M`` rows, ``K`` cols); at the first GEMM
    ``A`` is the hidden states tensor and at the second GEMM it is the
    SwiGLU intermediate.  ``B`` is the stacked expert weight tensor of
    shape ``(E, N, K)``.  ``C`` is the output cache of shape
    ``(M*topk, N)``.

    Tokens are permuted into expert-aligned blocks by
    ``moe_align_block_size`` before the launch.  ``sorted_token_ids``
    holds the permuted slot indices (< ``num_valid_tokens`` for real
    tokens, >= for padding) and ``expert_ids`` maps each ``BLOCK_SIZE_M``
    block to the expert index.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Skip blocks past the padded token count entirely.
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # offs_token // top_k recovers the source row in A for (token, slot) pairs.
    # For the second GEMM we pass top_k=1 so the divide is a no-op and the
    # kernel reads intermediate-cache rows directly.
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    """Launch :func:`fused_moe_kernel` with the right grid size.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor, shape ``(M, K)`` (hidden states or SwiGLU intermediate).
    B : torch.Tensor
        Stacked expert weights, shape ``(E, N, K)``.
    C : torch.Tensor
        Output buffer, shape ``(M*topk, N)`` for GEMM-1 or reshape-viewed
        ``(M*topk, hidden)`` for GEMM-2.
    topk_weights, topk_ids : torch.Tensor
        Router output, shapes ``(M, top_k)``.  ``topk_ids`` must be int32.
    sorted_token_ids, expert_ids, num_tokens_post_padded : torch.Tensor
        Outputs of :func:`moe_align_block_size`.
    mul_routed_weight : bool
        If ``True``, multiply the accumulator by ``topk_weights`` before
        writing -- used for the down GEMM to fold the routing weight into
        the output rows that ``moe_sum_reduce_triton`` then sums.
    top_k : int
        ``topk`` for GEMM-1; pass ``1`` for GEMM-2 so the in-kernel
        ``offs_token // top_k`` becomes an identity.
    config : dict
        Tile sizes; output of :func:`get_default_config`.
    compute_type : triton.language.dtype
        Dtype for the accumulator store.
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    def grid(META):
        return (
            triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
            * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
        )

    K = B.shape[2]
    even_Ks = (K % config["BLOCK_SIZE_K"]) == 0

    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        K,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(-2),
        C.stride(-1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        even_Ks=even_Ks,
        **config,
    )


# ---------------------------------------------------------------------------
# Activation (SwiGLU / GeGLU) kernel
# ---------------------------------------------------------------------------


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _apply_activation(x, ACTIVATION_TYPE: tl.constexpr):
    x = x.to(tl.float32)
    if ACTIVATION_TYPE == "silu":
        return x * tl.sigmoid(x)
    elif ACTIVATION_TYPE == "gelu":
        k = 0.7978845608028654  # sqrt(2/pi)
        return 0.5 * x * (1.0 + _tanh(k * (x + 0.044715 * x * x * x)))
    else:
        tl.static_assert(False, "Unsupported activation")
        return x


@triton.jit
def act_and_mul_kernel(
    gateup_output_ptr,
    down_input_ptr,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION_TYPE: tl.constexpr,
):
    """Per-slot SwiGLU activation.

    Input ``gateup_output`` has layout ``(M*topk, 2*inter)`` with the
    gate half in columns ``[0:inter]`` and the up half in columns
    ``[inter:2*inter]``.  Writes ``act(gate) * up`` to ``down_input`` of
    shape ``(M*topk, inter)``.
    """
    in_dtype = gateup_output_ptr.dtype.element_ty
    out_dtype = down_input_ptr.dtype.element_ty

    half = hidden_size // 2
    pid = tl.program_id(0)

    gate_row = gateup_output_ptr + pid * hidden_size
    up_row = gate_row + half
    out_row = down_input_ptr + pid * half

    for start_offset in tl.range(0, half, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < half
        gate = tl.load(gate_row + offset, mask=mask)
        up = tl.load(up_row + offset, mask=mask)
        activated = _apply_activation(gate, ACTIVATION_TYPE).to(in_dtype)
        out = activated * up
        tl.store(out_row + offset, out.to(out_dtype), mask=mask)


def act_and_mul_triton(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    activation: str = "silu",
) -> None:
    """Wrapper launching :func:`act_and_mul_kernel` per intermediate slot."""
    assert gateup_output.is_contiguous()
    assert down_input.is_contiguous()
    assert gateup_output.shape[0] == down_input.shape[0]
    assert gateup_output.shape[1] == 2 * down_input.shape[1]

    grid = (down_input.shape[0],)
    hidden_size = gateup_output.shape[1]
    act_and_mul_kernel[grid](
        gateup_output,
        down_input,
        hidden_size,
        BLOCK_SIZE=512,
        ACTIVATION_TYPE=activation,
    )


# ---------------------------------------------------------------------------
# Sum-reduce over the top-k dimension
# ---------------------------------------------------------------------------


@triton.jit
def moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    """Sum ``input`` over its ``topk`` dim, optionally scaling the result.

    ``input`` is ``(token_num, topk_num, hidden_dim)``; ``output`` is
    ``(token_num, hidden_dim)``.  The output dtype matches ``input``.
    """
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0.0,
        )
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )


def moe_sum_reduce_triton(
    input: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float = 1.0,
) -> None:
    """Launch :func:`moe_sum_reduce_kernel`."""
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 16

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )
    moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# Block-size / warp-count configuration (bf16 only; no quant, no marlin)
# ---------------------------------------------------------------------------


def get_default_config(M: int, E: int, N: int, K: int, top_k: int) -> Dict[str, int]:
    """Pick Triton tile sizes based on problem shape.

    Mirrors sglang's ``get_default_config`` for the unquantized path.
    For decode batch sizes (``M`` on the order of 1--64) we always fall
    into the ``M <= E`` branch since Qwen3-Omni has ``E == 128``.
    """
    if M <= E:
        return {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
