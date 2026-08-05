"""Python launchers over the vendored Marlin torch ops."""
from __future__ import annotations

import torch

from mstar.utils.fused_moe.align import moe_align_block_size
from mstar.utils.fused_moe.kernels import act_and_mul_triton, moe_sum_reduce_triton
from mstar.utils.marlin.scalar_type import UINT4B8_ID

_MARLIN_TILE = 16


def gptq_marlin_repack(
    b_q_weight: torch.Tensor, size_k: int, size_n: int, num_bits: int = 4
) -> torch.Tensor:
    perm = torch.empty(0, dtype=torch.int32, device=b_q_weight.device)
    return torch.ops._mstar_marlin_C.gptq_marlin_repack(
        b_q_weight, perm, size_k, size_n, num_bits
    )


def gptq_marlin_moe_repack(
    b_q_weight: torch.Tensor, size_k: int, size_n: int, num_bits: int = 4
) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    perm = torch.empty(0, dtype=torch.int32, device=b_q_weight.device)
    output = torch.empty(
        (num_experts, size_k // _MARLIN_TILE, size_n * (num_bits // 2)),
        device=b_q_weight.device,
        dtype=b_q_weight.dtype,
    )
    for e in range(num_experts):
        output[e] = torch.ops._mstar_marlin_C.gptq_marlin_repack(
            b_q_weight[e], perm, size_k, size_n, num_bits
        )
    return output


def _get_scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    scale_perm, scale_perm_single = _get_scale_perms()
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def marlin_moe_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int
) -> torch.Tensor:
    num_experts = s.shape[0]
    output = torch.empty_like(s)
    for e in range(num_experts):
        output[e] = marlin_permute_scales(s[e], size_k, size_n, group_size)
    return output


def marlin_make_workspace(device: torch.device, max_blocks_per_sm: int = 4) -> torch.Tensor:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int, device=device)


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1_marlin: torch.Tensor,
    w2_marlin: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace: torch.Tensor,
    *,
    activation: str = "silu",
    reduce_results: bool = True,
) -> torch.Tensor:
    assert hidden_states.is_contiguous() and hidden_states.dim() == 2
    assert hidden_states.dtype in (torch.bfloat16, torch.float16)
    M, K = hidden_states.shape
    E = w1_marlin.shape[0]
    top_k = topk_ids.shape[1]
    # w2 marlin is (E, N//16, K*num_bits//8) -> intermediate size N = shape[1]*16.
    N = w2_marlin.shape[1] * _MARLIN_TILE

    # Block-size selection (vLLM heuristic): smallest block that keeps ~>=0.9
    # expert occupancy.
    block_size_m = 8
    for block_size_m in (8, 16, 32, 48, 64):
        if M * top_k / E / block_size_m < 0.9:
            break

    # Marlin requires fp32 combine weights and int32 expert ids.
    topk_weights = topk_weights.to(torch.float32).contiguous()
    topk_ids_i32 = topk_ids.to(torch.int32).contiguous()
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids_i32, block_size_m, E
    )

    # Gate+up GEMM: (M, K) x experts -> (M*top_k, 2N).
    gate_up = torch.ops._mstar_marlin_C.moe_wna16_marlin_gemm(
        hidden_states, None, w1_marlin, w1_scale, workspace,
        sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
        block_size_m, top_k, False, UINT4B8_ID, M, 2 * N, K,
        True, False, True,
    )

    # SwiGLU: silu(gate) * up -> (M*top_k, N).
    down_in = torch.empty((M * top_k, N), device=hidden_states.device, dtype=hidden_states.dtype)
    act_and_mul_triton(gate_up, down_in, activation=activation)

    # Down GEMM (weighted): (M*top_k, N) x experts -> (M*top_k, K), routing
    # weight folded in (mul_topk_weights=True).
    down = torch.ops._mstar_marlin_C.moe_wna16_marlin_gemm(
        down_in, None, w2_marlin, w2_scale, workspace,
        sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights,
        block_size_m, 1, True, UINT4B8_ID, M * top_k, K, N,
        True, False, True,
    )

    cache3 = down.view(M, top_k, K)
    if not reduce_results:
        return cache3
    output = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
    return output
