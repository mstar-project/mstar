"""Token-to-expert alignment for the fused MoE Triton kernel.

The fused MoE kernel expects tokens sorted by expert index and padded to
multiples of ``BLOCK_SIZE_M`` per expert.  The sort / pad is produced by
vLLM's ``moe_align_block_size`` CUDA op, vendored (Apache-2.0) as
``csrc/moe_align_block_size.cu`` and JIT-compiled on first use against the
local CUDA toolkit -- no ``sgl_kernel`` / ``vllm`` dependency.

If the CUDA op cannot be built (no ``nvcc`` / no CUDA device) we fall back
to a vectorized PyTorch implementation with identical output semantics.
The fallback runs entirely on-device but is slower than the fused kernel;
it exists so the path is always *available*, not fast.
"""

from __future__ import annotations

import functools
import logging
import os

import torch
import triton

logger = logging.getLogger(__name__)

_CSRC = os.path.join(os.path.dirname(__file__), "csrc", "moe_align_block_size.cu")


@functools.lru_cache(maxsize=1)
def _cuda_op_available() -> bool:
    """JIT-compile and load the vendored CUDA op; return whether it worked.

    Cached so compilation is attempted once per process.  A failure (missing
    ``nvcc``, no CUDA device, ABI mismatch) is logged and the caller uses the
    torch fallback instead.
    """
    if not torch.cuda.is_available():
        return False
    try:
        from torch.utils.cpp_extension import load

        load(name="_mstar_moe_C", sources=[_CSRC], is_python_module=False, verbose=False)
        # Touch the op so a registration failure surfaces here, not at call time.
        _ = torch.ops._mstar_moe_C.moe_align_block_size
        return True
    except Exception as e:  # pragma: no cover -- depends on the build toolchain
        logger.warning(
            "fused MoE: could not build the CUDA moe_align_block_size op (%s); "
            "using the slower torch fallback.",
            e,
        )
        return False


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort ``topk_ids`` into expert-aligned blocks.

    ``topk_ids`` must be int32 and contiguous (the CUDA op reads it directly).

    Returns
    -------
    sorted_token_ids : torch.Tensor
        ``(max_num_tokens_padded,)`` int32.  Valid slot indices are in
        ``[0, topk_ids.numel())``; padding slots hold ``topk_ids.numel()``
        (so the kernel's ``token_mask = offs_token < num_valid_tokens``
        discards them).
    expert_ids : torch.Tensor
        ``(max_num_m_blocks,)`` int32.  Expert index for each
        ``BLOCK_SIZE_M`` tile.
    num_tokens_post_padded : torch.Tensor
        ``(1,)`` int32 scalar with the count of valid + padding slots;
        the Triton kernel early-returns on tiles past this count.
    """
    # vLLM's allocation: worst case is one partial block padded per expert.
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)

    if _cuda_op_available():
        torch.ops._mstar_moe_C.moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        )
    else:
        _moe_align_block_size_torch(
            topk_ids, block_size, num_experts, sorted_ids, expert_ids, num_tokens_post_pad
        )

    return sorted_ids, expert_ids, num_tokens_post_pad


def _moe_align_block_size_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    """Vectorized PyTorch equivalent of the CUDA op, filling the buffers
    in place with the same semantics.

    Assumes every entry of ``topk_ids`` is a valid expert in
    ``[0, num_experts)`` (always true for mstar's routed dispatch).
    """
    device = topk_ids.device
    flat = topk_ids.reshape(-1)
    numel = flat.numel()

    # Padding slots hold ``numel``; unused expert-id blocks hold 0.
    sorted_ids.fill_(numel)
    expert_ids.zero_()

    # Per-expert token counts, each rounded up to a multiple of block_size.
    counts = torch.bincount(flat, minlength=num_experts)[:num_experts]
    padded = ((counts + block_size - 1) // block_size) * block_size

    # Exclusive prefix sum of padded counts -> first slot of each expert.
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    cumsum[1:] = torch.cumsum(padded, dim=0)
    num_tokens_post_pad.copy_(cumsum[num_experts].reshape(1).to(torch.int32))

    # expert_ids: block -> expert, one entry per (padded) block per expert.
    nblocks = (padded // block_size).to(torch.int64)
    expert_index = torch.repeat_interleave(
        torch.arange(num_experts, device=device, dtype=torch.int32), nblocks
    )
    expert_ids[: expert_index.numel()] = expert_index

    # sorted_token_ids: group original token indices by expert into the
    # padded slot ranges. Intra-expert order is irrelevant to the GEMM kernel.
    order = torch.argsort(flat, stable=True)
    sorted_experts = flat[order].to(torch.int64)
    ucounts = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    ucounts[1:] = torch.cumsum(counts, dim=0)  # unpadded prefix over sorted tokens
    local_rank = torch.arange(numel, device=device, dtype=torch.int64) - ucounts[sorted_experts]
    dest = cumsum[sorted_experts] + local_rank
    sorted_ids[dest] = order.to(torch.int32)
