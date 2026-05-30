"""Token-to-expert alignment for the fused MoE Triton kernel.

The fused MoE kernel expects tokens sorted by expert index and padded to
multiples of ``BLOCK_SIZE_M`` per expert.  The sort / pad is produced by
sglang's ``moe_align_block_size`` CUDA op (from the ``sgl_kernel`` pip
package).  This module wraps that op with the same allocation pattern
sglang uses; if the ``sgl_kernel`` import fails we raise a
``RuntimeError`` at call time so the caller can fall back to the naive
per-expert dispatch path.
"""

from __future__ import annotations

import logging
from typing import Tuple

import torch
import triton

logger = logging.getLogger(__name__)

try:
    from sgl_kernel import moe_align_block_size as _sgl_moe_align_block_size
except ImportError as e:  # pragma: no cover -- exercised on boxes without sgl_kernel
    _sgl_moe_align_block_size = None
    logger.warning(f"Could not load _fused_experts: {e}")


def has_sgl_kernel() -> bool:
    """Whether the optional ``sgl_kernel`` dependency is available."""
    return _sgl_moe_align_block_size is not None


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort ``topk_ids`` into expert-aligned blocks.

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

    Notes
    -----
    ``topk_ids`` must be int32 and contiguous (the sglang CUDA op reads
    from it directly).  The ``num_experts + 1`` shift and ``cumsum``
    buffer width match sglang's own wrapper so the op sees the same
    memory layout it was built against.
    """
    if _sgl_moe_align_block_size is None:
        raise RuntimeError(
            "Fused MoE requires the optional 'sgl-kernel' package. "
            "Install with: pip install sgl-kernel "
            "(choose the wheel matching your torch/CUDA version: "
            "https://github.com/sgl-project/sglang/releases)."
        )

    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    # Matches sglang's extra slot for the "filtered / padding" expert id.
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)

    _sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad
