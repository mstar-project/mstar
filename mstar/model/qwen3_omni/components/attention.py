"""Qwen3-Omni attention: ``ParallelAttention`` + 3D MRoPE override.

Reuses ``ParallelAttention`` (which already supports per-head QK-norm,
fused QKV projection, and TP-sharded o_proj). The Qwen3-specific piece
is the 3D MRoPE path used by the Thinker — for ``use_mrope=True`` the
RoPE call goes through ``apply_interleaved_mrope`` with externally
provided ``cos_sin_3d`` instead of the cache handle. Talker uses
standard 1D RoPE (``use_mrope=False``) and inherits the parent's
``_apply_rope`` as-is.

Follows the same shape conventions as the shared attention:
  q: [tokens, num_heads, head_dim]
  k: [tokens, num_kv_heads, head_dim]
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from mstar.distributed.communication import TPCommGroup
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.components.distributed import ParallelAttention


class Qwen3OmniAttention(ParallelAttention):
    """TP-aware attention with QK-norm and pluggable 1D / 3D RoPE.

    When ``use_mrope=True`` (Thinker) the forward expects a
    ``cos_sin_3d`` tuple of ``(cos, sin)`` tensors and applies
    ``apply_interleaved_mrope``. When False (Talker), the parent's
    standard cache-handle RoPE is used.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-6,
        use_mrope: bool = False,
        comm_group: TPCommGroup | None = None,
    ):
        super().__init__(
            comm_group=comm_group,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_bias=False,
            o_bias=False,
            qk_norm=True,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
        )
        self.use_mrope = use_mrope

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        cos_sin_3d: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        mrope_section: Optional[list[int]] = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        q, k, v = self._project_qkv(hidden_states)
        q, k = self._apply_qk_norm(q, k)

        if self.use_mrope and cos_sin_3d is not None:
            from mstar.model.qwen3_omni.components.rope import apply_interleaved_mrope
            cos, sin = cos_sin_3d
            q, k = apply_interleaved_mrope(q, k, cos, sin)
        else:
            q, k = self._apply_rope(q, k, cache_handle)

        attn_output = cache_handle.run_attention(q=q, k=k, v=v)
        attn_output = attn_output.reshape(num_tokens, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)
