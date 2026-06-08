"""Ling-2.0 attention with QK-norm + partial 3D MRoPE + cache-handle attention.

Wraps mminf's :class:`BatchedCacheManager` for paged KV cache + masked
SDPA via FlashInfer, while keeping the architecture-specific bits
(packed QKV, per-head q_norm/k_norm before RoPE, partial 3D video_rope)
local to this module.

The forward expects the **packed** (num_tokens, hidden) layout that
mminf's engine uses everywhere — not the (B, T, H) layout the step-3a
unit-test scope had. Position handling is via an explicit
``position_ids`` argument (the model passes them through; we don't
read from ``cache_handle`` to keep this submodule unit-testable with a
mock cache).

Reference: vllm-omni's :class:`BailingMoeV2Attention`
(`/tmp/vllm-omni/.../ming_flash_omni/modeling_bailing_moe_v2.py:436-563`)
+ mminf's :class:`Attention`
(`mminf/model/components/attention.py`) for the cache-handle shape.
"""

from __future__ import annotations

import torch
from torch import nn

from mminf.engine.cache_manager import BatchedCacheManager
from mminf.model.components.norm import RMSNorm
from mminf.model.ming_omni_flash.components.rope import LingPartialMRotaryEmbedding


class LingAttention(nn.Module):
    """Ling-2.0 attention layer (packed-tokens, cache-handle-aware).

    Args mirror step 3a; the forward signature is now engine-facing:
    ``(hidden_states[num_tokens, hidden], cache_handle, position_ids)``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rotary: LingPartialMRotaryEmbedding,
        use_qkv_bias: bool = False,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must be divisible by "
                f"num_kv_heads={num_kv_heads} for GQA"
            )
        if rotary.head_dim != head_dim:
            raise ValueError(
                f"rotary.head_dim={rotary.head_dim} must equal head_dim={head_dim}"
            )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_groups = num_heads // num_kv_heads
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.scaling = head_dim ** -0.5

        # Packed QKV projection — matches the released ckpt's
        # ``query_key_value.weight`` shape `(num_heads + 2*num_kv_heads)*head_dim x hidden`,
        # rows ordered [Q heads, K heads, V heads].
        self.qkv_proj = nn.Linear(
            hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=use_qkv_bias,
        )
        self.dense = nn.Linear(self.q_size, hidden_size, bias=use_bias)

        # Per-head normalisation on q and k before rope.
        self.q_norm = RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, eps=rms_norm_eps)

        self.rotary = rotary

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Engine-facing forward (packed tokens, cache-aware).

        Args:
            hidden_states: ``(num_tokens, hidden_size)``.
            cache_handle: mminf's cache manager. Must have been
                ``set_layer_idx``-ed by the caller before this call.
                We call ``run_attention(q, k, v)`` for paged KV write +
                masked attention.
            position_ids: ``(num_tokens,)`` for 1D rope or
                ``(3, num_tokens)`` for 3D video_rope.

        Returns:
            ``(num_tokens, hidden_size)``.
        """
        num_tokens = hidden_states.shape[0]

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        # QK-norm: RMSNorm across head_dim, broadcast across heads. We
        # flatten to (num_tokens*num_heads_or_kv, head_dim) so mminf's
        # RMSNorm sees a contiguous last-dim normalization.
        q = self.q_norm(q.reshape(-1, self.head_dim)).view(
            num_tokens, self.num_heads, self.head_dim
        )
        k = self.k_norm(k.reshape(-1, self.head_dim)).view(
            num_tokens, self.num_kv_heads, self.head_dim
        )

        # Partial 3D rope. rotary expects (..., num_tokens, head_dim);
        # swap heads <-> tokens so the broadcast over the heads axis
        # works (rope cos/sin lives at (num_tokens, head_dim)).
        q = q.transpose(0, 1)  # (num_heads, num_tokens, head_dim)
        k = k.transpose(0, 1)
        q, k = self.rotary(q, k, position_ids)
        q = q.transpose(0, 1).contiguous()  # back to (num_tokens, num_heads, head_dim)
        k = k.transpose(0, 1).contiguous()

        # Engine-managed attention: paged KV write + masked SDPA via
        # the cache manager's pre-planned FlashInfer wrapper.
        attn_output = cache_handle.run_attention(q=q, k=k, v=v)
        attn_output = attn_output.reshape(num_tokens, self.q_size)
        return self.dense(attn_output)

    @staticmethod
    def head_norm_check(q_after_norm: torch.Tensor) -> float:
        """Diagnostic: returns max abs deviation of per-head RMS from 1.

        Used by the existing test that exercises q_norm independently.
        Kept as a static method so the test doesn't need a forward.
        """
        norms = q_after_norm.float().pow(2).mean(dim=-1).sqrt()
        return (norms - 1.0).abs().max().item()
