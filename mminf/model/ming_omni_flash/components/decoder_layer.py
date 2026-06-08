"""Ling-2.0 decoder layer (cache-aware, hybrid dense / MoE)."""

from __future__ import annotations

import torch
from torch import nn

from mminf.engine.cache_manager import BatchedCacheManager
from mminf.model.components.mlp import GatedMLP
from mminf.model.components.norm import RMSNorm
from mminf.model.ming_omni_flash.components.attention import LingAttention
from mminf.model.ming_omni_flash.components.moe import LingMoeBlock
from mminf.model.ming_omni_flash.components.rope import (
    LingPartialMRotaryEmbedding,
)


class LingDecoderLayer(nn.Module):
    """One Ling-2.0 decoder layer; layer_idx decides dense-vs-MoE FFN.

    Forward: pre-norm pattern, threads ``cache_handle`` to attention,
    threads optional modality masks to the MoE branch. Dense layers
    ignore the masks.

    See step 3b plan for full constructor docs.
    """

    def __init__(
        self,
        layer_idx: int,
        first_k_dense_replace: int,
        hidden_size: int,
        intermediate_size: int,
        moe_intermediate_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        num_experts: int,
        num_experts_per_tok: int,
        num_shared_experts: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        rotary: LingPartialMRotaryEmbedding,
        use_qkv_bias: bool = False,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.is_moe = layer_idx >= first_k_dense_replace

        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)

        self.self_attn = LingAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            rotary=rotary,
            use_qkv_bias=use_qkv_bias,
            use_bias=use_bias,
        )

        if self.is_moe:
            self.mlp: nn.Module = LingMoeBlock(
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_intermediate_size=moe_intermediate_size,
                num_shared_experts=num_shared_experts,
                n_group=n_group,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
            )
        else:
            self.mlp = GatedMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation="silu",
                bias=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, cache_handle, position_ids)
        h = residual + h

        residual = h
        h = self.post_attention_layernorm(h)
        if self.is_moe:
            h = self.mlp(h, image_mask=image_mask, audio_mask=audio_mask)
        else:
            h = self.mlp(h)
        return residual + h
