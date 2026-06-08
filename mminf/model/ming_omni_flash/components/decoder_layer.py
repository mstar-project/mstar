"""Ling-2.0 decoder layer (hybrid dense / MoE per ``first_k_dense_replace``).

Pre-norm transformer block:

    residual = h
    h = self_attn(input_layernorm(h), positions)
    h = residual + h
    residual = h
    h = post_attention_layernorm(h)
    h = mlp(h, [image_mask, audio_mask])   # MoE layers
        OR
    h = mlp(h)                              # dense layer 0
    h = residual + h

Why a new layer class instead of reusing
:class:`mminf.model.components.decoder_layer.DecoderLayer`: mminf's
existing layer calls ``self_attn(hidden, cache_handle=cache_handle)`` —
that KV-cache plumbing isn't wired up yet (step 3c). And the MoE path
needs the two modality-mask kwargs which the base class doesn't thread.

Reference: vllm-omni's :class:`BailingMoeV2DecoderLayer` at
``/tmp/vllm-omni/.../modeling_bailing_moe_v2.py:566-649``.
"""

from __future__ import annotations

import torch
from torch import nn

from mminf.model.components.mlp import GatedMLP
from mminf.model.components.norm import RMSNorm
from mminf.model.ming_omni_flash.components.attention import LingAttention
from mminf.model.ming_omni_flash.components.moe import LingMoeBlock
from mminf.model.ming_omni_flash.components.rope import (
    LingPartialMRotaryEmbedding,
)


class LingDecoderLayer(nn.Module):
    """One Ling-2.0 decoder layer; layer_idx decides dense-vs-MoE FFN.

    Args:
        layer_idx: 0-based layer index. Layers with
            ``layer_idx < first_k_dense_replace`` use the dense
            :class:`GatedMLP`; the rest use :class:`LingMoeBlock`.
        first_k_dense_replace: how many leading layers use a plain dense
            MLP. Released ckpt = 1.
        hidden_size, intermediate_size, moe_intermediate_size,
        num_attention_heads, num_kv_heads, head_dim, rms_norm_eps,
        num_experts, num_experts_per_tok, num_shared_experts, n_group,
        topk_group, routed_scaling_factor: passed through to MLP/MoE
        constructors.
        rotary: shared :class:`LingPartialMRotaryEmbedding` (one
            instance reused across all layers in the model).
        use_qkv_bias, use_bias: per attention config.
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
            # Dense layer-0 MLP — same SwiGLU shape but at the full
            # intermediate_size, not the per-expert moe_intermediate_size.
            self.mlp = GatedMLP(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                activation="silu",
                bias=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        image_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        h = self.self_attn(h, position_ids)
        h = residual + h

        residual = h
        h = self.post_attention_layernorm(h)
        if self.is_moe:
            h = self.mlp(h, image_mask=image_mask, audio_mask=audio_mask)
        else:
            # Dense layer ignores modality masks — there's only one
            # forward path.
            h = self.mlp(h)
        return residual + h
