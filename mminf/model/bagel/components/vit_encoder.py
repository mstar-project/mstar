# Copyright (c) 2024 The HuggingFace Inc. team.
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/huggingface/transformers/blob/main/LICENSE.
#
# This modified file is released under the same license.


import logging

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
from transformers.activations import ACT2FN

from mminf.model.bagel.config import BagelViTConfig

logger = logging.getLogger(__name__)

try:
    from flash_attn import flash_attn_varlen_func

    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    flash_attn_varlen_func = None
    _FLASH_ATTN_AVAILABLE = False
    logger.warning(
        "flash_attn is not available; BagelViT will fall back to torch SDPA "
        "with a block-diagonal mask. This is slower than flash-attn varlen."
    )


def _sdpa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    causal: bool,
    scale: float | None,
) -> torch.Tensor:
    # q/k/v: (total_seq_len, num_heads, head_dim). Build a block-diagonal mask
    # so packed images don't attend across boundaries, then run SDPA.
    total_len, num_heads, head_dim = q.shape
    seg_ids = torch.zeros(total_len, dtype=torch.int32, device=q.device)
    seg_ids[cu_seqlens[1:-1].long()] = 1
    seg_ids = torch.cumsum(seg_ids, dim=0)
    attn_mask = seg_ids[:, None] == seg_ids[None, :]
    if causal:
        attn_mask = attn_mask & torch.ones(
            total_len, total_len, dtype=torch.bool, device=q.device
        ).tril()

    q_b = q.transpose(0, 1).unsqueeze(0)
    k_b = k.transpose(0, 1).unsqueeze(0)
    v_b = v.transpose(0, 1).unsqueeze(0)
    out = scaled_dot_product_attention(
        q_b, k_b, v_b, attn_mask=attn_mask, scale=scale
    )
    return out.squeeze(0).transpose(0, 1).contiguous()

@torch.compiler.disable
def run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    # cu_seqlens isolates per-image attention when multiple images are packed.
    # flashinfer's varlen kernels silently miscompute at SigLIP2's head_dim=72.
    if _FLASH_ATTN_AVAILABLE:
        return flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=causal,
            softmax_scale=scale,
        )
    return _sdpa_varlen(q, k, v, cu_seqlens, causal=causal, scale=scale)


class RotaryEmbedding2D(torch.nn.Module):
    def __init__(self, dim, max_h, max_w, base=10000):
        super().__init__()
        freq = torch.arange(0, dim, 2, dtype=torch.int64).float() / dim
        inv_freq = 1.0 / (base ** freq)

        grid_h = torch.arange(0, max_h)
        grid_h = grid_h.to(inv_freq.dtype)
        grid_h = grid_h[:, None].repeat(1, max_w)

        grid_w = torch.arange(0, max_w)
        grid_w = grid_w.to(inv_freq.dtype)
        grid_w = grid_w[None, :].repeat(max_h, 1)

        cos_h, sin_h = self._forward_one_side(grid_h, inv_freq)
        cos_w, sin_w = self._forward_one_side(grid_w, inv_freq)

        self.register_buffer("cos_h", cos_h)
        self.register_buffer("sin_h", sin_h)
        self.register_buffer("cos_w", cos_w)
        self.register_buffer("sin_w", sin_w)

    def _forward_one_side(self, grid, inv_freq):
        freqs = grid[..., None] * inv_freq[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1).flatten(0, 1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    # unsqueeze due to the head dimension
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class BagelVisionEmbeddings(nn.Module):
    def __init__(self, config: BagelViTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches_per_side = self.image_size // self.patch_size
        self.num_patches = self.num_patches_per_side**2
        self.num_positions = self.num_patches
        if not config.rope:
            self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

    def convert_conv2d_to_linear(self, config: BagelViTConfig, meta=False):
        # Keep conversion device-safe so this helper works even if called with
        # legacy `meta=True` callers.
        device = self.patch_embedding.weight.device
        dtype = self.patch_embedding.weight.dtype
        linear_patch_embedding = nn.Linear(
            config.num_channels * self.patch_size ** 2,
            self.embed_dim,
            bias=True,
            device=device,
            dtype=dtype,
        )
        W = self.patch_embedding.weight.permute(0, 2, 3, 1).reshape(
            self.embed_dim, config.num_channels * self.patch_size ** 2
        )
        linear_patch_embedding.weight.data = W
        linear_patch_embedding.bias.data = self.patch_embedding.bias.data
        del self.patch_embedding
        self.patch_embedding = linear_patch_embedding

    def forward(
        self,
        packed_pixel_values: torch.FloatTensor,
        packed_flattened_position_ids: torch.LongTensor
    ) -> torch.Tensor:

        patch_embeds = self.patch_embedding(packed_pixel_values)
        if not self.config.rope:
            embeddings = patch_embeds + self.position_embedding(packed_flattened_position_ids)
        else:
            embeddings = patch_embeds
        return embeddings



class BagelViTAttention(nn.Module):
    def __init__(self, config: BagelViTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:

        total_q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(total_q_len, self.num_heads, self.head_dim)
        k = k.view(total_q_len, self.num_heads, self.head_dim)
        v = v.view(total_q_len, self.num_heads, self.head_dim)

        if self.config.rope:
            qh, qw = q[:, :, :self.head_dim // 2], q[:, :, self.head_dim // 2:]
            kh, kw = k[:, :, :self.head_dim // 2], k[:, :, self.head_dim // 2:]

            # TODO: replace this with flashinfer
            qh, kh = apply_rotary_pos_emb(qh, kh, cos_h, sin_h)
            qw, kw = apply_rotary_pos_emb(qw, kw, cos_w, sin_w)

            q = torch.cat([qh, qw], dim=-1)
            k = torch.cat([kh, kw], dim=-1)

        attn_output = run_attention(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            causal=False,
        )

        attn_output = attn_output.reshape(total_q_len, -1)

        attn_output = self.out_proj(attn_output)
        return attn_output


class BagelViTMLP(nn.Module):
    def __init__(self, config: BagelViTConfig):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class BagelViTEncoderLayer(nn.Module):
    def __init__(self, config: BagelViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = BagelViTAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = BagelViTMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cos_h=cos_h,
            sin_h=sin_h,
            cos_w=cos_w,
            sin_w=sin_w
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class BagelViTEncoder(nn.Module):
    def __init__(self, config: BagelViTConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [BagelViTEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None,
    ) -> torch.Tensor:

        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, cu_seqlens, max_seqlen,
                                          cos_h=cos_h, sin_h=sin_h, cos_w=cos_w, sin_w=sin_w)

        return hidden_states


class BagelVisionTransformer(nn.Module):
    def __init__(self, config: BagelViTConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = BagelVisionEmbeddings(config)
        if config.rope:
            max_size = config.image_size // config.patch_size
            dim_head = config.hidden_size // config.num_attention_heads
            self.rope = RotaryEmbedding2D(dim_head // 2, max_size, max_size)

        self.encoder = BagelViTEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        hidden_states = self.embeddings(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_flattened_position_ids
        )

        extra_inputs = {}
        if self.config.rope:
            extra_inputs.update(
                cos_h = self.rope.cos_h[packed_flattened_position_ids],
                sin_h = self.rope.sin_h[packed_flattened_position_ids],
                cos_w = self.rope.cos_w[packed_flattened_position_ids],
                sin_w = self.rope.sin_w[packed_flattened_position_ids]
            )

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
            **extra_inputs
        )
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state


class BagelVisionModel(nn.Module):
    def __init__(self, config: BagelViTConfig):
        super().__init__()
        self.vision_model = BagelVisionTransformer(config)

    def get_input_embeddings(self) -> nn.Module:
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        packed_pixel_values: torch.Tensor,
        packed_flattened_position_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
    ) -> torch.Tensor:

        return self.vision_model(
            packed_pixel_values=packed_pixel_values,
            packed_flattened_position_ids=packed_flattened_position_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
