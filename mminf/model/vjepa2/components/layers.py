"""Shared transformer building blocks for V-JEPA 2 encoder and predictor.

Ports ``VJEPA2MLP``, ``VJEPA2RopeAttention``, ``VJEPA2Layer`` from HuggingFace
``transformers/models/vjepa2/modeling_vjepa2.py``.  Uses eager (matmul+softmax)
attention — no SDPA auto-selection or gradient checkpointing — to keep
numerics bit-reproducible against the reference implementation.

Weight layout per layer (matches HF checkpoint keys):
    norm1.{weight,bias}
    attention.{query,key,value}.{weight,bias}
    attention.proj.{weight,bias}
    norm2.{weight,bias}
    mlp.fc1.{weight,bias}
    mlp.fc2.{weight,bias}
"""

from __future__ import annotations

import torch
from torch import nn

from mminf.model.vjepa2.components.rope_utils import rotate_queries_or_keys
from mminf.model.vjepa2.config import VJepa2Config

_ACT2FN = {
    "gelu": nn.GELU(),
    "gelu_new": nn.GELU(approximate="tanh"),
    "relu": nn.ReLU(),
    "silu": nn.SiLU(),
}


def _eager_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Bit-reproducible eager attention.

    Matches HF's ``eager_attention_forward`` — softmax done in fp32 then
    cast back to the input dtype.  No dropout (inference-only).
    """
    attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scaling
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output.transpose(1, 2).contiguous()


class VJEPA2MLP(nn.Module):
    def __init__(self, config: VJepa2Config, hidden_size: int, mlp_ratio: float):
        super().__init__()
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = nn.Linear(hidden_size, hidden_features, bias=True)
        self.activation = _ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(hidden_features, hidden_size, bias=True)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(hidden_state)))


class VJEPA2RopeAttention(nn.Module):
    """Self-attention with 3D rotary positional encoding.

    Q/K/V are separate ``nn.Linear`` projections (matches HF checkpoint key
    layout: ``attention.{query,key,value,proj}.*``).  RoPE is applied to
    queries and keys, split into depth/height/width axes derived from each
    token's position id.
    """

    def __init__(
        self,
        config: VJepa2Config,
        hidden_size: int,
        num_attention_heads: int,
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} is not divisible by num_attention_heads={num_attention_heads}")
        self.config = config
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.proj = nn.Linear(hidden_size, hidden_size)

        self.grid_size = config.crop_size // config.patch_size
        self.grid_depth = config.frames_per_clip // config.tubelet_size

        # Allocate equal share of head_dim to each of D/H/W, rounded to even.
        third = 2 * ((self.attention_head_size // 3) // 2)
        self.d_dim = third
        self.h_dim = third
        self.w_dim = third

        self.scaling = self.attention_head_size**-0.5

    def _get_frame_pos(self, ids: torch.Tensor) -> torch.Tensor:
        tokens_per_frame = self.grid_size * self.grid_size
        return ids // tokens_per_frame

    def _get_height_pos(self, ids: torch.Tensor) -> torch.Tensor:
        tokens_per_frame = self.grid_size * self.grid_size
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        return ids // self.grid_size

    def get_position_ids(
        self,
        x: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = x.device
        token_size = x.size(1)

        if masks is not None:
            ids = masks.unsqueeze(1).repeat(1, self.num_attention_heads, 1)
        else:
            ids = torch.arange(token_size, device=device)

        tokens_per_frame = self.grid_size * self.grid_size
        frame_ids = self._get_frame_pos(ids)
        height_ids = self._get_height_pos(ids)
        width_ids = (ids - tokens_per_frame * frame_ids) - self.grid_size * height_ids
        return frame_ids, height_ids, width_ids

    def apply_rotary_embeddings(
        self,
        qk: torch.Tensor,
        pos_ids: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        d_mask, h_mask, w_mask = pos_ids
        s = 0
        qkd = rotate_queries_or_keys(qk[..., s : s + self.d_dim], pos=d_mask)
        s += self.d_dim
        qkh = rotate_queries_or_keys(qk[..., s : s + self.h_dim], pos=h_mask)
        s += self.h_dim
        qkw = rotate_queries_or_keys(qk[..., s : s + self.w_dim], pos=w_mask)
        s += self.w_dim
        if s < self.attention_head_size:
            qkr = qk[..., s:]
            return torch.cat([qkd, qkh, qkw, qkr], dim=-1)
        return torch.cat([qkd, qkh, qkw], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.attention_head_size)
        q = self.query(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.key(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.value(hidden_states).view(hidden_shape).transpose(1, 2)

        pos_ids = self.get_position_ids(hidden_states, masks=position_mask)
        q = self.apply_rotary_embeddings(q, pos_ids)
        k = self.apply_rotary_embeddings(k, pos_ids)

        context = _eager_attention(q, k, v, self.scaling)
        context = context.reshape(*input_shape, self.all_head_size)
        return self.proj(context)


class VJEPA2Layer(nn.Module):
    """One transformer block: pre-norm self-attention + pre-norm MLP with residuals."""

    def __init__(
        self,
        config: VJepa2Config,
        hidden_size: int,
        num_attention_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = VJEPA2RopeAttention(config, hidden_size, num_attention_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.mlp = VJEPA2MLP(config, hidden_size=hidden_size, mlp_ratio=mlp_ratio)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, position_mask=position_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states
