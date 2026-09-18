"""Native Qwen3-family prompt encoder that returns tapped hidden states.

FLUX.2 [klein] and Z-Image condition their DiTs on intermediate hidden states of
a Qwen3 language model (klein: layers 9/18/27 of Qwen3-4B/8B concatenated), not on
its logits. This module is that encoder built from M*'s transformer components:
``DecoderLayer`` + ``GatedMLP`` + an HF-order RMSNorm with a GQA attention that carries
per-head q/k RMSNorm and plain RoPE, run only through the deepest tapped layer
(klein skips the last quarter of the LM).

Prompts arrive right-padded to one length (the pipelines pad to
``max_sequence_length`` and feed every position, pad tokens included, to the
DiT), so the padded positions' hidden states must be exactly what the reference
LM computes for them: a query at position ``i`` attends causally to keys ``<= i``
that are not padding. That mask is built once per batch and handed to SDPA, the
reference's own kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components import DecoderLayer, GatedMLP
from mstar.model.components.linear import FusedColumnLinear


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def qwen3_inv_freq(head_dim: int, theta: float) -> torch.Tensor:
    """HF's default RoPE ``inv_freq`` (``_compute_default_rope_parameters``), computed on the
    CPU as the reference does at module init: CPU and GPU ``pow`` differ in the last ulp for
    some entries, and that ulp survives into the bf16 sin table and the attention output."""
    steps = torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float)
    return 1.0 / (theta ** (steps / head_dim))


def qwen3_rotary_tables(
    positions: torch.Tensor, head_dim: int, theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF ``Qwen3RotaryEmbedding.forward``: fp32 ``cos, sin [S, head_dim]`` laid out ``[freqs | freqs]``,
    the angles from the same K=1 matmul on the positions' device."""
    inv_freq = qwen3_inv_freq(head_dim, theta).to(positions.device)
    freqs = (inv_freq[None, :, None] @ positions[None, None, :].float()).transpose(1, 2)[0]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def padded_causal_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """``[B, 1, S, S]`` boolean SDPA mask: causal, and keys that are padding are never
    attended (the reference ``create_causal_mask`` semantics for a right-padded batch)."""
    seq = attention_mask.shape[1]
    causal = torch.ones(seq, seq, dtype=torch.bool, device=attention_mask.device).tril()
    return causal[None, None] & attention_mask.bool()[:, None, None, :]


class Qwen3RMSNorm(nn.Module):
    """RMSNorm in the HF ``Qwen3RMSNorm`` rounding order: normalize in fp32, round to the
    input dtype, then multiply by the weight. In bf16 this is bit-exact with the reference
    encoder; ``mstar.model.components.RMSNorm`` rounds ``rsqrt`` first (portable path) or the
    whole product once (FlashInfer kernel), and either differs in the last bit, which the
    27 to 35 layers below a tap then amplify."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(x.dtype)


class Qwen3EncoderAttention(nn.Module):
    """GQA self-attention with per-head q/k RMSNorm (Qwen3), fused qkv GEMM, masked SDPA."""

    def __init__(
        self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float, rope_theta: float,
    ):
        super().__init__()
        self.num_heads, self.num_kv_heads, self.head_dim = num_heads, num_kv_heads, head_dim
        self.rope_theta = rope_theta
        self.qkv_proj = FusedColumnLinear(
            hidden_size, {"q": num_heads * head_dim, "k": num_kv_heads * head_dim, "v": num_kv_heads * head_dim},
            bias=False,
        )
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(head_dim, eps=eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=eps)
        # Set per forward by the encoder (shared across layers).
        self.rotary: tuple[torch.Tensor, torch.Tensor] | None = None
        self.attn_mask: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape
        q, k, v = self.qkv_proj(hidden_states).split(
            [self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim, self.num_kv_heads * self.head_dim],
            dim=-1,
        )
        q = self.q_norm(q.view(batch, seq, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(batch, seq, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = v.view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary
        cos, sin = cos.to(q.dtype)[None, None], sin.to(q.dtype)[None, None]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        groups = self.num_heads // self.num_kv_heads
        if groups > 1:
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask, is_causal=self.attn_mask is None)
        return self.o_proj(out.transpose(1, 2).reshape(batch, seq, self.num_heads * self.head_dim))


class Qwen3HiddenStateEncoder(nn.Module):
    """Embeddings + the first ``max(tap_layers)`` decoder layers of a Qwen3 LM; returns the
    hidden states after each tapped layer, concatenated along the feature dim
    (``hidden_states[k]`` in HF indexing == output of layer ``k``)."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        rope_theta: float,
        tap_layers: tuple[int, ...],
    ):
        super().__init__()
        if min(tap_layers) < 1:
            raise ValueError("tap layers are 1-indexed decoder layers (0 would be the raw embeddings)")
        self.tap_layers = tuple(tap_layers)
        self.head_dim, self.rope_theta = head_dim, rope_theta
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(
                self_attn=Qwen3EncoderAttention(
                    hidden_size, num_heads, num_kv_heads, head_dim, rms_norm_eps, rope_theta,
                ),
                mlp=GatedMLP(hidden_size, intermediate_size, activation="silu", bias=False),
                input_layernorm=Qwen3RMSNorm(hidden_size, eps=rms_norm_eps),
                post_attention_layernorm=Qwen3RMSNorm(hidden_size, eps=rms_norm_eps),
            )
            for _ in range(max(tap_layers))
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """``input_ids, attention_mask [B, S]`` (right padded) -> ``[B, S, len(taps) * hidden]``."""
        seq = input_ids.shape[1]
        rotary = qwen3_rotary_tables(torch.arange(seq, device=input_ids.device), self.head_dim, self.rope_theta)
        mask = padded_causal_mask(attention_mask)
        hidden = self.embed_tokens(input_ids)
        taps = []
        for layer_idx, layer in enumerate(self.layers, start=1):
            layer.self_attn.rotary = rotary
            layer.self_attn.attn_mask = mask
            hidden = layer(hidden)
            if layer_idx in self.tap_layers:
                taps.append(hidden)
        # stack(dim=1).permute(0, 2, 1, 3).reshape(...) == concatenation along the feature dim
        return torch.cat(taps, dim=-1)
