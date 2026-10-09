"""The Gemma-4 text tower LTX-2.5 encodes prompts with, native and forward-only.

A port of ``transformers``' ``Gemma4UnifiedTextModel`` (transformers >= 5.5; the
serving environment pins an older release without it) restricted to what the
pipeline uses: one causal pass over the prompt, returning every layer's hidden
state. No KV cache, no generation, no vision/audio towers.

The pipeline left-pads every prompt to 1024 tokens and lets the pads occupy
positions ``0 .. pad - 1``. Pad rows never reach the DiT — the connectors replace
them with learned registers — and causal attention keeps them out of every real
row, so this encoder runs the real tokens alone, at the positions the reference
gives them (``1024 - n .. 1023``). That is the same result for a fraction of the
compute.

Gemma-4 specifics, from ``modeling_gemma4_unified.py``:

* RMSNorm scales by ``weight`` (not Gemma-3's ``1 + weight``) and computes in fp32;
  the value norm has no weight at all.
* Attention scaling is 1.0 (the q/k norms carry it). Sliding layers have
  ``head_dim`` 256 and grouped KV heads; global layers have ``global_head_dim`` 512,
  one KV head, and reuse the key projection as values (``attention_k_eq_v``).
* Sliding layers use default RoPE; global layers use "proportional" RoPE, rotating
  only the first ``partial_rotary_factor`` of each head's frequencies.
* The 1024-token sliding window covers the whole prompt, so it never masks.
* Each layer's output is scaled by a learned ``layer_scalar``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.ltx2_5.config import GemmaTextConfig


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if with_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.weight is not None:
            normed = normed * self.weight.float()
        return normed.type_as(x)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def inv_freq_default(head_dim: int, theta: float) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))


def inv_freq_proportional(head_dim: int, theta: float, partial: float) -> torch.Tensor:
    """``_compute_proportional_rope_parameters``: the first ``partial`` of the angles
    rotate at ``theta^(-2i / head_dim)``; the rest have frequency 0."""
    angles = int(partial * head_dim // 2)
    rotated = 1.0 / (theta ** (torch.arange(0, 2 * angles, 2, dtype=torch.int64).float() / head_dim))
    return torch.cat([rotated, torch.zeros(head_dim // 2 - angles)])


class GemmaAttention(nn.Module):
    def __init__(self, cfg: GemmaTextConfig, is_global: bool):
        super().__init__()
        self.head_dim = cfg.global_head_dim if is_global else cfg.head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_global_key_value_heads if is_global else cfg.num_key_value_heads
        self.k_is_v = cfg.attention_k_eq_v and is_global
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = None if self.k_is_v else nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = GemmaRMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.v_norm = GemmaRMSNorm(self.head_dim, cfg.rms_norm_eps, with_scale=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, T, hidden]``; ``cos`` / ``sin``: ``[B, T, head_dim]`` in ``x``'s dtype."""
        shape = (*x.shape[:-1], -1, self.head_dim)
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
        q = self.q_norm(self.q_proj(x).view(shape))
        q = q * cos + rotate_half(q) * sin
        k_raw = self.k_proj(x).view(shape)
        v = k_raw if self.v_proj is None else self.v_proj(x).view(shape)
        k = self.k_norm(k_raw)
        k = k * cos + rotate_half(k) * sin
        v = self.v_norm(v)
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True, scale=1.0, enable_gqa=self.num_kv_heads != self.num_heads,
        )
        return self.o_proj(out.transpose(1, 2).flatten(2))


class GemmaMLP(nn.Module):
    def __init__(self, cfg: GemmaTextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaLayer(nn.Module):
    def __init__(self, cfg: GemmaTextConfig, is_global: bool):
        super().__init__()
        self.is_global = is_global
        self.self_attn = GemmaAttention(cfg, is_global)
        self.mlp = GemmaMLP(cfg)
        eps = cfg.rms_norm_eps
        self.input_layernorm = GemmaRMSNorm(cfg.hidden_size, eps)
        self.post_attention_layernorm = GemmaRMSNorm(cfg.hidden_size, eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(cfg.hidden_size, eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(cfg.hidden_size, eps)
        # a parameter, not a buffer, so the checkpoint loader reaches it
        self.layer_scalar = nn.Parameter(torch.ones(1), requires_grad=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), cos, sin))
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return x * self.layer_scalar


class Gemma4TextEncoder(nn.Module):
    """Prompt token ids -> the ``num_layers + 1`` hidden states the LTX connectors consume."""

    def __init__(self, cfg: GemmaTextConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.layers = nn.ModuleList(
            [GemmaLayer(cfg, layer_type == "full_attention") for layer_type in cfg.layer_types]
        )
        self.norm = GemmaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        # Not buffers: the module is built on the meta device and materialized with
        # to_empty, which would leave non-persistent buffers uninitialized.
        self._inv_freq: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    def _inv_freqs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """(sliding, global) RoPE frequencies, computed on the CPU as the reference
        initializes them, then moved."""
        if device not in self._inv_freq:
            cfg = self.cfg
            self._inv_freq[device] = (
                inv_freq_default(cfg.head_dim, cfg.sliding_rope_theta).to(device),
                inv_freq_proportional(
                    cfg.global_head_dim, cfg.global_rope_theta, cfg.global_partial_rotary_factor,
                ).to(device),
            )
        return self._inv_freq[device]

    @staticmethod
    def _rope(inv_freq: torch.Tensor, positions: torch.Tensor, dtype: torch.dtype):
        freqs = positions[..., None].float() * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """``input_ids`` / ``positions``: ``[B, T]``. Returns ``[B, T, hidden, layers + 1]``:
        the scaled embeddings, every layer's output, and the final norm in place of the
        last layer's raw output (what ``output_hidden_states`` returns)."""
        embed_scale = torch.tensor(self.cfg.hidden_size ** 0.5).to(self.dtype)
        x = self.embed_tokens(input_ids) * embed_scale
        sliding_freq, global_freq = self._inv_freqs(x.device)
        sliding = self._rope(sliding_freq, positions, x.dtype)
        global_ = self._rope(global_freq, positions, x.dtype)
        states = [x]
        for layer in self.layers:
            x = layer(x, *(global_ if layer.is_global else sliding))
            states.append(x)
        states[-1] = self.norm(x)
        return torch.stack(states, dim=-1)
