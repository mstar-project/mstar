"""Native Z-Image single-stream flow transformer (exact port of diffusers 0.39/0.40
``ZImageTransformer2DModel`` in its basic text-to-image mode).

One denoise step over a request:

    t -> sinusoidal(256, [cos | sin]) -> MLP(256 -> 1024 -> 256)          adaLN conditioning (bf16)
    image latents -> 2x2 patches (64 ch) -> x_embedder -> pad to a multiple of 32 with x_pad_token
    caption features (Qwen3 layer 35) -> RMSNorm -> cap_proj -> pad with cap_pad_token
    2 noise-refiner blocks (image, modulated) ; 2 context-refiner blocks (caption, unmodulated)
    unified [image | caption] -> 30 modulated blocks -> final adaLN + linear -> 16-channel latent

Blocks are sandwich-normed: ``x + tanh(gate) * norm2(attn(norm1(x) * (1 + scale)))`` and the
same for the SwiGLU feed-forward; every norm is the diffusers ``RMSNorm`` (normalize in
fp32, cast to the weight dtype, then multiply — reproduced by :class:`ScaledRMSNorm`).
Rotary embeddings are 3-axis complex tables (theta 256) applied as a complex product in
fp32. Attention runs through :func:`joint_attention` (SDPA or the ragged FlashInfer
resource); batching happens only at identical padded lengths, so no key mask is needed.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.diffusion.attention import RaggedAttentionFn, joint_attention
from mstar.model.components.linear import FusedColumnLinear
from mstar.model.z_image.config import ZImageTransformerConfig

# Sinusoidal timestep feature width (diffusers ``TimestepEmbedder.frequency_embedding_size``).
TIMESTEP_FREQ_DIM = 256


class ScaledRMSNorm(nn.Module):
    """diffusers ``RMSNorm``: ``x * rsqrt(mean(x^2) + eps)`` with an fp32 variance, cast to
    the weight's (half) dtype, then ``* weight``. Differs from ``torch.nn.RMSNorm`` in the
    last bit under bf16, which the port has to reproduce."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        if self.weight.dtype in (torch.float16, torch.bfloat16):
            x = x.to(self.weight.dtype)
        return x * self.weight


def timestep_features(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """``[cos | sin]`` sinusoidal features in fp32 (diffusers ``TimestepEmbedder.timestep_embedding``)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ZImageRoPE:
    """3-axis rotary tables as complex64 ``[S, head_dim / 2]`` (diffusers ``RopeEmbedder``):
    per axis ``polar(1, outer(arange(axis_len), 1 / theta^(arange(0, d, 2) / d)))`` with fp64
    frequencies cast to fp32, gathered by the integer ids and concatenated over the axes."""

    def __init__(self, theta: float, axes_dims: tuple[int, ...], axes_lens: tuple[int, ...]):
        self.tables = []
        for dim, length in zip(axes_dims, axes_lens, strict=True):
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
            freqs = torch.outer(torch.arange(length, dtype=torch.float64), freqs).float()
            self.tables.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.cat([table[ids[:, i].cpu()] for i, table in enumerate(self.tables)], dim=-1)


def apply_rotary_complex(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """``x [B, S, H, D]`` rotated by complex tables ``[S, D/2]``, in fp32, cast back (diffusers
    ``ZSingleStreamAttnProcessor.apply_rotary_emb``)."""
    x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    out = torch.view_as_real(x_c * freqs_cis[None, :, None, :]).flatten(3)
    return out.type_as(x)


class ZImageAttention(nn.Module):
    def __init__(self, config: ZImageTransformerConfig):
        super().__init__()
        dim, self.head_dim = config.dim, config.head_dim
        self.qkv = FusedColumnLinear(dim, {"q": dim, "k": dim, "v": dim}, bias=False)
        self.q_norm = ScaledRMSNorm(self.head_dim, 1e-5) if config.qk_norm else None
        self.k_norm = ScaledRMSNorm(self.head_dim, 1e-5) if config.qk_norm else None
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, ragged: RaggedAttentionFn | None) -> torch.Tensor:
        q, k, v = (t.unflatten(-1, (-1, self.head_dim)) for t in self.qkv(x).chunk(3, dim=-1))
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        dtype = q.dtype
        q = apply_rotary_complex(q, freqs_cis).to(dtype)
        k = apply_rotary_complex(k, freqs_cis).to(dtype)
        out = joint_attention(q, k, v, ragged).flatten(2, 3).to(dtype)
        return self.out(out)


class ZImageFeedForward(nn.Module):
    """``w2(silu(w1 x) * w3 x)`` with ``w1`` and ``w3`` fused into one GEMM."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.w13 = FusedColumnLinear(dim, {"w1": hidden, "w3": hidden}, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w13(x).split(self.hidden, dim=-1)
        return self.w2(F.silu(gate) * up)


class ZImageBlock(nn.Module):
    def __init__(self, config: ZImageTransformerConfig, modulation: bool):
        super().__init__()
        dim, eps = config.dim, config.norm_eps
        self.attn = ZImageAttention(config)
        self.ff = ZImageFeedForward(dim, config.ffn_hidden)
        self.attn_norm1 = ScaledRMSNorm(dim, eps)
        self.attn_norm2 = ScaledRMSNorm(dim, eps)
        self.ffn_norm1 = ScaledRMSNorm(dim, eps)
        self.ffn_norm2 = ScaledRMSNorm(dim, eps)
        self.adaln = nn.Linear(config.adaln_dim, 4 * dim, bias=True) if modulation else None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: torch.Tensor | None,
        ragged: RaggedAttentionFn | None,
    ) -> torch.Tensor:
        if self.adaln is not None:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaln(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
            x = x + gate_msa * self.attn_norm2(self.attn(self.attn_norm1(x) * scale_msa, freqs_cis, ragged))
            return x + gate_mlp * self.ffn_norm2(self.ff(self.ffn_norm1(x) * scale_mlp))
        x = x + self.attn_norm2(self.attn(self.attn_norm1(x), freqs_cis, ragged))
        return x + self.ffn_norm2(self.ff(self.ffn_norm1(x)))


class ZImageDiT(nn.Module):
    """The Z-Image flow transformer over an already laid-out ``[image | caption]`` batch.

    The caller (the denoise submodule) owns the token layout: patchified image tokens
    padded to a multiple of 32, caption features padded likewise, the per-row caption pad
    mask, and the rotary tables for the unified sequence — all per-shape derived state.
    """

    def __init__(self, config: ZImageTransformerConfig):
        super().__init__()
        self.config = config
        dim = config.dim
        self.freq_dim = TIMESTEP_FREQ_DIM
        self.time_in = nn.Linear(self.freq_dim, config.t_mid_size, bias=True)
        self.time_out = nn.Linear(config.t_mid_size, config.adaln_dim, bias=True)
        self.x_embedder = nn.Linear(config.patch_dim, dim, bias=True)
        self.cap_norm = ScaledRMSNorm(config.cap_feat_dim, config.norm_eps)
        self.cap_proj = nn.Linear(config.cap_feat_dim, dim, bias=True)
        self.x_pad_token = nn.Parameter(torch.zeros(1, dim))
        self.cap_pad_token = nn.Parameter(torch.zeros(1, dim))
        self.noise_refiner = nn.ModuleList(ZImageBlock(config, True) for _ in range(config.n_refiner_layers))
        self.context_refiner = nn.ModuleList(ZImageBlock(config, False) for _ in range(config.n_refiner_layers))
        self.layers = nn.ModuleList(ZImageBlock(config, True) for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_mod = nn.Linear(config.adaln_dim, dim, bias=True)
        self.final_proj = nn.Linear(dim, config.patch_dim, bias=True)

    @property
    def dtype(self) -> torch.dtype:
        return self.x_embedder.weight.dtype

    def embed_time(self, t: torch.Tensor) -> torch.Tensor:
        """``adaln_input [B, 256]`` from ``t = 1 - sigma`` (fp32): scaled by ``t_scale``, sinusoidal
        features cast to the weight dtype, then the two-layer MLP."""
        features = timestep_features(t * self.config.t_scale, self.freq_dim).to(self.dtype)
        return self.time_out(F.silu(self.time_in(features)))

    def forward(
        self,
        image_tokens: torch.Tensor,
        caption: torch.Tensor,
        caption_pad_mask: torch.Tensor,
        image_pad_mask: torch.Tensor,
        t: torch.Tensor,
        image_freqs: torch.Tensor,
        caption_freqs: torch.Tensor,
        ragged: RaggedAttentionFn | None = None,
    ) -> torch.Tensor:
        """``image_tokens [B, Lx, patch_dim]`` (padded to a multiple of 32), ``caption [B, Lc,
        cap_feat_dim]`` (padded likewise), boolean pad masks ``[B, Lx]`` / ``[B, Lc]`` (True at
        pad positions), ``t [B]`` fp32, complex rotary tables ``[Lx, D/2]`` / ``[Lc, D/2]``.
        Returns the velocity for the image tokens ``[B, Lx, patch_dim]`` (pads included)."""
        adaln_input = self.embed_time(t).type_as(image_tokens)
        x = self.x_embedder(image_tokens)
        x = torch.where(image_pad_mask[..., None], self.x_pad_token.to(x.dtype), x)
        for block in self.noise_refiner:
            x = block(x, image_freqs, adaln_input, ragged)
        cap = self.cap_proj(self.cap_norm(caption))
        cap = torch.where(caption_pad_mask[..., None], self.cap_pad_token.to(cap.dtype), cap)
        for block in self.context_refiner:
            cap = block(cap, caption_freqs, None, ragged)
        num_image = x.shape[1]
        unified = torch.cat([x, cap], dim=1)
        freqs = torch.cat([image_freqs, caption_freqs], dim=0)
        for block in self.layers:
            unified = block(unified, freqs, adaln_input, ragged)
        scale = 1.0 + self.final_mod(F.silu(adaln_input))
        out = self.final_norm(unified) * scale.unsqueeze(1)
        return self.final_proj(out)[:, :num_image]


def patchify_image(latent: torch.Tensor, patch: int) -> torch.Tensor:
    """``[C, H, W] -> [(H/p)(W/p), p*p*C]`` in the reference's patch-vector order (rows, then
    columns inside a patch, channel fastest)."""
    channels, height, width = latent.shape
    x = latent.view(channels, height // patch, patch, width // patch, patch)
    return x.permute(1, 3, 2, 4, 0).reshape((height // patch) * (width // patch), patch * patch * channels)


def unpatchify_image(tokens: torch.Tensor, grid: tuple[int, int], patch: int, channels: int) -> torch.Tensor:
    """Inverse of :func:`patchify_image` for the first ``h*w`` tokens: ``-> [C, H, W]``."""
    h, w = grid
    x = tokens[: h * w].view(h, w, patch, patch, channels)
    return x.permute(4, 0, 2, 1, 3).reshape(channels, h * patch, w * patch)
