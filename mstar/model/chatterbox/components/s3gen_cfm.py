"""S3Gen's flow-matching mel decoder: the causal 1-D UNet estimator and the
Euler solvers around it.

Reference: ``chatterbox/models/s3gen/decoder.py`` (``ConditionalDecoder``),
``chatterbox/models/s3gen/flow_matching.py`` (``CausalConditionalCFM``),
``chatterbox/models/s3gen/matcha/{decoder,transformer}.py`` and the diffusers
``BasicTransformerBlock``/``Attention`` semantics those rely on (LayerNorm
pre-norm, unbiased q/k/v, biased output projection, exact GELU feed-forward).

Two solvers: ``solve`` is the standard model's classifier-free-guided Euler
integration over a cosine time grid (conditional and unconditional halves
run as one 2B batch); ``solve_meanflow`` is Turbo's distilled mean-flow model
(linear grid, no guidance, the estimator sees the interval end ``r``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.chatterbox.config import S3GenCFMConfig, S3GenEstimatorConfig


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, scale: float = 1000.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device).float() * -emb)
        emb = self.scale * t.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.silu(self.linear_1(sample)))


class CausalConv1d(nn.Conv1d):
    """Convolution that only looks left (``kernel_size - 1`` zeros before the input)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__(in_channels, out_channels, kernel_size, padding=0)
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, self.causal_padding))


class _Transpose(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)


class CausalBlock1D(nn.Module):
    """Causal conv -> LayerNorm over channels -> Mish, masked on both sides."""

    def __init__(self, dim: int, dim_out: int):
        super().__init__()
        self.block = nn.Sequential(
            CausalConv1d(dim, dim_out, 3), _Transpose(), nn.LayerNorm(dim_out), _Transpose(), nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.block(x * mask) * mask


class CausalResnetBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, mask)
        h = h + self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class _SelfAttention(nn.Module):
    """Multi-head self-attention in the diffusers layout (``to_q/k/v`` unbiased, ``to_out.0`` biased)."""

    def __init__(self, dim: int, heads: int, dim_head: int):
        super().__init__()
        inner = heads * dim_head
        self.heads = heads
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(inner, dim, bias=True), nn.Dropout(0.0)])

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.to_q(x).view(b, t, self.heads, -1).transpose(1, 2)
        k = self.to_k(x).view(b, t, self.heads, -1).transpose(1, 2)
        v = self.to_v(x).view(b, t, self.heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, t, -1).to(q.dtype)
        return self.to_out[0](out)


class _GELUProj(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm self-attention + GELU feed-forward (diffusers ``BasicTransformerBlock``, layer_norm)."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = _SelfAttention(dim, num_heads, head_dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Module()
        self.ff.net = nn.ModuleList([_GELUProj(dim, dim * ff_mult), nn.Dropout(0.0), nn.Linear(dim * ff_mult, dim)])

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        x = self.attn1(self.norm1(x), attn_bias) + x
        h = self.norm3(x)
        for module in self.ff.net:
            h = module(h)
        return h + x


def _mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``[B, 1, T]`` float mask -> additive ``[B, 1, 1, T]`` key bias (0 keep, -1e10 drop)."""
    return ((1.0 - mask.to(dtype)) * -1.0e10).unsqueeze(1)


class ConditionalDecoder(nn.Module):
    """Velocity estimator ``v(x_t, t | mu, spk, cond)`` for 80-bin mels.

    Inputs are concatenated on the channel axis in the reference order
    ``[x, mu, spk (broadcast over time), cond]``; one resolution level, so the
    "down"/"up" samplers are plain causal convolutions.
    """

    def __init__(self, config: S3GenEstimatorConfig, meanflow: bool = False):
        super().__init__()
        self.config = config
        self.meanflow = meanflow
        channels = tuple(config.channels)
        time_embed_dim = channels[0] * 4
        self.time_embeddings = SinusoidalPosEmb(config.in_channels)
        self.time_mlp = TimestepEmbedding(config.in_channels, time_embed_dim)

        def blocks(dim: int) -> nn.ModuleList:
            return nn.ModuleList([
                TransformerBlock(dim, config.num_heads, config.attention_head_dim)
                for _ in range(config.n_blocks)
            ])

        self.down_blocks = nn.ModuleList()
        output_channel = config.in_channels
        for i, ch in enumerate(channels):
            input_channel, output_channel = output_channel, ch
            is_last = i == len(channels) - 1
            downsample = CausalConv1d(ch, ch, 3) if is_last else nn.Conv1d(ch, ch, 3, 2, 1)
            self.down_blocks.append(nn.ModuleList([
                CausalResnetBlock1D(input_channel, output_channel, time_embed_dim), blocks(output_channel), downsample,
            ]))
        self.mid_blocks = nn.ModuleList([
            nn.ModuleList([CausalResnetBlock1D(channels[-1], channels[-1], time_embed_dim), blocks(channels[-1])])
            for _ in range(config.num_mid_blocks)
        ])
        up_channels = channels[::-1] + (channels[0],)
        self.up_blocks = nn.ModuleList()
        for i in range(len(up_channels) - 1):
            input_channel, output_channel = up_channels[i] * 2, up_channels[i + 1]
            is_last = i == len(up_channels) - 2
            upsample = (
                CausalConv1d(output_channel, output_channel, 3) if is_last
                else nn.ConvTranspose1d(output_channel, output_channel, 4, 2, 1)
            )
            self.up_blocks.append(nn.ModuleList([
                CausalResnetBlock1D(input_channel, output_channel, time_embed_dim), blocks(output_channel), upsample,
            ]))
        self.final_block = CausalBlock1D(up_channels[-1], up_channels[-1])
        self.final_proj = nn.Conv1d(up_channels[-1], config.out_channels, 1)
        # Mean-flow: mixes the embeddings of the interval start ``t`` and end ``r``.
        self.time_embed_mixer = nn.Linear(time_embed_dim * 2, time_embed_dim, bias=False) if meanflow else None

    def embed_time(self, t: torch.Tensor, r: torch.Tensor | None) -> torch.Tensor:
        emb = self.time_mlp(self.time_embeddings(t).to(t.dtype))
        if self.time_embed_mixer is not None:
            if r is None:
                raise ValueError("the mean-flow estimator needs the interval end r")
            emb_r = self.time_mlp(self.time_embeddings(r).to(emb.dtype))
            emb = self.time_embed_mixer(torch.cat([emb, emb_r], dim=1))
        return emb

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        r: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``x, mu, cond``: ``[B, 80, T]``; ``mask``: ``[B, 1, T]``; ``t, r``: ``[B]`` or ``[1]``;
        ``spks``: ``[B, 80]``."""
        temb = self.embed_time(t, r)
        x = torch.cat([x, mu, spks.unsqueeze(-1).expand(-1, -1, x.shape[-1]), cond], dim=1)
        attn_bias = _mask_to_bias(mask, x.dtype)

        hiddens = []
        for resnet, transformer_blocks, downsample in self.down_blocks:
            x = resnet(x, mask, temb)
            x = x.transpose(1, 2).contiguous()
            for block in transformer_blocks:
                x = block(x, attn_bias)
            x = x.transpose(1, 2).contiguous()
            hiddens.append(x)
            x = downsample(x * mask)

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask, temb)
            x = x.transpose(1, 2).contiguous()
            for block in transformer_blocks:
                x = block(x, attn_bias)
            x = x.transpose(1, 2).contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            skip = hiddens.pop()
            x = torch.cat([x[:, :, : skip.shape[-1]], skip], dim=1)
            x = resnet(x, mask, temb)
            x = x.transpose(1, 2).contiguous()
            for block in transformer_blocks:
                x = block(x, attn_bias)
            x = x.transpose(1, 2).contiguous()
            x = upsample(x * mask)

        x = self.final_block(x, mask)
        return self.final_proj(x * mask) * mask


class CausalConditionalCFM(nn.Module):
    """Euler integration of the estimator from noise to mel."""

    def __init__(self, config: S3GenCFMConfig, estimator: ConditionalDecoder):
        super().__init__()
        self.config = config
        self.estimator = estimator

    def time_grid(self, n_timesteps: int, device: torch.device, dtype: torch.dtype, meanflow: bool) -> torch.Tensor:
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)
        if not meanflow and self.config.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return t_span

    def solve(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        noise: torch.Tensor,
        n_timesteps: int,
        cfg_rate: float | None = None,
    ) -> torch.Tensor:
        """Classifier-free-guided Euler solve; the unconditional half sees zero ``mu``/``spks``/``cond``."""
        rate = self.config.inference_cfg_rate if cfg_rate is None else cfg_rate
        x = noise
        b, _, t_len = mu.shape
        t_span = self.time_grid(n_timesteps, mu.device, mu.dtype, meanflow=False)
        mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
        spks_in = torch.cat([spks, torch.zeros_like(spks)], dim=0)
        cond_in = torch.cat([cond, torch.zeros_like(cond)], dim=0)
        mask_in = torch.cat([mask, mask], dim=0)
        for t, r in zip(t_span[:-1], t_span[1:], strict=True):
            t_in = t.expand(2 * b)
            dxdt = self.estimator(torch.cat([x, x], dim=0), mask_in, mu_in, t_in, spks_in, cond_in)
            dxdt, cfg_dxdt = dxdt.split([b, b], dim=0)
            dxdt = (1.0 + rate) * dxdt - rate * cfg_dxdt
            x = x + (r - t) * dxdt
        return x

    def solve_meanflow(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        noise: torch.Tensor,
        n_timesteps: int,
    ) -> torch.Tensor:
        """Mean-flow Euler solve (distilled with guidance baked in, so no CFG batch)."""
        x = noise
        b = mu.shape[0]
        t_span = self.time_grid(n_timesteps, mu.device, mu.dtype, meanflow=True)
        for t, r in zip(t_span[:-1], t_span[1:], strict=True):
            dxdt = self.estimator(x, mask, mu, t.expand(b), spks, cond, r=r.expand(b))
            x = x + (r - t) * dxdt
        return x
