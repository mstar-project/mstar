"""Native FLUX.2 [klein] rectified-flow transformer (exact port of diffusers 0.39/0.40
``Flux2Transformer2DModel``, restricted to what the klein checkpoints configure).

Layout of one denoise step over a request's tokens ``[txt (T) | img (L) | ref (R)]``:

    timestep -> sinusoidal(256) -> MLP -> temb            (bf16, like the reference)
    temb -> 3 modulation linears (double img / double txt / single), silu first
    img tokens -> img_in (128 -> D);   txt tokens -> txt_in (3 x Qwen3 hidden -> D)
    N double-stream blocks: separate img / txt streams, one joint attention over [txt | img]
    M single-stream blocks: parallel attention + SwiGLU over the concatenated stream
    drop txt tokens -> AdaLN (norm_out) -> proj_out (D -> 128)

Numerics contract: bf16 weights everywhere (the checkpoint has no fp32 islands);
the sinusoidal embedding and RoPE tables are fp32, rotary application and the
final cast follow the reference op-for-op. Attention goes through
:func:`mstar.model.components.diffusion.attention.joint_attention` — SDPA (the
reference kernel) or the engine's ragged FlashInfer resource when the submodule
binds one. RoPE tables are per-shape derived state supplied by the caller.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.diffusion.attention import RaggedAttentionFn, joint_attention
from mstar.model.components.diffusion.rope import apply_rotary_interleaved
from mstar.model.components.linear import FusedColumnLinear
from mstar.model.flux2_klein.config import Flux2TransformerConfig


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """diffusers ``get_timestep_embedding(flip_sin_to_cos=True, downscale_freq_shift=0)``:
    fp32 math from ``timesteps.float()``, ``[N, dim]`` fp32 laid out ``[cos | sin]``."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


class TimestepEmbedder(nn.Module):
    """``linear_in -> silu -> linear_out`` over the sinusoidal features (diffusers ``TimestepEmbedding``)."""

    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.linear_in = nn.Linear(in_channels, dim, bias=False)
        self.linear_out = nn.Linear(dim, dim, bias=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear_out(F.silu(self.linear_in(features)))


class SwiGLUFeedForward(nn.Module):
    """``linear_out(silu(a) * b)`` with ``[a | b] = linear_in(x)`` fused in one GEMM (diffusers
    ``Flux2FeedForward`` + ``Flux2SwiGLU``)."""

    def __init__(self, dim: int, inner_dim: int):
        super().__init__()
        self.inner_dim = inner_dim
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.linear_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.linear_in(x).split(self.inner_dim, dim=-1)
        return self.linear_out(F.silu(gate) * up)


def _modulate(norm: nn.LayerNorm, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """``(1 + scale) * LN(x) + shift`` with ``[B, 1, D]`` modulation broadcast over tokens."""
    return (1 + scale) * norm(x) + shift


def _heads(x: torch.Tensor, head_dim: int) -> torch.Tensor:
    return x.unflatten(-1, (-1, head_dim))


class Flux2JointAttention(nn.Module):
    """Double-stream attention: img and txt have their own q/k/v projections and per-head
    q/k RMSNorms; the two streams are concatenated ``[txt | img]`` for one joint attention
    and split again for their own output projections."""

    def __init__(self, dim: int, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        inner = num_heads * head_dim
        self.head_dim = head_dim
        # One GEMM per stream for q, k, v; the checkpoint's separate to_q/to_k/to_v (and
        # add_{q,k,v}_proj) load into the shards by name through the loader's stacked rules.
        self.img_qkv = FusedColumnLinear(dim, {"q": inner, "k": inner, "v": inner}, bias=False)
        self.txt_qkv = FusedColumnLinear(dim, {"q": inner, "k": inner, "v": inner}, bias=False)
        # torch's own RMSNorm: the reference op, bit for bit.
        self.img_q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.img_k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.txt_q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.txt_k_norm = nn.RMSNorm(head_dim, eps=eps)
        self.img_out = nn.Linear(inner, dim, bias=False)
        self.txt_out = nn.Linear(inner, dim, bias=False)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        ragged: RaggedAttentionFn | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_txt = txt.shape[1]
        q_i, k_i, v_i = self.img_qkv(img).chunk(3, dim=-1)
        q_t, k_t, v_t = self.txt_qkv(txt).chunk(3, dim=-1)
        q_i, k_i, v_i = _heads(q_i, self.head_dim), _heads(k_i, self.head_dim), _heads(v_i, self.head_dim)
        q_t, k_t, v_t = _heads(q_t, self.head_dim), _heads(k_t, self.head_dim), _heads(v_t, self.head_dim)
        q = torch.cat([self.txt_q_norm(q_t), self.img_q_norm(q_i)], dim=1)
        k = torch.cat([self.txt_k_norm(k_t), self.img_k_norm(k_i)], dim=1)
        v = torch.cat([v_t, v_i], dim=1)
        cos, sin = rope
        q = apply_rotary_interleaved(q, cos, sin)
        k = apply_rotary_interleaved(k, cos, sin)
        out = joint_attention(q, k, v, ragged).flatten(2, 3).to(q.dtype)
        txt_out, img_out = out[:, :num_txt], out[:, num_txt:]
        return self.img_out(img_out), self.txt_out(txt_out)


class Flux2DoubleBlock(nn.Module):
    """One double-stream block (diffusers ``Flux2TransformerBlock``)."""

    def __init__(self, config: Flux2TransformerConfig):
        super().__init__()
        dim, eps = config.hidden_size, config.eps
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Flux2JointAttention(dim, config.num_attention_heads, config.attention_head_dim, eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_ff = SwiGLUFeedForward(dim, config.mlp_hidden_dim)
        self.txt_ff = SwiGLUFeedForward(dim, config.mlp_hidden_dim)

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        mod_img: tuple[torch.Tensor, ...],
        mod_txt: tuple[torch.Tensor, ...],
        rope: tuple[torch.Tensor, torch.Tensor],
        ragged: RaggedAttentionFn | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod_img
        t_shift_a, t_scale_a, t_gate_a, t_shift_m, t_scale_m, t_gate_m = mod_txt
        img_attn, txt_attn = self.attn(
            _modulate(self.img_norm1, img, shift_a, scale_a),
            _modulate(self.txt_norm1, txt, t_shift_a, t_scale_a),
            rope, ragged,
        )
        img = img + gate_a * img_attn
        img = img + gate_m * self.img_ff(self.img_norm2(img) * (1 + scale_m) + shift_m)
        txt = txt + t_gate_a * txt_attn
        txt = txt + t_gate_m * self.txt_ff(self.txt_norm2(txt) * (1 + t_scale_m) + t_shift_m)
        return img, txt


class Flux2SingleBlock(nn.Module):
    """One single-stream "parallel" block (diffusers ``Flux2SingleTransformerBlock``):
    one fused input GEMM yields q, k, v and the SwiGLU input; one fused output GEMM
    consumes ``[attn | mlp]``."""

    def __init__(self, config: Flux2TransformerConfig):
        super().__init__()
        dim, eps = config.hidden_size, config.eps
        self.head_dim = config.attention_head_dim
        self.inner_dim = config.num_attention_heads * config.attention_head_dim
        self.mlp_hidden_dim = config.mlp_hidden_dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.qkv_mlp = nn.Linear(dim, 3 * self.inner_dim + 2 * self.mlp_hidden_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.out = nn.Linear(self.inner_dim + self.mlp_hidden_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mod: tuple[torch.Tensor, ...],
        rope: tuple[torch.Tensor, torch.Tensor],
        ragged: RaggedAttentionFn | None,
    ) -> torch.Tensor:
        shift, scale, gate = mod
        proj = self.qkv_mlp(_modulate(self.norm, x, shift, scale))
        qkv, mlp = proj.split([3 * self.inner_dim, 2 * self.mlp_hidden_dim], dim=-1)
        q, k, v = (_heads(t, self.head_dim) for t in qkv.chunk(3, dim=-1))
        cos, sin = rope
        q = apply_rotary_interleaved(self.q_norm(q), cos, sin)
        k = apply_rotary_interleaved(self.k_norm(k), cos, sin)
        attn = joint_attention(q, k, v, ragged).flatten(2, 3).to(q.dtype)
        gate_in, up = mlp.split(self.mlp_hidden_dim, dim=-1)
        out = self.out(torch.cat([attn, F.silu(gate_in) * up], dim=-1))
        return x + gate * out


class Flux2DiT(nn.Module):
    """The klein flow transformer. Built on the meta device and materialized by
    ``weight_loader.build_transformer``."""

    def __init__(self, config: Flux2TransformerConfig):
        super().__init__()
        self.config = config
        dim = config.hidden_size
        self.time_embed = TimestepEmbedder(config.timestep_guidance_channels, dim)
        self.guidance_embed = (
            TimestepEmbedder(config.timestep_guidance_channels, dim) if config.guidance_embeds else None
        )
        self.mod_double_img = nn.Linear(dim, 6 * dim, bias=False)
        self.mod_double_txt = nn.Linear(dim, 6 * dim, bias=False)
        self.mod_single = nn.Linear(dim, 3 * dim, bias=False)
        self.img_in = nn.Linear(config.in_channels, dim, bias=False)
        self.txt_in = nn.Linear(config.joint_attention_dim, dim, bias=False)
        self.double_blocks = nn.ModuleList(Flux2DoubleBlock(config) for _ in range(config.num_layers))
        self.single_blocks = nn.ModuleList(Flux2SingleBlock(config) for _ in range(config.num_single_layers))
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=config.eps)
        self.norm_out_mod = nn.Linear(dim, 2 * dim, bias=False)
        self.proj_out = nn.Linear(dim, config.patch_size * config.patch_size * config.out_channels, bias=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.img_in.weight.dtype

    def embed_time(self, timestep: torch.Tensor, guidance: torch.Tensor | None) -> torch.Tensor:
        """``temb [B, D]`` in the weights' dtype from the pipeline's ``timestep`` (sigma in
        the latents' dtype; the reference scales it by 1000 in that dtype first)."""
        t = timestep.to(self.dtype) * 1000
        temb = self.time_embed(sinusoidal_timestep_embedding(t, self.config.timestep_guidance_channels).to(t.dtype))
        if guidance is not None and self.guidance_embed is not None:
            g = guidance.to(self.dtype) * 1000
            temb = temb + self.guidance_embed(
                sinusoidal_timestep_embedding(g, self.config.timestep_guidance_channels).to(g.dtype)
            )
        return temb

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        timestep: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        guidance: torch.Tensor | None = None,
        ragged: RaggedAttentionFn | None = None,
    ) -> torch.Tensor:
        """``img [B, L(+R), in_channels]`` packed latent tokens (bf16), ``txt [B, T,
        joint_attention_dim]``, ``timestep [B]`` (sigma, latents' dtype), ``rope`` the fp32
        ``(cos, sin)`` tables over the joint ``[txt | img]`` layout. Returns the velocity
        ``[B, L(+R), out_channels]``; the caller drops reference-token rows."""
        temb = self.embed_time(timestep, guidance)
        mod_act = F.silu(temb)
        mod_img = tuple(m.unsqueeze(1) for m in self.mod_double_img(mod_act).chunk(6, dim=-1))
        mod_txt = tuple(m.unsqueeze(1) for m in self.mod_double_txt(mod_act).chunk(6, dim=-1))
        mod_single = tuple(m.unsqueeze(1) for m in self.mod_single(mod_act).chunk(3, dim=-1))

        num_txt = txt.shape[1]
        img = self.img_in(img)
        txt = self.txt_in(txt)
        for block in self.double_blocks:
            img, txt = block(img, txt, mod_img, mod_txt, rope, ragged)
        x = torch.cat([txt, img], dim=1)
        for block in self.single_blocks:
            x = block(x, mod_single, rope, ragged)
        x = x[:, num_txt:]
        scale, shift = self.norm_out_mod(mod_act).chunk(2, dim=1)
        x = self.norm_out(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return self.proj_out(x)
