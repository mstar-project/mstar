"""ZImage DiT transformer for Ming-flash-omni-2.0 image generation (step 9b).

Native mstar port of vllm-omni's ``z_image/z_image_transformer.py`` +
Ming's ``ming_zimage_transformer.py`` subclass. The upstream module is built
on vllm's tensor-parallel linears (``QKVParallelLinear`` / ``MergedColumn`` /
``RowParallel``), a custom fused ``Attention``, vllm's ``RotaryEmbedding``,
and ``CachedTransformer`` — none of which belong in the pure-torch mstar
modeling tree. This reimplementation:

  * uses plain ``nn.Linear`` with the **unfused** parameter names the released
    checkpoint actually ships (``attention.to_q/to_k/to_v``,
    ``feed_forward.w1/w3``), so the state dict loads with a direct ``copy_`` —
    no stacked-param remap (same approach as the byt5 mapper port);
  * reimplements the interleaved (GPT-J / ``is_neox_style=False``) RoPE that
    vllm's ``RotaryEmbedding(is_neox_style=False)`` applies, the GLIDE/DiT
    ``timestep_embedding``, and FP32 ``RMSNorm`` exactly;
  * runs attention through ``F.scaled_dot_product_attention``.

Architecture (released ckpt): dim=3840, 30 main layers + 2 noise-refiner + 2
context-refiner blocks, 30 heads (head_dim=128), 16-channel latents, 3D axial
RoPE with axes_dims=(32,48,48) summing to the 128-wide head. Caption features
(byt5 + connector, 2560-dim) are embedded, refined, then concatenated with the
patch-embedded image tokens into one unified sequence for the main blocks.

NOTE — attention masking divergence: vllm-omni *computes* the pad mask but
leaves it unapplied in attention ("we don't support multi prompts now"). This
port applies it (additive ``-inf`` on padded keys) so padded cap/image tokens
cannot leak into real positions. For the dominant batch-size-1 text-to-image
path with sequences already a multiple of ``SEQ_MULTI_OF`` the two are
numerically identical; they only diverge when caption padding is non-zero,
where applying the mask is the correct behavior.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


# ============================================================
# Primitives (native equivalents of the vllm-omni helpers)
# ============================================================


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """GLIDE/DiT sinusoidal timestep embedding (cos-then-sin, log-spaced).

    Mirrors ``vllm_omni...timestep_embedding`` byte-for-byte so the adaLN
    conditioning matches the validated serving path.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class RMSNorm(nn.Module):
    """FP32 RMSNorm with a learnable scale (matches vllm-omni's forward_native)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight.to(torch.float32) * out
        return out.to(input_dtype)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """GPT-J style rotate: (-x_odd, x_even) interleaved back together."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb_interleaved(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply interleaved (is_neox_style=False) RoPE to ``[B, S, H, D]``.

    ``cos``/``sin`` are ``[B, S, D/2]`` (per-axis concatenated half-frequencies
    from :class:`RopeEmbedder`); each entry is duplicated to the adjacent pair
    to match the interleaved convention, broadcasting over the head axis.
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    # [B, S, D/2] -> [B, S, 1, D] with each freq duplicated to its pair.
    cos_r = cos[..., None, :].repeat_interleave(2, dim=-1)
    sin_r = sin[..., None, :].repeat_interleave(2, dim=-1)
    x_rot = x[..., :ro_dim]
    rotated = x_rot * cos_r + _rotate_half_interleaved(x_rot) * sin_r
    if ro_dim < x.shape[-1]:
        return torch.cat([rotated, x[..., ro_dim:]], dim=-1)
    return rotated


class RopeEmbedder:
    """Per-axis (3D axial) RoPE frequency table, matching vllm-omni's."""

    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: tuple[int, ...] = (16, 56, 56),
        axes_lens: tuple[int, ...] = (64, 128, 128),
    ) -> None:
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must match"
        self.cos_cached: list[torch.Tensor] | None = None
        self.sin_cached: list[torch.Tensor] | None = None

    @staticmethod
    def precompute_freqs(dim, end, theta: float = 256.0):
        cos_list, sin_list = [], []
        for d, e in zip(dim, end, strict=True):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
            timestep = torch.arange(e, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            cos_list.append(torch.cos(freqs))
            sin_list.append(torch.sin(freqs))
        return cos_list, sin_list

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device
        if self.cos_cached is None:
            self.cos_cached, self.sin_cached = self.precompute_freqs(self.axes_dims, self.axes_lens, theta=self.theta)
            self.cos_cached = [c.to(device) for c in self.cos_cached]
            self.sin_cached = [s.to(device) for s in self.sin_cached]
        elif self.cos_cached[0].device != device:
            self.cos_cached = [c.to(device) for c in self.cos_cached]
            self.sin_cached = [s.to(device) for s in self.sin_cached]

        cos_result, sin_result = [], []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            cos_result.append(self.cos_cached[i][index])
            sin_result.append(self.sin_cached[i][index])
        return torch.cat(cos_result, dim=-1), torch.cat(sin_result, dim=-1)


# ============================================================
# Modules
# ============================================================


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size: int, mid_size: int | None = None, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].bias.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        return self.mlp(t_freq)


class ZImageAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        # Unfused projections — the checkpoint ships to_q/to_k/to_v separately.
        self.to_q = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        self.scale = 1.0 / (self.head_dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        query = self.to_q(hidden_states).unflatten(-1, (self.num_heads, self.head_dim))
        key = self.to_k(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))
        value = self.to_v(hidden_states).unflatten(-1, (self.num_kv_heads, self.head_dim))

        query = self.norm_q(query)
        key = self.norm_k(key)

        query = apply_rotary_emb_interleaved(query, cos, sin)
        key = apply_rotary_emb_interleaved(key, cos, sin)
        dtype = query.dtype

        # [B, S, H, D] -> [B, H, S, D] for SDPA.
        q = query.transpose(1, 2)
        k = key.transpose(1, 2).to(dtype)
        v = value.transpose(1, 2).to(dtype)

        attn_bias = None
        if attention_mask is not None:
            # bool [B, S] keep-mask -> additive [B, 1, 1, S].
            if attention_mask.dtype == torch.bool:
                attn_bias = torch.zeros(bsz, 1, 1, seqlen, dtype=dtype, device=q.device)
                attn_bias = attn_bias.masked_fill(~attention_mask[:, None, None, :], float("-inf"))
            else:
                attn_bias = attention_mask

        enable_gqa = self.num_kv_heads != self.num_heads
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, scale=self.scale, enable_gqa=enable_gqa
        )
        out = out.transpose(1, 2).flatten(2, 3).to(dtype)
        return self.to_out[0](out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        # Unfused SwiGLU gate/up (checkpoint ships w1 + w3 separately).
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.attention = ZImageAttention(dim, n_heads, n_kv_heads, eps=1e-5)
        self.feed_forward = FeedForward(dim, hidden_dim=int(dim / 3 * 8))
        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)
        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = (
                self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            )
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(self.attention_norm1(x) * scale_msa, attn_mask, cos, sin)
            x = x + gate_msa * self.attention_norm2(attn_out)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn_out = self.attention(self.attention_norm1(x), attn_mask, cos, sin)
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        return self.linear(x)


class ZImageTransformer2DModel(nn.Module):
    """Native Z-Image DiT (pure torch). See module docstring for divergences."""

    def __init__(
        self,
        all_patch_size: tuple[int, ...] = (2,),
        all_f_patch_size: tuple[int, ...] = (1,),
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        n_kv_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
        axes_lens: tuple[int, ...] = (1024, 512, 512),
    ) -> None:
        super().__init__()
        assert len(all_patch_size) == len(all_f_patch_size)
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = tuple(all_patch_size)
        self.all_f_patch_size = tuple(all_f_patch_size)
        self.dim = dim
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.t_scale = t_scale

        all_x_embedder = {}
        all_final_layer = {}
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size, strict=True):
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = nn.Linear(
                f_patch_size * patch_size * patch_size * in_channels, dim, bias=True
            )
            all_final_layer[f"{patch_size}-{f_patch_size}"] = FinalLayer(
                dim, patch_size * patch_size * f_patch_size * self.out_channels
            )
        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(1000 + i, dim, n_heads, n_kv_heads, norm_eps, modulation=True)
                for i in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(i, dim, n_heads, n_kv_heads, norm_eps, modulation=False)
                for i in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )
        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.layers = nn.ModuleList(
            [ZImageTransformerBlock(i, dim, n_heads, n_kv_heads, norm_eps, modulation=True) for i in range(n_layers)]
        )
        self.axes_dims = tuple(axes_dims)
        self.axes_lens = tuple(axes_lens)
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=self.axes_dims, axes_lens=self.axes_lens)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def unpatchify(self, x: list[torch.Tensor], size: list[tuple], patch_size: int, f_patch_size: int):
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        for i in range(bsz):
            Fr, H, W = size[i]
            ori_len = (Fr // pF) * (H // pH) * (W // pW)
            x[i] = (
                x[i][:ori_len]
                .view(Fr // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, Fr, H, W)
            )
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [
            torch.arange(x0, x0 + span, dtype=torch.int32, device=device)
            for x0, span in zip(start, size, strict=True)
        ]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_image: list[torch.Tensor],
        all_cap_feats: list[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out, all_image_size, all_image_pos_ids, all_image_pad_mask = [], [], [], []
        all_cap_pos_ids, all_cap_pad_mask, all_cap_feats_out = [], [], []

        for image, cap_feat in zip(all_image, all_cap_feats, strict=True):
            # ---- Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1), start=(1, 0, 0), device=device
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            all_cap_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                    ]
                )
            )
            all_cap_feats_out.append(torch.cat([cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)], dim=0))

            # ---- Image
            C, Fr, H, W = image.size()
            all_image_size.append((Fr, H, W))
            F_tokens, H_tokens, W_tokens = Fr // pF, H // pH, W // pW
            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF
            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device)
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            all_image_pos_ids.append(torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0))
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ]
                )
            )
            all_image_out.append(torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0))

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def _unified_prepare(self, x, x_cos, x_sin, cap_feats, cap_cos, cap_sin, x_item_seqlens, cap_item_seqlens):
        bsz = x.shape[0]
        device = x.device
        unified, unified_cos, unified_sin = [], [], []
        for i in range(bsz):
            x_len, cap_len = x_item_seqlens[i], cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_cos.append(torch.cat([x_cos[i][:x_len], cap_cos[i][:cap_len]]))
            unified_sin.append(torch.cat([x_sin[i][:x_len], cap_sin[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens, strict=True)]
        unified_max = max(unified_item_seqlens)
        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_cos = pad_sequence(unified_cos, batch_first=True, padding_value=0.0)
        unified_sin = pad_sequence(unified_sin, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1
        return unified, unified_cos, unified_sin, unified_attn_mask

    def forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size
        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # ---- x embed + noise refine
        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max = max(x_item_seqlens)
        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)
        adaln_input = t.type_as(x)
        x_pad_mask = torch.cat(x_inner_pad_mask)
        x = torch.where(x_pad_mask.unsqueeze(1).expand_as(x), self.x_pad_token.expand(x.shape[0], -1), x)
        x = list(x.split(x_item_seqlens, dim=0))
        x_cos, x_sin = self.rope_embedder(torch.cat(x_pos_ids, dim=0))
        x_cos = list(x_cos.split(x_item_seqlens, dim=0))
        x_sin = list(x_sin.split(x_item_seqlens, dim=0))
        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_cos = pad_sequence(x_cos, batch_first=True, padding_value=0.0)
        x_sin = pad_sequence(x_sin, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1
        for layer in self.noise_refiner:
            x = layer(x, x_attn_mask, x_cos, x_sin, adaln_input)

        # ---- cap embed + context refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max = max(cap_item_seqlens)
        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_pad_mask = torch.cat(cap_inner_pad_mask)
        cap_feats = torch.where(
            cap_pad_mask.unsqueeze(1).expand_as(cap_feats),
            self.cap_pad_token.expand(cap_feats.shape[0], -1),
            cap_feats,
        )
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_cos, cap_sin = self.rope_embedder(torch.cat(cap_pos_ids, dim=0))
        cap_cos = list(cap_cos.split(cap_item_seqlens, dim=0))
        cap_sin = list(cap_sin.split(cap_item_seqlens, dim=0))
        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_cos = pad_sequence(cap_cos, batch_first=True, padding_value=0.0)
        cap_sin = pad_sequence(cap_sin, batch_first=True, padding_value=0.0)
        cap_attn_mask = torch.zeros((bsz, cap_max), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_attn_mask, cap_cos, cap_sin)

        # ---- unify + main blocks
        unified, unified_cos, unified_sin, unified_attn_mask = self._unified_prepare(
            x, x_cos, x_sin, cap_feats, cap_cos, cap_sin, x_item_seqlens, cap_item_seqlens
        )
        for layer in self.layers:
            unified = layer(unified, unified_attn_mask, unified_cos, unified_sin, adaln_input)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        return self.unpatchify(unified, x_size, patch_size, f_patch_size), {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Direct state-dict load — our unfused layout matches the checkpoint.

        Unlike vllm-omni (which fuses to_qkv / w13 and remaps), we keep
        to_q/to_k/to_v + w1/w3 separate, so the released DiT weights copy in
        verbatim. Returns the set of param names covered so callers can assert
        completeness.
        """
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            if param.shape != loaded_weight.shape:
                raise ValueError(
                    f"Shape mismatch loading ZImage DiT weight {name}: "
                    f"param {tuple(param.shape)} vs checkpoint {tuple(loaded_weight.shape)}"
                )
            with torch.no_grad():
                param.copy_(loaded_weight)
            loaded_params.add(name)
        return loaded_params


class MingZImageTransformer2DModel(ZImageTransformer2DModel):
    """ZImage DiT with Ming's reference-latent (img2img) support.

    Ming's img2img path concatenates a VAE-encoded reference latent along the
    frame axis before patchification, then drops the reference-frame prediction
    from the unpatchified output. ``ref_latent`` is threaded through as an
    explicit forward arg (the upstream reads it from a global forward-context;
    mstar passes it directly).
    """

    def forward(
        self,
        x: list[torch.Tensor],
        t: torch.Tensor,
        cap_feats: list[torch.Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
        ref_latent: list[torch.Tensor] | None = None,
    ):
        self._dropping_ref = ref_latent is not None
        if ref_latent is not None:
            per_item = ref_latent[0].unsqueeze(1).to(dtype=x[0].dtype, device=x[0].device)  # [C, 1, H, W]
            x = [torch.cat([img, per_item], dim=1) for img in x]
        return super().forward(x, t, cap_feats, patch_size=patch_size, f_patch_size=f_patch_size)

    def unpatchify(self, x, size, patch_size, f_patch_size):
        out = super().unpatchify(x, size, patch_size, f_patch_size)
        if getattr(self, "_dropping_ref", False):
            # Drop the reference frame (F==2 -> keep first frame only).
            return [t[:, :1, :, :] for t in out]
        return out


__all__ = ["ZImageTransformer2DModel", "MingZImageTransformer2DModel"]
