"""MiniCPM-o's vision tower: a navit SigLIP over packed image slices, then a
one-layer perceiver resampler that turns each slice into 64 LLM-width tokens.

Ported from the checkpoint's ``modeling_navit_siglip.py`` and the
``Resampler`` in ``modeling_minicpmo.py`` (Apache-2.0).

An image arrives as slices (the source image plus up to ``max_slice_nums``
crops), each an ``(h, w)`` grid of 14x14 patches. Every slice attends within
itself, so a forward packs all slices of all requests and the ragged
attention resource keeps them apart; the resampler's 64 queries per slice
cross-attend that slice's patches through the ragged cross-attention resource.

Grid arithmetic (position buckets, sincos gather indices, segment lengths)
is host-side, done by ``slice_layout`` in the submodule's ``prepare_inputs``,
so the forward is shape-only in the packed patch count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.resources.attn.ragged.config import cross_label
from mstar.model.components.linear import FusedColumnLinear
from mstar.model.minicpm_o.config import (
    PATCHES,
    QUERIES,
    RESAMPLER_ATTN,
    VISION_ATTN,
    ResamplerConfig,
    VisionConfig,
)

# ----------------------------------------------------------------------------
# Host-side grid helpers
# ----------------------------------------------------------------------------


def navit_position_ids(h: int, w: int, num_patches_per_side: int) -> torch.Tensor:
    """Each patch's row in the square position table: its fractional
    coordinate bucketized onto the table's grid, not interpolated.

    The same float32 ops as upstream, on the CPU, so the buckets match
    exactly at their boundaries.
    """
    side = num_patches_per_side
    boundaries = torch.arange(1 / side, 1.0, 1 / side)
    frac_h = torch.arange(0, 1 - 1e-6, 1 / h)
    frac_w = torch.arange(0, 1 - 1e-6, 1 / w)
    bucket_h = torch.bucketize(frac_h, boundaries, right=True)
    bucket_w = torch.bucketize(frac_w, boundaries, right=True)
    return (bucket_h[:, None] * side + bucket_w).flatten()


def sincos_omega(embed_dim: int) -> torch.Tensor:
    """The resampler's 1D sincos frequencies for one axis (a quarter of the
    width each for sin and cos), float32 as upstream's numpy builds them."""
    dim = embed_dim // 2
    omega = np.arange(dim // 2, dtype=np.float32)
    omega /= dim / 2.0
    return torch.from_numpy(1.0 / 10000 ** omega)


@dataclass(frozen=True)
class SliceLayout:
    """Where one packed forward's patches come from; all host tensors."""
    # patches per slice, in pack order
    seq_lengths: tuple[int, ...]
    # [total_patches] row of the navit position table per patch
    position_ids: torch.Tensor
    # [total_patches, 2] float (row, col) of each patch in its slice, for
    # the resampler's 2D sincos keys
    grid_coords: torch.Tensor


def slice_layout(tgt_sizes: list[tuple[int, int]], vision: VisionConfig) -> SliceLayout:
    """``tgt_sizes`` is each slice's ``(h, w)`` in patches."""
    pos, coords = [], []
    for h, w in tgt_sizes:
        pos.append(navit_position_ids(h, w, vision.num_patches_per_side))
        rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        coords.append(torch.stack([rows.flatten(), cols.flatten()], dim=-1))
    return SliceLayout(
        seq_lengths=tuple(h * w for h, w in tgt_sizes),
        position_ids=torch.cat(pos) if pos else torch.zeros(0, dtype=torch.long),
        grid_coords=(torch.cat(coords) if coords else torch.zeros(0, 2, dtype=torch.long)).float(),
    )


# ----------------------------------------------------------------------------
# SigLIP
# ----------------------------------------------------------------------------


class SiglipAttention(nn.Module):
    """Bidirectional attention within each slice, through the ragged resource."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        d = config.hidden_size
        self.qkv_proj = FusedColumnLinear(d, {"q": d, "k": d, "v": d}, bias=True)
        self.out_proj = nn.Linear(d, d)
        self.ragged = None

    def bind_resources(self, resources: dict) -> None:
        self.ragged = resources.get(VISION_ATTN)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.ragged is None:
            raise RuntimeError(
                f"MiniCPM-o's vision tower has no {VISION_ATTN!r} resource; declare a "
                "RaggedAttentionSpec for the vision_encoder node"
            )
        n, d = hidden_states.shape
        q, k, v = self.qkv_proj(hidden_states).view(n, 3, self.num_heads, self.head_dim).unbind(1)
        out = self.ragged.run(q, k, v, label=PATCHES)
        return self.out_proj(out.reshape(n, d))


class SiglipMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        if config.hidden_act != "gelu_pytorch_tanh":
            raise NotImplementedError(f"vision activation {config.hidden_act!r}")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        return hidden_states + self.mlp(self.layer_norm2(hidden_states))


class SiglipEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Conv2d(
            config.num_channels, config.hidden_size,
            kernel_size=config.patch_size, stride=config.patch_size,
        )
        self.position_embedding = nn.Embedding(config.num_patches_per_side ** 2, config.hidden_size)

    def forward(self, patches: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """``[N, 3, p, p]`` pixel patches -> ``[N, hidden]``."""
        x = self.patch_embedding(patches.to(self.patch_embedding.weight.dtype)).flatten(1)
        return x + self.position_embedding(position_ids)


class SiglipEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.layers = nn.ModuleList(SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers))


class NavitSiglip(nn.Module):
    """Parameter paths mirror the checkpoint's ``vpm.*``."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder.layers:
            hidden_states = layer(hidden_states)
        return self.post_layernorm(hidden_states)


# ----------------------------------------------------------------------------
# Resampler
# ----------------------------------------------------------------------------


class ResamplerAttention(nn.Module):
    """``nn.MultiheadAttention``'s parameters (packed ``in_proj``), with the
    attention itself through the ragged cross-attention resource."""

    def __init__(self, config: ResamplerConfig):
        super().__init__()
        d = config.embed_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.in_proj_weight = nn.Parameter(torch.empty(3 * d, d))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d))
        self.out_proj = nn.Linear(d, d)
        self.ragged = None

    def bind_resources(self, resources: dict) -> None:
        self.ragged = resources.get(RESAMPLER_ATTN)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """``query``: ``[num_slices * Q, d]``; ``key``/``value``: ``[total_patches, d]``."""
        if self.ragged is None:
            raise RuntimeError(
                f"MiniCPM-o's resampler has no {RESAMPLER_ATTN!r} resource; declare a "
                "RaggedCrossAttentionSpec for the vision_encoder node"
            )
        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        h, hd = self.num_heads, self.head_dim
        q = F.linear(query, w_q, b_q).view(-1, h, hd)
        k = F.linear(key, w_k, b_k).view(-1, h, hd)
        v = F.linear(value, w_v, b_v).view(-1, h, hd)
        out = self.ragged.run(q, k, v, label=cross_label(QUERIES, PATCHES))
        return self.out_proj(out.reshape(-1, h * hd))


class Resampler(nn.Module):
    """Parameter paths mirror the checkpoint's ``resampler.*``."""

    def __init__(self, config: ResamplerConfig):
        super().__init__()
        self.config = config
        d = config.embed_dim
        self.query = nn.Parameter(torch.empty(config.num_queries, d))
        self.kv_proj = nn.Linear(config.kv_dim, d, bias=False)
        self.attn = ResamplerAttention(config)
        self.ln_q = nn.LayerNorm(d, eps=1e-6)
        self.ln_kv = nn.LayerNorm(d, eps=1e-6)
        self.ln_post = nn.LayerNorm(d, eps=1e-6)
        self.proj = nn.Parameter(torch.empty(d, d))
        # not a checkpoint tensor; kept float32 whatever the module's dtype
        self.register_buffer("omega", sincos_omega(d), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.omega = self.omega.float()
        return self

    def reset_buffers(self) -> None:
        """Recompute ``omega``, which ``to_empty`` leaves uninitialized."""
        self.omega.copy_(sincos_omega(self.config.embed_dim))

    def sincos(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """2D sincos keys for ``[N, 2]`` (row, col): column half first, as
        upstream's w-first meshgrid lays it out. Computed in float32 per patch
        rather than from a fixed-size table, so no slice shape outgrows it."""
        angles = grid_coords.float()[:, :, None] * self.omega  # [N, 2, D/4]
        row, col = angles.unbind(1)
        return torch.cat([col.sin(), col.cos(), row.sin(), row.cos()], dim=-1)

    def forward(self, x: torch.Tensor, grid_coords: torch.Tensor, num_slices: int) -> torch.Tensor:
        """``[total_patches, kv_dim]`` -> ``[num_slices * Q, d]``, slice-major."""
        x = self.ln_kv(self.kv_proj(x))
        pos = self.sincos(grid_coords).to(x.dtype)
        # every slice asks with the same queries
        q = self.ln_q(self.query).repeat(num_slices, 1)
        out = self.attn(q, x + pos, x)
        return self.ln_post(out) @ self.proj


class MiniCPMOVision(nn.Module):
    """SigLIP then the resampler, over every slice of a packed forward."""

    def __init__(self, vision: VisionConfig, resampler: ResamplerConfig):
        super().__init__()
        self.vpm = NavitSiglip(vision)
        self.resampler = Resampler(resampler)

    def forward(
        self,
        patches: torch.Tensor,
        position_ids: torch.Tensor,
        grid_coords: torch.Tensor,
        num_slices: int,
    ) -> torch.Tensor:
        hidden = self.vpm.encode(self.vpm.embeddings(patches, position_ids))
        return self.resampler(hidden, grid_coords, num_slices)
