"""Muse Glimmer's perception encoder: a windowed ViT over packed patches.

Ported from transformers' ``modeling_muse_glimmer.py`` and ``vision_utils.py``
(Apache-2.0). The tower reorders each frame's patches so every 32x32-patch
window is contiguous, runs three windowed layers to one full-frame layer (and
the last layer full), restores raster order, folds 2x2 blocks into the channel
dim and projects to the LLM's width.

Every grid-derived index is built on the host, on the CPU (``tower_inputs``),
so the forward is tensor-in, tensor-out and the worker's main thread never
waits on the device. The two attention
layouts are two labels of one ragged attention resource, declared as step
segments: one per window, and one per frame.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.norm import rms_norm_no_weight
from mstar.model.components.vit import (
    AxialVisionRope,
    RaggedVisionAttention,
    bilinear_table_taps,
)
from mstar.model.muse.config import (
    VISION_FULL,
    VISION_WINDOW,
    WINDOW_ATTENTION,
    MuseGlimmerVisionConfig,
)

# ----------------------------------------------------------------------------
# Grid helpers (host-side; grids are ``(t, h, w)`` in patches)
# ----------------------------------------------------------------------------


class TowerInputs(NamedTuple):
    """Everything the tower needs besides the pixels, for a packed set of grids."""
    interp_indices: torch.Tensor   # [P, 4] into the learned position table
    interp_weights: torch.Tensor   # [P, 4]
    window_index: torch.Tensor     # [P] raster -> window order
    rope_positions: torch.Tensor   # [P, 2] (w + 1, h + 1), in window order
    output_index: torch.Tensor     # [P] window order -> merge-block order
    window_lengths: tuple[int, ...]
    frame_lengths: tuple[int, ...]


def _half_pixel_src(index: torch.Tensor, size: torch.Tensor, side: int) -> torch.Tensor:
    """Where positions on an axis of length ``size`` sample a ``side``-long
    table: half-pixel centres (``align_corners=False``)."""
    return (index.float() + 0.5) * side / size - 0.5


def _interpolation(grid, side: int):
    """The learned table resampled onto each grid, with zero padding, as
    ``F.grid_sample(padding_mode="zeros")``: edges are attenuated."""
    rows, cols, heights, widths = [], [], [], []
    for t, h, w in grid:
        r = torch.arange(h).repeat_interleave(w).repeat(t)
        c = torch.arange(w).repeat(h * t)
        rows.append(r)
        cols.append(c)
        heights.append(torch.full_like(r, h))
        widths.append(torch.full_like(r, w))
    return bilinear_table_taps(
        _half_pixel_src(torch.cat(rows), torch.cat(heights), side),
        _half_pixel_src(torch.cat(cols), torch.cat(widths), side),
        side, zero_pad=True,
    )


def _windows(grid, window: int):
    """Raster -> window order, and each window's length. Windows tile each
    frame from the top left; edge windows are partial. Frames never mix."""
    index, lengths, offset = [], [], 0
    for t, h, w in grid:
        frame = torch.arange(h * w).view(h, w)
        for f in range(t):
            for r0 in range(0, h, window):
                for c0 in range(0, w, window):
                    block = frame[r0:r0 + window, c0:c0 + window].reshape(-1)
                    index.append(block + offset + f * h * w)
                    lengths.append(block.numel())
        offset += t * h * w
    return torch.cat(index), tuple(lengths)


def _merge_order(grid, merge: int):
    """Raster -> 2x2-block order, block-major over the merged grid."""
    index, offset = [], 0
    for t, h, w in grid:
        perm = torch.arange(h * w).view(h // merge, merge, w // merge, merge)
        perm = perm.permute(0, 2, 1, 3).reshape(-1)
        frames = torch.arange(t)[:, None] * h * w
        index.append((perm[None, :] + frames).reshape(-1) + offset)
        offset += t * h * w
    return torch.cat(index)


def tower_inputs(grid: list[tuple[int, int, int]], config: MuseGlimmerVisionConfig) -> TowerInputs:
    """The tower's index tensors for ``grid``, built once per request, on
    the CPU: the node's `preprocess` moves them."""
    indices, weights = _interpolation(grid, config.pos_emb_height)
    window_index, window_lengths = _windows(grid, config.window_patches)
    raster_pos = torch.cat([
        torch.stack([
            torch.arange(w).repeat(h * t), torch.arange(h).repeat_interleave(w).repeat(t),
        ], dim=-1)
        for t, h, w in grid
    ])
    # the reference's positions start at 1, and put w before h
    rope_positions = raster_pos[window_index] + 1
    # after the block loop: back to raster, then into merge blocks
    output_index = torch.argsort(window_index)[_merge_order(grid, config.merge_size)]
    return TowerInputs(
        interp_indices=indices,
        interp_weights=weights,
        window_index=window_index,
        rope_positions=rope_positions,
        output_index=output_index,
        window_lengths=window_lengths,
        frame_lengths=tuple(h * w for t, h, w in grid for _ in range(t)),
    )


# ----------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------


class VisionAttention(RaggedVisionAttention):
    """Attention within each window or each frame (``label``)."""

    def __init__(self, config: MuseGlimmerVisionConfig, label: str):
        super().__init__(config.hidden_size, config.num_attention_heads, label)
        dim = config.hidden_size
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)

    def project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (x.shape[0], self.num_heads, self.head_dim)
        return self.q_proj(x).view(shape), self.k_proj(x).view(shape), self.v_proj(x).view(shape)


class VisionMLP(nn.Module):
    def __init__(self, config: MuseGlimmerVisionConfig):
        super().__init__()
        if config.hidden_act != "gelu":
            raise ValueError(f"the vision MLP's activation is gelu here, not {config.hidden_act!r}")
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class VisionEncoderLayer(nn.Module):
    def __init__(self, config: MuseGlimmerVisionConfig, label: str):
        super().__init__()
        # eps hard-coded in the reference, not `layer_norm_eps`
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.attn = VisionAttention(config, label)
        self.mlp = VisionMLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class PatchEmbedder(nn.Module):
    def __init__(self, config: MuseGlimmerVisionConfig):
        super().__init__()
        self.patch_embedding = nn.Linear(config.patch_dim, config.hidden_size, bias=False)
        self.position_embedding_table = nn.Embedding(
            config.pos_emb_height * config.pos_emb_width, config.hidden_size,
        )


class MuseGlimmerVisionModel(nn.Module):
    """``vision_tower``, ``vision_adapter`` and ``vision_projection`` in one
    module: patches in, LLM-width embeddings out, at unit RMS."""

    def __init__(self, config: MuseGlimmerVisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedder = PatchEmbedder(config)
        # the reference keeps the frequencies fp32 and rounds only cos / sin
        self.rotary_emb = AxialVisionRope(config.head_dim, config.rope_theta, keep_fp32=True)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            VisionEncoderLayer(
                config, VISION_WINDOW if kind == WINDOW_ATTENTION else VISION_FULL,
            )
            for kind in config.layer_types
        )
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        merged = config.hidden_size * config.merge_size ** 2
        if merged != config.out_hidden_size:
            raise ValueError(f"pixel shuffle gives {merged} channels, config says {config.out_hidden_size}")
        self.adapter_fc1 = nn.Linear(merged, config.projector_hidden_size, bias=False)
        self.adapter_fc2 = nn.Linear(config.projector_hidden_size, config.projector_hidden_size, bias=False)
        self.projection = nn.Linear(config.projector_hidden_size, config.text_hidden_size, bias=False)
        # hard-coded in the reference, not `layer_norm_eps`
        self.out_eps = 1e-5

    def embed(
        self, pixel_values: torch.Tensor, interp_indices: torch.Tensor,
        interp_weights: torch.Tensor, window_index: torch.Tensor,
        rope_positions: torch.Tensor, **_: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Everything before the block loop: patch embed, resampled position
        table, ``ln_pre``, window order, and the rope for the blocks."""
        pe = self.patch_embedder
        x = pe.patch_embedding(pixel_values.to(pe.patch_embedding.weight.dtype))
        pos = (pe.position_embedding_table(interp_indices) * interp_weights[:, :, None]).sum(1)
        x = self.ln_pre(x + pos.to(x.dtype))[window_index]
        cos, sin = self.rotary_emb(rope_positions, x.dtype)
        return x, cos, sin

    def encode(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """The block loop, ``[patches, hidden]`` in and out."""
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x

    def project(self, x: torch.Tensor, output_index: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        """``ln_post``, the ``merge_size`` x ``merge_size`` pixel shuffle and the adapter.

        The shuffle interleaves channels (``out[c * m + k] = x_k[c]``, ``m =
        merge_size ** 2``), not concatenates them, which is the layout the
        adapter was trained on.
        """
        x = self.ln_post(x)[output_index]
        d, m = x.shape[-1], self.config.merge_size ** 2
        x = x.view(-1, m, d).transpose(1, 2).reshape(-1, m * d)
        x = F.gelu(self.adapter_fc2(F.gelu(self.adapter_fc1(x))))
        return rms_norm_no_weight(self.projection(x), self.out_eps)
