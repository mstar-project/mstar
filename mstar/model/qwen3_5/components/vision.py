"""Qwen3.5's vision tower: a ViT over packed image patches.

Ported from transformers' ``modeling_qwen3_5.py`` and ``vision_utils.py``
(Apache-2.0). The grid helpers are ported rather than imported: they are pure
tensor maths with no model state, and ``transformers.vision_utils`` is an
internal module whose shape can change between releases.

Simplifications against upstream, all of them things Qwen3.5 does not use:
bilinear-only position resampling (no bicubic), no temporal rotary axis, and
no deepstack, so the tower returns merged embeddings and nothing else.

Images arrive packed: ``pixel_values`` is ``[total_patches, patch_numel]`` and
``grid_thw`` gives each image's ``(t, h, w)`` patch grid. Attention is per
image, which is what ``cu_seqlens`` marks off.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.qwen3_5.config import Qwen3_5VisionConfig

# ----------------------------------------------------------------------------
# Grid helpers
# ----------------------------------------------------------------------------


def vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    """``[num_segments + 1]`` int32 boundaries; one segment per frame.

    Each frame attends to itself alone, so a ``t``-frame entry contributes
    ``t`` segments of ``h * w``.
    """
    seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0],
    )
    return F.pad(seqlens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)


def vision_position_ids(
    grid_thw: torch.Tensor, spatial_merge_size: int,
) -> torch.Tensor:
    """``[total_patches, 2]`` (h, w) indices in spatial-merge-block order.

    The merger later folds each ``m x m`` block into one token, so patches are
    emitted block-major here and the rotary sees the same order the merger
    will consume.
    """
    device = grid_thw.device
    out = []
    for t, h, w in grid_thw.tolist():
        hpos, wpos = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        block = (
            h // spatial_merge_size, spatial_merge_size,
            w // spatial_merge_size, spatial_merge_size,
        )
        hpos = hpos.reshape(block).transpose(1, 2).flatten()
        wpos = wpos.reshape(block).transpose(1, 2).flatten()
        out.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0)


def _axis_taps_weights(
    index: torch.Tensor, size: torch.Tensor, side: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear taps into a ``side``-long table, for positions on an axis of
    length ``size``. ``align_corners=True`` and border padding, matching the
    ``F.interpolate`` call upstream reproduces."""
    index = index.to(torch.float32)
    # closed form of `linspace(0, side - 1, size)[index]`; clamp avoids a
    # divide-by-zero at size == 1, where index is 0 anyway
    src = index * (side - 1) / torch.clamp(size - 1, min=1)
    floor = torch.floor(src)
    offsets = torch.arange(0, 2, device=index.device)
    taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)
    distance = (src[:, None] - floor[:, None] - offsets).abs()
    return taps, (1 - distance).clamp(min=0)


def vision_interpolation(
    grid_thw: torch.Tensor, num_grid_per_side: int, spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather indices and weights that resample the square learned position
    table onto each image's ``(h, w)`` grid.

    Returns ``[total_patches, 4]`` each — the 2x2 bilinear neighbourhood per
    patch, as the outer product of the two axes' taps.
    """
    side, merge = num_grid_per_side, spatial_merge_size
    device = grid_thw.device

    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(
        F.pad(counts.cumsum(0)[:-1], (1, 0)), counts,
    )
    # position within one frame's flat patch run, repeating across frames
    within = (
        torch.arange(int(counts.sum()), device=device) - starts
    ) % (heights * widths)

    # undo the spatial-merge-block ordering to recover (row, col)
    blocks_w = widths // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    col = block_col * merge + in_col

    h_taps, h_w = _axis_taps_weights(row, heights, side)
    w_taps, w_w = _axis_taps_weights(col, widths, side)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
    return indices, weights


# ----------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class VisionRotaryEmbedding(nn.Module):
    """Axial 2D rope: the same frequencies drive the H and W axes, and the two
    halves are concatenated to cover the whole head — no partial rotation."""

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        spatial_dim = config.head_dim // 2
        inv_freq = 1.0 / (
            config.rope_theta
            ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [N, 2] -> freqs [N, 2, dim/4]
        freqs = position_ids[..., None].float() * self.inv_freq.float()
        return self._recompose(freqs.cos()), self._recompose(freqs.sin())

    @staticmethod
    def _recompose(freq: torch.Tensor) -> torch.Tensor:
        freq_hw = torch.cat([freq[:, 0], freq[:, 1]], dim=-1)
        return torch.cat([freq_hw, freq_hw], dim=-1)


class VisionPatchEmbed(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.in_channels = config.in_channels
        self.temporal_patch_size = config.temporal_patch_size
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_size
        kernel = (config.temporal_patch_size, config.patch_size, config.patch_size)
        self.proj = nn.Conv3d(
            config.in_channels, config.hidden_size,
            kernel_size=kernel, stride=kernel, bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(
            -1, self.in_channels, self.temporal_patch_size,
            self.patch_size, self.patch_size,
        )
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


class VisionPatchMerger(nn.Module):
    """Folds each ``m x m`` block of patches into one LLM-width token."""

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.merged_size = config.hidden_size * config.merge_unit
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_size, self.merged_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.merged_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.merged_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionMLP(nn.Module):
    """Plain, not gated: one up projection, an activation, one down."""

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True,
        )
        self.linear_fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True,
        )
        if config.hidden_act not in ("gelu_pytorch_tanh", "gelu_new"):
            raise NotImplementedError(
                f"vision activation {config.hidden_act!r} is not wired up"
            )
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionAttention(nn.Module):
    """Full attention within each frame, over packed patches.

    Not the engine's attention resource: there is no KV cache and nothing
    persists across steps, so this runs straight through SDPA. Segments are
    split by ``cu_seqlens`` rather than masked, which keeps the cost linear in
    patches instead of quadratic across the whole packed batch.
    """

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(seq_len, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        # rope in fp32, as upstream does, then back
        dtype = q.dtype
        c, s = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
        qf, kf = q.float(), k.float()
        q = ((qf * c) + (_rotate_half(qf) * s)).to(dtype)
        k = ((kf * c) + (_rotate_half(kf) * s)).to(dtype)

        # [1, heads, tokens, head_dim]
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outs = [
            F.scaled_dot_product_attention(qs, ks, vs, is_causal=False)
            for qs, ks, vs in zip(
                *(torch.split(t, lengths, dim=2) for t in (q, k, v)), strict=True,
            )
        ]
        out = torch.cat(outs, dim=2)
        return self.proj(out.transpose(1, 2).reshape(seq_len, -1).contiguous())


class VisionBlock(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens, cos, sin,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3_5VisionModel(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        if config.deepstack_visual_indexes:
            raise NotImplementedError(
                "deepstack is not wired up: this tower returns merged "
                "embeddings only. No released Qwen3.5 sets this, and neither "
                "does transformers' Qwen3.5 — port it from Qwen3-VL."
            )
        self.config = config
        self.patch_embed = VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size,
        )
        self.rotary_pos_emb = VisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList(
            VisionBlock(config) for _ in range(config.depth)
        )
        self.merger = VisionPatchMerger(config)

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """``[total_patches, patch_numel]`` in, ``[merged_tokens, out_hidden]``
        out — one token per ``spatial_merge_size ** 2`` block of patches."""
        merge = self.config.spatial_merge_size
        indices, weights = vision_interpolation(
            grid_thw, self.config.num_grid_per_side, merge,
        )
        position_ids = vision_position_ids(grid_thw, merge)
        cu_seqlens = vision_cu_seqlens(grid_thw)

        hidden = self.patch_embed(pixel_values)
        # bilinear resample of the learned table, as a weighted gather
        pos = (self.pos_embed(indices) * weights[:, :, None]).sum(1)
        hidden = hidden + pos.to(hidden.dtype)

        cos, sin = self.rotary_pos_emb(position_ids)
        cos, sin = cos.to(hidden.dtype), sin.to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, cu_seqlens, cos, sin)
        return self.merger(hidden)
