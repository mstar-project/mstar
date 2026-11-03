"""The Qwen3-VL family's ViT, the parts every tower shares: a ViT over packed
image patches with a learned position grid resampled to each image, axial 2D
rope, and attention within each frame through a ragged attention resource.

Ported from transformers' ``modeling_qwen3_5.py`` and ``vision_utils.py``
(Apache-2.0); the grid helpers are copied because ``transformers.vision_utils``
is internal. Not here, as the towers differ in them: the patch mergers and
deepstack. Bicubic resampling and the temporal rotary axis are dropped, as no
tower using this sets them.

Images arrive packed: ``pixel_values`` is ``[total_patches, patch_numel]`` and
``grid_thw`` gives each image's ``(t, h, w)`` patch grid.
"""
from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn


class ViTConfig(Protocol):
    """What the shared modules read off a tower's config."""

    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    hidden_act: str
    rope_theta: float

    @property
    def head_dim(self) -> int: ...


# ----------------------------------------------------------------------------
# Grid helpers
# ----------------------------------------------------------------------------


def vision_seq_lengths(grid: list[tuple[int, int, int]]) -> tuple[int, ...]:
    """One attending segment per frame, on the host: ``t`` segments of
    ``h * w`` per entry, planned by the ragged attention resource."""
    return tuple(h * w for t, h, w in grid for _ in range(t))


def vision_position_ids(
    grid: list[tuple[int, int, int]],
    spatial_merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    """``[total_patches, 2]`` (h, w) indices in spatial-merge-block order.

    Block-major, so the rotary sees the order the merger will fold.
    """
    out = []
    for t, h, w in grid:
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
    grid: list[tuple[int, int, int]],
    num_grid_per_side: int,
    spatial_merge_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather indices and weights that resample the square learned position
    table onto each image's ``(h, w)`` grid.

    Returns ``[total_patches, 4]`` each — the 2x2 bilinear neighbourhood per
    patch, as the outer product of the two axes' taps.
    """
    side, merge = num_grid_per_side, spatial_merge_size

    grid_thw = torch.tensor(grid, dtype=torch.long, device=device)
    counts = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
    heights = torch.repeat_interleave(grid_thw[:, 1], counts)
    widths = torch.repeat_interleave(grid_thw[:, 2], counts)
    starts = torch.repeat_interleave(
        F.pad(counts.cumsum(0)[:-1], (1, 0)), counts,
    )
    # position within one frame's flat patch run, repeating across frames;
    # the total comes off the host grid to avoid a device read
    total_patches = sum(t * h * w for t, h, w in grid)
    within = (
        torch.arange(total_patches, device=device) - starts
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

    def __init__(self, config: ViTConfig):
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
    def __init__(self, config: ViTConfig):
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


class VisionMLP(nn.Module):
    """Plain, not gated: one up projection, an activation, one down."""

    def __init__(self, config: ViTConfig):
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

    The ragged attention resource plans the per-frame layout, so this forward
    never sees the boundaries. That is what lets the tower compile: splitting
    per frame here gave each frame its own symint, and inductor then refused
    (``CantSplit``) to codegen packed runs of three or more images.
    """

    def __init__(self, config: ViTConfig, attn_key: str):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_key = attn_key
        self.ragged_attn = None

    def bind_resources(self, resources: dict) -> None:
        """See ``NodeSubmodule.bind_node_resources``. ``.get``: a deployment
        that never builds the vision node binds nothing here."""
        self.ragged_attn = resources.get(self.attn_key)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.ragged_attn is None:
            raise RuntimeError(
                "the vision tower has no ragged attention resource; declare a "
                f"RaggedAttentionSpec under {self.attn_key!r} for its node"
            )
        seq_len = hidden_states.shape[0]
        # [tokens, heads, head_dim]: the ragged kernel's layout, no transposes
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

        out = self.ragged_attn.run(q, k, v)
        return self.proj(out.reshape(seq_len, -1).contiguous())


class VisionBlock(nn.Module):
    def __init__(self, config: ViTConfig, attn_key: str):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config, attn_key)
        self.mlp = VisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cos, sin,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))
