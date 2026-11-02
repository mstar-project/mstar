"""The Qwen3-VL family's ViT, and the parts other towers over packed patches
share with it (Muse Glimmer's): a learned position grid resampled to each
image (`bilinear_table_taps`), axial 2D rope, and attention through a ragged
attention resource (`VisionAttention`).

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

from mstar.model.components.linear import FusedColumnLinear


class ViTBlockConfig(Protocol):
    """What the rotary table and the transformer block read off a tower's config."""

    hidden_size: int
    intermediate_size: int
    num_heads: int
    hidden_act: str
    rope_theta: float

    @property
    def head_dim(self) -> int: ...


class ViTConfig(ViTBlockConfig, Protocol):
    """What the patch embedding reads besides."""

    in_channels: int
    patch_size: int
    temporal_patch_size: int


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


def _axis_taps(src: torch.Tensor, side: int, zero_pad: bool) -> tuple[torch.Tensor, torch.Tensor]:
    floor = torch.floor(src)
    offsets = torch.arange(0, 2, device=src.device)
    taps = floor.long()[:, None] + offsets
    weights = (1 - (src[:, None] - floor[:, None] - offsets).abs()).clamp(min=0)
    if zero_pad:
        weights = weights * ((taps >= 0) & (taps <= side - 1))
    return taps.clamp(0, side - 1), weights


def bilinear_table_taps(
    row_src: torch.Tensor, col_src: torch.Tensor, side: int, zero_pad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[P, 4]`` indices into a ``side x side`` table and their weights,
    bilinear around each patch's fp32 source coordinates in the table.

    The caller maps patch positions to source coordinates, which is where
    ``align_corners`` lives. ``zero_pad``: a tap off the table weighs zero
    (``grid_sample``'s ``padding_mode="zeros"``); otherwise it reads the edge.
    """
    h_taps, h_w = _axis_taps(row_src, side, zero_pad)
    w_taps, w_w = _axis_taps(col_src, side, zero_pad)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_w[:, :, None] * w_w[:, None, :]).reshape(-1, 4)
    return indices, weights


def _align_corners_src(index: torch.Tensor, size: torch.Tensor, side: int) -> torch.Tensor:
    """Where positions on an axis of length ``size`` sample a ``side``-long
    table with ``align_corners=True``, as the ``F.interpolate`` call upstream
    reproduces: the closed form of ``linspace(0, side - 1, size)[index]``; the
    clamp avoids a divide-by-zero at size == 1, where index is 0 anyway."""
    return index.to(torch.float32) * (side - 1) / torch.clamp(size - 1, min=1)


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

    # border padding: a tap past the table reads its edge
    return bilinear_table_taps(
        _align_corners_src(row, heights, side), _align_corners_src(col, widths, side), side, zero_pad=False,
    )


# ----------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class VisionRotaryEmbedding(nn.Module):
    """Axial 2D rope: the same frequencies drive the H and W axes, and the two
    halves are concatenated to cover the whole head — no partial rotation.

    ``keep_fp32``: the frequencies stay fp32 through a cast of the module (the
    engine's ``.to(bfloat16)`` would otherwise round them).
    """

    def __init__(self, config: ViTBlockConfig, keep_fp32: bool = False):
        super().__init__()
        self._spatial_dim = config.head_dim // 2
        self._theta = config.rope_theta
        self._keep_fp32 = keep_fp32
        self.register_buffer("inv_freq", self._inv_freq(None), persistent=False)

    def _inv_freq(self, device) -> torch.Tensor:
        d = self._spatial_dim
        return 1.0 / (self._theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if self._keep_fp32:
            self.inv_freq = self._inv_freq(self.inv_freq.device)
        return result

    @torch.no_grad()
    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``[N, 2]`` positions in, ``[N, head_dim]`` cos and sin out, in fp32,
        or rounded to ``dtype``."""
        freqs = position_ids[..., None].float() * self.inv_freq.float()  # [N, 2, dim/4]
        cos, sin = self._recompose(freqs.cos()), self._recompose(freqs.sin())
        if dtype is None:
            return cos, sin
        return cos.to(dtype), sin.to(dtype)

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

    # the activations the towers here use: tanh-approximated GELU (Qwen3-VL) or exact (Muse Glimmer)
    _GELU = {"gelu_pytorch_tanh": "tanh", "gelu_new": "tanh", "gelu": "none"}

    def __init__(self, config: ViTBlockConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True,
        )
        self.linear_fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True,
        )
        if config.hidden_act not in self._GELU:
            raise ValueError(
                f"vision activation {config.hidden_act!r} is not wired up"
            )
        self.act_fn = nn.GELU(approximate=self._GELU[config.hidden_act])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionAttention(nn.Module):
    """Non-causal attention over packed patches, through the plan the ragged
    attention resource under ``attn_key`` holds for ``label`` (frames by
    default; a windowed tower declares windows and frames as two labels).

    The resource plans the layout, so this forward never sees the boundaries.
    That is what lets the tower compile: splitting per frame here gave each
    frame its own symint, and inductor then refused (``CantSplit``) to codegen
    packed runs of three or more images.

    q, k and v are one GEMM. A checkpoint that stores them fused loads as is,
    one that keeps them apart loads each into its rows (``q`` / ``k`` / ``v``
    shards, via the loader's stacked-param rules).
    """

    def __init__(self, config: ViTBlockConfig, attn_key: str, label: str | None = None):
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = FusedColumnLinear(hidden, {"q": hidden, "k": hidden, "v": hidden}, bias=True)
        # initialised as the nn.Linear it replaces (a checkpoint overwrites it)
        self.qkv.reset_parameters()
        self.proj = nn.Linear(hidden, hidden)
        self.attn_key = attn_key
        self.label = label
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
            raise ValueError(
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

        out = self.ragged_attn.run(q, k, v, label=self.label)
        return self.proj(out.reshape(seq_len, -1).contiguous())


class VisionBlock(nn.Module):
    """Pre-norm attention and MLP. ``eps`` is the tower's LayerNorm epsilon."""

    def __init__(self, config: ViTBlockConfig, attn_key: str, label: str | None = None, eps: float = 1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.attn = VisionAttention(config, attn_key, label)
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
