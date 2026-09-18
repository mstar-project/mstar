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
image, which is what the per-frame segment lengths mark off.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.qwen3_5.config import VISION_ATTN, Qwen3_5VisionConfig

# ----------------------------------------------------------------------------
# Grid helpers
# ----------------------------------------------------------------------------


def vision_seq_lengths(grid: list[tuple[int, int, int]]) -> tuple[int, ...]:
    """One attending segment per frame, on the host.

    Each frame attends to itself alone, so a ``t``-frame entry contributes
    ``t`` segments of ``h * w``. The submodule turns these into the step's
    ``Segment`` list and the ragged attention resource plans them; nothing in
    the tower reads them.
    """
    return tuple(h * w for t, h, w in grid for _ in range(t))


def vision_position_ids(
    grid: list[tuple[int, int, int]],
    spatial_merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    """``[total_patches, 2]`` (h, w) indices in spatial-merge-block order.

    The merger later folds each ``m x m`` block into one token, so patches are
    emitted block-major here and the rotary sees the same order the merger
    will consume.
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
    # position within one frame's flat patch run, repeating across frames.
    # The total comes off the host grid rather than `counts.sum()`, which
    # would be a device read this function has no other reason to make.
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

    The engine's ragged attention resource does the isolating: the submodule
    declares one segment per frame and the manager plans a varlen layout, so
    this hands it the whole packed run and never sees the boundaries.

    The layout being in the plan rather than in this forward is what lets the
    tower compile. Splitting the packed run per frame here put each frame's
    patch count into the graph as its own symint, which made the token
    dimension a sum like ``s45 + s63 + 1792``; inductor then would not prove
    ``hidden_size * n`` divisible by ``n`` when it fused a block's residual add
    into the next block's norm, and refused to codegen any packed run of three
    or more images (``CantSplit``).
    """

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.ragged_attn = None

    def bind_resources(self, resources: dict) -> None:
        """See ``NodeSubmodule.bind_node_resources``. ``.get``: a deployment
        that never builds the vision node binds nothing here."""
        self.ragged_attn = resources.get(VISION_ATTN)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.ragged_attn is None:
            raise RuntimeError(
                "Qwen3.5's vision tower has no ragged attention resource; "
                f"declare a RaggedAttentionSpec under {VISION_ATTN!r} for the "
                "vision_encoder node"
            )
        seq_len = hidden_states.shape[0]
        # [tokens, heads, head_dim] throughout — the layout the ragged kernel
        # takes, so nothing transposes on the way in or out
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
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config)
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
        self,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """``[total_patches, patch_numel]`` in, ``[merged_tokens, out_hidden]``
        out — one token per ``spatial_merge_size ** 2`` block of patches.

        Everything the grid decides is built by the caller (see
        ``vision_grid_inputs``) and handed over ready-made. The grid must not
        be read here: ``grid_thw.tolist()`` breaks the graph, and the h and w
        it yields come back as unbacked symints, which makes the token
        dimension a polynomial like ``s11*s50 + 768``. Inductor then cannot
        prove ``hidden_size * n`` divisible by ``n`` and refuses to codegen
        (``CantSplit``) for any packed multi-image run.

        Which patches attend together is not an argument here either: the
        submodule declares one segment per frame and the ragged attention
        resource plans the layout, outside the graph.
        """
        n = pixel_values.shape[0]
        # Same grid, separate inputs: without this each leading dim gets its
        # own symbol and nothing downstream lines up.
        torch._check(indices.shape[0] == n)
        torch._check(weights.shape[0] == n)
        torch._check(position_ids.shape[0] == n)

        hidden = self.patch_embed(pixel_values)
        # bilinear resample of the learned table, as a weighted gather
        pos = (self.pos_embed(indices) * weights[:, :, None]).sum(1)
        hidden = hidden + pos.to(hidden.dtype)

        cos, sin = self.rotary_pos_emb(position_ids)
        cos, sin = cos.to(hidden.dtype), sin.to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, cos, sin)
        return self.merger(hidden)
