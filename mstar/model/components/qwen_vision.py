"""The Qwen vision tower (Qwen3.5, Qwen3-VL): a ViT over packed image patches.

Ported from transformers' ``modeling_qwen3_5.py``, ``modeling_qwen3_vl.py`` and
``vision_utils.py`` (Apache-2.0); the grid helpers are copied because
``transformers.vision_utils`` is internal. Qwen3-VL adds deepstack: mergers after some
blocks whose features the language model adds to its first layers' hidden states at
the image rows (``deepstack_visual_indexes``; empty for Qwen3.5). Not ported: bicubic
resampling and the temporal rotary axis, which neither model uses.

Images arrive packed: ``pixel_values`` is ``[total_patches, patch_numel]`` and
``grid_thw`` gives each image's ``(t, h, w)`` patch grid.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# the tower's cacheless attention resource, by default
VISION_ATTN = "vision_attn"


@dataclass
class QwenVisionConfig:
    """The ViT tower's config, read off ``config.json``'s ``vision_config``."""

    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    num_position_embeddings: int
    out_hidden_size: int

    hidden_act: str = "gelu_pytorch_tanh"
    rope_theta: float = 10_000.0
    # Blocks after which a deepstack merger taps the hidden states (Qwen3-VL); empty on
    # every released Qwen3.5 size.
    deepstack_visual_indexes: tuple[int, ...] = ()

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_grid_per_side(self) -> int:
        """The learned position table is a square grid."""
        return int(self.num_position_embeddings**0.5)

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge_size**2

    @classmethod
    def from_hf(cls, path: str | Path) -> "QwenVisionConfig":
        config = cls.from_hf_or_none(path)
        if config is None:
            raise ValueError(f"{path} has no `vision_config`")
        return config

    @classmethod
    def from_hf_or_none(cls, path: str | Path) -> "QwenVisionConfig | None":
        """None when the checkpoint is text-only."""
        path = Path(path)
        if path.is_dir():
            path = path / "config.json"
        raw = json.loads(path.read_text()).get("vision_config")
        if raw is None:
            return None
        merged = {**raw, **raw.get("rope_parameters", {})}
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in merged.items() if k in known}
        if "deepstack_visual_indexes" in values:
            values["deepstack_visual_indexes"] = tuple(values["deepstack_visual_indexes"])
        return cls(**values)

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

    def __init__(self, config: QwenVisionConfig):
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
    def __init__(self, config: QwenVisionConfig):
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
    """Folds each ``m x m`` block of patches into one LLM-width token. The norm runs
    per patch before the fold, or (``postshuffle_norm``, Qwen3-VL's deepstack mergers)
    over the folded token."""

    def __init__(self, config: QwenVisionConfig, postshuffle_norm: bool = False):
        super().__init__()
        self.merged_size = config.hidden_size * config.merge_unit
        self.postshuffle_norm = postshuffle_norm
        self.norm = nn.LayerNorm(self.merged_size if postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_size, self.merged_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.merged_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.postshuffle_norm:
            x = self.norm(x.view(-1, self.merged_size))
        else:
            x = self.norm(x).view(-1, self.merged_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionMLP(nn.Module):
    """Plain, not gated: one up projection, an activation, one down."""

    def __init__(self, config: QwenVisionConfig):
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

    def __init__(self, config: QwenVisionConfig, resource_key: str = VISION_ATTN):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.resource_key = resource_key
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.ragged_attn = None

    def bind_resources(self, resources: dict) -> None:
        """See ``NodeSubmodule.bind_node_resources``. A node without the resource
        keeps what was bound before (an owner's own per-frame attention, or nothing:
        a deployment that never builds the vision node)."""
        self.ragged_attn = resources.get(self.resource_key, self.ragged_attn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        if self.ragged_attn is None:
            raise RuntimeError(
                "the Qwen vision tower has no ragged attention resource; "
                f"declare a RaggedAttentionSpec under {self.resource_key!r} for "
                "its node"
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


class FrameSDPA:
    """Stands in for the ragged resource where none is declared (a model's SDPA mode, the
    reference's own kernel): SDPA within each frame of ``lengths`` (``vision_seq_lengths``),
    set before each tower call."""

    def __init__(self, lengths=()):
        self.lengths = list(lengths)

    def run(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # [1, heads, frame, head_dim] per frame, the call transformers makes (and so its kernel)
        out = [
            F.scaled_dot_product_attention(*(t.transpose(0, 1)[None] for t in segment))[0].transpose(0, 1)
            for segment in zip(q.split(self.lengths), k.split(self.lengths), v.split(self.lengths), strict=True)
        ]
        return torch.cat(out)


class VisionBlock(nn.Module):
    def __init__(self, config: QwenVisionConfig, resource_key: str = VISION_ATTN):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = VisionAttention(config, resource_key)
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


class QwenVisionModel(nn.Module):
    """``rope_dtype``: the rotary tables' dtype into the blocks, fp32 as transformers keeps
    them. ``None`` rounds them to the activations' dtype instead, as Qwen3.5's tower always
    has (a known difference from transformers there)."""

    def __init__(self, config: QwenVisionConfig, resource_key: str = VISION_ATTN,
                 rope_dtype: torch.dtype | None = torch.float32):
        super().__init__()
        self.config = config
        self.rope_dtype = rope_dtype
        self.patch_embed = VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size,
        )
        self.rotary_pos_emb = VisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList(
            VisionBlock(config, resource_key) for _ in range(config.depth)
        )
        self.merger = VisionPatchMerger(config)
        self.deepstack_visual_indexes = tuple(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            VisionPatchMerger(config, postshuffle_norm=True)
            for _ in self.deepstack_visual_indexes
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """``[total_patches, patch_numel]`` in, ``[merged_tokens, out_hidden]``
        out — one token per ``spatial_merge_size ** 2`` block of patches; with
        deepstack, also each deepstack merger's ``[merged_tokens, out_hidden]``.

        Grid-derived inputs are built by the caller
        (``VisionEncoderSubmodule.prepare_inputs``): ``grid_thw.tolist()`` here
        would break the graph and make h, w unbacked symints, and inductor then
        refuses (``CantSplit``) to codegen packed multi-image runs.
        """
        hidden, cos, sin = self.embed(pixel_values, indices, weights, position_ids)
        if not self.deepstack_visual_indexes:
            return self.merger(self.encode(hidden, cos, sin))
        hidden, deepstack = self.encode_with_deepstack(hidden, cos, sin)
        return self.merger(hidden), deepstack

    def embed(
        self,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patch embed plus the resampled position table, and the rope for
        the blocks: everything before the block loop, per patch."""
        n = pixel_values.shape[0]
        # one symbol for every leading dim, not one each
        torch._check(indices.shape[0] == n)
        torch._check(weights.shape[0] == n)
        torch._check(position_ids.shape[0] == n)

        hidden = self.patch_embed(pixel_values)
        # bilinear resample of the learned table, as a weighted gather
        pos = (self.pos_embed(indices) * weights[:, :, None]).sum(1)
        hidden = hidden + pos.to(hidden.dtype)

        cos, sin = self.rotary_pos_emb(position_ids)
        dtype = hidden.dtype if self.rope_dtype is None else self.rope_dtype
        return hidden, cos.to(dtype), sin.to(dtype)

    def encode(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        """The block loop, ``[patches, hidden]`` in and out -- the part a
        CUDA graph can hold (see ``VisionEncoderSubmodule``)."""
        for block in self.blocks:
            hidden = block(hidden, cos, sin)
        return hidden

    def encode_with_deepstack(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """The block loop, tapping each ``deepstack_visual_indexes`` block's output
        through its merger (``Qwen3VLVisionModel.forward``)."""
        features = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, cos, sin)
            if index in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(index)]
                features.append(merger(hidden))
        return hidden, features
