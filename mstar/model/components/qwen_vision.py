"""Qwen-VL vision towers over the shared ViT parts (``components.qwen3_vit``).

``QwenVisionModel`` is Qwen3-VL's: a learned position grid, Qwen3-VL's patch mergers and
deepstack, and the ``vision_config`` reader Qwen3.5 shares. Deepstack: mergers after some
blocks whose features the language model adds to its first layers' hidden states at the image
rows (``deepstack_visual_indexes``; empty for Qwen3.5). Parameter names follow transformers'
``Qwen3VLVisionModel``, so its weights load by name.

``Qwen25VisionModel`` is Qwen2.5-VL's: no position table, RMSNorm and gated-SiLU blocks, and
window attention (most blocks attend within windows of merge units, ``fullatt_block_indexes``
over whole frames). Parameter names follow ``Qwen2_5_VisionTransformerPretrainedModel`` except
the merger's (``ln_q`` / ``mlp.0`` / ``mlp.2``), which a loader maps.
"""
from __future__ import annotations

import functools
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn

from mstar.model.components.mlp import GatedMLP
from mstar.model.components.norm import QwenRMSNorm
from mstar.model.components.qwen3_vit import (
    VisionBlock,
    VisionPatchEmbed,
    VisionRotaryEmbedding,
    vision_position_ids,
    window_order,
)
from mstar.model.keys import VISION_ATTN, VISION_FULL, VISION_WINDOW


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


class VisionPatchMerger(nn.Module):
    """Folds each ``m x m`` block of patches into one LLM-width token. The norm runs
    per patch before the fold, or (``postshuffle_norm``, Qwen3-VL's deepstack mergers)
    over the folded token."""

    def __init__(
        self, config: QwenVisionConfig | Qwen25VisionConfig, postshuffle_norm: bool = False,
        norm: Callable[[int], nn.Module] | None = None,
    ):
        super().__init__()
        self.merged_size = config.hidden_size * config.merge_unit
        self.postshuffle_norm = postshuffle_norm
        norm = norm or functools.partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm(self.merged_size if postshuffle_norm else config.hidden_size)
        self.linear_fc1 = nn.Linear(self.merged_size, self.merged_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.merged_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.postshuffle_norm:
            x = self.norm(x.view(-1, self.merged_size))
        else:
            x = self.norm(x).view(-1, self.merged_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class ConvPatchEmbed(VisionPatchEmbed):
    """The patch embed as transformers runs it, a strided conv. ``VisionPatchEmbed``'s
    matmul is faster on long videos but rounds differently in bf16 at Qwen3-VL's width."""

    lora_unservable = {"proj": "run as transformers' conv, never its own forward"}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        # the conv itself: ``proj``'s own forward is the shared matmul
        return nn.Conv3d.forward(self.proj, x.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


class QwenVisionModel(nn.Module):
    """Rounds as transformers' tower does: the patch embed is its conv, and the rotary
    tables go into the blocks fp32 (Qwen3.5's and Qwen3-Omni's towers round them to the
    activations' dtype)."""

    def __init__(self, config: QwenVisionConfig, attn_key: str = VISION_ATTN):
        super().__init__()
        self.config = config
        self.patch_embed = ConvPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.rotary_pos_emb = VisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList(VisionBlock(config, attn_key) for _ in range(config.depth))
        self.merger = VisionPatchMerger(config)
        self.deepstack_visual_indexes = tuple(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            VisionPatchMerger(config, postshuffle_norm=True) for _ in self.deepstack_visual_indexes
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``[total_patches, patch_numel]`` in; ``[merged_tokens, out_hidden]`` and one such
        tensor per deepstack merger out (``Qwen3VLVisionModel.forward``). Grid-derived inputs
        come from the caller (``qwen3_vit``'s grid helpers), so the tower has no host reads."""
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

        features = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, cos, sin)
            if index in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(index)]
                features.append(merger(hidden))
        return self.merger(hidden), features


# ----------------------------------------------------------------------------- Qwen2.5-VL


@dataclass
class Qwen25VisionConfig:
    """Qwen2.5-VL's tower config, read off ``config.json``'s ``vision_config``."""

    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    out_hidden_size: int
    window_size: int                       # in pixels
    fullatt_block_indexes: tuple[int, ...]
    hidden_act: str = "silu"
    rope_theta: float = 10_000.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge_size**2

    @property
    def window_units(self) -> int:
        """A window's side in merge units."""
        return self.window_size // self.spatial_merge_size // self.patch_size

    @classmethod
    def from_dict(cls, raw: dict) -> "Qwen25VisionConfig":
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in raw.items() if k in known}
        values["fullatt_block_indexes"] = tuple(values["fullatt_block_indexes"])
        if values.get("hidden_act", "silu") != "silu":
            raise ValueError(f"Qwen2.5-VL vision activation {values['hidden_act']!r} is not wired up")
        return cls(**values)


# transformers' merger names -> ``VisionPatchMerger``'s; every other name matches
_QWEN25_MERGER = {
    "merger.ln_q.": "merger.norm.", "merger.mlp.0.": "merger.linear_fc1.", "merger.mlp.2.": "merger.linear_fc2.",
}


# the tower's prefixes in a Qwen2.5-VL checkpoint, as published and as transformers 5 saves it
QWEN25_VISION_PREFIXES = ("visual.", "model.visual.")


def qwen25_vision_key(name: str) -> str:
    """A ``Qwen2_5_VisionTransformerPretrainedModel`` parameter's name (or a checkpoint key under its prefix) in
    ``Qwen25VisionModel``."""
    for prefix in QWEN25_VISION_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for theirs, ours in _QWEN25_MERGER.items():
        if name.startswith(theirs):
            return ours + name[len(theirs):]
    return name


class Qwen25TowerInputs(NamedTuple):
    """Everything the tower needs besides the pixels, for a packed set of grids, on the host."""

    rope_positions: torch.Tensor     # [P, 2] (h, w), in window order
    window_index: torch.Tensor       # [P] merge-block order -> window order
    restore_index: torch.Tensor      # [N] merged tokens, window order -> merge-block order
    window_lengths: tuple[int, ...]  # patches per window
    frame_lengths: tuple[int, ...]   # patches per frame


def qwen25_tower_inputs(grid: Sequence[tuple[int, int, int]], config: Qwen25VisionConfig) -> Qwen25TowerInputs:
    """The tower's index tensors for ``grid`` (``get_window_index`` and ``rot_pos_emb``). The
    processor's patches come in merge-block order, so the windows tile the merged grid and a
    window's patches are its merge units' patches."""
    merge, unit = config.spatial_merge_size, config.merge_unit
    units, unit_lengths = window_order([(t, h // merge, w // merge) for t, h, w in grid], config.window_units)
    window_index = (units[:, None] * unit + torch.arange(unit)).reshape(-1)
    return Qwen25TowerInputs(
        rope_positions=vision_position_ids(list(grid), merge)[window_index],
        window_index=window_index,
        restore_index=torch.argsort(units),
        window_lengths=tuple(n * unit for n in unit_lengths),
        frame_lengths=tuple(h * w for t, h, w in grid for _ in range(t)),
    )


class Qwen25VisionModel(nn.Module):
    """Qwen2.5-VL's tower (``Qwen2_5_VisionTransformerPretrainedModel``): the patches go into
    window order, each block attends within its label's layout (``window`` / ``full``), the
    merger folds each merge block, and the merged tokens go back to the processor's order.

    The blocks attend through ``attn_key``, a ragged resource or a stand-in such as
    ``FrameSDPA``, planned per label from ``Qwen25TowerInputs``' lengths."""

    def __init__(self, config: Qwen25VisionConfig, attn_key: str = VISION_ATTN):
        super().__init__()
        self.config = config
        self.patch_embed = ConvPatchEmbed(config, bias=False)
        # derived, not in the checkpoint: fp32 through every move and cast, as the reference keeps it
        self.rotary_pos_emb = VisionRotaryEmbedding(config, keep_fp32=True)
        norm = functools.partial(QwenRMSNorm, eps=1e-6)
        full = set(config.fullatt_block_indexes)
        self.blocks = nn.ModuleList(
            VisionBlock(
                config, attn_key, VISION_FULL if index in full else VISION_WINDOW, norm=norm,
                mlp=GatedMLP(config.hidden_size, config.intermediate_size, activation="silu", bias=True),
            )
            for index in range(config.depth)
        )
        self.merger = VisionPatchMerger(config, norm=norm)

    def forward(
        self, pixel_values: torch.Tensor, rope_positions: torch.Tensor, window_index: torch.Tensor,
        restore_index: torch.Tensor,
    ) -> torch.Tensor:
        """``[total_patches, patch_numel]`` in merge-block order -> ``[merged_tokens, out_hidden]``."""
        hidden = self.patch_embed(pixel_values)[window_index]
        cos, sin = self.rotary_pos_emb(rope_positions)
        for block in self.blocks:
            hidden = block(hidden, cos, sin)
        return self.merger(hidden)[restore_index]
