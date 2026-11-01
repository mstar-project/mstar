"""Qwen3.5's vision tower: the shared Qwen3-VL ViT (``components.qwen3_vit``)
with Qwen3.5's patch merger and no deepstack.

Images arrive packed: ``pixel_values`` is ``[total_patches, patch_numel]`` and
``grid_thw`` gives each image's ``(t, h, w)`` patch grid.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.model.components.qwen3_vit import VisionBlock, VisionPatchEmbed, VisionRotaryEmbedding
from mstar.model.qwen3_5.config import VISION_ATTN, Qwen3_5VisionConfig


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
            VisionBlock(config, VISION_ATTN) for _ in range(config.depth)
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

        Grid-derived inputs are built by the caller
        (``VisionEncoderSubmodule.prepare_inputs``): ``grid_thw.tolist()`` here
        would break the graph and make h, w unbacked symints, and inductor then
        refuses (``CantSplit``) to codegen packed multi-image runs.
        """
        hidden, cos, sin = self.embed(pixel_values, indices, weights, position_ids)
        return self.merger(self.encode(hidden, cos, sin))

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
        return hidden, cos.to(hidden.dtype), sin.to(hidden.dtype)

    def encode(
        self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        """The block loop, ``[patches, hidden]`` in and out -- the part a
        CUDA graph can hold (see ``VisionEncoderSubmodule``)."""
        for block in self.blocks:
            hidden = block(hidden, cos, sin)
        return hidden
