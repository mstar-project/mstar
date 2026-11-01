"""Qwen3-Omni's vision tower: the shared Qwen3-VL ViT (``components.qwen3_vit``)
plus deepstack, i.e. the hidden states after a few chosen blocks, each folded
by its own patch merger, which the Thinker adds to its first layers' vision
rows.

Parameter names follow the checkpoint's ``thinker.visual.*`` with the prefix
stripped, so it loads by name: ``merger`` and ``merger_list.{i}`` are
``Qwen3OmniMoeVisionPatchMerger`` (``ln_q``, ``mlp.0``, ``mlp.2``).
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.model.components.qwen3_vit import VisionBlock, VisionPatchEmbed, VisionRotaryEmbedding
from mstar.model.qwen3_omni.config import VisionEncoderConfig


class OmniPatchMerger(nn.Module):
    """Folds each ``m x m`` block of patches into one Thinker-width token.

    The main merger normalizes each patch before folding; the deepstack ones
    (``postshuffle``) normalize the folded vector.
    """

    def __init__(self, config: VisionEncoderConfig, postshuffle: bool):
        super().__init__()
        self.merged_size = config.hidden_size * config.merge_unit
        self.postshuffle = postshuffle
        self.ln_q = nn.LayerNorm(self.merged_size if postshuffle else config.hidden_size, eps=1e-6)
        self.mlp = nn.ModuleList([
            nn.Linear(self.merged_size, self.merged_size),
            nn.GELU(),
            nn.Linear(self.merged_size, config.out_hidden_size),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.merged_size) if self.postshuffle else x
        x = self.ln_q(x).view(-1, self.merged_size)
        for layer in self.mlp:
            x = layer(x)
        return x


class Qwen3OmniVisionTower(nn.Module):
    def __init__(self, config: VisionEncoderConfig, attn_key: str):
        super().__init__()
        self.config = config
        self.patch_embed = VisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.rotary_pos_emb = VisionRotaryEmbedding(config)
        self.blocks = nn.ModuleList(VisionBlock(config, attn_key) for _ in range(config.depth))
        self.merger = OmniPatchMerger(config, postshuffle=False)
        self.merger_list = nn.ModuleList(
            OmniPatchMerger(config, postshuffle=True) for _ in config.deepstack_visual_indexes
        )

    def _apply(self, fn, recurse=True):
        """Keep the rotary frequencies fp32 through a dtype cast (the worker
        casts every submodule to its node's dtype), as the reference does: in
        bf16 the far patches of a large image rotate by visibly wrong angles."""
        out = super()._apply(fn, recurse)
        rotary = self.rotary_pos_emb
        if rotary.inv_freq.dtype != torch.float32 and rotary.inv_freq.device.type != "meta":
            rotary.inv_freq = VisionRotaryEmbedding(self.config).inv_freq.to(rotary.inv_freq.device)
        return out

    def embed(
        self,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Patch embed plus the resampled position table, and the rope for
        the blocks: everything before the block loop, per patch. Grid-derived
        inputs come from the caller, so the tower has no host reads."""
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
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """The block loop, ``[patches, hidden]`` in and out, with the hidden
        states after each deepstack block (unmerged)."""
        taps = []
        for i, block in enumerate(self.blocks):
            hidden = block(hidden, cos, sin)
            if i in self.config.deepstack_visual_indexes:
                taps.append(hidden)
        return hidden, taps

    def merge(
        self, hidden: torch.Tensor, taps: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """The merged embeddings and the merged deepstack features."""
        return self.merger(hidden), [m(t) for m, t in zip(self.merger_list, taps, strict=True)]

    def forward(
        self,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """``[total_patches, patch_numel]`` in; ``[merged_tokens, out_hidden]``
        and one such deepstack tensor per deepstack block out."""
        hidden, cos, sin = self.embed(pixel_values, indices, weights, position_ids)
        return self.merge(*self.encode(hidden, cos, sin))
