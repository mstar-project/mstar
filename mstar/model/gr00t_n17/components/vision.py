"""GR00T N1.7's vision encoder: Qwen3.5's ViT tower, driven a block at a time
so the DeepStack features can be read off layers 5, 11 and 17.

The tower is ``qwen3_5``'s, unchanged: ``Qwen3_5VisionModel`` splits into
``embed`` -> blocks -> ``merger`` precisely so a caller can run the loop
itself, and it builds only with ``deepstack_visual_indexes`` empty, which is
how ``Gr00tN17Config`` stores it. The three DeepStack mergers are GR00T's own,
because Qwen3.5 has no DeepStack: they differ from the tower's merger in one
place, the LayerNorm runs *after* the 2x2 shuffle and over the shuffled width.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.model.gr00t_n17.config import Gr00tN17Config
from mstar.model.qwen3_5.components.vision import Qwen3_5VisionModel
from mstar.model.qwen3_5.config import Qwen3_5VisionConfig


class DeepStackMerger(nn.Module):
    """``use_postshuffle_norm=True`` in transformers' Qwen3-VL patch merger."""

    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()
        self.merged_size = config.hidden_size * config.merge_unit
        self.norm = nn.LayerNorm(self.merged_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_size, self.merged_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.merged_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.merged_size))
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Gr00tN17VisionEncoder(nn.Module):
    """Packed image patches in; the embeddings the LLM scatters onto its image
    rows, plus one DeepStack feature per tap, each of the same shape.

    Grid-derived inputs come from the caller, as the tower requires
    (``VisionEncoderSubmodule.prepare_inputs``, ``qwen3_5/submodules.py``).
    """

    def __init__(self, config: Gr00tN17Config):
        super().__init__()
        vision = config.vision
        taps = config.deepstack_visual_indexes
        # GR00T's config.json names the taps, the backbone's names the depth
        if taps and max(taps) >= vision.depth:
            raise ValueError(
                f"DeepStack taps {taps} do not fit a {vision.depth}-block tower"
            )
        self.tower = Qwen3_5VisionModel(vision)
        self.deepstack_merger_list = nn.ModuleList(
            DeepStackMerger(vision) for _ in taps
        )
        # block index -> which merger reads it
        self._merger_of = {layer: k for k, layer in enumerate(taps)}

    def forward(
        self,
        pixel_values: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden, _, _ = self.tower.embed(
            pixel_values, indices, weights, position_ids,
        )
        # `embed` rounds the rope to the hidden dtype; transformers' Qwen3-VL, which
        # GR00T runs, keeps it fp32 into the blocks. In bf16 the rounding alone put
        # vision_embeds at 1.25x the bf16-vs-fp32 floor, so the blocks get the fp32 table.
        cos, sin = self.tower.rotary_pos_emb(position_ids)
        features = []
        for layer, block in enumerate(self.tower.blocks):
            hidden = block(hidden, cos, sin)
            merger = self._merger_of.get(layer)
            if merger is not None:
                features.append(self.deepstack_merger_list[merger](hidden))
        return self.tower.merger(hidden), features
