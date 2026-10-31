"""Qwen3.5's vision tower: the shared Qwen tower (``components/qwen_vision.py``)
under its Qwen3.5 names. Qwen3.5 has no deepstack, and a checkpoint that sets it is
refused here."""
from __future__ import annotations

from mstar.model.components.qwen_vision import (  # noqa: F401
    QwenVisionModel,
    VisionAttention,
    VisionBlock,
    VisionMLP,
    VisionPatchEmbed,
    VisionPatchMerger,
    VisionRotaryEmbedding,
    vision_interpolation,
    vision_position_ids,
    vision_seq_lengths,
)
from mstar.model.qwen3_5.config import VISION_ATTN, Qwen3_5VisionConfig


class Qwen3_5VisionModel(QwenVisionModel):
    def __init__(self, config: Qwen3_5VisionConfig):
        if config.deepstack_visual_indexes:
            raise NotImplementedError(
                "deepstack is not wired up for Qwen3.5: no released Qwen3.5 sets it, "
                "and neither does transformers' Qwen3.5"
            )
        # the key Qwen3.5 declares its tower's resource under; rotary tables in the
        # activations' dtype, as this tower has always run (transformers keeps fp32)
        super().__init__(config, resource_key=VISION_ATTN, rope_dtype=None)
