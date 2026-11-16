"""Independently placeable, eager Pixtral node over prepared image patches."""

import torch

from mstar.engine.resources import NodeResourceSpec, RaggedAttentionConfig, RaggedAttentionSpec
from mstar.model.components.block_loop_encoder import PackedBlockLoopEncoder
from mstar.model.components.qwen3_vit import vision_position_ids, vision_seq_lengths
from mstar.model.keys import VISION_ATTN
from mstar.model.ministral3.components.vision import Ministral3Vision, PixtralConfig
from mstar.model.ministral3.image import IMAGE_GRIDS
from mstar.model.multimodal import MEDIA_LAYOUT
from mstar.model.submodule_base import NodeInputs

VISION_NODE = "vision_encoder"
VISION_OUTPUT = "image_embeds"


def vision_node_resources(config: PixtralConfig) -> list[NodeResourceSpec]:
    return [RaggedAttentionSpec(
        resource_key=VISION_ATTN, nodes={VISION_NODE}, config=RaggedAttentionConfig(
            num_qo_heads=config.num_attention_heads, num_kv_heads=config.num_attention_heads,
            head_dim=config.head_dim,
        ),
    )]


class Ministral3VisionEncoder(PackedBlockLoopEncoder):
    # Eager execution uses the shared encoder's batching without a capture recipe.
    def __init__(self, config: PixtralConfig, *, model: Ministral3Vision | None = None):
        super().__init__(model if model is not None else Ministral3Vision(config), config, VISION_OUTPUT)

    def get_autocast_dtype(self) -> torch.dtype:
        return torch.bfloat16

    def get_piecewise_accelerator_graph_configs(self, *args, **kwargs):
        return {}

    def can_batch(self, batch, model_inputs):
        # Eager execution needs no capture bucket; max_batch_size still limits selection.
        return True

    def grids(self, fwd_info):
        # Decoder pad runs end at each row delimiter; attention spans whole images.
        layout = fwd_info.step_metadata.get(MEDIA_LAYOUT) or {}
        return [tuple(grid) for grid in layout.get(IMAGE_GRIDS, ())]

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs):
        grids = self.grids(fwd_info)
        merge = self.config.spatial_merge_size
        if not grids or any(t != 1 or min(h, w) < 1 or h % merge or w % merge for t, h, w in grids):
            raise ValueError("Pixtral requires image grids divisible by spatial_merge_size")
        pixels = torch.cat(inputs["pixel_values"])
        lengths = vision_seq_lengths(grids)
        columns = self.config.num_channels * self.config.patch_size ** 2
        if pixels.shape != (sum(lengths), columns):
            raise ValueError("Pixtral packed patches disagree with the declared image grids")
        return NodeInputs(
            input_seq_len=sum(lengths),
            tensor_inputs={"pixel_values": pixels, "position_ids": vision_position_ids(grids, merge)},
            resource_step_info={"main": lengths},
        )
