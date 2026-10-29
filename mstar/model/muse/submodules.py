"""Muse Glimmer's node submodules: the LLM and the vision encoder.

Both build on the shared scaffolds (`SampledTokenDecoder`,
`PackedBlockLoopEncoder`). Positions are plain 1D and advance one per token,
images included, so the position resource's own counters place every step.
"""

from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import PiecewiseAcceleratorGraphConfig
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.components.block_loop_encoder import PackedBlockLoopEncoder
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.muse.components.vision import tower_inputs
from mstar.model.muse.config import (
    ATTN_FULL,
    ATTN_SLIDING,
    KV_CACHE,
    VISION_FULL,
    VISION_WINDOW,
)
from mstar.model.submodule_base import NodeInputs


class LLMSubmodule(SampledTokenDecoder):
    # either attention resource's plan picks each prefill row's last token
    ATTN_KEY = ATTN_FULL
    # max-autotune until the default compile level (mstar serve -O) is measured on this model
    CAPTURE_COMPILE_MODE = "max-autotune-no-cudagraphs"

    def layer_steps(self) -> dict[str, ResourceStep]:
        # one KV cache, read through a windowed and an unwindowed resource
        return {
            KV_CACHE: KVStep(),
            ATTN_SLIDING: AttentionStep(causal=True),
            ATTN_FULL: AttentionStep(causal=True),
        }


class VisionEncoderSubmodule(PackedBlockLoopEncoder):
    """The perception encoder: every image of a prompt packed into one call.

    The grid maths runs in `prepare_inputs`, and both attention layouts, the
    windows and the whole frames, are declared as step segments, so the
    forward never sees a boundary.
    """

    SEGMENT_LABELS = (VISION_WINDOW, VISION_FULL)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> NodeInputs:
        grid = self.grids(fwd_info)
        ti = tower_inputs(grid, self.config)
        return NodeInputs(
            input_seq_len=sum(t * h * w for t, h, w in grid),
            tensor_inputs={
                "pixel_values": torch.cat(inputs["pixel_values"], dim=0),
                "interp_indices": ti.interp_indices,
                "interp_weights": ti.interp_weights,
                "window_index": ti.window_index,
                "rope_positions": ti.rope_positions,
                "output_index": ti.output_index,
            },
            resource_step_info={VISION_WINDOW: ti.window_lengths, VISION_FULL: ti.frame_lengths},
        )

    def get_piecewise_accelerator_graph_configs(
        self, device: torch.device, autocast_dtype: torch.dtype,
        tp_world_size: int = 1, **kwargs,
    ) -> dict[str, PiecewiseAcceleratorGraphConfig]:
        # eager for now
        return {}
