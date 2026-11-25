"""Muse Glimmer's node submodules: the LLM and the vision encoder.

Both build on the shared scaffolds (`DeclaredTokenDecoder`,
`PackedBlockLoopEncoder`). Positions are plain 1D and advance one per token,
images included, so the position resource's own counters place every step.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.state import (
    ArrayStep,
    SubmoduleStep,
    kernel_setting,
)
from mstar.model.components.block_loop_encoder import PackedBlockLoopEncoder
from mstar.model.components.sampled_token_decoder import DeclaredTokenDecoder
from mstar.model.muse.components.vision import tower_inputs
from mstar.model.muse.config import (
    VISION_ATTN,
    VISION_FRAME_TOKENS,
    VISION_FULL,
    VISION_WINDOW,
    VISION_WINDOW_TOKENS,
)
from mstar.model.submodule_base import NodeInputs


class LLMSubmodule(DeclaredTokenDecoder):
    pass


class VisionEncoderSubmodule(PackedBlockLoopEncoder):
    """The perception encoder: every image of a prompt packed into one call.

    The grid maths runs in `prepare_inputs`, and both attention layouts, the
    windows and the whole frames, are declared as step segments, so the
    forward never sees a boundary. The block loop replays as a captured
    region, one prompt per replay, its windows and frames planned per replay;
    patch embed and the 2x2 merge run eagerly around it.
    """

    SEGMENT_LABELS = (VISION_WINDOW, VISION_FULL)
    # `embed` ends on `ln_pre`, which autocast runs and returns in fp32, so
    # the block loop's residual stream and rope tables are fp32
    BLOCK_LOOP_DTYPE = torch.float32
    # One request a step: ``window_index`` and ``output_index`` index a
    # request's own patches, so two requests' do not pack by concatenation,
    # and the config has no ``spatial_merge_size`` to cut the output apart by.
    MAX_BATCH_SIZE = 1
    # the array each label's segments lay out
    _TOKENS = {VISION_WINDOW: VISION_WINDOW_TOKENS, VISION_FULL: VISION_FRAME_TOKENS}

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.vision = handles[VISION_WINDOW_TOKENS]

    def _declare(
        self, rows: Sequence[tuple[Any, Mapping[str, Sequence[int]]]], attention: bool = True,
    ) -> SubmoduleStep:
        segments = super()._declare(rows).segments
        state = {
            array: ArrayStep(segments=tuple(segment for segment in segments if segment.label == label))
            for label, array in self._TOKENS.items()
        } if attention else {}
        return SubmoduleStep(segments=segments, state=state)

    def max_segments_per_step(self) -> int:
        return kernel_setting(self.vision, VISION_ATTN, "max_segments_per_step")

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
