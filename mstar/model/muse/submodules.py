"""The Muse node submodules: the LLM (and Spark's, over its experts) and the
vision encoder; Spark's audio encoder is a `ChunkedCausalEncoder` as it is.

They build on the shared scaffolds (`SampledTokenDecoder`,
`PackedBlockLoopEncoder`). Positions are plain 1D and
advance one per token, media included, so the position resource's own
counters place every step.
"""

from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
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
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine, NodeInputs


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


class SparkLLMSubmodule(LLMSubmodule):
    """The LLM node over Spark's experts: their expert-load log, and a
    replay's padding rows kept out of them, as the node group's ``ep_config``
    asks (``log_expert_load_every``, ``mask_padding``)."""

    def preprocess(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine, inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        out = super().preprocess(graph_walk, engine_inputs, inputs)
        ids = out["input_ids"]
        out.update(self.model.model.moe_step_inputs(engine_inputs, ids.shape[0], ids.device))
        return out

    def decoder_kwargs(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        num_tokens: int, **tensors: torch.Tensor,
    ) -> dict[str, Any]:
        return {"num_valid_tokens": tensors["num_valid_tokens"]} if "num_valid_tokens" in tensors else {}


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

