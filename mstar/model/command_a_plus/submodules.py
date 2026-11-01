"""Packed prefill, one-token-per-request decode, and the image encoder for Command A+."""

from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig
from mstar.engine.resources import AttentionStep, KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.command_a_plus.components.vision import CommandAPlusVisionEncoder
from mstar.model.command_a_plus.config import GLOBAL_ATTN, KV_CACHE, LOCAL_ATTN
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.submodule_base import NodeInputs, NodeSubmodule


class CommandAPlusVisionSubmodule(NodeSubmodule):
    """Turns each request's image tiles into the embeddings for its ``<|IMG_PATCH|>`` tokens."""

    disable_torch_compile = True

    # On GB200 a tile costs ~1.3 ms from 64 tiles up and ~26 MiB of activations,
    # so larger calls only grow the peak; a step's tile count is otherwise unbounded.
    TILES_PER_CALL = 64

    def __init__(self, encoder: CommandAPlusVisionEncoder):
        super().__init__()
        self.encoder = encoder

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        tiles = torch.cat(inputs["pixel_values"]).to(self.get_device())
        return NodeInputs(tensor_inputs={"pixel_values": tiles})

    def preprocess(self, graph_walk, engine_inputs, inputs) -> dict[str, Any]:
        tiles = [inp.tensor_inputs["pixel_values"] for inp in inputs]
        return {"pixel_values": torch.cat(tiles), "tile_counts": [len(t) for t in tiles]}

    def forward(self, graph_walk, engine_inputs, pixel_values, **kwargs) -> NameToTensorList:
        return {"image_embeds": [self.encoder(pixel_values)]}

    def can_batch(self, batch, model_inputs) -> bool:
        return True

    def forward_batched(
        self, graph_walk, engine_inputs, pixel_values, tile_counts, **kwargs,
    ) -> dict[str, NameToTensorList]:
        tokens = self.encoder.config.tokens_per_tile
        embeds = torch.cat([self.encoder(chunk) for chunk in pixel_values.split(self.TILES_PER_CALL)])
        embeds = embeds.split([n * tokens for n in tile_counts])
        return {
            rid: {"image_embeds": [e]}
            for rid, e in zip(engine_inputs.request_ids, embeds, strict=True)
        }


class CommandAPlusLLMSubmodule(SampledTokenDecoder):
    """The Command A+ text stack; image rows are spliced in at their pads."""

    # Decode replays CUDA graphs of the same eager kernels; torch.compile stays
    # off until compiled numerics get their own parity checks.
    disable_torch_compile = True
    PREFILL_WALK = "prefill"
    # both attention resources plan the same packing
    ATTN_KEY = LOCAL_ATTN
    # the ladder it captured before moving onto the scaffold: wider decode
    # steps are unmeasured on this MoE (graph memory, fused-MoE tiles)
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    def layer_steps(self) -> dict[str, ResourceStep]:
        # one KV cache, read through a windowed and an unwindowed resource
        return {
            KV_CACHE: KVStep(),
            LOCAL_ATTN: AttentionStep(causal=True),
            GLOBAL_ATTN: AttentionStep(causal=True),
        }

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        # Prefill stays eager: its captures are unmeasured on this model. Decode
        # captures uncompiled, as the eager forward runs (disable_torch_compile).
        decode = [
            config for config in super().get_accelerator_graph_configs(device, tp_world_size)
            if config.capture_graph_walk == self.DECODE_WALK
        ]
        for config in decode:
            config.compile = False
        return decode
