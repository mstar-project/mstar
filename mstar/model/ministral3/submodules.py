"""Ministral's decoder node on the shared text and media sampled-token workflow."""

from collections.abc import Sequence

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import CommGroup
from mstar.engine.resources import AttentionStep, KVStep, ResourceStep
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.keys import KV_CACHE, ROPE
from mstar.model.ministral3.components.language_model import Ministral3DenseLM
from mstar.model.ministral3.config import ATTN, Ministral3TextConfig
from mstar.model.multimodal import MediaEncoder
from mstar.model.submodule_base import ARNodeInputs


class Ministral3Decoder(SampledTokenDecoder):
    """Native tensor stack plus engine-managed sampling, positions and KV.

    Inherits the text prefill/decode/mixed walks' batching, capture declarations,
    token feedback and stop contract.
    """

    ATTN_KEY = ATTN

    def __init__(self, config: Ministral3TextConfig, *, comm_group: CommGroup | None = None,
                 encoders: Sequence[MediaEncoder] = (), model: Ministral3DenseLM | None = None):
        model = model if model is not None else Ministral3DenseLM(
            config, attn_key=ATTN, kv_key=KV_CACHE, pos_key=ROPE, comm_group=comm_group,
        )
        super().__init__(model, config, encoders=encoders)

    def get_autocast_dtype(self) -> torch.dtype:
        # Also supplies the engine's attention/KV dtype when model autocast is null.
        return torch.bfloat16

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True)}

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        # Rows arrive under their individual walk, even in a mixed batch.
        if graph_walk not in {self.PREFILL_WALK, self.DECODE_WALK, *self._media_walks}:
            raise ValueError(f"Ministral 3 decoder does not support walk {graph_walk!r}")
        allowed = {"text_inputs"}
        if graph_walk in self._media_walks:
            allowed.update(e.output for e in self._encoders)
        if extra := inputs.keys() - allowed:
            raise ValueError(f"Ministral 3 decoder got unexpected inputs: {sorted(extra)}")
        return super().prepare_inputs(graph_walk, fwd_info, inputs, **kwargs)
