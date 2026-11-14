"""Dense Qwen3's one node, on the shared `SampledTokenDecoder`."""
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.keys import KV_CACHE
from mstar.model.qwen3.config import ATTN


class Qwen3LMSubmodule(SampledTokenDecoder):
    """``model`` is a `Qwen3DenseLM`."""

    ATTN_KEY = ATTN

    def resident_walks(self) -> frozenset[str]:
        # decode's preprocess reads only the loop-back tokens; its positions and KV
        # locations are planned by the resources
        return frozenset({self.DECODE_WALK})

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True)}
