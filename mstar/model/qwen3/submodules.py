"""Dense Qwen3's one node, on the shared `DeclaredTokenDecoder`."""
from mstar.model.components.sampled_token_decoder import DeclaredTokenDecoder


class Qwen3LMSubmodule(DeclaredTokenDecoder):
    """``model`` is a `Qwen3DenseLM` that attends through the node's ``kv_cache`` array."""

    def resident_walks(self) -> frozenset[str]:
        # decode's preprocess reads only the loop-back tokens; its positions and KV
        # locations are planned by the resources
        return frozenset({self.DECODE_WALK})
