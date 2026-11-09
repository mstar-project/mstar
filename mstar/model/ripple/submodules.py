"""Ripple's one node, on the shared `SampledTokenDecoder`."""
from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.ripple.config import ATTN_FULL, ATTN_SLIDING, KV_CACHE


class RippleLLMSubmodule(SampledTokenDecoder):
    # either attention resource's plan picks each prefill row's last token
    ATTN_KEY = ATTN_FULL
    # Tokens a step runs (the chunking budget). As on Qwen3.5, whose measurement stops at 9B.
    MAX_BATCH_TOKENS = 2048
    # At the default min_chunk_tokens no step, prefill or mixed, passes the budget
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048]

    def layer_steps(self) -> dict[str, ResourceStep]:
        # one KV cache, read through a windowed and an unwindowed resource
        return {
            KV_CACHE: KVStep(),
            ATTN_SLIDING: AttentionStep(causal=True),
            ATTN_FULL: AttentionStep(causal=True),
        }
