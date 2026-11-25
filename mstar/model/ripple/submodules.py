"""Ripple's one node, on the shared `DeclaredTokenDecoder`."""
from mstar.model.components.sampled_token_decoder import DeclaredTokenDecoder


class RippleLLMSubmodule(DeclaredTokenDecoder):
    # Tokens a step runs (the chunking budget). As on Qwen3.5, whose measurement stops at 9B.
    MAX_BATCH_TOKENS = 2048
    # At the default min_chunk_tokens no step, prefill or mixed, passes the budget
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048]
