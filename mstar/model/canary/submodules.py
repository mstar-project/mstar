"""Canary's text decoder node (KV, attention, cross-attention, positions,
sampler); the encoder node is the shared ``FastConformerEncoderSubmodule``."""

from __future__ import annotations

from mstar.engine.resources import AttentionStep, KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.canary.config import ATTN, CONTEXT_LABEL, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE, POS, PROMPT_LENGTH
from mstar.model.components.encoder_decoder import EncoderDecoderNode


class CanaryDecoderSubmodule(EncoderDecoderNode):
    """Autoregressive Canary decoder: the task prompt at prefill, then one
    token per step."""

    PREFILL_WALK = "prefill"
    ATTN_KEY = ATTN
    POS_KEY = POS
    CROSS_KV_CACHE = CROSS_KV_CACHE
    CROSS_ATTN = CROSS_ATTN
    CONTEXT_LABEL = CONTEXT_LABEL
    PREFILL_PROMPT_TOKENS = PROMPT_LENGTH
    # keeps one prefill's context reservation well inside the cache
    MAX_PREFILL_BATCH_SIZE = 32
    # off: with torch.compile, Canary's throughput was 6-15% lower at c=1-64 (H100 A/B)
    disable_torch_compile = True
    # up to the 64-request cap (`configs/canary.yaml`); the scaffold's ladder runs to 128
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True)}

    def resident_walks(self) -> frozenset[str]:
        # one token a row in the main stream; the context is read, not written,
        # and positions are read in the forward, not in `preprocess`
        return frozenset({self.DECODE_WALK})
