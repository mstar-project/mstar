"""Canary's text decoder node (KV, attention, cross-attention, positions,
sampler); the encoder node is the shared ``FastConformerEncoderSubmodule``."""

from __future__ import annotations

from mstar.model.canary.config import (
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    KV_CACHE,
    POS,
    SAMPLER,
    CanaryModelConfig,
)
from mstar.model.canary.decoder import CanaryDecoderModel
from mstar.model.components.encoder_decoder import EncoderDecoderNode
from mstar.model.submodule_base import ARNodeSubmodule


class CanaryDecoderSubmodule(EncoderDecoderNode, ARNodeSubmodule):
    """Autoregressive Canary decoder: the task prompt at prefill, then one
    token per step. Prefill isn't captured: its shape follows the clip."""

    KV_CACHE = KV_CACHE
    ATTN = ATTN
    POS = POS
    SAMPLER = SAMPLER
    CROSS_KV_CACHE = CROSS_KV_CACHE
    CROSS_ATTN = CROSS_ATTN
    CONTEXT_LABEL = CONTEXT_LABEL
    # Prefill isn't captured, so no capture bucket bounds its batch; this does,
    # keeping one prefill's context reservation well inside the cache.
    MAX_PREFILL_BATCH_SIZE = 32
    # off: with torch.compile, Canary's throughput was 6-15% lower at c=1-64 (H100 A/B)
    disable_torch_compile = True

    def __init__(self, decoder: CanaryDecoderModel, config: CanaryModelConfig):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self.stop_token_ids = frozenset({config.eos_token_id})
