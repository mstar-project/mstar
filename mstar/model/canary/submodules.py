"""Canary's text decoder node (two arrays and a sampler); the encoder node is the
shared ``FastConformerEncoderSubmodule``."""

from __future__ import annotations

from mstar.model.canary.config import CONTEXT_LABEL, CROSS_KV_CACHE, KV_CACHE
from mstar.model.components.encoder_decoder import EncoderDecoderNode


class CanaryDecoderSubmodule(EncoderDecoderNode):
    """Autoregressive Canary decoder: the task prompt at prefill, then one
    token per step."""

    PREFILL_WALK = "prefill"
    KV_CACHE = KV_CACHE
    CROSS_KV_CACHE = CROSS_KV_CACHE
    CONTEXT_LABEL = CONTEXT_LABEL
    # Prefill isn't captured, so no capture bucket bounds its batch; this does,
    # keeping one prefill's context reservation well inside the cache.
    MAX_PREFILL_BATCH_SIZE = 32
    # off: with torch.compile, Canary's throughput was 6-15% lower at c=1-64 (H100 A/B)
    disable_torch_compile = True
    # the decode ladder and token path Canary had before the scaffold: decode
    # reads its ids from the routed token and emits a tensor a request; the
    # device loop-back is a measured change of its own
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    device_loopback = None
