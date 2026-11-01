"""Canary's text decoder node (KV, attention, cross-attention, positions,
sampler); the encoder node is the shared ``FastConformerEncoderSubmodule``."""

from __future__ import annotations

from mstar.engine.resources import AttentionStep, KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.canary.config import ATTN, CONTEXT_LABEL, CROSS_ATTN, CROSS_KV_CACHE, KV_CACHE
from mstar.model.components.encoder_decoder import EncoderDecoderNode


class CanaryDecoderSubmodule(EncoderDecoderNode):
    """Autoregressive Canary decoder: the task prompt at prefill, then one
    token per step."""

    PREFILL_WALK = "prefill"
    ATTN_KEY = ATTN
    CROSS_KV_CACHE = CROSS_KV_CACHE
    CROSS_ATTN = CROSS_ATTN
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
    reads_device_loopback = False

    def inline_client_signals(self, graph_walk: str) -> dict[str, str]:
        return {}

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True)}
