"""Phi-4-multimodal's LLM node, on the shared `SampledTokenDecoder`.

The rotation (partial, LongRoPE) and the modality LoRAs are resources the
layers reach by key, so the node declares only its KV and attention steps; the
position, sampler and LoRA steps are the scaffold's and the engine's.
"""

from __future__ import annotations

from mstar.engine.resources import AttentionStep, KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.phi4mm.config import ATTN, KV_CACHE


class Phi4MMLLMSubmodule(SampledTokenDecoder):
    ATTN_KEY = ATTN

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True)}
