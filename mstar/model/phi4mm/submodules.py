"""Phi-4-multimodal's LLM node, on the shared `SampledTokenDecoder`.

The rotation (partial, LongRoPE) and the modality LoRAs are resources the
layers reach by key, so the node declares only its KV and attention steps; the
position, sampler and LoRA steps are the scaffold's and the engine's.
"""

from __future__ import annotations

import torch

from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, check_inductor_options
from mstar.engine.resources import AttentionStep, KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.phi4mm.config import ATTN, KV_CACHE


class Phi4MMLLMSubmodule(SampledTokenDecoder):
    ATTN_KEY = ATTN
    # An image prompt is ~600-3000 tokens at everyday sizes (up to ~9.7k at 36 crops),
    # so a step of a few of them outgrows the scaffold's 4096-token ladder and ran
    # eagerly (13 of 19 image prefill steps at c=32, ~200 ms each).
    PREFILL_MEDIA_TOKEN_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 12288, 16384]
    PREFILL_MEDIA_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True)}

    def resident_walks(self) -> frozenset[str]:
        # the layers rotate inside the graph at the position resource's planned
        # table ids, and decode's `preprocess` reads only the loop-back tokens
        return frozenset({self.DECODE_WALK})

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        configs = super().get_accelerator_graph_configs(device, tp_world_size)
        for c in configs:
            if c.capture_graph_walk == self.DECODE_WALK:
                # cuBLAS for the decode GEMMs, as Cosmos3's decode: max-autotune
                # picks Triton templates for down_proj and qkv_proj that run
                # 1.2-1.3x slower in the graph at 32 rows (H200).
                c.inductor_options = check_inductor_options({"max_autotune_gemm_backends": "ATEN"})
        return configs
