"""Phi-4-multimodal's LLM node, on the shared `DeclaredTokenDecoder`.

The rotation (partial, LongRoPE) is the KV array's declared `Rope` and the modality
LoRAs are a resource the layers reach by key, so the node declares only its array's
and its sampler's steps; the LoRA step is the engine's.
"""

from __future__ import annotations

import torch

from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, check_inductor_options
from mstar.model.components.sampled_token_decoder import DeclaredTokenDecoder


class Phi4MMLLMSubmodule(DeclaredTokenDecoder):
    # An image prompt is ~600-3000 tokens at everyday sizes (up to ~9.7k at 36 crops),
    # so a step of a few of them outgrows the scaffold's 4096-token ladder and ran
    # eagerly (13 of 19 image prefill steps at c=32, ~200 ms each).
    PREFILL_MEDIA_TOKEN_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 12288, 16384]
    PREFILL_MEDIA_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]

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
