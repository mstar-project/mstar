"""Phi-4-multimodal's LLM node, on the shared `SampledTokenDecoder`.

The rotation (partial, LongRoPE) is the KV array's declared `Rope` and the modality
LoRAs are a resource the layers reach by key, so the node declares only its array's
and its sampler's steps; the LoRA step is the engine's.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, check_inductor_options
from mstar.engine.state import (
    ArrayStep,
    SampleStep,
    Segment,
    SubmoduleStep,
    at_last_token,
    last_sampled,
    position,
    sample,
)
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.phi4mm.config import KV_CACHE, SAMPLER
from mstar.model.submodule_base import ModelInputsFromEngine


class Phi4MMLLMSubmodule(SampledTokenDecoder):
    # An image prompt is ~600-3000 tokens at everyday sizes (up to ~9.7k at 36 crops),
    # so a step of a few of them outgrows the scaffold's 4096-token ladder and ran
    # eagerly (13 of 19 image prefill steps at c=32, ~200 ms each).
    PREFILL_MEDIA_TOKEN_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 12288, 16384]
    PREFILL_MEDIA_CAPTURE_BATCH_SIZES = [1, 2, 4, 8]

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.kv = handles[KV_CACHE]
        self.sampler = handles[SAMPLER]

    def node_step(
        self, segments: list[Segment], prefill_tokens: Mapping[int, torch.Tensor], advance: tuple[int, ...] | None,
    ) -> SubmoduleStep:
        return SubmoduleStep(
            segments=segments,
            state={KV_CACHE: ArrayStep(advance=advance), SAMPLER: SampleStep(prompt_tokens=prefill_tokens)},
        )

    def next_position(self, rid_handle: int) -> int:
        return position(self.kv, rid_handle, "main")

    def loopback_tokens(self, engine_inputs: ModelInputsFromEngine) -> torch.Tensor:
        return last_sampled(self.sampler)

    def last_token_hidden(self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor) -> torch.Tensor:
        return at_last_token(self.kv, hidden)

    def sample_tokens(self, engine_inputs: ModelInputsFromEngine, logits: torch.Tensor) -> torch.Tensor:
        return sample(self.sampler, logits)

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
