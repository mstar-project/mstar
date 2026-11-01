"""The LLM node of Canary-Qwen: dense Qwen3 on the shared `SampledTokenDecoder`,
the whole prompt prefilled as one row with the encoder's audio frames spliced
in at its placeholders. The encoder node is the shared
``FastConformerEncoderSubmodule`` with the projection into the LLM's width."""
from __future__ import annotations

import torch

from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig
from mstar.engine.resources import AttentionStep, KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.model.canary_qwen.config import ATTN, KV_CACHE
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder


class CanaryQwenLLMSubmodule(SampledTokenDecoder):
    ATTN_KEY = ATTN
    # off: with torch.compile, Canary-Qwen's throughput was 7-15% lower at c=1-64 (H100 A/B)
    disable_torch_compile = True
    # the ladder it captured before moving onto the scaffold
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    # Prefill isn't captured (below); this caps its batch, so one prefill's
    # reservation stays well inside the cache.
    PREFILL_MEDIA_CAPTURE_BATCH_SIZES = [16]
    # decode reads its ids back from the routed token and emits a tensor a
    # request, as before the scaffold; the device path is a measured change
    reads_device_loopback = False

    def inline_client_signals(self, graph_walk: str) -> dict[str, str]:
        return {}

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True)}

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        # decode only, as before the scaffold: a prefill's length follows its clip
        return [
            config for config in super().get_accelerator_graph_configs(device, tp_world_size)
            if config.capture_graph_walk == self.DECODE_WALK
        ]
