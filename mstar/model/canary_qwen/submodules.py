"""The LLM node of Canary-Qwen: dense Qwen3 on the shared `SampledTokenDecoder`,
the whole prompt prefilled as one row with the encoder's audio frames spliced
in at its placeholders. The encoder node is the shared
``FastConformerEncoderSubmodule`` with the projection into the LLM's width."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.engine.state import ArrayStep, SampleStep, Segment, SubmoduleStep, at_last_token, position, sample
from mstar.model.canary_qwen.config import KV_CACHE, SAMPLER
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.submodule_base import ModelInputsFromEngine


class CanaryQwenLLMSubmodule(SampledTokenDecoder):
    # off: with torch.compile, Canary-Qwen's throughput was 7-15% lower at c=1-64 (H100 A/B)
    disable_torch_compile = True
    # the ladder it captured before moving onto the scaffold
    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
    # decode only, as before the scaffold: a prefill's length follows its clip
    CAPTURE_PREFILL = False
    # Prefill isn't captured; this caps its batch, so one prefill's
    # reservation stays well inside the cache.
    PREFILL_MEDIA_CAPTURE_BATCH_SIZES = [16]
    # decode reads its ids back from the routed token and emits a tensor a
    # request, as before the scaffold; the device path is a measured change
    device_loopback = None

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

    def last_token_hidden(self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor) -> torch.Tensor:
        return at_last_token(self.kv, hidden)

    def sample_tokens(self, engine_inputs: ModelInputsFromEngine, logits: torch.Tensor) -> torch.Tensor:
        return sample(self.sampler, logits)
