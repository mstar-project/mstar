"""Ripple's one node, on the shared `SampledTokenDecoder`."""
from collections.abc import Mapping
from typing import Any

import torch

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
from mstar.model.ripple.config import KV_CACHE, SAMPLER
from mstar.model.submodule_base import ModelInputsFromEngine


class RippleLLMSubmodule(SampledTokenDecoder):
    # Tokens a step runs (the chunking budget). As on Qwen3.5, whose measurement stops at 9B.
    MAX_BATCH_TOKENS = 2048
    # At the default min_chunk_tokens no step, prefill or mixed, passes the budget
    PREFILL_TOKEN_BUCKETS = [32, 64, 128, 256, 512, 1024, 2048]

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
