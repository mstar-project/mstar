"""The LLM node of Canary-Qwen: dense Qwen3 with the encoder's audio
embeddings spliced into the prompt at prefill. The encoder node is the shared
``FastConformerEncoderSubmodule`` with the projection into the LLM's width."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.canary_qwen.config import ATTN, DECODE_LOOP, DECODE_WALK, KV_CACHE, ROPE, SAMPLER
from mstar.model.components.qwen3_lm import Qwen3DenseLM
from mstar.model.components.token_step import TokenStepNode
from mstar.model.multimodal import splice_media
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    BatchedModelOutput,
    ModelInputsFromEngine,
)
from mstar.utils.h2d import to_device_nonblocking


class CanaryQwenLLMSubmodule(TokenStepNode, ARNodeSubmodule):
    """Dispatches on graph_walk:
      - prefill: embed the chat prompt, put the audio embeddings over its
        placeholder tokens, fill the KV cache, sample the first token.
      - decode: embed the previous token, single-step decode.

    Decode is captured per batch size; prefill runs eager (its length
    follows the clip).
    """

    DECODE_WALK = DECODE_WALK
    DECODE_LOOP = DECODE_LOOP
    SAMPLER = SAMPLER
    # Prefill isn't captured; this bounds its batch, so one prefill's
    # reservation stays well inside the cache.
    MAX_PREFILL_BATCH_SIZE = 16
    # off: with torch.compile, Canary-Qwen's throughput was 7-15% lower at c=1-64 (H100 A/B)
    disable_torch_compile = True

    def __init__(self, llm: Qwen3DenseLM, audio_token_id: int, stop_token_ids: set[int]):
        super().__init__()
        self.model = llm
        self.audio_token_id = audio_token_id
        self.stop_token_ids = frozenset(stop_token_ids)

    def splice_audio(self, input_ids: torch.Tensor, audio_embeds: torch.Tensor) -> torch.Tensor:
        """Prompt embeddings with the k-th audio frame over the k-th
        placeholder token (``process_prompt`` writes one per frame). No host
        reads: this runs on the GPU thread, ahead of the step in flight."""
        slots = input_ids == self.audio_token_id
        # the placeholder id is past the trained vocabulary; embed 0 there
        embeds = self.model.embed(torch.where(slots, torch.zeros_like(input_ids), input_ids))
        return splice_media(embeds, slots, audio_embeds)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        device = self.get_device()
        token_ids = to_device_nonblocking(inputs["text_inputs"][0], device).reshape(-1)
        if graph_walk == DECODE_WALK:
            return ARNodeInputs(input_seq_len=1, input_ids=token_ids)
        audio_embeds = inputs["encoder_states"][0].to(device)
        return ARNodeInputs(
            input_seq_len=token_ids.shape[0],
            input_ids=token_ids,
            input_embeds=self.splice_audio(token_ids, audio_embeds),
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                KV_CACHE: KVStep(),
                ATTN: AttentionStep(causal=True),
                SAMPLER: SamplerStep(apply_penalty=False),
                ROPE: PositionStep(),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        if graph_walk == DECODE_WALK:
            return {"input_ids": torch.cat([inp.input_ids for inp in inputs])}
        return {"input_embeds": torch.cat([inp.input_embeds for inp in inputs], dim=0)}

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None,
        input_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]
        attn: AttentionManager = engine_inputs.resources[ATTN]
        if input_embeds is None:
            input_embeds = self.model.embed(input_ids)
        hidden = self.model(input_embeds, label="main")
        if graph_walk != DECODE_WALK:
            hidden = attn.select_last_hidden(hidden)
        return sampler.sample(engine_inputs.request_ids, logits=self.model.logits(hidden))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {"new_token": [self._forward(graph_walk, engine_inputs, input_ids, input_embeds)]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(graph_walk, engine_inputs, input_ids, input_embeds)
        # row i is request i: one clone and one D2H copy per step, not per request
        return BatchedModelOutput(row_outputs={"new_token": new_tokens}, check_stop_buffers={"new_token": new_tokens})

