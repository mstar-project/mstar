"""The decoder node of an encoder-decoder speech model (Canary): the
encoder's output added once to a cross-attention context at prefill, then
one token per decode step."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.state import (
    ArrayStep,
    SampleStep,
    Segment,
    SubmoduleStep,
    at_last_token,
    last_sampled,
    position,
    positions,
    sample,
)
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine


class EncoderDecoderNode(SampledTokenDecoder):
    """A `SampledTokenDecoder` whose prefill walk also takes the encoder's
    output, ``encoder_states``, and whose decoder cross-attends to it.

    It adds the cross-attention part only: the prefill row carries the
    encoder output, each step lists the context array's segments (the
    encoder output's rows on the prefill that adds them, zero after; the
    commit turns that reservation into resident pages), and the decoder gets
    the encoder output and its absolute positions as forward arguments. The
    context is an array of its own, so the self-attention resource never
    plans over it. Only decode captures: a prefill's context follows its clip.

    A subclass names its self-attention array, its context array and the
    context's label, and ``MAX_PREFILL_BATCH_SIZE``, which bounds a prefill
    step's context reservation since no capture does. The decoder module
    provides ``embed(ids)``, ``logits(hidden)`` and
    ``forward(input_embeds, *, label, position_ids, encoder_states=None)``, which
    adds ``encoder_states`` to the context array before its layers read it.
    """

    KV_CACHE: str
    CROSS_KV_CACHE: str
    CONTEXT_LABEL: str
    CAPTURE_PREFILL = False

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.kv, self.sampler = handles[self.KV_CACHE], handles[self.SAMPLER]

    def node_step(
        self, segments: list[Segment], prefill_tokens: Mapping[int, torch.Tensor], advance: tuple[int, ...] | None,
    ) -> SubmoduleStep:
        return SubmoduleStep(
            segments=segments,
            state={self.KV_CACHE: ArrayStep(advance=advance), self.SAMPLER: SampleStep(prompt_tokens=prefill_tokens)},
        )

    def next_position(self, rid_handle: int) -> int:
        return position(self.kv, rid_handle, "main")

    def loopback_tokens(self, engine_inputs: ModelInputsFromEngine) -> torch.Tensor:
        return last_sampled(self.sampler)

    def last_token_hidden(self, engine_inputs: ModelInputsFromEngine, hidden: torch.Tensor) -> torch.Tensor:
        return at_last_token(self.kv, hidden)

    def sample_tokens(self, engine_inputs: ModelInputsFromEngine, logits: torch.Tensor) -> torch.Tensor:
        return sample(self.sampler, logits)

    def max_batch_size(self, graph_walk: str) -> int | None:
        if graph_walk == self.PREFILL_WALK:
            return self.MAX_PREFILL_BATCH_SIZE
        return super().max_batch_size(graph_walk)

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs: Any,
    ) -> ARNodeInputs:
        row = super().prepare_inputs(graph_walk, fwd_info, inputs, **kwargs)
        if graph_walk == self.PREFILL_WALK:
            row.tensor_inputs["encoder_states"] = inputs["encoder_states"][0]
        return row

    @staticmethod
    def _context_span(inp: ARNodeInputs) -> int:
        encoder_states = inp.tensor_inputs.get("encoder_states")
        return 0 if encoder_states is None else encoder_states.shape[0]

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        **kwargs: Any,
    ) -> SubmoduleStep:
        step = super().declare_step(graph_walk, request_ids, inputs, **kwargs)
        context = tuple(
            Segment(request_id=rid, label=self.CONTEXT_LABEL, span=self._context_span(inp))
            for rid, inp in zip(request_ids, inputs, strict=True)
        )
        return replace(step, state={**step.state, self.CROSS_KV_CACHE: ArrayStep(segments=context)})

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        out = super().preprocess(graph_walk, engine_inputs, inputs)
        if graph_walk == self.PREFILL_WALK:
            # segment order, which is how the context plan laid out the pages
            out["encoder_states"] = torch.cat([inp.tensor_inputs["encoder_states"] for inp in inputs], dim=0)
        return out

    def decoder_kwargs(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        num_tokens: int, **tensors: torch.Tensor,
    ) -> dict[str, Any]:
        # the array's planned position ids (a slot buffer when captured), and
        # on prefill the encoder output
        return {"position_ids": positions(self.kv)[:num_tokens], **tensors}
