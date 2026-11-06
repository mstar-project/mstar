"""The decoder node of an encoder-decoder speech model (Canary): the
encoder's output added once to a cross-attention context at prefill, then
one token per decode step. The model supplies the decoder, its stop token and
its logit rules."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.state import (
    ArrayStep,
    SampleStep,
    Segment,
    SlotLease,
    SubmoduleStep,
    at_last_token,
    positions,
    sample,
)
from mstar.model.components.token_step import TokenStepNode
from mstar.model.submodule_base import ARNodeInputs, BatchedModelOutput, ModelInputsFromEngine
from mstar.utils.h2d import to_device_nonblocking


class EncoderDecoderNode(TokenStepNode):
    """Mixin for an ``ARNodeSubmodule`` over ``self.decoder``, which has
    ``add_context(encoder_states)``, ``embed(input_ids, position_ids)``,
    ``forward(input_embeds)`` and ``lm_head(hidden)``.

    Dispatches on graph_walk:
      - prefill: project ``encoder_states`` to per-layer cross-attention K/V
        and add them to the context array, fill the self-attention KV cache
        with the prompt, and sample the first token.
      - decode: embed the previous token, single-step decode.

    The context array grows by the encoder output at prefill and is declared
    zero-span (read-only) after. Only decode is captured. A subclass names its
    self-attention array, its context array, the context's label and its
    sampler (the encoder output gets an array of its own, so the self-attention
    resource never plans over it).
    """

    KV_CACHE: str
    CROSS_KV_CACHE: str
    CONTEXT_LABEL: str

    def bind_state(self, handles: Mapping[str, Any]) -> None:
        self.kv, self.sampler = handles[self.KV_CACHE], handles[self.SAMPLER]

    def process_logits(self, logits: torch.Tensor, graph_walk: str) -> torch.Tensor:
        """The model's rules on the logits before sampling; none by default."""
        return logits

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        device = self.get_device()
        token_ids = to_device_nonblocking(inputs["text_inputs"][0], device).reshape(-1)
        tensor_inputs = {}
        if graph_walk == self.PREFILL_WALK:
            tensor_inputs["encoder_states"] = inputs["encoder_states"][0].to(device)
        return ARNodeInputs(input_seq_len=token_ids.shape[0], input_ids=token_ids, tensor_inputs=tensor_inputs)

    @staticmethod
    def _context_span(inp: ARNodeInputs) -> int:
        """How far this step grows the request's encoder-context array: the
        encoder output on the prefill that adds it, zero after."""
        encoder_states = inp.tensor_inputs.get("encoder_states")
        return 0 if encoder_states is None else encoder_states.shape[0]

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        context_segments = tuple(
            Segment(request_id=rid, label=self.CONTEXT_LABEL, span=self._context_span(inp))
            for rid, inp in zip(request_ids, inputs, strict=True)
        )
        return SubmoduleStep(
            # the default, for the self-attention array
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            state={
                self.KV_CACHE: ArrayStep(),
                # real span on the prefill that adds the context, zero after;
                # the commit turns that reservation into resident pages
                self.CROSS_KV_CACHE: ArrayStep(segments=context_segments),
                self.SAMPLER: SampleStep(apply_penalty=False),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        preprocessed: dict[str, torch.Tensor | Any] = {"input_ids": torch.cat([inp.input_ids for inp in inputs])}
        if graph_walk == self.PREFILL_WALK:
            # segment order, which is how the context plan laid out the pages
            preprocessed["encoder_states"] = torch.cat(
                [inp.tensor_inputs["encoder_states"] for inp in inputs], dim=0,
            )
        return preprocessed

    def _forward(self, graph_walk: str, input_ids: torch.Tensor, encoder_states: torch.Tensor | None) -> torch.Tensor:
        if encoder_states is not None:
            # prefill only: fill the context pages this step reserved before
            # the layers read them
            self.decoder.add_context(encoder_states)
        hidden = self.decoder(self.decoder.embed(input_ids, positions(self.kv)))
        if graph_walk == self.PREFILL_WALK:
            # packed prefill: one hidden per request, at its last token
            hidden = at_last_token(self.kv, hidden)
        return sample(self.sampler, self.process_logits(self.decoder.lm_head(hidden), graph_walk))

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {"new_token": [self._forward(graph_walk, input_ids, encoder_states)]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        **kwargs,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(graph_walk, input_ids, encoder_states)
        # row i is request i: one clone and one D2H copy per step, not per request
        return BatchedModelOutput(row_outputs={"new_token": new_tokens}, check_stop_buffers={"new_token": new_tokens})
