"""The decoder node of an encoder-decoder speech model (Whisper, Canary): the
encoder's output written once into a cross-attention context at prefill, then
one token per decode step. The model supplies the decoder, its stop token and
its logit rules."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.position.manager import PositionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.components.token_step import TokenStepNode
from mstar.model.submodule_base import ARNodeInputs, BatchedModelOutput, ModelInputsFromEngine
from mstar.utils.h2d import to_device_nonblocking


class EncoderDecoderNode(TokenStepNode):
    """Mixin for an ``ARNodeSubmodule`` over ``self.decoder``, which has
    ``write_cross_kv(encoder_states)``, ``embed(input_ids, position_ids)``,
    ``forward(input_embeds, *, label)`` and ``lm_head(hidden)``.

    Dispatches on graph_walk:
      - prefill: project ``encoder_states`` to per-layer cross-attention K/V
        and write them into the context stream, fill the self-attention KV
        cache with the prompt, and sample the first token.
      - decode: embed the previous token, single-step decode.

    The context stream grows by the encoder output at prefill and is declared
    zero-span (read-only) after. Only decode is captured. A subclass binds the
    resource keys below to its model's (the encoder output gets a KV cache of
    its own, so the self-attention resource never plans over it).
    """

    PREFILL_WALK = "prefill"
    KV_CACHE: str
    ATTN: str
    POS: str
    CROSS_KV_CACHE: str
    CROSS_ATTN: str
    CONTEXT_LABEL: str

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
        """How far this step grows the request's encoder-context stream: the
        encoder output on the prefill that writes it, zero after."""
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
            # the default, for the resources over the self-attention cache
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                self.KV_CACHE: KVStep(),
                self.ATTN: AttentionStep(causal=True),
                # real span on the prefill that writes the context, zero after;
                # the commit turns that reservation into resident pages
                self.CROSS_KV_CACHE: KVStep(segments=context_segments),
                self.CROSS_ATTN: AttentionStep(segments=context_segments, causal=False),
                self.SAMPLER: SamplerStep(apply_penalty=False),
                self.POS: PositionStep(),
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

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None,
    ) -> torch.Tensor:
        attn: AttentionManager = engine_inputs.resources[self.ATTN]
        pos: PositionManager = engine_inputs.resources[self.POS]
        sampler: SamplerResource = engine_inputs.resources[self.SAMPLER]
        if encoder_states is not None:
            # prefill only: fill the context pages this step reserved before
            # the layers read them
            self.decoder.write_cross_kv(encoder_states)
        hidden = self.decoder(self.decoder.embed(input_ids, pos.pos_ids("main")), label="main")
        if graph_walk == self.PREFILL_WALK:
            # packed prefill: one hidden per request, at its last token
            hidden = attn.select_last_hidden(hidden)
        logits = self.process_logits(self.decoder.lm_head(hidden), graph_walk)
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {"new_token": [self._forward(graph_walk, engine_inputs, input_ids, encoder_states)]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        encoder_states: torch.Tensor | None = None,
        **kwargs,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(graph_walk, engine_inputs, input_ids, encoder_states)
        # row i is request i: one clone and one D2H copy per step, not per request
        return BatchedModelOutput(row_outputs={"new_token": new_tokens}, check_stop_buffers={"new_token": new_tokens})
