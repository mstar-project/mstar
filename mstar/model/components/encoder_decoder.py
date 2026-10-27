"""The decoder node of an encoder-decoder speech model (Canary): the
encoder's output written once into a cross-attention context at prefill, then
one token per decode step. The model supplies the decoder, its stop token and
its logit rules."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import AcceleratorGraphConfig, PackedAcceleratorGraphConfig
from mstar.engine.resources import AttentionStep, KVStep, PositionStep, SamplerStep, Segment, SlotLease, SubmoduleStep
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.position.manager import PositionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.components.token_step import TokenStepNode
from mstar.model.submodule_base import ARNodeInputs, BatchedModelOutput, ModelInputsFromEngine
from mstar.utils.h2d import to_device_nonblocking


class _FixedPromptConfig(PackedAcceleratorGraphConfig):
    """One packed bucket per batch size, of ``prompt_length`` tokens a row:
    every prompt has that length, so a bucket of any other density would
    never be chosen."""

    def __init__(self, prompt_length: int, **kwargs):
        super().__init__(**kwargs)
        self.prompt_length = prompt_length

    def get_total_tokens(self, bs: int) -> list[int]:
        return [n for n in super().get_total_tokens(bs) if n == self.prompt_length * bs]


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
    zero-span (read-only) after. A subclass binds the resource keys below to
    its model's (the encoder output gets a KV cache of its own, so the
    self-attention resource never plans over it).

    Decode is captured per batch size. Prefill writes the context in
    ``preprocess``, outside any graph (``KVStep.eager_write``), so when every
    prompt is ``PROMPT_LENGTH`` tokens, prefill is captured too, in packed
    buckets sized by the prompt alone.
    """

    PREFILL_WALK = "prefill"
    # the decoder prompt's length when it is the same for every request; None
    # leaves prefill eager
    PROMPT_LENGTH: int | None = None
    KV_CACHE: str
    ATTN: str
    POS: str
    CROSS_KV_CACHE: str
    CROSS_ATTN: str
    CONTEXT_LABEL: str

    def process_logits(self, logits: torch.Tensor, graph_walk: str) -> torch.Tensor:
        """The model's rules on the logits before sampling; none by default."""
        return logits

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        configs = super().get_accelerator_graph_configs(device, tp_world_size)
        if self.PROMPT_LENGTH is None:
            return configs
        cap = self.MAX_PREFILL_BATCH_SIZE
        batch_sizes = [bs for bs in self.DECODE_CAPTURE_BATCH_SIZES if cap is None or bs <= cap]
        prefill = _FixedPromptConfig(
            prompt_length=self.PROMPT_LENGTH,
            capture_graph_walk=self.PREFILL_WALK,
            # Packed, as Whisper's prefill is: a packed bucket is leased only after
            # `prepare_inputs`, so speculation never pre-plans template rows, which
            # can't have the clip's context. Template and padding rows carry none.
            make_node_input=lambda n: ARNodeInputs(
                input_ids=torch.zeros(n, dtype=torch.long, device=device), input_seq_len=n,
            ),
            capture_token_lengths=[self.PROMPT_LENGTH * bs for bs in batch_sizes],
            capture_batch_sizes=batch_sizes,
            # compiling it measured within run-to-run noise for Canary at c=64
            # (H100) and added over a minute to a cold start
            compile=False,
        )
        return [*configs, prefill]

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
                # the commit turns that reservation into resident pages. The
                # write is `preprocess`'s, outside any graph.
                self.CROSS_KV_CACHE: KVStep(
                    segments=context_segments, eager_write=graph_walk == self.PREFILL_WALK,
                ),
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
        if graph_walk == self.PREFILL_WALK:
            # Fill the context pages this step reserved, here rather than in the
            # forward: the context is as long as the clip, so a captured prefill
            # holds the prompt alone. Segment order, which is how the context
            # plan laid out the pages; a replay's padding rows have no context.
            states = [inp.tensor_inputs["encoder_states"] for inp in inputs if "encoder_states" in inp.tensor_inputs]
            if states:
                self.decoder.write_cross_kv(torch.cat(states, dim=0))
        return {"input_ids": torch.cat([inp.input_ids for inp in inputs])}

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        attn: AttentionManager = engine_inputs.resources[self.ATTN]
        pos: PositionManager = engine_inputs.resources[self.POS]
        sampler: SamplerResource = engine_inputs.resources[self.SAMPLER]
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
        **kwargs,
    ) -> NameToTensorList:
        return {"new_token": [self._forward(graph_walk, engine_inputs, input_ids)]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> BatchedModelOutput:
        new_tokens = self._forward(graph_walk, engine_inputs, input_ids)
        # row i is request i: one clone and one D2H copy per step, not per request
        return BatchedModelOutput(row_outputs={"new_token": new_tokens}, check_stop_buffers={"new_token": new_tokens})
