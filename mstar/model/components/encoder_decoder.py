"""The decoder node of an encoder-decoder speech model (Canary): the
encoder's output added once to a cross-attention context at prefill, then
one token per decode step."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig,
    PackedAcceleratorGraphConfig,
    geometric_batch_sizes,
)
from mstar.engine.state import (
    ArrayStep,
    Segment,
    SubmoduleStep,
    positions,
)
from mstar.model.components.sampled_token_decoder import DeclaredTokenDecoder
from mstar.model.submodule_base import ARNodeInputs, ModelInputsFromEngine


class EncoderDecoderNode(DeclaredTokenDecoder):
    """A `DeclaredTokenDecoder` whose prefill walk also takes the encoder's
    output, ``encoder_states``, and whose decoder cross-attends to it.

    It adds the cross-attention part only: the prefill row carries the
    encoder output, each step lists the context array's segments (the
    encoder output's rows on the prefill that adds them, zero after; the
    commit turns that reservation into resident pages), and the decoder gets
    the absolute positions as a forward argument. The context is an array of
    its own, so the self-attention resource never plans over it. The prefill
    adds the context eagerly in ``preprocess`` (`ArrayStep.eager_add`), so its
    graph is sized by the prompt alone: one packed capture per prefill batch
    size up to ``MAX_PREFILL_BATCH_SIZE``, of ``PREFILL_PROMPT_TOKENS`` a row.

    A subclass names its self-attention array, its context array and the
    context's label, ``PREFILL_PROMPT_TOKENS`` and ``MAX_PREFILL_BATCH_SIZE``,
    which bounds a prefill step's context reservation and its captures. The
    decoder module provides ``embed(ids)``, ``logits(hidden)``,
    ``add_context(encoder_states)``, which adds the encoder output to the
    context array, and ``forward(input_embeds, *, label, position_ids)``.
    """

    KV_CACHE: str
    CROSS_KV_CACHE: str
    CONTEXT_LABEL: str
    # the scaffold's text-prefill and mixed captures; this node captures its own prefill
    CAPTURE_PREFILL = False
    # tokens of a prefill row's prompt, which every task's prompt has
    PREFILL_PROMPT_TOKENS: int

    def __init__(self, model: torch.nn.Module, config: Any, max_context: int, **kwargs: Any):
        """``max_context``: the encoder frames of the longest clip, the context
        span of a prefill capture's template rows, so a capture runs at the
        longest context it replays (as Whisper's does)."""
        super().__init__(model, config, **kwargs)
        self.max_context = max_context

    def get_accelerator_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[AcceleratorGraphConfig]:
        dtype = next(self.model.parameters()).dtype

        def row(n: int) -> ARNodeInputs:
            # a template row's context is only measured (its span), so one broadcast row
            return ARNodeInputs(
                input_ids=torch.zeros(n, dtype=torch.long, device=device), input_seq_len=n,
                tensor_inputs={"encoder_states": torch.zeros(
                    1, self.config.hidden_size, dtype=dtype, device=device,
                ).expand(self.max_context, -1)},
            )

        # A replay's padding row: no tokens and no context, so `preprocess`
        # projects nothing for it. One object, as a batched config's padding is
        padding = ARNodeInputs(
            input_ids=torch.zeros(0, dtype=torch.long, device=device), input_seq_len=0,
            tensor_inputs={"encoder_states": torch.zeros(0, self.config.hidden_size, dtype=dtype, device=device)},
        )

        sizes = geometric_batch_sizes(self.MAX_PREFILL_BATCH_SIZE)
        return [
            *super().get_accelerator_graph_configs(device, tp_world_size),
            PackedAcceleratorGraphConfig(
                capture_graph_walk=self.PREFILL_WALK,
                capture_token_lengths={bs: [bs * self.PREFILL_PROMPT_TOKENS] for bs in sizes},
                make_node_input=lambda n: row(n) if n else padding,
                capture_batch_sizes=sizes,
                compile=False,
            ),
        ]

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
        return replace(step, state={
            **step.state,
            self.CROSS_KV_CACHE: ArrayStep(segments=context, eager_add=graph_walk == self.PREFILL_WALK),
        })

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        out = super().preprocess(graph_walk, engine_inputs, inputs)
        if graph_walk == self.PREFILL_WALK:
            # outside the graph; in segment order, which is how the context
            # plan laid out the pages
            self.model.add_context(torch.cat([inp.tensor_inputs["encoder_states"] for inp in inputs], dim=0))
        return out

    def decoder_kwargs(
        self, graph_walk: str, engine_inputs: ModelInputsFromEngine,
        num_tokens: int, **tensors: torch.Tensor,
    ) -> dict[str, Any]:
        # the array's planned position ids (a slot buffer when captured)
        return {"position_ids": positions(self.kv)[:num_tokens], **tensors}
