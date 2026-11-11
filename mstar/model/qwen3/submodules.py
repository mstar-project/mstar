"""Dense Qwen3's one node, on the shared `SampledTokenDecoder`. Under
``model_kwargs.draft`` every decode step is a verify step (speculative
decoding): the drafter in the node proposes ``k`` drafts and the target checks
them in the same forward, on the scaffold's verify path."""
from __future__ import annotations

from typing import Any

from mstar.engine.resources.attn.config import AttentionStep
from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.step import ResourceStep
from mstar.engine.resources.taps.resource import step_taps
from mstar.model.components.hosted_drafter import HostedDrafter
from mstar.model.components.sampled_token_decoder import SampledTokenDecoder
from mstar.model.keys import KV_CACHE
from mstar.model.qwen3.config import ATTN


class Qwen3LMSubmodule(SampledTokenDecoder):
    """``model`` is a `Qwen3DenseLM`. ``drafter`` (a hosted drafter module) makes
    decode verify its drafts, k a step as the engine binds."""

    ATTN_KEY = ATTN

    def __init__(self, model, config, drafter: HostedDrafter | None = None, accelerator_graph: bool = True):
        super().__init__(model, config, drafter=drafter, accelerator_graph=accelerator_graph)

    def resident_walks(self) -> frozenset[str]:
        # decode's preprocess reads only the loop-back tokens; its positions and KV
        # locations are planned by the resources. An uncaptured node has no resident
        # step, so a draft block's `mode: auto` verifies synchronously.
        return frozenset({self.DECODE_WALK}) if self.accelerator_graph else frozenset()

    def layer_steps(self) -> dict[str, ResourceStep]:
        return {KV_CACHE: KVStep(), ATTN: AttentionStep(causal=True)}

    def decoder_kwargs(self, graph_walk, engine_inputs, num_tokens, **tensors) -> dict[str, Any]:
        # the drafter's taps: the target hands it the residual stream entering its layers
        tap_key = None if self.drafter is None else self.drafter.needs.tap_key
        return {} if tap_key is None else {"taps": step_taps(engine_inputs, tap_key)}

    def after_prompt_rows(self, engine_inputs, input_ids, hidden, embeds, tokens, decoder_kwargs, tensors) -> None:
        # the drafter's K/V of the prompt, from the taps the target just wrote
        self.drafter.prefill(engine_inputs, input_ids, tokens)
