"""Dense Qwen3's one node, on the shared `DeclaredTokenDecoder`. Under
``model_kwargs.draft`` every decode step is a verify step (speculative
decoding): the drafter in the node proposes ``k`` drafts and the target checks
them in the same forward, on the scaffold's verify path."""
from __future__ import annotations

from mstar.model.components.hosted_drafter import HostedDrafter
from mstar.model.components.sampled_token_decoder import DeclaredTokenDecoder


class Qwen3LMSubmodule(DeclaredTokenDecoder):
    """``model`` is a `Qwen3DenseLM` that attends through the node's ``kv_cache``
    array. ``drafter`` (a hosted drafter module) makes decode verify its drafts,
    k a step as the engine binds."""

    # kept for the narrower type: the node reads its hosted drafter's declaration (``needs.tap_key``)
    def __init__(self, model, config, drafter: HostedDrafter | None = None, accelerator_graph: bool = True):
        super().__init__(model, config, drafter=drafter, accelerator_graph=accelerator_graph)

    def resident_walks(self) -> frozenset[str]:
        # decode's preprocess reads only the loop-back tokens; its positions and KV
        # locations are planned by the resources. An uncaptured node has no resident
        # step, so a draft block's `mode: auto` verifies synchronously.
        return frozenset({self.DECODE_WALK}) if self.accelerator_graph else frozenset()
