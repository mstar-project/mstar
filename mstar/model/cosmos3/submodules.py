"""NodeSubmodule wrappers for the Cosmos3 generator nodes.

Two nodes:
  Cosmos3DiTSubmodule         -- dual-pathway DiT (KV_CACHE). Dispatches by
                                 graph_walk between ``prefill`` (the
                                 understanding tower writes the text-condition
                                 KV) and ``image_gen`` (one denoising step of
                                 the generation tower per loop iteration,
                                 attending to the frozen understanding KV plus
                                 the current generation tokens).
  Cosmos3VAEDecoderSubmodule  -- Wan VAE decode (STATELESS): final latents to
                                 pixels.

The compute bodies (patchify, timestep scatter, mRoPE, joint attention, Euler
step, VAE decode) are wired separately; these wrappers fix the node structure
and the engine-facing contract.
"""

from __future__ import annotations

import logging

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)


class Cosmos3DiTSubmodule(ARNodeSubmodule):
    """Dual-pathway DiT node (understanding tower + generation denoiser)."""

    def __init__(self, transformer, config):
        super().__init__()
        self.transformer = transformer
        self.config = config

    def get_needed_cache_labels(
        self, graph_walk: str, per_request_info: dict[str, CurrentForwardPassInfo],
    ) -> list[str] | None:
        # The understanding K/V lives under a single label that the generation
        # loop reads read-only across all denoise steps.
        return ["main"]

    def prepare_inputs(self, graph_walk, fwd_info, inputs, seen_token_mask, pos_info={}) -> ARNodeInputs:
        raise NotImplementedError("Cosmos3 DiT prepare_inputs not yet wired")

    def preprocess(self, graph_walk, engine_inputs: ModelInputsFromEngine, inputs) -> dict:
        raise NotImplementedError("Cosmos3 DiT preprocess not yet wired")

    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, **kwargs):
        raise NotImplementedError("Cosmos3 DiT forward not yet wired")


class Cosmos3VAEDecoderSubmodule(NodeSubmodule):
    """Wan VAE decode node: final denoised latents -> pixel frames."""

    def __init__(self, vae, config):
        super().__init__()
        self.vae = vae
        self.config = config

    def prepare_inputs(self, graph_walk, fwd_info, inputs, **kwargs) -> NodeInputs:
        raise NotImplementedError("Cosmos3 VAE prepare_inputs not yet wired")

    def forward(self, graph_walk, engine_inputs: ModelInputsFromEngine, **kwargs):
        raise NotImplementedError("Cosmos3 VAE forward not yet wired")
