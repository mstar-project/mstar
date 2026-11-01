"""The prompt encoder node of a DiT model: how its requests batch."""

from __future__ import annotations

from torch import nn

from mstar.model.submodule_base import NodeSubmodule


class PromptEncoderSubmodule(NodeSubmodule):
    """Base for a DiT model's ``text_encoder`` node. Prompts pad to one length, so any of
    them can share a step, up to ``max_batch_size``.

    A model passes its deployment's ``text_encoder_max_batch_size`` here. 1 runs every
    prompt alone while the DiT and VAE keep batching: a GEMM's rounding can depend on its
    row count, and a few-step sampler carries that into the image, so a prompt encoded in a
    batch gives another image than the same prompt alone. A batch of one is the path a lone
    request takes, which is why that setting leaves single-request images as they are.
    """

    disable_torch_compile = True

    def __init__(self, encoder: nn.Module, config, max_batch_size: int = 16):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self._max_batch_size = max_batch_size

    def can_batch(self, batch, model_inputs) -> bool:
        return len(model_inputs) > 1

    def max_batch_size(self, graph_walk: str):
        return self._max_batch_size
