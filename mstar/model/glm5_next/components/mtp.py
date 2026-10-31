"""GLM-5.3-Flash MTP module: the layer-45 draft model."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.glm5_next.components.decoder_layer import Glm5NextPlainDecoderLayer
from mstar.model.glm5_next.components.language_model import build_rmsnorm
from mstar.model.glm5_next.config import Glm5NextModelConfig


class Glm5NextSharedHead(nn.Module):
    """The MTP module's final norm, named for the checkpoint's ``shared_head.norm``.
    The head itself is the trunk's ``lm_head``, which the caller applies."""

    def __init__(self, config: Glm5NextModelConfig) -> None:
        super().__init__()
        self.norm = build_rmsnorm(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class Glm5NextMTPModule(nn.Module):
    """One draft iteration: fuse (token embedding, previous hidden), then run the
    MTP decoder layer."""

    def __init__(
        self, config: Glm5NextModelConfig, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        self.kv_plane = len(config.full_attn_layer_indices)
        self.enorm = build_rmsnorm(config)
        self.hnorm = build_rmsnorm(config)
        self.eh_proj = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=False
        )
        self.transformer_layer = Glm5NextPlainDecoderLayer(
            config,
            layer_idx=config.num_hidden_layers,
            kv_plane=self.kv_plane,
            comm_group=comm_group,
        )
        self.shared_head = Glm5NextSharedHead(config)

    def fuse(
        self, token_embeds: torch.Tensor, prev_hidden: torch.Tensor
    ) -> torch.Tensor:
        return self.eh_proj(
            torch.cat([self.enorm(token_embeds), self.hnorm(prev_hidden)], dim=-1)
        )

    def forward(
        self,
        token_embeds: torch.Tensor,
        prev_hidden: torch.Tensor,
        row: int | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The shared-head norm output: the LM head's input, and also the next
        draft step's ``prev_hidden`` (vLLM's Glm5NextMTP recycles the normed
        state). ``row`` and ``latents`` run one query per request inside the
        step's block (``Glm5NextPlainDecoderLayer.forward``)."""
        hidden_states = self.fuse(token_embeds, prev_hidden)
        hidden_states = self.transformer_layer(hidden_states, row=row, latents=latents)
        return self.shared_head(hidden_states)
