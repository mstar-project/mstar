"""GLM-5.3-Flash MTP module — the layer-45 draft model (M2 wiring target)."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.glm5_next.components.decoder_layer import Glm5NextPlainDecoderLayer
from mstar.model.glm5_next.components.language_model import build_rmsnorm
from mstar.model.glm5_next.config import Glm5NextModelConfig

# Glue sub-keys under ``layers.45.`` that map 1:1 onto module attributes;
# the loader's remap keeps its own copy (it must import without this
# package's engine-touching modules) — ``weight_loader.MTP_GLUE_PREFIXES``
# is asserted equal here so the two cannot drift.
MTP_GLUE_PREFIXES = ("enorm", "hnorm", "eh_proj", "shared_head")


def _assert_loader_glue_contract() -> None:
    from mstar.model.glm5_next import weight_loader

    assert weight_loader.MTP_GLUE_PREFIXES == MTP_GLUE_PREFIXES, (
        "MTP glue prefixes drifted between components/mtp.py and "
        f"weight_loader.py: {MTP_GLUE_PREFIXES} vs "
        f"{weight_loader.MTP_GLUE_PREFIXES}"
    )


class Glm5NextSharedHead(nn.Module):
    """Named to mirror the checkpoint's ``shared_head.norm`` key: the MTP
    module's final norm. The actual head weight is the trunk's ``lm_head``
    (not duplicated in the checkpoint), applied by the caller."""

    def __init__(self, config: Glm5NextModelConfig) -> None:
        super().__init__()
        self.norm = build_rmsnorm(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class Glm5NextMTPModule(nn.Module):
    """One draft iteration: fuse (token embedding, previous hidden) and run
    the layer-45 plain-residual decoder layer.
    """

    def __init__(
        self, config: Glm5NextModelConfig, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        _assert_loader_glue_contract()
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
        dsa_ctx=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(head_input, raw_hidden)`` — see the module docstring."""
        if dsa_ctx is not None:
            raise NotImplementedError(
                "glm5_next MTP has no DSA engine path yet (identity regime "
                "only); pass dsa_ctx=None"
            )
        hidden_states = self.fuse(token_embeds, prev_hidden)
        hidden_states = self.transformer_layer(
            hidden_states
        )
        return self.shared_head(hidden_states), hidden_states
