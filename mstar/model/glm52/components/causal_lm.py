"""Assembled GLM-5.2 text backbone."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.glm52.components.decoder_layer import Glm52DecoderLayer
from mstar.model.glm52.components.language_model import (
    build_embedding,
    build_lm_head,
    build_rmsnorm,
)
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.dsa import Glm52DsaForwardContext


class Glm52LanguageModel(nn.Module):
    def __init__(
        self, config: Glm52ModelConfig, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        self.embed_tokens = build_embedding(config, comm_group=comm_group)
        self.layers = nn.ModuleList(
            [
                Glm52DecoderLayer(config, layer_idx, comm_group=comm_group)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = build_rmsnorm(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
        dsa_ctx: Glm52DsaForwardContext | None = None,
        return_prenorm: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``dsa_ctx`` (engine DSA threading, None when dsa_long_context is
        off): layer order IS the IndexShare order — each FULL layer
        overwrites ``dsa_ctx.last_selection`` and the SHARED layers between
        it and the next FULL layer consume that value.

        ``return_prenorm``: additionally return the raw last-layer hidden
        BEFORE the final norm as ``(normed, prenorm)``. The MTP plane pairs
        drafts against this raw stream — its ``hnorm`` is a LEARNED norm
        over the trunk's un-normalized hidden (DeepSeek-V3 MTP glue);
        feeding it the final-norm output double-norms the fusion input and
        was measured to zero out draft acceptance (0.00 over 3584
        request-steps, 2026-08-09 bench)."""
        hidden_states = self.embed_tokens(input_ids)
        for layer_idx, decoder_layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            hidden_states = decoder_layer(
                hidden_states, cache_handle, position_ids, dsa_ctx=dsa_ctx
            )
        cache_handle.advance_seq_lens()
        normed = self.norm(hidden_states)
        if return_prenorm:
            return normed, hidden_states
        return normed


class Glm52ForCausalLM(nn.Module):
    def __init__(
        self, config: Glm52ModelConfig, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Glm52LanguageModel(config, comm_group=comm_group)
        self.lm_head = build_lm_head(config, comm_group=comm_group)
        # M3: the layer-78 draft module exists only when drafting is on, so
        # flag-off keeps the parameter set (and load) byte-identical to M1.
        self.mtp = None
        if config.mtp_num_draft_tokens > 0:
            from mstar.model.glm52.components.mtp import Glm52MTPModule

            self.mtp = Glm52MTPModule(config, comm_group=comm_group)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, cache_handle, position_ids)
        return self.lm_head(hidden_states)

    def load_weights(self, weights, **kwargs) -> set[str]:
        from mstar.model.glm52.weight_loader import load_glm52_hf_weights

        return load_glm52_hf_weights(
            self, weights, self.config.n_routed_experts,
            quant_config=self.config.quantization_config,
            fp8_experts=(
                self.config.quantization_config is not None
                and self.config.moe_fp8_resident
            ),
            num_hidden_layers=self.config.num_hidden_layers,
            load_mtp=self.mtp is not None,
        )
