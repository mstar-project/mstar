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
        BEFORE the final norm as ``(normed, prenorm)``. Both streams reach
        the MTP step because WHICH one the plane's learned ``hnorm`` pairs
        against is an open A/B — see ``Glm52LLMSubmodule._mtp_pair_rows``.
        (vLLM pairs post-norm on this checkpoint and scores higher. The
        0.00-acceptance number that once condemned post-norm here was
        re-attributed by 474a95e9 to the read plan never loading the MTP
        weights; do not cite it as pairing evidence.)"""
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

        loaded = load_glm52_hf_weights(
            self, weights, self.config.n_routed_experts,
            quant_config=self.config.quantization_config,
            fp8_experts=(
                self.config.quantization_config is not None
                and self.config.moe_fp8_resident
            ),
            num_hidden_layers=self.config.num_hidden_layers,
            load_mtp=self.mtp is not None,
        )
        if self.mtp is not None:
            missing = {
                f"mtp.{name}" for name, _ in self.mtp.named_parameters()
            } - loaded
            if missing:
                raise RuntimeError(
                    f"MTP drafting is on but {len(missing)} mtp.* "
                    f"parameters received no checkpoint weights (e.g. "
                    f"{sorted(missing)[:3]}) — the weight stream is "
                    "missing the MTP layer's keys, typically a read plan "
                    "built without load_mtp. Drafting from uninitialized "
                    "memory is silent 0.00 acceptance; refuse to serve."
                )
        return loaded
