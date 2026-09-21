"""Assembled GLM-5.2 text backbone."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.glm52.components.decoder_layer import Glm52DecoderLayer
from mstar.model.glm52.components.language_model import (
    build_embedding,
    build_lm_head,
    build_rmsnorm,
)
from mstar.model.glm52.components.rope import Glm52RotaryEmbedding
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
        # cos/sin for the whole forward are computed ONCE here and handed to
        # every layer (rope_cos_sin) — same dims/base as each layer's own
        # rotary, which stays the fallback for single-layer callers.
        self.rotary = Glm52RotaryEmbedding(rotary_dim=config.qk_rope_head_dim, base=config.rope_theta)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        dsa_ctx: Glm52DsaForwardContext | None = None,
        return_prenorm: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``dsa_ctx`` (engine DSA threading, None when dsa_long_context is
        off): layer order IS the IndexShare order — each FULL layer
        overwrites ``dsa_ctx.last_selection`` and the SHARED layers between
        it and the next FULL layer consume that value.
        """
        hidden_states = self.embed_tokens(input_ids)
        rope_cos_sin = self.rotary.cos_sin(position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states, position_ids, dsa_ctx=dsa_ctx, rope_cos_sin=rope_cos_sin,
            )
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
        # Build the draft module only when drafting is on, so the parameter
        # set and the weight load are unchanged when it is off.
        self.mtp = None
        if config.mtp_num_draft_tokens > 0:
            from mstar.model.glm52.components.mtp import Glm52MTPModule

            self.mtp = Glm52MTPModule(config, comm_group=comm_group)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, position_ids)
        return self.lm_head(hidden_states)

    def load_weights(self, weights, **kwargs) -> set[str]:
        from mstar.model.glm52.weight_loader import load_glm52_hf_weights

        # The indexer exists only on the DSA engine path (Glm52MLAAttention
        # builds none flag-off), so its keys are skipped before the dequant
        # stream rather than dequantized and dropped as unmatched.
        load_indexer = self.config.dsa_long_context
        loaded = load_glm52_hf_weights(
            self, weights, self.config.n_routed_experts,
            quant_config=self.config.quantization_config,
            fp8_experts=(
                self.config.quantization_config is not None
                and self.config.moe_fp8_resident
            ),
            num_hidden_layers=self.config.num_hidden_layers,
            load_indexer=load_indexer,
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
        if load_indexer:
            missing = {
                name for name, _ in self.named_parameters() if ".self_attn.indexer." in name
            } - loaded
            if missing:
                raise RuntimeError(
                    f"dsa_long_context is on but {len(missing)} indexer "
                    f"parameters received no checkpoint weights (e.g. "
                    f"{sorted(missing)[:3]}) — the weight stream is missing "
                    "the FULL-layer indexer keys. Selecting from uninitialized "
                    "memory is silent garbage; refuse to serve."
                )
        return loaded
