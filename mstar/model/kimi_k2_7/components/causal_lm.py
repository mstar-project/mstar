"""Assembled Kimi-K2.7 text backbone."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.kimi_k2_7.components.decoder_layer import KimiDecoderLayer
from mstar.model.kimi_k2_7.components.language_model import (
    build_embedding,
    build_lm_head,
    build_rmsnorm,
)
from mstar.model.kimi_k2_7.config import KimiK2Config


class KimiLanguageModel(nn.Module):
    def __init__(
        self, config: KimiK2Config, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        self.embed_tokens = build_embedding(config, comm_group=comm_group)
        self.layers = nn.ModuleList(
            [
                KimiDecoderLayer(config, layer_idx, comm_group=comm_group)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = build_rmsnorm(config)

    def forward(self, input_ids: torch.Tensor, *, label: str) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        # The label and layer index are cursors on the shared resources: bind
        # the label once, advance the index per layer. Passing them as
        # arguments instead would make inductor specialize on the int.
        self.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, decoder_layer in enumerate(self.layers):
            decoder_layer.self_attn.attend.set_layer_idx(layer_idx)
            hidden_states = decoder_layer(hidden_states)
        # the advance is the runner's now, off the step declaration
        return self.norm(hidden_states)


class KimiForCausalLM(nn.Module):
    def __init__(
        self, config: KimiK2Config, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.model = KimiLanguageModel(config, comm_group=comm_group)
        self.lm_head = build_lm_head(config, comm_group=comm_group)

    def forward(
        self, input_ids: torch.Tensor, *, label: str, **kwargs,
    ) -> torch.Tensor:
        return self.lm_head(self.model(input_ids, label=label))

    def load_weights(self, weights, **kwargs) -> set[str]:
        from mstar.model.kimi_k2_7.weight_loader import load_kimi_hf_weights

        packed_experts = (
            self.config.quantization_config is not None
            and self.config.moe_in_kernel_dequant
        )
        return load_kimi_hf_weights(
            self, weights, self.config.n_routed_experts,
            quant_config=self.config.quantization_config,
            packed_experts=packed_experts,
        )
