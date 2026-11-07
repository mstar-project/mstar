"""Assembled Kimi K3 text backbone: embedding, 93 hybrid layers on attention residuals, the
output read, final norm and LM head."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ColumnParallelLinear, VocabParallelEmbedding
from mstar.model.components.norm import RMSNorm
from mstar.model.kimi_k3.components.attn_res import attn_res_read, score_weight
from mstar.model.kimi_k3.components.decoder_layer import KimiK3DecoderLayer
from mstar.model.kimi_k3.config import KimiK3ModelConfig


class KimiK3LanguageModel(nn.Module):
    def __init__(self, config: KimiK3ModelConfig, comm_group: CommGroup | None = None) -> None:
        super().__init__()
        self.eps = config.rms_norm_eps
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.hidden_size,
            comm_group=comm_group, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([
            KimiK3DecoderLayer(config, idx, comm_group=comm_group)
            for idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_attn_res_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_attn_res_proj = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, input_ids: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        """``input_ids (T,)`` flat batch -> final normed hidden ``(T, hidden)``, or only its
        ``rows``."""
        prefix = self.embed_tokens(input_ids)
        blocks = None
        last = len(self.layers) - 1
        for idx, layer in enumerate(self.layers):
            prefix, blocks = layer(prefix, blocks, rows=rows if idx == last else None)
        out = attn_res_read(prefix, blocks, score_weight(
            self.output_attn_res_norm.weight, self.output_attn_res_proj.weight), self.eps)
        return self.norm(out)


class KimiK3ForCausalLM(nn.Module):
    """The ``LLM`` node's weights root: ``model.*`` and ``lm_head`` under the checkpoint's
    ``language_model.`` prefix."""

    def __init__(self, config: KimiK3ModelConfig, comm_group: CommGroup | None = None) -> None:
        super().__init__()
        self.config = config
        self.model = KimiK3LanguageModel(config, comm_group=comm_group)
        self.lm_head = ColumnParallelLinear(
            comm_group or CommGroup.trivial(), input_size=config.hidden_size,
            output_size=config.vocab_size, bias=False, gather_output=True)

    def forward(self, input_ids: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        return self.lm_head(self.model(input_ids, rows=rows))

    def kda_conv_dtype(self) -> torch.dtype:
        """The KDA projections' dtype, which the pool's conv tail must match."""
        for layer in self.model.layers:
            if layer.is_linear_attention:
                return layer.self_attn.q_proj.weight.dtype
        raise RuntimeError("no KDA layer in the schedule")  # unreachable: the config refuses it

    def load_weights(self, weights) -> set[str]:
        from mstar.model.kimi_k3.weight_loader import load_kimi_k3_weights

        return load_kimi_k3_weights(self, weights, self.config)
