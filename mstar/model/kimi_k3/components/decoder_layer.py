"""Kimi K3 decoder layer: KDA or gated MLA, then the dense or latent-MoE MLP, on block
attention residuals (``attn_res.py``)."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ParallelKimiDeltaAttention
from mstar.model.components.norm import RMSNorm
from mstar.model.kimi_k3.components.attention import KimiMLAAttention
from mstar.model.kimi_k3.components.attn_res import attn_res_read, push_block, score_weight
from mstar.model.kimi_k3.components.moe import build_mlp
from mstar.model.kimi_k3.config import KimiK3ModelConfig


class KimiK3DecoderLayer(nn.Module):
    def __init__(
        self, config: KimiK3ModelConfig, layer_idx: int, comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        hidden, eps = config.hidden_size, config.rms_norm_eps
        self.layer_idx = layer_idx
        self.eps = eps
        self.is_linear_attention = not config.is_full_attention_layer(layer_idx)
        if self.is_linear_attention:
            self.self_attn = ParallelKimiDeltaAttention(config.kda_config(), comm_group=comm_group)
            # this layer's blocks in the recurrent pool: its index among the KDA layers
            self.kda_pos = config.kda_layers.index(layer_idx)
        else:
            self.self_attn = KimiMLAAttention(
                config, comm_group=comm_group, layer_idx=layer_idx,
                kv_plane=config.full_attn_layers.index(layer_idx))
            self.kda_pos = None
        # the checkpoint's names: ``mlp`` on the dense layer, ``block_sparse_moe`` elsewhere
        if config.is_moe_layer(layer_idx):
            self.block_sparse_moe = build_mlp(config, layer_idx, comm_group)
        else:
            self.mlp = build_mlp(config, layer_idx, comm_group)
        self.input_layernorm = RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden, eps=eps)
        # the two AttnRes read sites; only their norms' weights are read
        self.self_attention_res_norm = RMSNorm(hidden, eps=eps)
        self.self_attention_res_proj = nn.Linear(hidden, 1, bias=False)
        self.mlp_res_norm = RMSNorm(hidden, eps=eps)
        self.mlp_res_proj = nn.Linear(hidden, 1, bias=False)
        self.block_start = config.is_attn_res_block_start(layer_idx)

    @property
    def ffn(self) -> nn.Module:
        return self.block_sparse_moe if hasattr(self, "block_sparse_moe") else self.mlp

    def forward(
        self,
        prefix: torch.Tensor,
        blocks: torch.Tensor | None,
        rows: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``prefix (T, hidden)``, the partial sum since the last block start, and the residual
        stack ``blocks (T, m, hidden)`` -> both after this layer. With ``rows``, only those
        tokens go on past the attention (the last layer feeds nothing but the sampled rows)."""
        x = attn_res_read(prefix, blocks, score_weight(
            self.self_attention_res_norm.weight, self.self_attention_res_proj.weight), self.eps)
        if self.block_start:
            blocks, prefix = push_block(blocks, prefix), None
        h = self.input_layernorm(x)
        if self.is_linear_attention:
            a = self.self_attn(h, self.kda_pos)
        else:
            a = self.self_attn(h)
        prefix = a if prefix is None else prefix + a
        if rows is not None:
            prefix = prefix.index_select(0, rows)
            blocks = None if blocks is None else blocks.index_select(0, rows)
        x = attn_res_read(prefix, blocks, score_weight(
            self.mlp_res_norm.weight, self.mlp_res_proj.weight), self.eps)
        return prefix + self.ffn(self.post_attention_layernorm(x)), blocks
