"""Kimi K3 decoder layer: KDA or gated MLA, then the dense or latent-MoE MLP, on block
attention residuals (``attn_res.py``)."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ParallelKimiDeltaAttention
from mstar.model.components.norm import RMSNorm
from mstar.model.kimi_k3.components import attn_res
from mstar.model.kimi_k3.components.attention import KimiMLAAttention
from mstar.model.kimi_k3.components.attn_res import attn_res_read_norm
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
        # stack blocks the two reads see: the block starts before this layer, and this one's
        self.blocks_in = -(-layer_idx // config.attn_res_block_size)
        self.blocks_out = self.blocks_in + int(self.block_start)
        # the reads as FlashInfer's one launch where it serves (attn_res.attn_res_read_norm)
        self.attn_res_flashinfer = config.attn_res_backend == "auto"

    @property
    def ffn(self) -> nn.Module:
        return self.block_sparse_moe if hasattr(self, "block_sparse_moe") else self.mlp

    def forward(
        self,
        prefix: torch.Tensor,
        pending: torch.Tensor | None,
        stack: torch.Tensor,
        rows: torch.Tensor | None = None,
        token_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The stream in, ``prefix + pending`` (``(T, hidden)``; the partial sum since the
        last block start, its last addend not yet added: the first read adds it), and the
        residual stack ``stack (T, num_blocks, hidden)`` (token-major) with its first ``blocks_in``
        blocks filled -> ``(prefix, pending, stack)`` after this layer, ``pending`` its MLP's output.
        A block start writes its stream into ``stack[:, blocks_in]``. With ``rows``, only those
        tokens go on past the attention (the last layer feeds nothing but the sampled rows),
        and the stack returned holds only them."""
        h, prefix = self.read_attn(prefix, pending, stack)
        if self.is_linear_attention:
            a = self.self_attn(h, self.kda_pos)
        else:
            a = self.self_attn(h)
        h, prefix, stack = self.read_mlp(prefix, a, stack, rows)
        if token_valid is None or not hasattr(self, "block_sparse_moe"):
            return prefix, self.ffn(h), stack
        if rows is not None:
            token_valid = token_valid.index_select(0, rows)
        # ``token_valid``: a padded step's rows that are real; the routed experts skip the others
        return prefix, self.block_sparse_moe(h, token_valid=token_valid), stack

    # The tensor work around the attention, which the submodule may compile as regions
    # (``compile_regions``); the attention and its resources stay outside them.

    def read_attn(
        self, prefix: torch.Tensor, pending: torch.Tensor | None, stack: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The attention's normed input (the first read, adding ``pending``), and the stream
        after it: None at a block start, which writes it into ``stack[:, blocks_in]``."""
        x, prefix = attn_res_read_norm(
            stack, self.blocks_in, prefix, self.self_attention_res_norm.weight,
            self.self_attention_res_proj.weight, self.eps, self.input_layernorm, addend=pending,
            write=self.blocks_in if self.block_start else -1, flashinfer=self.attn_res_flashinfer)
        return x, None if self.block_start else prefix

    def read_mlp(
        self, prefix: torch.Tensor | None, a: torch.Tensor, stack: torch.Tensor,
        rows: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The MLP's normed input (the second read, adding the attention's output ``a``), the
        stream after it, and the stack, cut to ``rows`` when given."""
        if rows is not None:
            prefix = (a if prefix is None else prefix + a).index_select(0, rows)
            stack = stack[:, :self.blocks_out].index_select(0, rows)
            a = None
        elif prefix is None:
            # a block start: the partial sum restarts at the attention's output. FlashInfer's read adds
            # the next addend into the stream in place, so there the stream is a tensor this forward owns
            prefix, a = (a.clone() if self.attn_res_flashinfer and attn_res.fi_serves(stack, a, ()) else a), None
        x, prefix = attn_res_read_norm(
            stack, self.blocks_out, prefix, self.mlp_res_norm.weight, self.mlp_res_proj.weight,
            self.eps, self.post_attention_layernorm, addend=a, flashinfer=self.attn_res_flashinfer)
        return x, prefix, stack
