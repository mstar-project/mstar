"""One Kimi K3 decoder layer with Block Attention Residual bookkeeping (spec B.5).

The layer transforms ``(prefix, blocks, pending)``: ``prefix [T, H]`` is the running intra-block
partial sum, ``blocks [T, m, H]`` the residual stack and ``pending [T, H]`` (or ``None``) an
addend of the prefix whose residual add is deferred into the next AttnRes read, where it costs
no launch of its own. Attention is either KDA or gated
MLA; the FFN is the dense SiTU MLP (layer 0) or the latent MoE.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.model.kimi_k3.components.attn_res import AttnResRead
from mstar.model.kimi_k3.components.common import KimiRMSNorm
from mstar.model.kimi_k3.reference.kda import KDAState


class KimiK3DecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        layer_idx: int,
        hidden_size: int,
        self_attn: nn.Module,
        mlp: nn.Module,
        is_kda: bool,
        is_moe: bool,
        attn_res_block_size: int | None,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_kda = is_kda
        self.is_moe = is_moe
        self.self_attn = self_attn
        if is_moe:
            self.block_sparse_moe = mlp
        else:
            self.mlp = mlp
        self.input_layernorm = KimiRMSNorm(hidden_size, eps=norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(hidden_size, eps=norm_eps)
        self.use_attn_res = attn_res_block_size is not None
        self.attn_res_block_size = attn_res_block_size
        if self.use_attn_res:
            self.self_attention_res = AttnResRead(hidden_size, eps=norm_eps)
            self.mlp_res = AttnResRead(hidden_size, eps=norm_eps)

    @property
    def ffn(self) -> nn.Module:
        return self.block_sparse_moe if self.is_moe else self.mlp

    def _pre_attn(self, prefix: torch.Tensor, blocks: torch.Tensor, pending: torch.Tensor | None):
        if self.use_attn_res:
            # the read applies the input norm itself and folds in the pending residual add
            x, prefix = self.self_attention_res.read(prefix, blocks, out_norm=self.input_layernorm, add=pending)
            if self.layer_idx % self.attn_res_block_size == 0:
                blocks = torch.cat([blocks, prefix.unsqueeze(1)], dim=1)
                prefix = None
        else:
            if pending is not None:
                prefix = prefix + pending
            x = self.input_layernorm(prefix)
        return x, prefix, blocks

    def _post_attn(self, prefix: torch.Tensor | None, blocks: torch.Tensor, a: torch.Tensor):
        if prefix is None:  # first layer of a residual block: the attention output starts the prefix
            prefix, a = a, None
        if self.use_attn_res:
            x, prefix = self.mlp_res.read(prefix, blocks, out_norm=self.post_attention_layernorm, add=a)
        else:
            if a is not None:
                prefix = prefix + a
            x = self.post_attention_layernorm(prefix)
        # the FFN's residual add is left pending for the next read (the next layer's, or the output's)
        return prefix, blocks, self.ffn(x)

    def forward(
        self, prefix: torch.Tensor, blocks: torch.Tensor, pending: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Paged path (attention reads its state from the bound resources).

        ``pending`` is the previous layer's FFN output whose residual add into ``prefix`` has
        not happened yet; it is folded into this layer's first read. Returns
        ``(prefix, blocks, pending)`` with this layer's FFN output pending in turn, so the
        caller's running prefix is ``prefix + pending``."""
        x, prefix, blocks = self._pre_attn(prefix, blocks, pending)
        a = self.self_attn(x)
        return self._post_attn(prefix, blocks, a)

    def forward_dense(
        self,
        prefix: torch.Tensor,
        blocks: torch.Tensor,
        state: KDAState | torch.Tensor | None,
        pending: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, KDAState | torch.Tensor, torch.Tensor]:
        """Explicit-state path for one sequence (tests / eager reference); same contract as
        :meth:`forward` plus the state."""
        x, prefix, blocks = self._pre_attn(prefix, blocks, pending)
        if self.is_kda:
            a, new_state = self.self_attn.forward_dense(x, state)
        else:
            a, latent_new = self.self_attn.forward_dense(x, state)
            new_state = latent_new if state is None else torch.cat([state, latent_new], 0)
        prefix, blocks, pending = self._post_attn(prefix, blocks, a)
        return prefix, blocks, new_state, pending
