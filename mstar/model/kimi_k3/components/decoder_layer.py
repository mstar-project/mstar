"""One Kimi K3 decoder layer with Block Attention Residual bookkeeping (spec B.5).

The layer transforms ``(prefix, blocks)``: ``prefix [T, H]`` is the running intra-block
partial sum and ``blocks [T, m, H]`` the residual stack. Attention is either KDA or gated
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

    def _pre_attn(self, prefix: torch.Tensor, blocks: torch.Tensor):
        if self.use_attn_res:
            x = self.self_attention_res(prefix, blocks)
            if self.layer_idx % self.attn_res_block_size == 0:
                blocks = torch.cat([blocks, prefix.unsqueeze(1)], dim=1)
                prefix = None
        else:
            x = prefix
        return self.input_layernorm(x), prefix, blocks

    def _post_attn(self, prefix: torch.Tensor | None, blocks: torch.Tensor, a: torch.Tensor):
        prefix = a if prefix is None else prefix + a
        x = self.mlp_res(prefix, blocks) if self.use_attn_res else prefix
        f = self.ffn(self.post_attention_layernorm(x))
        return prefix + f, blocks

    def forward(self, prefix: torch.Tensor, blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Paged path (attention reads its state from the bound resources)."""
        x, prefix, blocks = self._pre_attn(prefix, blocks)
        a = self.self_attn(x)
        return self._post_attn(prefix, blocks, a)

    def forward_dense(
        self, prefix: torch.Tensor, blocks: torch.Tensor, state: KDAState | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, KDAState | torch.Tensor]:
        """Explicit-state path for one sequence (tests / eager reference)."""
        x, prefix, blocks = self._pre_attn(prefix, blocks)
        if self.is_kda:
            a, new_state = self.self_attn.forward_dense(x, state)
        else:
            a, latent_new = self.self_attn.forward_dense(x, state)
            new_state = latent_new if state is None else torch.cat([state, latent_new], 0)
        prefix, blocks = self._post_attn(prefix, blocks, a)
        return prefix, blocks, new_state
