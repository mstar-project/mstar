"""Block Attention Residuals as a module (spec B.6): holds the ``*_res_norm.weight`` and
``*_res_proj.weight`` parameters under their checkpoint names and performs one read.

The read mixes the residual stack ``blocks [T, m, H]`` and the running prefix ``[T, H]``.
This is the eager/reference implementation (fp32 online mixture); a fused Triton kernel
replaces ``attn_res_read`` later without changing the module contract.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.model.kimi_k3.reference.attn_res import attn_res_read


class AttnResRead(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        # names chosen so ``<prefix>_norm.weight`` / ``<prefix>_proj.weight`` map 1:1
        self.norm = nn.Module()
        self.norm.weight = nn.Parameter(torch.ones(hidden_size))
        self.proj = nn.Module()
        self.proj.weight = nn.Parameter(torch.zeros(1, hidden_size))
        self.eps = eps

    def score_weight(self) -> torch.Tensor:
        return self.norm.weight.float() * self.proj.weight.reshape(-1).float()

    def forward(self, prefix: torch.Tensor, blocks: torch.Tensor | None) -> torch.Tensor:
        if prefix.is_cuda and blocks is not None and blocks.shape[1] > 0:
            from mstar.model.kimi_k3.components.attn_res_kernel import attn_res_read_triton

            return attn_res_read_triton(prefix, blocks, self.score_weight(), self.eps)
        return attn_res_read(prefix, blocks, self.score_weight(), self.eps)
