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
        """``res_norm.weight * res_proj.weight`` as one fp32 vector. Cached after the first
        call on CUDA (the weights are fixed once loaded): recomputing it was three launches
        per read, 186 reads per decode step."""
        if not self.training and self.norm.weight.is_cuda:
            cached = getattr(self, "_score_weight_cache", None)
            if cached is None or cached.device != self.norm.weight.device:
                cached = (self.norm.weight.float() * self.proj.weight.reshape(-1).float()).contiguous()
                self._score_weight_cache = cached
            return cached
        return self.norm.weight.float() * self.proj.weight.reshape(-1).float()

    def _apply(self, fn, recurse=True):
        self._score_weight_cache = None  # weights may move or change dtype
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self._score_weight_cache = None
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(
        self, prefix: torch.Tensor, blocks: torch.Tensor | None, out_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        """The read, optionally followed by ``out_norm`` (a ``KimiRMSNorm``): on CUDA the norm
        is folded into the read's mixing kernel (same arithmetic, one launch less)."""
        if prefix.is_cuda and prefix.dtype in (torch.bfloat16, torch.float16) and (
            out_norm is not None or (blocks is not None and blocks.shape[1] > 0)
        ):
            from mstar.model.kimi_k3.components.attn_res_kernel import attn_res_read_triton

            if out_norm is None:
                return attn_res_read_triton(prefix, blocks, self.score_weight(), self.eps)
            return attn_res_read_triton(
                prefix, blocks, self.score_weight(), self.eps,
                out_norm_weight=out_norm.weight, out_eps=out_norm.variance_epsilon,
            )
        x = attn_res_read(prefix, blocks, self.score_weight(), self.eps)
        return x if out_norm is None else out_norm(x)
