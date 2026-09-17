"""GLM-5.3-Flash mHC: manifold-constrained hyper-connections."""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.glm5_next.sinkhorn_kernel import (
    fused_sinkhorn_available,
    sinkhorn_normalize_fused,
)

# GLM-5.3-Flash checkpoint constants (config.json). Layers 0-44 carry mHC
# params; the MTP layer 45 is plain-residual and never touches this file.
GLM5_NEXT_HC_MULT = 4
GLM5_NEXT_HC_EPS = 1e-6
GLM5_NEXT_HC_SINKHORN_ITERS = 20
GLM5_NEXT_RMS_NORM_EPS = 1e-5

# M3 Phase 2: fuse the mHC Sinkhorn (~78 launches/site -> 1) on CUDA. Set =0 to
# force the pure-torch reference (also the CPU path). sinkhorn_kernel guards its
# own triton import, so this file still imports on a CPU-only box (ground rule 4).
_FUSED_SINKHORN = os.environ.get("MSTAR_GLM53_FUSED_SINKHORN", "1") == "1"


class UnweightedRMSNorm(nn.Module):
    """RMSNorm with no learned weight, over the last (flattened 4*hidden) axis."""

    def __init__(self, eps: float = GLM5_NEXT_RMS_NORM_EPS) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


def sinkhorn_normalize(
    matrix: torch.Tensor, num_iters: int = GLM5_NEXT_HC_SINKHORN_ITERS,
    eps: float = GLM5_NEXT_HC_EPS,
) -> torch.Tensor:
    """Project a positive ``[..., H, H]`` matrix onto the doubly-stochastic manifold."""
    matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    for _ in range(num_iters - 1):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


def _dispatch_sinkhorn(
    matrix: torch.Tensor, num_iters: int, eps: float
) -> torch.Tensor:
    """CUDA -> fused kernel (one launch); CPU or disabled -> the reference above."""
    if _FUSED_SINKHORN and fused_sinkhorn_available(matrix):
        return sinkhorn_normalize_fused(matrix, num_iters, eps)
    return sinkhorn_normalize(matrix, num_iters, eps)


class Glm5NextHyperConnection(nn.Module):
    """One mHC site: learned collapse/placement/mixing weights from the live streams."""

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int = GLM5_NEXT_HC_MULT,
        hc_eps: float = GLM5_NEXT_HC_EPS,
        hc_sinkhorn_iters: int = GLM5_NEXT_HC_SINKHORN_ITERS,
        rms_norm_eps: float = GLM5_NEXT_RMS_NORM_EPS,
    ) -> None:
        super().__init__()
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.input_norm = UnweightedRMSNorm(eps=rms_norm_eps)
        mix = (2 + hc_mult) * hc_mult
        self.fn = nn.Parameter(torch.empty(mix, hc_mult * hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # One learned scale per mapping output: pre, post, comb.
        self.scale = nn.Parameter(torch.empty(3))
        # fp32 copy of ``fn`` built once by ``process_weights_after_loading``
        # (the Glm52MoEGate idiom). Plain attribute, not a buffer, so
        # ``model.to(bf16)`` cannot downcast it; ``forward`` falls back to a
        # per-call ``.float()`` (bit-identical) until it exists or after a
        # device move.
        self._fn_fp32: torch.Tensor | None = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """HF ``_init_weights`` for this module; checkpoint load overwrites."""
        self.fn.normal_(mean=0.0, std=0.02)
        self.base.zero_()
        self.scale.fill_(1.0)

    def finalize_weights(self) -> None:
        """Cache the fp32 ``fn``; call after the weights are loaded."""
        self._fn_fp32 = self.fn.detach().float()

    def process_weights_after_loading(self, device) -> None:
        """Post-load hook the generic module walk invokes (quantization
        ``process_weights_after_loading`` protocol)."""
        del device  # fn already carries the right device
        self.finalize_weights()

    def forward(
        self, hidden_streams: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        w = self._fn_fp32
        if w is None or w.device != self.fn.device:
            w = self.fn.float()
        pre_w, post_w, comb_w = F.linear(flat, w).split(
            [hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = _dispatch_sinkhorn(comb, self.hc_sinkhorn_iters, self.hc_eps)
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


def expand_streams(hidden_states: torch.Tensor, hc_mult: int = GLM5_NEXT_HC_MULT) -> torch.Tensor:
    """Model entry: replicate embeddings ``[B, S, D]`` into ``[B, S, hc_mult, D]``."""
    return hidden_states.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()


def update_streams(
    residual: torch.Tensor,
    sublayer_out: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Write a sublayer output back into the streams: place by ``post``, mix by ``comb``."""
    dtype = residual.dtype
    return post.to(dtype).unsqueeze(-1) * sublayer_out.unsqueeze(-2) + torch.matmul(
        comb.to(dtype).transpose(-1, -2), residual)


class Glm5NextHyperHead(nn.Module):
    """Model exit: collapse streams ``[B, S, H, D] -> [B, S, D]``."""

    def forward(self, hidden_streams: torch.Tensor) -> torch.Tensor:
        return hidden_streams.mean(dim=2)
