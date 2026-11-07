"""Fused Sinkhorn-Knopp kernel for the mHC comb matrix."""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ModuleNotFoundError:  # no triton: the pure-torch reference covers this install
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _sinkhorn_kernel(
        m_ptr, num_iters: "tl.constexpr", H: "tl.constexpr", eps,
    ):
        """One program per ``[H, H]`` matrix; the whole matrix stays in registers for all
        iterations.
        """
        pid = tl.program_id(0)
        rows = tl.arange(0, H)
        cols = tl.arange(0, H)
        offs = pid * H * H + rows[:, None] * H + cols[None, :]
        m = tl.load(m_ptr + offs).to(tl.float32)  # [H, H]

        # ``+ eps`` (a Python float, fp64 in Triton) promotes the divide to fp64; casting back
        # matches the fp32 reference and keeps the loop-carried dtype fixed, as Triton requires.
        csum = tl.sum(m, axis=0)               # [H], per-column sum over rows
        m = (m / (csum[None, :] + eps)).to(tl.float32)
        for _ in range(num_iters - 1):
            rsum = tl.sum(m, axis=1)           # [H], per-row sum over cols
            m = (m / (rsum[:, None] + eps)).to(tl.float32)
            csum = tl.sum(m, axis=0)
            m = (m / (csum[None, :] + eps)).to(tl.float32)

        tl.store(m_ptr + offs, m)

    def sinkhorn_normalize_fused(
        matrix: torch.Tensor, num_iters: int, eps: float,
    ) -> torch.Tensor:
        """The whole Sinkhorn loop in one launch over ``[..., H, H]`` fp32 on CUDA, into a new
        tensor like the reference."""
        H = matrix.shape[-1]
        out = matrix.reshape(-1, H, H).contiguous().clone()
        n = out.shape[0]
        if n:
            _sinkhorn_kernel[(n,)](out, num_iters, H, eps)
        return out.reshape(matrix.shape)

else:

    def sinkhorn_normalize_fused(matrix, num_iters, eps):  # pragma: no cover
        raise RuntimeError("sinkhorn_normalize_fused needs triton (CUDA path only)")


def fused_sinkhorn_available(matrix: torch.Tensor) -> bool:
    """Fused path is usable: triton present, CUDA tensor, square power-of-two H."""
    if not (_HAS_TRITON and matrix.is_cuda):
        return False
    H = matrix.shape[-1]
    return matrix.shape[-2] == H and H > 0 and (H & (H - 1)) == 0
